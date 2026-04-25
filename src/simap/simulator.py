from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from openap import aero

from .config import AircraftConfig, mode_for_s
from .lateral_dynamics import (
    LateralGuidanceConfig,
    compute_lateral_command,
    lateral_rates,
    wrap_angle_rad,
)
from .coupled_descent_planner import CoupledDescentPlanResult
from .openap_adapter import openap_dT
from .path_geometry import ReferencePath
from .weather import ConstantWeather, WeatherProvider


@dataclass(frozen=True)
class State:
    t_s: float
    s_m: float
    h_m: float
    v_tas_mps: float
    east_m: float
    north_m: float
    psi_rad: float
    phi_rad: float = 0.0

    @classmethod
    def on_reference_path(
        cls,
        *,
        t_s: float,
        s_m: float,
        h_m: float,
        v_tas_mps: float,
        reference_path: ReferencePath,
        bank_rad: float = 0.0,
        heading_offset_rad: float = 0.0,
        cross_track_m: float = 0.0,
    ) -> "State":
        east_m, north_m = reference_path.position_ne(s_m)
        normal_hat = reference_path.normal_hat(s_m)
        psi_rad = wrap_angle_rad(reference_path.track_angle_rad(s_m) + heading_offset_rad)
        return cls(
            t_s=t_s,
            s_m=s_m,
            h_m=h_m,
            v_tas_mps=v_tas_mps,
            east_m=east_m + cross_track_m * normal_hat[0],
            north_m=north_m + cross_track_m * normal_hat[1],
            psi_rad=psi_rad,
            phi_rad=bank_rad,
        )


@dataclass(frozen=True)
class CoupledDescentPlanSample:
    s_m: float
    h_m: float
    v_tas_mps: float
    v_cas_mps: float
    gamma_rad: float
    thrust_n: float
    time_from_threshold_s: float
    elapsed_from_tod_s: float


@dataclass(frozen=True)
class CoupledDescentPlanProfile:
    plan: CoupledDescentPlanResult

    def __post_init__(self) -> None:
        if len(self.plan) < 2:
            raise ValueError("descent plan requires at least two samples")
        if np.any(np.diff(self.plan.s_m) <= 0.0):
            raise ValueError("descent plan s_m must be strictly increasing")
        if np.any(np.diff(self.plan.t_s) < 0.0):
            raise ValueError("descent plan t_s must be nondecreasing")

    @property
    def start_s_m(self) -> float:
        return float(self.plan.s_m[-1])

    @property
    def duration_s(self) -> float:
        return float(self.plan.t_s[-1])

    def sample(self, s_m: float) -> CoupledDescentPlanSample:
        s_val = float(np.clip(s_m, float(self.plan.s_m[0]), float(self.plan.s_m[-1])))
        time_from_threshold_s = float(np.interp(s_val, self.plan.s_m, self.plan.t_s))
        return CoupledDescentPlanSample(
            s_m=s_val,
            h_m=float(np.interp(s_val, self.plan.s_m, self.plan.h_m)),
            v_tas_mps=float(np.interp(s_val, self.plan.s_m, self.plan.v_tas_mps)),
            v_cas_mps=float(np.interp(s_val, self.plan.s_m, self.plan.v_cas_mps)),
            gamma_rad=float(np.interp(s_val, self.plan.s_m, self.plan.gamma_rad)),
            thrust_n=float(np.interp(s_val, self.plan.s_m, self.plan.thrust_n)),
            time_from_threshold_s=time_from_threshold_s,
            elapsed_from_tod_s=float(self.duration_s - time_from_threshold_s),
        )


@dataclass(frozen=True)
class SimulationRequest:
    cfg: AircraftConfig
    plan: CoupledDescentPlanResult
    reference_path: ReferencePath
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    guidance: LateralGuidanceConfig = field(default_factory=LateralGuidanceConfig)
    dt_s: float = 1.0
    max_time_factor: float = 3.0
    threshold_tolerance_m: float = 1.0
    initial_state: State | None = None

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.max_time_factor <= 0.0:
            raise ValueError("max_time_factor must be positive")
        if self.threshold_tolerance_m < 0.0:
            raise ValueError("threshold_tolerance_m must be nonnegative")
        if self.reference_path.total_length_m + 1e-9 < float(self.plan.tod_m):
            raise ValueError("reference path must extend at least to the planned top of descent")
        if np.any(np.diff(self.plan.s_m) <= 0.0):
            raise ValueError("simulation requires a strictly increasing longitudinal s grid")
        if self.initial_state is not None:
            if not 0.0 <= self.initial_state.s_m <= float(self.plan.tod_m):
                raise ValueError("initial state s_m must lie on the planned interval")


@dataclass(frozen=True)
class SimulationResult:
    t_s: np.ndarray
    s_m: np.ndarray
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    v_cas_mps: np.ndarray
    gamma_rad: np.ndarray
    thrust_n: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    psi_rad: np.ndarray
    phi_rad: np.ndarray
    ground_track_rad: np.ndarray
    ground_speed_mps: np.ndarray
    alongtrack_speed_mps: np.ndarray
    cross_track_m: np.ndarray
    track_error_rad: np.ndarray
    curvature_cmd_inv_m: np.ndarray
    phi_req_rad: np.ndarray
    phi_max_rad: np.ndarray
    mode: tuple[str, ...]
    success: bool
    message: str
    max_abs_cross_track_m: float
    max_abs_track_error_rad: float
    min_alongtrack_speed_mps: float
    max_bank_command_ratio: float
    final_threshold_error_m: float

    def __len__(self) -> int:
        return int(len(self.t_s))

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t_s": self.t_s,
                "s_m": self.s_m,
                "h_m": self.h_m,
                "v_tas_mps": self.v_tas_mps,
                "v_cas_mps": self.v_cas_mps,
                "gamma_rad": self.gamma_rad,
                "thrust_n": self.thrust_n,
                "east_m": self.east_m,
                "north_m": self.north_m,
                "lat_deg": self.lat_deg,
                "lon_deg": self.lon_deg,
                "psi_rad": self.psi_rad,
                "phi_rad": self.phi_rad,
                "ground_track_rad": self.ground_track_rad,
                "ground_speed_mps": self.ground_speed_mps,
                "alongtrack_speed_mps": self.alongtrack_speed_mps,
                "cross_track_m": self.cross_track_m,
                "track_error_rad": self.track_error_rad,
                "curvature_cmd_inv_m": self.curvature_cmd_inv_m,
                "phi_req_rad": self.phi_req_rad,
                "phi_max_rad": self.phi_max_rad,
                "mode": np.asarray(self.mode, dtype=object),
            }
        )


def _cas_from_tas(
    *,
    weather: WeatherProvider,
    s_m: float,
    h_m: float,
    t_s: float,
    v_tas_mps: float,
) -> float:
    delta_isa_K = float(weather.delta_isa_K(s_m, h_m, t_s))
    return float(aero.tas2cas(v_tas_mps, h_m, dT=openap_dT(delta_isa_K)))


def _initial_state(request: SimulationRequest, profile: CoupledDescentPlanProfile) -> State:
    if request.initial_state is None:
        sample = profile.sample(profile.start_s_m)
        if all(hasattr(profile.plan, field_name) for field_name in ("east_m", "north_m", "psi_rad", "phi_rad")):
            return State(
                t_s=0.0,
                s_m=sample.s_m,
                h_m=sample.h_m,
                v_tas_mps=sample.v_tas_mps,
                east_m=float(profile.plan.east_m[-1]),
                north_m=float(profile.plan.north_m[-1]),
                psi_rad=float(profile.plan.psi_rad[-1]),
                phi_rad=float(profile.plan.phi_rad[-1]),
            )
        return State.on_reference_path(
            t_s=0.0,
            s_m=sample.s_m,
            h_m=sample.h_m,
            v_tas_mps=sample.v_tas_mps,
            reference_path=request.reference_path,
        )

    sample = profile.sample(request.initial_state.s_m)
    return State(
        t_s=float(request.initial_state.t_s),
        s_m=float(request.initial_state.s_m),
        h_m=sample.h_m,
        v_tas_mps=sample.v_tas_mps,
        east_m=float(request.initial_state.east_m),
        north_m=float(request.initial_state.north_m),
        psi_rad=float(request.initial_state.psi_rad),
        phi_rad=float(request.initial_state.phi_rad),
    )


def simulate_plan(request: SimulationRequest) -> SimulationResult:
    """Replay a descent plan through the lateral simulation model.

    The coupled planner is authoritative over the optimized node profile. This
    replay helper advances a time-stepped lateral model while resampling
    altitude, speed, gamma, and thrust from the planned profile at the current
    along-track position.
    """

    profile = CoupledDescentPlanProfile(request.plan)
    state = _initial_state(request, profile)
    max_time_s = max(float(request.dt_s), profile.duration_s * float(request.max_time_factor))

    t_s: list[float] = []
    s_m: list[float] = []
    h_m: list[float] = []
    v_tas_mps: list[float] = []
    v_cas_mps: list[float] = []
    gamma_rad: list[float] = []
    thrust_n: list[float] = []
    east_m: list[float] = []
    north_m: list[float] = []
    psi_rad: list[float] = []
    phi_rad: list[float] = []
    ground_track_rad: list[float] = []
    ground_speed_mps: list[float] = []
    alongtrack_speed_mps: list[float] = []
    cross_track_m: list[float] = []
    track_error_rad: list[float] = []
    curvature_cmd_inv_m: list[float] = []
    phi_req_rad: list[float] = []
    phi_max_rad: list[float] = []
    mode_names: list[str] = []

    success = False
    message = f"simulation exceeded {max_time_s:.1f} s before reaching the threshold"

    while True:
        sample = profile.sample(state.s_m)
        mode = mode_for_s(request.cfg, state.s_m)
        command = compute_lateral_command(
            s_m=state.s_m,
            east_m=state.east_m,
            north_m=state.north_m,
            h_m=state.h_m,
            t_s=state.t_s,
            psi_rad=state.psi_rad,
            v_tas_mps=state.v_tas_mps,
            cfg=request.cfg,
            mode=mode,
            reference_path=request.reference_path,
            weather=request.weather,
            guidance=request.guidance,
        )

        t_s.append(float(state.t_s))
        s_m.append(float(state.s_m))
        h_m.append(float(state.h_m))
        v_tas_mps.append(float(state.v_tas_mps))
        v_cas_mps.append(
            _cas_from_tas(
                weather=request.weather,
                s_m=state.s_m,
                h_m=state.h_m,
                t_s=state.t_s,
                v_tas_mps=state.v_tas_mps,
            )
        )
        gamma_rad.append(float(sample.gamma_rad))
        thrust_n.append(float(sample.thrust_n))
        east_m.append(float(state.east_m))
        north_m.append(float(state.north_m))
        psi_rad.append(float(state.psi_rad))
        phi_rad.append(float(state.phi_rad))
        ground_track_rad.append(float(command.ground_track_rad))
        ground_speed_mps.append(float(command.ground_speed_mps))
        alongtrack_speed_mps.append(float(command.alongtrack_speed_mps))
        cross_track_m.append(float(command.cross_track_m))
        track_error_rad.append(float(command.track_error_rad))
        curvature_cmd_inv_m.append(float(command.curvature_cmd_inv_m))
        phi_req_rad.append(float(command.phi_req_rad))
        phi_max_rad.append(float(command.phi_max_rad))
        mode_names.append(mode.name)

        if state.s_m <= request.threshold_tolerance_m:
            success = True
            message = "reached the threshold"
            break
        if state.t_s >= max_time_s:
            break

        step_dt_s = float(min(request.dt_s, max_time_s - state.t_s))
        if command.alongtrack_speed_mps > 1e-9:
            step_dt_s = float(min(step_dt_s, state.s_m / command.alongtrack_speed_mps))
        if step_dt_s <= 0.0:
            break

        psi_dot_rps, phi_dot_rps = lateral_rates(
            phi_rad=state.phi_rad,
            phi_req_rad=command.phi_req_rad,
            tau_phi_s=mode.tau_phi_s,
            p_max_rps=mode.p_max_rps,
            v_tas_mps=state.v_tas_mps,
        )
        next_phi_rad = float(state.phi_rad + phi_dot_rps * step_dt_s)
        if (command.phi_req_rad - state.phi_rad) * (command.phi_req_rad - next_phi_rad) < 0.0:
            next_phi_rad = float(command.phi_req_rad)
        next_state_s_m = float(max(0.0, state.s_m - command.alongtrack_speed_mps * step_dt_s))
        next_sample = profile.sample(next_state_s_m)
        state = State(
            t_s=float(state.t_s + step_dt_s),
            s_m=next_state_s_m,
            h_m=float(next_sample.h_m),
            v_tas_mps=float(next_sample.v_tas_mps),
            east_m=float(state.east_m + command.east_dot_mps * step_dt_s),
            north_m=float(state.north_m + command.north_dot_mps * step_dt_s),
            psi_rad=wrap_angle_rad(state.psi_rad + psi_dot_rps * step_dt_s),
            phi_rad=next_phi_rad,
        )

    east_arr = np.asarray(east_m, dtype=float)
    north_arr = np.asarray(north_m, dtype=float)
    latlon_arr = np.asarray(
        [
            request.reference_path.latlon_from_ne(float(east_val), float(north_val))
            for east_val, north_val in zip(east_arr, north_arr, strict=True)
        ],
        dtype=float,
    )
    threshold_east_m, threshold_north_m = request.reference_path.position_ne(0.0)
    phi_max_arr = np.asarray(phi_max_rad, dtype=float)
    phi_req_arr = np.asarray(phi_req_rad, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bank_ratio = np.divide(np.abs(phi_req_arr), phi_max_arr, out=np.zeros_like(phi_req_arr), where=phi_max_arr > 0.0)

    return SimulationResult(
        t_s=np.asarray(t_s, dtype=float),
        s_m=np.asarray(s_m, dtype=float),
        h_m=np.asarray(h_m, dtype=float),
        v_tas_mps=np.asarray(v_tas_mps, dtype=float),
        v_cas_mps=np.asarray(v_cas_mps, dtype=float),
        gamma_rad=np.asarray(gamma_rad, dtype=float),
        thrust_n=np.asarray(thrust_n, dtype=float),
        east_m=east_arr,
        north_m=north_arr,
        lat_deg=latlon_arr[:, 0],
        lon_deg=latlon_arr[:, 1],
        psi_rad=np.asarray(psi_rad, dtype=float),
        phi_rad=np.asarray(phi_rad, dtype=float),
        ground_track_rad=np.asarray(ground_track_rad, dtype=float),
        ground_speed_mps=np.asarray(ground_speed_mps, dtype=float),
        alongtrack_speed_mps=np.asarray(alongtrack_speed_mps, dtype=float),
        cross_track_m=np.asarray(cross_track_m, dtype=float),
        track_error_rad=np.asarray(track_error_rad, dtype=float),
        curvature_cmd_inv_m=np.asarray(curvature_cmd_inv_m, dtype=float),
        phi_req_rad=phi_req_arr,
        phi_max_rad=phi_max_arr,
        mode=tuple(mode_names),
        success=success,
        message=message,
        max_abs_cross_track_m=float(np.max(np.abs(cross_track_m))),
        max_abs_track_error_rad=float(np.max(np.abs(track_error_rad))),
        min_alongtrack_speed_mps=float(np.min(alongtrack_speed_mps)),
        max_bank_command_ratio=float(np.max(bank_ratio)),
        final_threshold_error_m=float(np.hypot(east_arr[-1] - threshold_east_m, north_arr[-1] - threshold_north_m)),
    )


__all__ = [
    "CoupledDescentPlanProfile",
    "CoupledDescentPlanSample",
    "SimulationRequest",
    "SimulationResult",
    "State",
    "simulate_plan",
]


LongitudinalPlanSample = CoupledDescentPlanSample
LongitudinalPlanProfile = CoupledDescentPlanProfile
