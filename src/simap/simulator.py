from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from openap import aero

from .backends import PerformanceBackend
from .config import AircraftConfig, mode_for_s
from .lateral_dynamics import LateralGuidanceConfig, compute_lateral_command, lateral_rates, wrap_angle_rad
from .longitudinal_dynamics import LongitudinalState, longitudinal_rhs
from .longitudinal_profiles import FeasibilityConfig, ScalarProfile, build_feasible_cas_schedule
from .path_geometry import ReferencePath
from .weather import ConstantWeather, WeatherProvider


@dataclass(frozen=True)
class Scenario:
    altitude_profile: ScalarProfile
    raw_speed_schedule_cas: ScalarProfile
    reference_path: ReferencePath
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    lateral_guidance: LateralGuidanceConfig = field(default_factory=LateralGuidanceConfig)


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
class Trajectory:
    t_s: np.ndarray
    s_m: np.ndarray
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    v_cas_mps: np.ndarray
    gs_mps: np.ndarray
    h_ref_m: np.ndarray
    v_ref_cas_mps: np.ndarray
    mode: tuple[str, ...]
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    heading_rad: np.ndarray
    bank_rad: np.ndarray
    cross_track_m: np.ndarray

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
                "gs_mps": self.gs_mps,
                "h_ref_m": self.h_ref_m,
                "v_ref_cas_mps": self.v_ref_cas_mps,
                "mode": np.asarray(self.mode, dtype=object),
                "lat_deg": self.lat_deg,
                "lon_deg": self.lon_deg,
                "heading_rad": self.heading_rad,
                "bank_rad": self.bank_rad,
                "cross_track_m": self.cross_track_m,
            }
        )


def coupled_rhs(
    state: State,
    cfg: AircraftConfig,
    perf: PerformanceBackend,
    scenario: Scenario,
    feasible_speed_schedule_cas: ScalarProfile,
) -> np.ndarray:
    mode = mode_for_s(cfg, state.s_m)
    command = compute_lateral_command(
        s_m=state.s_m,
        east_m=state.east_m,
        north_m=state.north_m,
        h_m=state.h_m,
        t_s=state.t_s,
        psi_rad=state.psi_rad,
        v_tas_mps=state.v_tas_mps,
        cfg=cfg,
        mode=mode,
        reference_path=scenario.reference_path,
        weather=scenario.weather,
        guidance=scenario.lateral_guidance,
    )
    s_dot_mps = -command.alongtrack_speed_mps
    long_rates = longitudinal_rhs(
        state=LongitudinalState(
            t_s=state.t_s,
            s_m=state.s_m,
            h_m=state.h_m,
            v_tas_mps=state.v_tas_mps,
        ),
        cfg=cfg,
        perf=perf,
        altitude_profile=scenario.altitude_profile,
        speed_schedule_cas=feasible_speed_schedule_cas,
        weather=scenario.weather,
        track_angle_rad=scenario.reference_path.track_angle_rad(state.s_m),
        bank_rad=state.phi_rad,
        s_dot_mps=s_dot_mps,
    )
    psi_dot_rps, phi_dot_rps = lateral_rates(
        phi_rad=state.phi_rad,
        phi_req_rad=command.phi_req_rad,
        tau_phi_s=mode.tau_phi_s,
        p_max_rps=mode.p_max_rps,
        v_tas_mps=state.v_tas_mps,
    )
    return np.asarray(
        [
            long_rates[0],
            long_rates[1],
            long_rates[2],
            command.east_dot_mps,
            command.north_dot_mps,
            psi_dot_rps,
            phi_dot_rps,
        ],
        dtype=float,
    )


class ApproachSimulator:
    def __init__(
        self,
        cfg: AircraftConfig,
        perf: PerformanceBackend,
        scenario: Scenario,
    ) -> None:
        self.cfg = cfg
        self.perf = perf
        self.scenario = scenario
        self.altitude_profile = scenario.altitude_profile
        self.raw_speed_schedule_cas = scenario.raw_speed_schedule_cas
        self.weather = scenario.weather
        self.reference_path = scenario.reference_path
        self.feasible_speed_schedule_cas = build_feasible_cas_schedule(
            raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
            altitude_profile=scenario.altitude_profile,
            cfg=cfg,
            perf=perf,
            feasibility=scenario.feasibility,
        )

    def step(self, state: State, dt_s: float) -> State:
        y0 = np.asarray(
            [
                state.s_m,
                state.h_m,
                state.v_tas_mps,
                state.east_m,
                state.north_m,
                state.psi_rad,
                state.phi_rad,
            ],
            dtype=float,
        )

        def f(y: np.ndarray, t_s: float) -> np.ndarray:
            step_state = State(
                t_s=t_s,
                s_m=float(y[0]),
                h_m=float(y[1]),
                v_tas_mps=float(y[2]),
                east_m=float(y[3]),
                north_m=float(y[4]),
                psi_rad=float(y[5]),
                phi_rad=float(y[6]),
            )
            return coupled_rhs(
                state=step_state,
                cfg=self.cfg,
                perf=self.perf,
                scenario=self.scenario,
                feasible_speed_schedule_cas=self.feasible_speed_schedule_cas,
            )

        k1 = f(y0, state.t_s)
        k2 = f(y0 + 0.5 * dt_s * k1, state.t_s + 0.5 * dt_s)
        k3 = f(y0 + 0.5 * dt_s * k2, state.t_s + 0.5 * dt_s)
        k4 = f(y0 + dt_s * k3, state.t_s + dt_s)
        y1 = y0 + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return State(
            t_s=state.t_s + dt_s,
            s_m=max(0.0, float(y1[0])),
            h_m=max(0.0, float(y1[1])),
            v_tas_mps=max(1.0, float(y1[2])),
            east_m=float(y1[3]),
            north_m=float(y1[4]),
            psi_rad=wrap_angle_rad(float(y1[5])),
            phi_rad=float(y1[6]),
        )

    def run(
        self,
        initial: State,
        *,
        dt_s: float = 1.0,
        t_max_s: float = 4_000.0,
    ) -> Trajectory:
        rows: dict[str, list[float] | list[str]] = {
            "t_s": [],
            "s_m": [],
            "h_m": [],
            "v_tas_mps": [],
            "v_cas_mps": [],
            "gs_mps": [],
            "h_ref_m": [],
            "v_ref_cas_mps": [],
            "mode": [],
            "lat_deg": [],
            "lon_deg": [],
            "heading_rad": [],
            "bank_rad": [],
            "cross_track_m": [],
        }

        state = initial
        while state.t_s <= t_max_s and state.s_m > 1.0:
            mode = mode_for_s(self.cfg, state.s_m)
            command = compute_lateral_command(
                s_m=state.s_m,
                east_m=state.east_m,
                north_m=state.north_m,
                h_m=state.h_m,
                t_s=state.t_s,
                psi_rad=state.psi_rad,
                v_tas_mps=state.v_tas_mps,
                cfg=self.cfg,
                mode=mode,
                reference_path=self.reference_path,
                weather=self.weather,
                guidance=self.scenario.lateral_guidance,
            )
            delta_isa_K = self.weather.delta_isa_K(state.s_m, state.h_m, state.t_s)
            v_cas_mps = float(aero.tas2cas(state.v_tas_mps, state.h_m, dT=delta_isa_K))
            h_ref_m = self.altitude_profile.value(state.s_m)
            v_ref_cas_mps = self.feasible_speed_schedule_cas.value(state.s_m)
            lat_deg, lon_deg = self.reference_path.latlon_from_ne(state.east_m, state.north_m)

            rows["t_s"].append(state.t_s)
            rows["s_m"].append(state.s_m)
            rows["h_m"].append(state.h_m)
            rows["v_tas_mps"].append(state.v_tas_mps)
            rows["v_cas_mps"].append(v_cas_mps)
            rows["gs_mps"].append(command.ground_speed_mps)
            rows["h_ref_m"].append(h_ref_m)
            rows["v_ref_cas_mps"].append(v_ref_cas_mps)
            rows["mode"].append(mode.name)
            rows["lat_deg"].append(lat_deg)
            rows["lon_deg"].append(lon_deg)
            rows["heading_rad"].append(state.psi_rad)
            rows["bank_rad"].append(state.phi_rad)
            rows["cross_track_m"].append(command.cross_track_m)

            state = self.step(state, dt_s)

        return Trajectory(
            t_s=np.asarray(rows["t_s"], dtype=float),
            s_m=np.asarray(rows["s_m"], dtype=float),
            h_m=np.asarray(rows["h_m"], dtype=float),
            v_tas_mps=np.asarray(rows["v_tas_mps"], dtype=float),
            v_cas_mps=np.asarray(rows["v_cas_mps"], dtype=float),
            gs_mps=np.asarray(rows["gs_mps"], dtype=float),
            h_ref_m=np.asarray(rows["h_ref_m"], dtype=float),
            v_ref_cas_mps=np.asarray(rows["v_ref_cas_mps"], dtype=float),
            mode=tuple(str(value) for value in rows["mode"]),
            lat_deg=np.asarray(rows["lat_deg"], dtype=float),
            lon_deg=np.asarray(rows["lon_deg"], dtype=float),
            heading_rad=np.asarray(rows["heading_rad"], dtype=float),
            bank_rad=np.asarray(rows["bank_rad"], dtype=float),
            cross_track_m=np.asarray(rows["cross_track_m"], dtype=float),
        )
