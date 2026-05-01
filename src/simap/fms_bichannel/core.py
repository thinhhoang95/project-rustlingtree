from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import mode_for_s
from ..fms import (
    FMSRequest,
    FMSResult,
    HoldAwareFMSRequest,
    plan_fms_descent,
    plan_hold_aware_fms_descent,
    simulate_fms_descent,
    simulate_hold_aware_fms_descent,
)
from ..lateral_dynamics import (
    LateralGuidanceConfig,
    compute_lateral_command,
    lateral_rates,
    wrap_angle_rad,
)
from ..path_geometry import ReferencePath


@dataclass(frozen=True)
class FMSBiChannelState:
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
        heading_offset_rad: float = 0.0,
        cross_track_m: float = 0.0,
        bank_rad: float = 0.0,
    ) -> "FMSBiChannelState":
        east_m, north_m = reference_path.position_ne(s_m)
        normal_hat = reference_path.normal_hat(s_m)
        return cls(
            t_s=float(t_s),
            s_m=float(s_m),
            h_m=float(h_m),
            v_tas_mps=float(v_tas_mps),
            east_m=float(east_m + cross_track_m * normal_hat[0]),
            north_m=float(north_m + cross_track_m * normal_hat[1]),
            psi_rad=wrap_angle_rad(reference_path.track_angle_rad(s_m) + heading_offset_rad),
            phi_rad=float(bank_rad),
        )


@dataclass(frozen=True)
class FMSBiChannelRequest:
    base_request: FMSRequest | HoldAwareFMSRequest
    guidance: LateralGuidanceConfig = field(default_factory=LateralGuidanceConfig)
    initial_state: FMSBiChannelState | None = None


@dataclass(frozen=True)
class FMSBiChannelResult:
    longitudinal: FMSResult
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
    success: bool
    message: str
    max_abs_cross_track_m: float
    max_abs_track_error_rad: float
    max_bank_command_ratio: float
    final_threshold_error_m: float

    def __len__(self) -> int:
        return len(self.longitudinal)

    @property
    def t_s(self) -> np.ndarray:
        return self.longitudinal.t_s

    @property
    def s_m(self) -> np.ndarray:
        return self.longitudinal.s_m

    @property
    def h_m(self) -> np.ndarray:
        return self.longitudinal.h_m

    @property
    def v_tas_mps(self) -> np.ndarray:
        return self.longitudinal.v_tas_mps

    @property
    def v_cas_mps(self) -> np.ndarray:
        return self.longitudinal.v_cas_mps

    def to_pandas(self) -> pd.DataFrame:
        frame = self.longitudinal.to_pandas()
        for column, values in {
            "east_m": self.east_m,
            "north_m": self.north_m,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "psi_rad": self.psi_rad,
            "phi_rad": self.phi_rad,
            "ground_track_rad": self.ground_track_rad,
            "lateral_ground_speed_mps": self.ground_speed_mps,
            "alongtrack_speed_mps": self.alongtrack_speed_mps,
            "cross_track_m": self.cross_track_m,
            "track_error_rad": self.track_error_rad,
            "curvature_cmd_inv_m": self.curvature_cmd_inv_m,
            "phi_req_rad": self.phi_req_rad,
            "phi_max_rad": self.phi_max_rad,
        }.items():
            frame[column] = values
        return frame


def _base_fms_request(request: FMSRequest | HoldAwareFMSRequest) -> FMSRequest:
    if isinstance(request, HoldAwareFMSRequest):
        return request.base_request
    return request


def _initial_state(
    *,
    request: FMSBiChannelRequest,
    longitudinal: FMSResult,
) -> FMSBiChannelState:
    base = _base_fms_request(request.base_request)
    if request.initial_state is not None:
        return request.initial_state
    return FMSBiChannelState.on_reference_path(
        t_s=float(longitudinal.t_s[0]),
        s_m=float(longitudinal.s_m[0]),
        h_m=float(longitudinal.h_m[0]),
        v_tas_mps=float(longitudinal.v_tas_mps[0]),
        reference_path=base.reference_path,
    )


def _lateral_response(
    *,
    request: FMSBiChannelRequest,
    longitudinal: FMSResult,
) -> FMSBiChannelResult:
    base = _base_fms_request(request.base_request)
    state = _initial_state(request=request, longitudinal=longitudinal)

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

    for idx in range(len(longitudinal)):
        state = FMSBiChannelState(
            t_s=float(longitudinal.t_s[idx]),
            s_m=float(longitudinal.s_m[idx]),
            h_m=float(longitudinal.h_m[idx]),
            v_tas_mps=float(longitudinal.v_tas_mps[idx]),
            east_m=state.east_m,
            north_m=state.north_m,
            psi_rad=state.psi_rad,
            phi_rad=state.phi_rad,
        )
        mode = mode_for_s(base.cfg, state.s_m)
        command = compute_lateral_command(
            s_m=state.s_m,
            east_m=state.east_m,
            north_m=state.north_m,
            h_m=state.h_m,
            t_s=state.t_s,
            psi_rad=state.psi_rad,
            v_tas_mps=state.v_tas_mps,
            cfg=base.cfg,
            mode=mode,
            reference_path=base.reference_path,
            weather=base.weather,
            guidance=request.guidance,
        )

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

        if idx + 1 >= len(longitudinal):
            continue
        step_dt_s = float(max(0.0, longitudinal.t_s[idx + 1] - longitudinal.t_s[idx]))
        if step_dt_s <= 0.0:
            continue
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
        state = FMSBiChannelState(
            t_s=float(longitudinal.t_s[idx + 1]),
            s_m=float(longitudinal.s_m[idx + 1]),
            h_m=float(longitudinal.h_m[idx + 1]),
            v_tas_mps=float(longitudinal.v_tas_mps[idx + 1]),
            east_m=float(state.east_m + command.east_dot_mps * step_dt_s),
            north_m=float(state.north_m + command.north_dot_mps * step_dt_s),
            psi_rad=wrap_angle_rad(state.psi_rad + psi_dot_rps * step_dt_s),
            phi_rad=next_phi_rad,
        )

    east_arr = np.asarray(east_m, dtype=float)
    north_arr = np.asarray(north_m, dtype=float)
    latlon_arr = base.reference_path.latlon_from_ne_many(east_arr, north_arr)
    threshold_east_m, threshold_north_m = base.reference_path.position_ne(0.0)
    phi_max_arr = np.asarray(phi_max_rad, dtype=float)
    phi_req_arr = np.asarray(phi_req_rad, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bank_ratio = np.divide(np.abs(phi_req_arr), phi_max_arr, out=np.zeros_like(phi_req_arr), where=phi_max_arr > 0.0)

    max_abs_cross_track_m = float(np.max(np.abs(cross_track_m))) if cross_track_m else 0.0
    max_abs_track_error_rad = float(np.max(np.abs(track_error_rad))) if track_error_rad else 0.0
    final_threshold_error_m = (
        float(np.hypot(east_arr[-1] - threshold_east_m, north_arr[-1] - threshold_north_m))
        if len(east_arr)
        else 0.0
    )
    return FMSBiChannelResult(
        longitudinal=longitudinal,
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
        success=bool(longitudinal.success),
        message=longitudinal.message,
        max_abs_cross_track_m=max_abs_cross_track_m,
        max_abs_track_error_rad=max_abs_track_error_rad,
        max_bank_command_ratio=float(np.max(bank_ratio)) if len(bank_ratio) else 0.0,
        final_threshold_error_m=final_threshold_error_m,
    )


def simulate_fms_bichannel(request: FMSBiChannelRequest) -> FMSBiChannelResult:
    if isinstance(request.base_request, HoldAwareFMSRequest):
        longitudinal = simulate_hold_aware_fms_descent(request.base_request)
    else:
        longitudinal = simulate_fms_descent(request.base_request)
    return _lateral_response(request=request, longitudinal=longitudinal)


def plan_fms_bichannel(
    request: FMSBiChannelRequest,
    *,
    tod_tolerance_m: float = 5.0,
    max_tod_iterations: int = 40,
) -> FMSBiChannelResult:
    if isinstance(request.base_request, HoldAwareFMSRequest):
        longitudinal = plan_hold_aware_fms_descent(
            request.base_request,
            tod_tolerance_m=tod_tolerance_m,
            max_tod_iterations=max_tod_iterations,
        )
    else:
        longitudinal = plan_fms_descent(
            request.base_request,
            tod_tolerance_m=tod_tolerance_m,
            max_tod_iterations=max_tod_iterations,
        )
    return _lateral_response(request=request, longitudinal=longitudinal)


__all__ = [
    "FMSBiChannelRequest",
    "FMSBiChannelResult",
    "FMSBiChannelState",
    "plan_fms_bichannel",
    "simulate_fms_bichannel",
]
