from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openap import aero

from .config import AircraftConfig, ModeConfig, bank_limit_rad
from .path_geometry import ReferencePath
from .weather import WeatherProvider


def wrap_angle_rad(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


@dataclass(frozen=True)
class LateralGuidanceConfig:
    lookahead_m: float = 1_500.0
    cross_track_gain: float = 1.0
    track_error_gain: float = 2.0


@dataclass(frozen=True)
class LateralCommand:
    east_dot_mps: float
    north_dot_mps: float
    ground_speed_mps: float
    alongtrack_speed_mps: float
    ground_track_rad: float
    cross_track_m: float
    track_error_rad: float
    curvature_cmd_inv_m: float
    phi_req_rad: float
    phi_max_rad: float


def compute_lateral_command(
    *,
    s_m: float,
    east_m: float,
    north_m: float,
    h_m: float,
    t_s: float,
    psi_rad: float,
    v_tas_mps: float,
    cfg: AircraftConfig,
    mode: ModeConfig,
    reference_path: ReferencePath,
    weather: WeatherProvider,
    guidance: LateralGuidanceConfig,
) -> LateralCommand:
    wind_east_mps, wind_north_mps = weather.wind_ne_mps(s_m, h_m, t_s)
    east_dot_mps = float(v_tas_mps * np.cos(psi_rad) + wind_east_mps)
    north_dot_mps = float(v_tas_mps * np.sin(psi_rad) + wind_north_mps)
    ground_speed_mps = float(np.hypot(east_dot_mps, north_dot_mps))
    ground_track_rad = wrap_angle_rad(np.arctan2(north_dot_mps, east_dot_mps))

    ref_east_m, ref_north_m = reference_path.position_ne(s_m)
    tangent_hat = reference_path.tangent_hat(s_m)
    normal_hat = reference_path.normal_hat(s_m)
    ref_track_rad = reference_path.track_angle_rad(s_m)
    ref_curvature_inv_m = reference_path.curvature(s_m)

    error_vector = np.asarray([east_m - ref_east_m, north_m - ref_north_m], dtype=float)
    cross_track_m = float(np.dot(error_vector, normal_hat))
    track_error_rad = wrap_angle_rad(ground_track_rad - ref_track_rad)
    alongtrack_speed_mps = float(max(0.0, np.dot(np.asarray([east_dot_mps, north_dot_mps]), tangent_hat)))

    lookahead_m = max(1.0, guidance.lookahead_m)
    curvature_feedback = (
        -(guidance.cross_track_gain * cross_track_m) / (lookahead_m**2)
        - (guidance.track_error_gain * track_error_rad) / lookahead_m
    )
    curvature_cmd_inv_m = ref_curvature_inv_m + curvature_feedback
    phi_req_rad = float(np.arctan(max(ground_speed_mps, 1.0) ** 2 * curvature_cmd_inv_m / aero.g0))
    delta_isa_K = weather.delta_isa_K(s_m, h_m, t_s)
    v_cas_mps = float(aero.tas2cas(v_tas_mps, h_m, dT=delta_isa_K))
    phi_max_rad = bank_limit_rad(cfg, mode, v_cas_mps)
    phi_req_rad = float(np.clip(phi_req_rad, -phi_max_rad, phi_max_rad))

    return LateralCommand(
        east_dot_mps=east_dot_mps,
        north_dot_mps=north_dot_mps,
        ground_speed_mps=ground_speed_mps,
        alongtrack_speed_mps=alongtrack_speed_mps,
        ground_track_rad=ground_track_rad,
        cross_track_m=cross_track_m,
        track_error_rad=track_error_rad,
        curvature_cmd_inv_m=curvature_cmd_inv_m,
        phi_req_rad=phi_req_rad,
        phi_max_rad=phi_max_rad,
    )


def lateral_rates(
    *,
    phi_rad: float,
    phi_req_rad: float,
    tau_phi_s: float,
    p_max_rps: float,
    v_tas_mps: float,
) -> tuple[float, float]:
    phi_dot_rps = float(np.clip((phi_req_rad - phi_rad) / tau_phi_s, -p_max_rps, p_max_rps))
    psi_dot_rps = float(aero.g0 * np.tan(phi_rad) / max(v_tas_mps, 1.0))
    return psi_dot_rps, phi_dot_rps
