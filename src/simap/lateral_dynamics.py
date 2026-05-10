from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openap import aero

from .config import AircraftConfig, ModeConfig, bank_limit_rad
from .path_geometry import ReferencePath
from .openap_adapter import openap_dT
from .weather import WeatherProvider


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to the principal interval ``[-pi, pi]``.

    The implementation uses ``atan2(sin(angle), cos(angle))`` so it works
    reliably for very large positive or negative inputs instead of repeatedly
    subtracting ``2*pi``.

    Parameters
    ----------
    angle_rad:
        Any real-valued angle in radians.

    Returns
    -------
    float
        The equivalent wrapped angle in radians.

    Examples
    --------
    >>> wrap_angle_rad(3 * np.pi)
    3.141592653589793
    >>> wrap_angle_rad(-3 * np.pi / 2)
    1.5707963267948966

    Notes
    -----
    This helper is used throughout lateral guidance whenever a heading, track
    angle, or track error must be compared without discontinuities at the
    ``-pi`` / ``pi`` boundary.
    """
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


@dataclass(frozen=True)
class LateralGuidanceConfig:
    lookahead_m: float = 1_500.0
    cross_track_gain: float = 1.0
    track_error_gain: float = 2.0
    min_lookahead_m: float = 150.0
    max_los_angle_rad: float = float(np.deg2rad(89.0))
    integration_step_s: float = 0.5


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
    """Compute the instantaneous lateral guidance command for the aircraft.

    The controller uses a nonlinear lookahead guidance law. It measures
    cross-track and track-angle error at the closest tangent point on the
    reference path, selects a target point farther along the active path, then
    commands curvature from the line-of-sight angle between the current ground
    track and that target point. This behaves like the path-capture part of a
    transport LNAV law: cross-track error changes the desired intercept
    geometry, while curved-path feed-forward naturally appears because the
    lookahead point lies on the curved reference path.
    """
    wind_east_mps, wind_north_mps = weather.wind_ne_mps(s_m, h_m, t_s)
    east_dot_mps = float(v_tas_mps * np.cos(psi_rad) + wind_east_mps)
    north_dot_mps = float(v_tas_mps * np.sin(psi_rad) + wind_north_mps)
    ground_speed_mps = float(np.hypot(east_dot_mps, north_dot_mps))
    ground_track_rad = wrap_angle_rad(np.arctan2(north_dot_mps, east_dot_mps))

    ref_s_m = reference_path.project_s_m(east_m, north_m)
    ref_east_m, ref_north_m = reference_path.position_ne(ref_s_m)
    tangent_hat = reference_path.tangent_hat(ref_s_m)
    normal_hat = reference_path.normal_hat(ref_s_m)
    ref_track_rad = reference_path.track_angle_rad(ref_s_m)

    error_vector = np.asarray([east_m - ref_east_m, north_m - ref_north_m], dtype=float)
    cross_track_m = float(np.dot(error_vector, normal_hat))
    track_error_rad = wrap_angle_rad(ground_track_rad - ref_track_rad)
    alongtrack_speed_mps = float(max(0.0, np.dot(np.asarray([east_dot_mps, north_dot_mps]), tangent_hat)))
    along_path_offset_m = float(np.dot(error_vector, tangent_hat))

    base_lookahead_m = max(1.0, float(guidance.lookahead_m))
    min_lookahead_m = float(np.clip(guidance.min_lookahead_m, 1.0, base_lookahead_m))
    lookahead_m = float(max(min_lookahead_m, min(base_lookahead_m, max(min_lookahead_m, ref_s_m))))
    if ref_s_m <= min_lookahead_m and along_path_offset_m > 0.0:
        lookahead_m = base_lookahead_m
        target_east_m = float(ref_east_m + (along_path_offset_m + lookahead_m) * tangent_hat[0])
        target_north_m = float(ref_north_m + (along_path_offset_m + lookahead_m) * tangent_hat[1])
    else:
        target_s_m = float(max(0.0, ref_s_m - lookahead_m))
        target_east_m, target_north_m = reference_path.position_ne(target_s_m)
    los_track_rad = wrap_angle_rad(np.arctan2(target_north_m - north_m, target_east_m - east_m))
    los_error_rad = wrap_angle_rad(los_track_rad - ground_track_rad)
    los_limit_rad = float(np.clip(guidance.max_los_angle_rad, np.deg2rad(5.0), np.deg2rad(120.0)))
    los_error_rad = float(np.clip(los_error_rad, -los_limit_rad, los_limit_rad))

    l1_gain = max(0.05, float(guidance.track_error_gain) * np.sqrt(max(0.05, float(guidance.cross_track_gain))))
    curvature_cmd_inv_m = float(l1_gain * np.sin(los_error_rad) / lookahead_m)
    phi_req_rad = float(np.arctan(max(ground_speed_mps, 1.0) ** 2 * curvature_cmd_inv_m / aero.g0))
    delta_isa_K = weather.delta_isa_K(s_m, h_m, t_s)
    v_cas_mps = float(aero.tas2cas(v_tas_mps, h_m, dT=openap_dT(delta_isa_K)))
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
    """Convert a bank request into heading-rate and roll-rate commands.

    The roll loop is modeled as a first-order response:

    ``phi_dot = (phi_req - phi) / tau_phi``

    and then clipped to the maximum roll-rate magnitude ``p_max_rps``. The
    resulting bank angle is used to compute the coordinated-turn heading rate:

    ``psi_dot = g * tan(phi) / Vtas``

    where ``g`` is standard gravity and ``Vtas`` is the true airspeed.

    Parameters
    ----------
    phi_rad:
        Current bank angle.
    phi_req_rad:
        Requested bank angle from the lateral guidance law.
    tau_phi_s:
        Time constant for the bank response.
    p_max_rps:
        Absolute roll-rate limit.
    v_tas_mps:
        Current true airspeed.

    Returns
    -------
    tuple[float, float]
        ``(psi_dot_rps, phi_dot_rps)``.

    Examples
    --------
    A 0.2 rad bank request with a 10 s roll time constant produces a
    0.02 rad/s roll rate when it is within the roll-rate limit:

    >>> lateral_rates(
    ...     phi_rad=0.0,
    ...     phi_req_rad=0.2,
    ...     tau_phi_s=10.0,
    ...     p_max_rps=0.05,
    ...     v_tas_mps=70.0,
    ... )
    (0.0, 0.02)

    If the requested correction is too aggressive, the roll-rate command is
    clipped:

    >>> lateral_rates(
    ...     phi_rad=0.2,
    ...     phi_req_rad=1.0,
    ...     tau_phi_s=5.0,
    ...     p_max_rps=0.05,
    ...     v_tas_mps=70.0,
    ... )[1]
    0.05

    Notes
    -----
    The turn-rate output depends on the *current* bank angle, not the requested
    one. That means the heading response lags behind the bank command until the
    roll loop catches up.
    """
    phi_dot_rps = float(np.clip((phi_req_rad - phi_rad) / tau_phi_s, -p_max_rps, p_max_rps))
    psi_dot_rps = float(aero.g0 * np.tan(phi_rad) / max(v_tas_mps, 1.0))
    return psi_dot_rps, phi_dot_rps
