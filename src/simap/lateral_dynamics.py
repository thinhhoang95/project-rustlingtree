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

    This is the main path-following function in the module. It combines:

    - the current aircraft state ``(east_m, north_m, h_m, psi_rad, v_tas_mps)``
    - the reference path geometry at the path coordinate ``s_m``
    - the local wind and atmosphere from ``weather``
    - the bank-limits and phase-dependent limits from ``cfg`` and ``mode``

    The result is a dataclass containing the estimated ground velocity, cross
    track error, track error, commanded curvature, requested bank angle, and
    the bank limit that was applied. In practice this is the lateral guidance
    output consumed by a downstream roll / heading-rate controller.

    Parameters
    ----------
    s_m, east_m, north_m, h_m, t_s, psi_rad, v_tas_mps:
        Current aircraft state and simulation time.
    cfg, mode:
        Aircraft and phase configuration used to compute the allowable bank
        angle.
    reference_path:
        The path being tracked.
    weather:
        Provider for wind and ISA deviation.
    guidance:
        Lookahead and feedback gains for the lateral controller.

    Returns
    -------
    LateralCommand
        A summary of the current lateral control request.

    Examples
    --------
    A straight eastbound path with zero wind and the aircraft 10 m north of
    the path is the simplest "turn back to centerline" case. With
    ``s_m = 500 m``, ``psi_rad = 0``, and ``v_tas_mps = 70``, the returned
    command is approximately:

    - ``ground_speed_mps = 70.0``
    - ``cross_track_m = 10.0``
    - ``track_error_rad = 0.0``
    - ``phi_req_rad = -0.0022`` rad

    The exact ``phi_max_rad`` value depends on the supplied aircraft and mode
    limits, so it is not fixed by the guidance law itself.

    Notes
    -----
    - The path geometry is evaluated at the closest projected point on the
      reference path, not at the current longitudinal ``s_m``. This keeps the
      controller responsive when the aircraft drifts far from the scheduled
      descent station.
    - ``cross_track_m`` is measured using the path normal, so its sign depends
      on the path orientation.
    - ``track_error_rad`` compares the ground-track angle to the local path
      tangent, not the aircraft heading. A crosswind can therefore create a
      track error even when the heading is aligned with the path.
    - ``phi_req_rad`` is clipped to the current bank limit, so the returned
      value is always commandable for the current phase and speed.
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
    ref_curvature_inv_m = reference_path.curvature(ref_s_m)

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
