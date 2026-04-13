from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openap import aero

from .backends import PerformanceBackend
from .config import AircraftConfig, clamp_cas_to_mode_limits, mode_for_s
from .longitudinal_profiles import ScalarProfile, longitudinal_deceleration_limit_mps2, path_angle_rad
from .weather import WeatherProvider, alongtrack_wind_mps


@dataclass(frozen=True)
class LongitudinalState:
    t_s: float
    s_m: float
    h_m: float
    v_tas_mps: float


@dataclass(frozen=True)
class LongitudinalCommand:
    s_dot_mps: float
    hdot_cmd_mps: float
    vdot_cmd_mps2: float
    vdot_mps2: float


def longitudinal_command(
    state: LongitudinalState,
    cfg: AircraftConfig,
    perf: PerformanceBackend,
    altitude_profile: ScalarProfile,
    speed_schedule_cas: ScalarProfile,
    weather: WeatherProvider,
    *,
    track_angle_rad: float = 0.0,
    bank_rad: float = 0.0,
    s_dot_mps: float | None = None,
) -> LongitudinalCommand:
    s_m = state.s_m
    h_m = state.h_m
    v_tas_mps = max(1.0, state.v_tas_mps)
    mode = mode_for_s(cfg, s_m)

    delta_isa_K = weather.delta_isa_K(s_m, h_m, state.t_s)
    if s_dot_mps is None:
        wind_mps = alongtrack_wind_mps(weather, track_angle_rad, s_m, h_m, state.t_s)
        gs_along_mps = max(1.0, v_tas_mps + wind_mps)
        s_dot_mps = -gs_along_mps

    h_ref_m = altitude_profile.value(s_m)
    gamma_ref_rad = path_angle_rad(altitude_profile, s_m)
    hdot_ff = altitude_profile.slope(s_m) * s_dot_mps
    hdot_cmd = hdot_ff + cfg.k_h_sinv * (h_ref_m - h_m)
    hdot_cmd = float(np.clip(hdot_cmd, mode.vs_min_mps, mode.vs_max_mps))

    v_ref_cas_mps = clamp_cas_to_mode_limits(mode, speed_schedule_cas.value(s_m))
    v_ref_tas_mps = float(aero.cas2tas(v_ref_cas_mps, h_m, dT=delta_isa_K))
    vdot_cmd = float((v_ref_tas_mps - v_tas_mps) / mode.tau_v_s)

    a_dec_max = longitudinal_deceleration_limit_mps2(
        mode=mode,
        cfg=cfg,
        perf=perf,
        v_tas_mps=v_tas_mps,
        h_m=h_m,
        vs_mps=hdot_cmd,
        bank_rad=bank_rad,
        delta_isa_K=delta_isa_K,
        gamma_ref_rad=gamma_ref_rad,
    )
    vdot = float(np.clip(vdot_cmd, -a_dec_max, cfg.a_acc_max_mps2))
    return LongitudinalCommand(
        s_dot_mps=float(s_dot_mps),
        hdot_cmd_mps=hdot_cmd,
        vdot_cmd_mps2=vdot_cmd,
        vdot_mps2=vdot,
    )


def longitudinal_rhs(
    state: LongitudinalState,
    cfg: AircraftConfig,
    perf: PerformanceBackend,
    altitude_profile: ScalarProfile,
    speed_schedule_cas: ScalarProfile,
    weather: WeatherProvider,
    *,
    track_angle_rad: float = 0.0,
    bank_rad: float = 0.0,
    s_dot_mps: float | None = None,
) -> np.ndarray:
    """Compute the longitudinal state derivatives for one guidance step.

    This is the core right-hand-side used by the longitudinal and coupled
    simulators. It turns the current aircraft state, target altitude profile,
    target CAS schedule, weather, and performance model into the derivatives of
    along-track distance, altitude, and true airspeed.

    The function is intentionally control-law flavored rather than purely
    ballistic:

    - altitude rate follows the commanded glidepath with a proportional
      correction toward the reference altitude profile,
    - the speed target is taken from the CAS schedule, converted to TAS at the
      current atmosphere, and filtered through the mode time constant, and
    - the speed derivative is capped by the available deceleration margin so
      the model does not demand physically impossible slowdown.

    Parameters
    ----------
    state:
        Current longitudinal state. `s_m` is along-track distance to the
        threshold, `h_m` is altitude, and `v_tas_mps` is true airspeed.
    cfg:
        Aircraft configuration, including mode thresholds, vertical-speed
        limits, and acceleration limits.
    perf:
        Performance backend used to estimate drag and idle thrust.
    altitude_profile:
        Reference altitude profile over along-track distance.
    speed_schedule_cas:
        Reference calibrated-airspeed schedule over along-track distance.
    weather:
        Provider for wind and ISA deviation.
    track_angle_rad:
        Reference track direction used to project wind into the along-track
        axis. This matters when `s_dot_mps` is not supplied.
    bank_rad:
        Bank angle passed to the drag model.
    s_dot_mps:
        Optional externally imposed along-track rate. The coupled 3D simulator
        supplies this so the longitudinal model stays consistent with the
        lateral guidance solution. When omitted, the function derives the rate
        from TAS plus along-track wind.

    Returns
    -------
    np.ndarray
        A length-3 float array with `[s_dot_mps, h_dot_mps, v_tas_dot_mps]`.
        The altitude component is clipped to the active mode's vertical-speed
        limits, and the speed component is clipped to the available deceleration
        ceiling and `cfg.a_acc_max_mps2`.

    Examples
    --------
    With a flat altitude reference, a speed schedule that matches the current
    speed, zero wind, and a backend that does not force any climb or slowdown,
    the derivative is purely along-track:

    - input: `state = LongitudinalState(t_s=0.0, s_m=1000.0, h_m=450.0,
      v_tas_mps=72.0)`, `track_angle_rad=0.0`, `s_dot_mps=None`
    - output: `np.array([-72.0, 0.0, 0.0])`

    If the coupled simulator supplies `s_dot_mps=-50.0`, the first component
    follows that command instead:

    - input: same state and atmosphere, but `s_dot_mps=-50.0`
    - output: `np.array([-50.0, 0.0, 0.0])`

    Nuances
    --------
    - In this package, decreasing `s_m` means moving toward the runway
      threshold, so a negative `s_dot_mps` corresponds to forward motion.
    - `state.v_tas_mps` is floored at 1 m/s before any atmosphere or drag
      calculations to avoid divide-by-zero behavior.
    - `speed_schedule_cas` is clamped to the current mode's CAS limits before
      being converted to TAS.
    - The deceleration ceiling is conservative: it compares drag against idle
      thrust and subtracts the longitudinal gravity component for the reference
      path angle.
    """
    command = longitudinal_command(
        state=state,
        cfg=cfg,
        perf=perf,
        altitude_profile=altitude_profile,
        speed_schedule_cas=speed_schedule_cas,
        weather=weather,
        track_angle_rad=track_angle_rad,
        bank_rad=bank_rad,
        s_dot_mps=s_dot_mps,
    )
    return np.asarray([command.s_dot_mps, command.hdot_cmd_mps, command.vdot_mps2], dtype=float)
