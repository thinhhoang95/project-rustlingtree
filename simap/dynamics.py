from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openap import aero

from .backends import PerformanceBackend
from .config import AircraftConfig, mode_for_x
from .profiles import ScalarProfile, path_angle_rad
from .weather import WeatherProvider


@dataclass(frozen=True)
class State:
    t_s: float
    x_m: float
    h_m: float
    v_tas_mps: float


def rhs(
    state: State,
    cfg: AircraftConfig,
    perf: PerformanceBackend,
    altitude_profile: ScalarProfile,
    speed_schedule_cas: ScalarProfile,
    weather: WeatherProvider,
) -> np.ndarray:
    x_m = state.x_m # along-track distance profile
    h_m = state.h_m # altitude by along-track distance profile h(x)
    v_tas_mps = max(1.0, state.v_tas_mps)
    mode = mode_for_x(cfg, x_m)

    # Along-track state dynamics
    wind_mps = weather.alongtrack_mps(x_m, h_m, state.t_s) # along-track wind speed (m/s)
    delta_isa_K = weather.delta_isa_K(x_m, h_m, state.t_s) # deviation from ISA temperature (K)
    gs_mps = max(1.0, v_tas_mps + wind_mps) # ground_speed = true air speed + along track wind, 1m/s minimum
    xdot = -gs_mps

    # Altitude state dynamics
    h_ref_m = altitude_profile.value(x_m) # reference altitude given from profile
    gamma_ref_rad = path_angle_rad(altitude_profile, x_m) # convert reference altitude to reference gliding angle (3 deg for example)
    hdot_ff = -gs_mps * np.tan(gamma_ref_rad) # reference vertical speed
    hdot_cmd = hdot_ff + cfg.k_h_sinv * (h_ref_m - h_m) # commanded vertical speed, basically a PI controller for altitude
    hdot_cmd = float(np.clip(hdot_cmd, mode.vs_min_mps, mode.vs_max_mps))

    # Speed state dynamics
    v_ref_cas_mps = speed_schedule_cas.value(x_m) # CAS by along-track distance profile
    v_ref_tas_mps = float(aero.cas2tas(v_ref_cas_mps, h_m, dT=delta_isa_K)) # TAS by along-track distance profile
    vdot_cmd = (v_ref_tas_mps - v_tas_mps) / mode.tau_v_s # commanded speed, with "speed lag" factor dictated by tau_v_s (1st order rise time)

    drag_newtons = perf.drag_newtons(
        mode=mode,
        mass_kg=cfg.mass_kg,
        wing_area_m2=cfg.wing_area_m2,
        v_tas_mps=v_tas_mps,
        h_m=h_m,
        vs_mps=hdot_cmd,
        delta_isa_K=delta_isa_K,
    )
    idle_thrust_newtons = perf.idle_thrust_newtons(v_tas_mps, h_m, delta_isa_K=delta_isa_K)

    a_dec_max = max(
        0.0,
        (drag_newtons - idle_thrust_newtons) / cfg.mass_kg - aero.g0 * abs(np.sin(gamma_ref_rad)),
    )
    vdot = float(np.clip(vdot_cmd, -a_dec_max, cfg.a_acc_max_mps2))
    
    return np.asarray([xdot, hdot_cmd, vdot], dtype=float)
