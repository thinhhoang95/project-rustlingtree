from __future__ import annotations

import numpy as np
from openap import aero

from ..backends import PerformanceBackend
from ..config import AircraftConfig, ModeConfig, mode_for_s
from ..openap_adapter import openap_dT
from ..weather import WeatherProvider, alongtrack_wind_mps


def quasi_steady_cl(
    *,
    mass_kg: float,
    wing_area_m2: float,
    v_tas_mps: float,
    h_m: float,
    gamma_rad: float,
    delta_isa_K: float = 0.0,
) -> float:
    v_tas_mps = max(1.0, float(v_tas_mps))
    _, rho, _ = aero.atmos(h_m, dT=openap_dT(delta_isa_K))
    dynamic_pressure = 0.5 * float(rho) * v_tas_mps**2
    lift_newtons = mass_kg * aero.g0 * float(np.cos(gamma_rad))
    return float(lift_newtons / max(dynamic_pressure * wing_area_m2, 1e-6))


def mode_and_atmosphere(
    *,
    cfg: AircraftConfig,
    weather: WeatherProvider,
    s_m: float,
    h_m: float,
    t_s: float,
) -> tuple[ModeConfig, float]:
    mode = mode_for_s(cfg, s_m)
    delta_isa_K = weather.delta_isa_K(s_m, h_m, t_s)
    return mode, float(delta_isa_K)


def distance_state_derivatives(
    *,
    s_m: float,
    h_m: float,
    v_tas_mps: float,
    t_s: float,
    gamma_rad: float,
    thrust_n: float,
    cfg: AircraftConfig,
    perf: PerformanceBackend,
    weather: WeatherProvider,
    reference_track_rad: float = 0.0,
) -> np.ndarray:
    mode, delta_isa_K = mode_and_atmosphere(
        cfg=cfg,
        weather=weather,
        s_m=s_m,
        h_m=h_m,
        t_s=t_s,
    )
    drag_n = perf.drag_newtons(
        mode=mode,
        mass_kg=cfg.mass_kg,
        wing_area_m2=cfg.wing_area_m2,
        v_tas_mps=v_tas_mps,
        h_m=h_m,
        gamma_rad=gamma_rad,
        delta_isa_K=delta_isa_K,
    )
    cos_gamma = float(np.clip(np.cos(gamma_rad), 0.05, None))
    v_tas = max(1.0, float(v_tas_mps))
    ground_speed = max(
        1.0,
        v_tas + alongtrack_wind_mps(weather, reference_track_rad, s_m, h_m, t_s),
    )
    dhds = -float(np.tan(gamma_rad))
    dvds = -(((float(thrust_n) - drag_n) / cfg.mass_kg) - aero.g0 * float(np.sin(gamma_rad))) / (
        v_tas * cos_gamma
    )
    dtds = 1.0 / ground_speed
    return np.asarray([dhds, dvds, dtds], dtype=float)


__all__ = ["distance_state_derivatives", "mode_and_atmosphere", "quasi_steady_cl"]
