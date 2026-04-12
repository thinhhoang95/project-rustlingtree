from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openap import aero

from .config import AircraftConfig, clamp_cas_for_s, mode_for_s
from .openap_adapter import wrap_default
from .units import km_to_m


@dataclass(frozen=True)
class FeasibilityConfig:
    planning_tailwind_mps: float = 0.0
    planning_delta_isa_K: float = 0.0
    distance_step_m: float = 250.0


@dataclass(frozen=True)
class ScalarProfile:
    s_m: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        s = np.asarray(self.s_m, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if s.ndim != 1 or y.ndim != 1:
            raise ValueError("ScalarProfile expects one-dimensional arrays")
        if len(s) != len(y):
            raise ValueError("s_m and y must have the same length")
        if len(s) < 2:
            raise ValueError("ScalarProfile requires at least two nodes")
        if np.any(np.diff(s) <= 0.0):
            raise ValueError("s_m must be strictly increasing")
        object.__setattr__(self, "s_m", s)
        object.__setattr__(self, "y", y)

    def value(self, s_m: float) -> float:
        s = float(np.clip(s_m, self.s_m[0], self.s_m[-1]))
        return float(np.interp(s, self.s_m, self.y))

    def slope(self, s_m: float) -> float:
        s = float(np.clip(s_m, self.s_m[0], self.s_m[-1]))
        index = int(np.searchsorted(self.s_m, s))
        index = max(1, min(index, len(self.s_m) - 1))
        ds = self.s_m[index] - self.s_m[index - 1]
        dy = self.y[index] - self.y[index - 1]
        return float(dy / ds)


def path_angle_rad(altitude_profile: ScalarProfile, s_m: float) -> float:
    return float(np.arctan(altitude_profile.slope(s_m)))


def build_simple_glidepath(
    threshold_elevation_m: float,
    intercept_distance_m: float,
    intercept_altitude_m: float,
    *,
    glide_deg: float = 3.0,
    n: int = 400,
) -> ScalarProfile:
    s = np.linspace(0.0, intercept_distance_m, n, dtype=float)
    h = threshold_elevation_m + np.tan(np.deg2rad(glide_deg)) * s
    h = np.minimum(h, intercept_altitude_m)
    return ScalarProfile(s_m=s, y=h)


def build_speed_schedule_from_wrap(
    wrap,
    s_nodes_km: tuple[float, ...] = (0.0, 8.0, 30.0, 60.0),
) -> ScalarProfile:
    v_des_mps = wrap_default(wrap, "descent_const_vcas")
    v_final_mps = wrap_default(wrap, "finalapp_vcas")
    v_land_mps = wrap_default(wrap, "landing_speed")

    s_km = np.asarray(s_nodes_km, dtype=float)
    v_mps = np.asarray([v_land_mps, v_final_mps, v_des_mps, v_des_mps], dtype=float)
    if len(s_km) != len(v_mps):
        raise ValueError("s_nodes_km must provide four schedule anchors")
    return ScalarProfile(s_m=km_to_m(s_km), y=v_mps)


def _build_distance_grid(max_s_m: float, step_m: float, *extra_nodes: np.ndarray) -> np.ndarray:
    if step_m <= 0.0:
        raise ValueError("distance_step_m must be positive")
    uniform = np.arange(0.0, max_s_m + step_m, step_m, dtype=float)
    nodes = [uniform, np.asarray([0.0, max_s_m], dtype=float)]
    nodes.extend(np.asarray(extra, dtype=float) for extra in extra_nodes)
    merged = np.unique(np.concatenate(nodes))
    return merged[(merged >= 0.0) & (merged <= max_s_m)]


def build_feasible_cas_schedule(
    raw_speed_schedule_cas: ScalarProfile,
    altitude_profile: ScalarProfile,
    cfg: AircraftConfig,
    perf,
    feasibility: FeasibilityConfig,
) -> ScalarProfile:
    max_s_m = float(max(raw_speed_schedule_cas.s_m[-1], altitude_profile.s_m[-1]))
    s_grid = _build_distance_grid(
        max_s_m,
        feasibility.distance_step_m,
        raw_speed_schedule_cas.s_m,
        altitude_profile.s_m,
    )
    altitudes = np.asarray([altitude_profile.value(s) for s in s_grid], dtype=float)
    feasible_tas = np.empty_like(s_grid)
    feasible_cas = np.empty_like(s_grid)

    initial_cas = clamp_cas_for_s(cfg, 0.0, raw_speed_schedule_cas.value(0.0))
    feasible_tas[0] = float(aero.cas2tas(initial_cas, altitudes[0], dT=feasibility.planning_delta_isa_K))
    feasible_cas[0] = initial_cas

    for index in range(1, len(s_grid)):
        s_upstream = float(s_grid[index])
        ds_m = float(s_grid[index] - s_grid[index - 1])
        h_m = float(altitudes[index])
        raw_cas = clamp_cas_for_s(cfg, s_upstream, raw_speed_schedule_cas.value(s_upstream))
        raw_tas = float(aero.cas2tas(raw_cas, h_m, dT=feasibility.planning_delta_isa_K))
        mode = mode_for_s(cfg, s_upstream)
        gamma_rad = path_angle_rad(altitude_profile, s_upstream)
        downstream_tas = float(feasible_tas[index - 1])
        gs_plan = max(1.0, downstream_tas + feasibility.planning_tailwind_mps)
        vs_plan = -gs_plan * np.tan(gamma_rad)
        drag_newtons = perf.drag_newtons(
            mode=mode,
            mass_kg=cfg.mass_kg,
            wing_area_m2=cfg.wing_area_m2,
            v_tas_mps=downstream_tas,
            h_m=h_m,
            vs_mps=vs_plan,
            delta_isa_K=feasibility.planning_delta_isa_K,
        )
        idle_thrust_newtons = perf.idle_thrust_newtons(
            downstream_tas,
            h_m,
            delta_isa_K=feasibility.planning_delta_isa_K,
        )
        a_dec_max = max(
            0.0,
            (drag_newtons - idle_thrust_newtons) / cfg.mass_kg - aero.g0 * abs(np.sin(gamma_rad)),
        )
        feasible_upstream_tas = downstream_tas + a_dec_max * ds_m / gs_plan
        feasible_tas[index] = min(raw_tas, feasible_upstream_tas)
        feasible_cas[index] = clamp_cas_for_s(
            cfg,
            s_upstream,
            float(aero.tas2cas(feasible_tas[index], h_m, dT=feasibility.planning_delta_isa_K)),
        )
        feasible_tas[index] = float(
            aero.cas2tas(feasible_cas[index], h_m, dT=feasibility.planning_delta_isa_K)
        )

    return ScalarProfile(s_m=s_grid, y=feasible_cas)
