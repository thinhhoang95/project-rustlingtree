from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openap import aero

from .config import AircraftConfig, mode_for_x
from .openap_adapter import wrap_default
from .units import km_to_m


@dataclass(frozen=True)
class FeasibilityConfig:
    planning_tailwind_mps: float = 0.0
    planning_delta_isa_K: float = 0.0
    distance_step_m: float = 250.0


@dataclass(frozen=True)
class ScalarProfile:
    x_m: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("ScalarProfile expects one-dimensional arrays")
        if len(x) != len(y):
            raise ValueError("x_m and y must have the same length")
        if len(x) < 2:
            raise ValueError("ScalarProfile requires at least two nodes")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("x_m must be strictly increasing")
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "y", y)

    def value(self, x_m: float) -> float:
        x = float(np.clip(x_m, self.x_m[0], self.x_m[-1]))
        return float(np.interp(x, self.x_m, self.y))

    def slope(self, x_m: float) -> float:
        x = float(np.clip(x_m, self.x_m[0], self.x_m[-1]))
        index = int(np.searchsorted(self.x_m, x))
        index = max(1, min(index, len(self.x_m) - 1))
        dx = self.x_m[index] - self.x_m[index - 1]
        dy = self.y[index] - self.y[index - 1]
        return float(dy / dx)


@dataclass(frozen=True)
class Centerline:
    x_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=float)
        lat = np.asarray(self.lat_deg, dtype=float)
        lon = np.asarray(self.lon_deg, dtype=float)
        if len(x) != len(lat) or len(x) != len(lon):
            raise ValueError("centerline arrays must have the same length")
        if len(x) < 2:
            raise ValueError("Centerline requires at least two nodes")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("centerline x_m must be strictly increasing")
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "lat_deg", lat)
        object.__setattr__(self, "lon_deg", lon)

    def latlon(self, x_m: float) -> tuple[float, float]:
        x = float(np.clip(x_m, self.x_m[0], self.x_m[-1]))
        lat = float(np.interp(x, self.x_m, self.lat_deg))
        lon = float(np.interp(x, self.x_m, self.lon_deg))
        return lat, lon


def path_angle_rad(altitude_profile: ScalarProfile, x_m: float) -> float:
    return float(np.arctan(altitude_profile.slope(x_m)))


def build_simple_glidepath(
    threshold_elevation_m: float,
    intercept_distance_m: float,
    intercept_altitude_m: float,
    *,
    glide_deg: float = 3.0,
    n: int = 400,
) -> ScalarProfile:
    x = np.linspace(0.0, intercept_distance_m, n, dtype=float)
    h = threshold_elevation_m + np.tan(np.deg2rad(glide_deg)) * x
    h = np.minimum(h, intercept_altitude_m)
    return ScalarProfile(x_m=x, y=h)


def build_speed_schedule_from_wrap(wrap, x_nodes_km: tuple[float, ...] = (0.0, 8.0, 30.0, 60.0)) -> ScalarProfile:
    v_des_mps = wrap_default(wrap, "descent_const_vcas")
    v_final_mps = wrap_default(wrap, "finalapp_vcas")
    v_land_mps = wrap_default(wrap, "landing_speed")

    x_km = np.asarray(x_nodes_km, dtype=float)
    v_mps = np.asarray([v_land_mps, v_final_mps, v_des_mps, v_des_mps], dtype=float)
    if len(x_km) != len(v_mps):
        raise ValueError("x_nodes_km must provide four schedule anchors")
    return ScalarProfile(x_m=km_to_m(x_km), y=v_mps)


def _build_distance_grid(max_x_m: float, step_m: float, *extra_nodes: np.ndarray) -> np.ndarray:
    if step_m <= 0.0:
        raise ValueError("distance_step_m must be positive")
    uniform = np.arange(0.0, max_x_m + step_m, step_m, dtype=float)
    nodes = [uniform, np.asarray([0.0, max_x_m], dtype=float)]
    nodes.extend(np.asarray(extra, dtype=float) for extra in extra_nodes)
    merged = np.unique(np.concatenate(nodes))
    return merged[(merged >= 0.0) & (merged <= max_x_m)]


def build_feasible_cas_schedule(
    raw_speed_schedule_cas: ScalarProfile,
    altitude_profile: ScalarProfile,
    cfg: AircraftConfig,
    perf,
    feasibility: FeasibilityConfig,
) -> ScalarProfile:
    max_x_m = float(max(raw_speed_schedule_cas.x_m[-1], altitude_profile.x_m[-1]))
    x_grid = _build_distance_grid(
        max_x_m,
        feasibility.distance_step_m,
        raw_speed_schedule_cas.x_m,
        altitude_profile.x_m,
    )
    altitudes = np.asarray([altitude_profile.value(x) for x in x_grid], dtype=float)
    feasible_tas = np.empty_like(x_grid)
    feasible_cas = np.empty_like(x_grid)

    initial_cas = raw_speed_schedule_cas.value(0.0)
    feasible_tas[0] = float(aero.cas2tas(initial_cas, altitudes[0], dT=feasibility.planning_delta_isa_K))
    feasible_cas[0] = initial_cas

    for index in range(1, len(x_grid)):
        x_upstream = float(x_grid[index])
        ds_m = float(x_grid[index] - x_grid[index - 1])
        h_m = float(altitudes[index])
        raw_cas = raw_speed_schedule_cas.value(x_upstream)
        raw_tas = float(aero.cas2tas(raw_cas, h_m, dT=feasibility.planning_delta_isa_K))
        mode = mode_for_x(cfg, x_upstream)
        gamma_rad = path_angle_rad(altitude_profile, x_upstream)
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
        feasible_cas[index] = float(
            aero.tas2cas(feasible_tas[index], h_m, dT=feasibility.planning_delta_isa_K)
        )

    return ScalarProfile(x_m=x_grid, y=feasible_cas)
