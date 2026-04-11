from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from openap import aero

from .backends import PerformanceBackend
from .config import AircraftConfig, mode_for_x
from .dynamics import State, rhs
from .profiles import Centerline, FeasibilityConfig, ScalarProfile, build_feasible_cas_schedule
from .weather import ConstantWeather, WeatherProvider


@dataclass(frozen=True)
class Scenario:
    altitude_profile: ScalarProfile
    raw_speed_schedule_cas: ScalarProfile
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    centerline: Centerline | None = None
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)


@dataclass(frozen=True)
class Trajectory:
    t_s: np.ndarray
    x_m: np.ndarray
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    v_cas_mps: np.ndarray
    gs_mps: np.ndarray
    h_ref_m: np.ndarray
    v_ref_cas_mps: np.ndarray
    mode: tuple[str, ...]
    lat_deg: np.ndarray | None = None
    lon_deg: np.ndarray | None = None

    def __len__(self) -> int:
        return int(len(self.t_s))

    def to_pandas(self) -> pd.DataFrame:
        columns: dict[str, object] = {
            "t_s": self.t_s,
            "x_m": self.x_m,
            "h_m": self.h_m,
            "v_tas_mps": self.v_tas_mps,
            "v_cas_mps": self.v_cas_mps,
            "gs_mps": self.gs_mps,
            "h_ref_m": self.h_ref_m,
            "v_ref_cas_mps": self.v_ref_cas_mps,
            "mode": np.asarray(self.mode, dtype=object),
        }
        if self.lat_deg is not None and self.lon_deg is not None:
            columns["lat_deg"] = self.lat_deg
            columns["lon_deg"] = self.lon_deg
        return pd.DataFrame(columns)


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
        self.centerline = scenario.centerline
        self.feasible_speed_schedule_cas = build_feasible_cas_schedule(
            raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
            altitude_profile=scenario.altitude_profile,
            cfg=cfg,
            perf=perf,
            feasibility=scenario.feasibility,
        )

    def step(self, state: State, dt_s: float) -> State:
        y0 = np.asarray([state.x_m, state.h_m, state.v_tas_mps], dtype=float)

        def f(y: np.ndarray, t_s: float) -> np.ndarray:
            step_state = State(
                t_s=t_s,
                x_m=float(y[0]),
                h_m=float(y[1]),
                v_tas_mps=float(y[2]),
            )
            return rhs(
                state=step_state,
                cfg=self.cfg,
                perf=self.perf,
                altitude_profile=self.altitude_profile,
                speed_schedule_cas=self.feasible_speed_schedule_cas,
                weather=self.weather,
            )

        k1 = f(y0, state.t_s)
        k2 = f(y0 + 0.5 * dt_s * k1, state.t_s + 0.5 * dt_s)
        k3 = f(y0 + 0.5 * dt_s * k2, state.t_s + 0.5 * dt_s)
        k4 = f(y0 + dt_s * k3, state.t_s + dt_s)
        y1 = y0 + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return State(
            t_s=state.t_s + dt_s,
            x_m=max(0.0, float(y1[0])),
            h_m=max(0.0, float(y1[1])),
            v_tas_mps=max(1.0, float(y1[2])),
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
            "x_m": [],
            "h_m": [],
            "v_tas_mps": [],
            "v_cas_mps": [],
            "gs_mps": [],
            "h_ref_m": [],
            "v_ref_cas_mps": [],
            "mode": [],
        }
        latitudes: list[float] | None = [] if self.centerline is not None else None
        longitudes: list[float] | None = [] if self.centerline is not None else None

        state = initial
        while state.t_s <= t_max_s and state.x_m > 1.0:
            wind_mps = self.weather.alongtrack_mps(state.x_m, state.h_m, state.t_s)
            delta_isa_K = self.weather.delta_isa_K(state.x_m, state.h_m, state.t_s)
            gs_mps = max(1.0, state.v_tas_mps + wind_mps)
            v_cas_mps = float(aero.tas2cas(state.v_tas_mps, state.h_m, dT=delta_isa_K))
            h_ref_m = self.altitude_profile.value(state.x_m)
            v_ref_cas_mps = self.feasible_speed_schedule_cas.value(state.x_m)
            mode_name = mode_for_x(self.cfg, state.x_m).name

            rows["t_s"].append(state.t_s)
            rows["x_m"].append(state.x_m)
            rows["h_m"].append(state.h_m)
            rows["v_tas_mps"].append(state.v_tas_mps)
            rows["v_cas_mps"].append(v_cas_mps)
            rows["gs_mps"].append(gs_mps)
            rows["h_ref_m"].append(h_ref_m)
            rows["v_ref_cas_mps"].append(v_ref_cas_mps)
            rows["mode"].append(mode_name)

            if self.centerline is not None and latitudes is not None and longitudes is not None:
                lat_deg, lon_deg = self.centerline.latlon(state.x_m)
                latitudes.append(lat_deg)
                longitudes.append(lon_deg)

            state = self.step(state, dt_s)

        return Trajectory(
            t_s=np.asarray(rows["t_s"], dtype=float),
            x_m=np.asarray(rows["x_m"], dtype=float),
            h_m=np.asarray(rows["h_m"], dtype=float),
            v_tas_mps=np.asarray(rows["v_tas_mps"], dtype=float),
            v_cas_mps=np.asarray(rows["v_cas_mps"], dtype=float),
            gs_mps=np.asarray(rows["gs_mps"], dtype=float),
            h_ref_m=np.asarray(rows["h_ref_m"], dtype=float),
            v_ref_cas_mps=np.asarray(rows["v_ref_cas_mps"], dtype=float),
            mode=tuple(str(value) for value in rows["mode"]),
            lat_deg=None if latitudes is None else np.asarray(latitudes, dtype=float),
            lon_deg=None if longitudes is None else np.asarray(longitudes, dtype=float),
        )
