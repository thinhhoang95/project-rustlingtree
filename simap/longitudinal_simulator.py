from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from openap import aero

from .backends import PerformanceBackend
from .config import AircraftConfig, mode_for_s
from .longitudinal_dynamics import LongitudinalState, longitudinal_rhs
from .longitudinal_profiles import FeasibilityConfig, ScalarProfile, build_feasible_cas_schedule
from .weather import ConstantWeather, WeatherProvider, alongtrack_wind_mps


@dataclass(frozen=True)
class LongitudinalScenario:
    altitude_profile: ScalarProfile
    raw_speed_schedule_cas: ScalarProfile
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    reference_track_rad: float = 0.0


@dataclass(frozen=True)
class LongitudinalTrajectory:
    t_s: np.ndarray
    s_m: np.ndarray
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    v_cas_mps: np.ndarray
    gs_mps: np.ndarray
    h_ref_m: np.ndarray
    v_ref_cas_mps: np.ndarray
    mode: tuple[str, ...]

    def __len__(self) -> int:
        return int(len(self.t_s))

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t_s": self.t_s,
                "s_m": self.s_m,
                "h_m": self.h_m,
                "v_tas_mps": self.v_tas_mps,
                "v_cas_mps": self.v_cas_mps,
                "gs_mps": self.gs_mps,
                "h_ref_m": self.h_ref_m,
                "v_ref_cas_mps": self.v_ref_cas_mps,
                "mode": np.asarray(self.mode, dtype=object),
            }
        )


class LongitudinalApproachSimulator:
    def __init__(
        self,
        cfg: AircraftConfig,
        perf: PerformanceBackend,
        scenario: LongitudinalScenario,
    ) -> None:
        self.cfg = cfg
        self.perf = perf
        self.scenario = scenario
        self.altitude_profile = scenario.altitude_profile
        self.raw_speed_schedule_cas = scenario.raw_speed_schedule_cas
        self.weather = scenario.weather
        self.feasible_speed_schedule_cas = build_feasible_cas_schedule(
            raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
            altitude_profile=scenario.altitude_profile,
            cfg=cfg,
            perf=perf,
            feasibility=scenario.feasibility,
        )

    def step(self, state: LongitudinalState, dt_s: float) -> LongitudinalState:
        y0 = np.asarray([state.s_m, state.h_m, state.v_tas_mps], dtype=float)

        def f(y: np.ndarray, t_s: float) -> np.ndarray:
            step_state = LongitudinalState(
                t_s=t_s,
                s_m=float(y[0]),
                h_m=float(y[1]),
                v_tas_mps=float(y[2]),
            )
            return longitudinal_rhs(
                state=step_state,
                cfg=self.cfg,
                perf=self.perf,
                altitude_profile=self.altitude_profile,
                speed_schedule_cas=self.feasible_speed_schedule_cas,
                weather=self.weather,
                track_angle_rad=self.scenario.reference_track_rad,
            )

        k1 = f(y0, state.t_s)
        k2 = f(y0 + 0.5 * dt_s * k1, state.t_s + 0.5 * dt_s)
        k3 = f(y0 + 0.5 * dt_s * k2, state.t_s + 0.5 * dt_s)
        k4 = f(y0 + dt_s * k3, state.t_s + dt_s)
        y1 = y0 + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return LongitudinalState(
            t_s=state.t_s + dt_s,
            s_m=max(0.0, float(y1[0])),
            h_m=max(0.0, float(y1[1])),
            v_tas_mps=max(1.0, float(y1[2])),
        )

    def run(
        self,
        initial: LongitudinalState,
        *,
        dt_s: float = 1.0,
        t_max_s: float = 4_000.0,
    ) -> LongitudinalTrajectory:
        rows: dict[str, list[float] | list[str]] = {
            "t_s": [],
            "s_m": [],
            "h_m": [],
            "v_tas_mps": [],
            "v_cas_mps": [],
            "gs_mps": [],
            "h_ref_m": [],
            "v_ref_cas_mps": [],
            "mode": [],
        }

        state = initial
        while state.t_s <= t_max_s and state.s_m > 1.0:
            wind_mps = alongtrack_wind_mps(
                self.weather,
                self.scenario.reference_track_rad,
                state.s_m,
                state.h_m,
                state.t_s,
            )
            delta_isa_K = self.weather.delta_isa_K(state.s_m, state.h_m, state.t_s)
            gs_mps = max(1.0, state.v_tas_mps + wind_mps)
            v_cas_mps = float(aero.tas2cas(state.v_tas_mps, state.h_m, dT=delta_isa_K))
            h_ref_m = self.altitude_profile.value(state.s_m)
            v_ref_cas_mps = self.feasible_speed_schedule_cas.value(state.s_m)
            mode_name = mode_for_s(self.cfg, state.s_m).name

            rows["t_s"].append(state.t_s)
            rows["s_m"].append(state.s_m)
            rows["h_m"].append(state.h_m)
            rows["v_tas_mps"].append(state.v_tas_mps)
            rows["v_cas_mps"].append(v_cas_mps)
            rows["gs_mps"].append(gs_mps)
            rows["h_ref_m"].append(h_ref_m)
            rows["v_ref_cas_mps"].append(v_ref_cas_mps)
            rows["mode"].append(mode_name)

            state = self.step(state, dt_s)

        return LongitudinalTrajectory(
            t_s=np.asarray(rows["t_s"], dtype=float),
            s_m=np.asarray(rows["s_m"], dtype=float),
            h_m=np.asarray(rows["h_m"], dtype=float),
            v_tas_mps=np.asarray(rows["v_tas_mps"], dtype=float),
            v_cas_mps=np.asarray(rows["v_cas_mps"], dtype=float),
            gs_mps=np.asarray(rows["gs_mps"], dtype=float),
            h_ref_m=np.asarray(rows["h_ref_m"], dtype=float),
            v_ref_cas_mps=np.asarray(rows["v_ref_cas_mps"], dtype=float),
            mode=tuple(str(value) for value in rows["mode"]),
        )
