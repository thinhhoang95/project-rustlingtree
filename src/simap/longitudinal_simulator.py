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
    """Inputs that define a longitudinal approach simulation.

    The scenario bundles the reference altitude profile, the raw calibrated-
    airspeed schedule, and the ambient conditions used by the simulator. The
    simulator turns this into a feasible speed schedule during initialization.

    Parameters
    ----------
    altitude_profile:
        Reference altitude over along-track distance.
    raw_speed_schedule_cas:
        Raw calibrated-airspeed schedule before feasibility filtering.
    weather:
        Weather model used during simulation. Defaults to still air and ISA.
    feasibility:
        Settings used to build the feasible speed schedule.
    reference_track_rad:
        Track angle used to project wind into the along-track axis.

    Example
    -------
    >>> scenario = LongitudinalScenario(
    ...     altitude_profile=altitude_profile,
    ...     raw_speed_schedule_cas=speed_schedule,
    ...     reference_track_rad=0.0,
    ... )

    Nuance
    -------
    The `weather` and `feasibility` values are not just bookkeeping: they
    directly affect the feasible speed schedule built by
    `LongitudinalApproachSimulator.__init__`.
    """
    altitude_profile: ScalarProfile
    raw_speed_schedule_cas: ScalarProfile
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    reference_track_rad: float = 0.0


@dataclass(frozen=True)
class LongitudinalTrajectory:
    """Sampled output of a longitudinal simulation run.

    Each field stores one sampled series from the simulation. All arrays should
    have the same length, and `mode` stores the active mode name at each sample.

    Example
    -------
    >>> traj = LongitudinalTrajectory(
    ...     t_s=np.asarray([0.0, 1.0]),
    ...     s_m=np.asarray([12_000.0, 11_930.0]),
    ...     h_m=np.asarray([3_500.0, 3_498.0]),
    ...     v_tas_mps=np.asarray([72.0, 71.5]),
    ...     v_cas_mps=np.asarray([145.0, 144.0]),
    ...     gs_mps=np.asarray([72.0, 71.0]),
    ...     h_ref_m=np.asarray([3_500.0, 3_497.0]),
    ...     v_ref_cas_mps=np.asarray([145.0, 144.0]),
    ...     mode=("approach", "approach"),
    ... )

    Nuance
    -------
    The arrays are stored exactly as produced by `LongitudinalApproachSimulator.run`;
    `to_pandas()` is the convenience method for analysis and plotting.
    """
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
        """Return the number of recorded samples.

        The trajectory length is defined by the time series length.

        Example
        -------
        >>> len(traj)
        2

        Nuance
        -------
        The simulator writes all arrays together, so the returned length should
        match the length of every other field as well.
        """
        return int(len(self.t_s))

    def to_pandas(self) -> pd.DataFrame:
        """Convert the trajectory to a tabular representation.

        This is the preferred form for plotting, inspection, and downstream
        analysis in pandas.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns `t_s`, `s_m`, `h_m`, `v_tas_mps`,
            `v_cas_mps`, `gs_mps`, `h_ref_m`, `v_ref_cas_mps`, and `mode`.

        Example
        -------
        >>> df = traj.to_pandas()
        >>> list(df.columns)
        ['t_s', 's_m', 'h_m', 'v_tas_mps', 'v_cas_mps', 'gs_mps',
         'h_ref_m', 'v_ref_cas_mps', 'mode']

        Nuance
        -------
        The `mode` column is explicitly stored as object dtype so the string
        labels are preserved without implicit numeric coercion.
        """
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
    """Integrate the longitudinal approach model with fixed-step RK4.

    The simulator combines a longitudinal control law, the aircraft performance
    backend, and the scenario definition to produce a time history from an
    initial state. It precomputes a feasible CAS schedule at construction time
    and then uses that schedule during stepping and trajectory generation.

    Example
    -------
    >>> simulator = LongitudinalApproachSimulator(cfg=cfg, perf=perf, scenario=scenario)
    >>> traj = simulator.run(initial_state, dt_s=1.0, t_max_s=3_000.0)

    Nuance
    -------
    The simulator is not a pure integrator: it clamps altitude, speed, and
    along-track progression to keep the model in physically valid bounds.
    """
    def __init__(
        self,
        cfg: AircraftConfig,
        perf: PerformanceBackend,
        scenario: LongitudinalScenario,
    ) -> None:
        """Build a simulator and precompute the feasible CAS schedule.

        Parameters
        ----------
        cfg:
            Aircraft parameters and mode thresholds.
        perf:
            Performance backend used for drag and idle-thrust estimates.
        scenario:
            Simulation inputs including altitude profile, raw speed schedule,
            weather, feasibility settings, and track angle.

        Example
        -------
        >>> simulator = LongitudinalApproachSimulator(cfg=cfg, perf=perf, scenario=scenario)
        >>> isinstance(simulator.feasible_speed_schedule_cas.value(12_000.0), float)
        True

        Nuance
        -------
        The feasible speed schedule is built once here because it depends on the
        performance backend and the scenario feasibility settings, not just the
        aircraft configuration.
        """
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
        """Advance the longitudinal state by one Runge-Kutta step.

        This evaluates `longitudinal_rhs` four times and uses the classic RK4
        update to produce the next state.

        Parameters
        ----------
        state:
            Current longitudinal state.
        dt_s:
            Step size in seconds.

        Returns
        -------
        LongitudinalState
            The updated state, with nonnegative altitude and at least 1 m/s TAS.

        Example
        -------
        >>> next_state = simulator.step(state, dt_s=0.0)
        >>> next_state == state
        True

        Nuance
        -------
        This method integrates only the dynamic state. It does not collect
        samples or stop at the runway threshold; `run()` handles trajectory
        recording and termination.
        """
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
        """Run the simulation and collect a full trajectory.

        The simulator records the current state before each step, then advances
        until either the time limit is reached or the along-track distance drops
        to 1 m or below.

        Parameters
        ----------
        initial:
            Initial longitudinal state.
        dt_s:
            Fixed integration step in seconds.
        t_max_s:
            Maximum simulated time.

        Returns
        -------
        LongitudinalTrajectory
            The sampled trajectory, including state, ground speed, reference
            altitude, reference CAS, and mode at each recorded time.

        Example
        -------
        >>> traj = simulator.run(initial, dt_s=1.0, t_max_s=0.0)
        >>> len(traj)
        1
        >>> list(traj.to_pandas().columns)
        ['t_s', 's_m', 'h_m', 'v_tas_mps', 'v_cas_mps', 'gs_mps',
         'h_ref_m', 'v_ref_cas_mps', 'mode']

        Nuance
        -------
        Sampling happens before stepping, so the final state after the last
        integration step is not appended unless it is reached at the start of a
        loop iteration. This makes the trajectory a history of commanded
        states, not a terminal event log.
        """
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
