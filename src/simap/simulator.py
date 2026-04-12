from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from openap import aero

from .backends import PerformanceBackend
from .config import AircraftConfig, mode_for_s
from .lateral_dynamics import LateralGuidanceConfig, compute_lateral_command, lateral_rates, wrap_angle_rad
from .longitudinal_dynamics import LongitudinalState, longitudinal_rhs
from .longitudinal_profiles import FeasibilityConfig, ScalarProfile, build_feasible_cas_schedule
from .path_geometry import ReferencePath
from .weather import ConstantWeather, WeatherProvider


@dataclass(frozen=True)
class Scenario:
    altitude_profile: ScalarProfile
    raw_speed_schedule_cas: ScalarProfile
    reference_path: ReferencePath
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    lateral_guidance: LateralGuidanceConfig = field(default_factory=LateralGuidanceConfig)


@dataclass(frozen=True)
class State:
    t_s: float
    s_m: float
    h_m: float
    v_tas_mps: float
    east_m: float
    north_m: float
    psi_rad: float
    phi_rad: float = 0.0

    @classmethod
    def on_reference_path(
        cls,
        *,
        t_s: float,
        s_m: float,
        h_m: float,
        v_tas_mps: float,
        reference_path: ReferencePath,
        bank_rad: float = 0.0,
        heading_offset_rad: float = 0.0,
        cross_track_m: float = 0.0,
    ) -> "State":
        """Create an initial state anchored to a reference path location.

        The helper looks up the path position at ``s_m``, shifts the aircraft
        sideways by ``cross_track_m`` along the path normal, and sets the
        initial heading to the local path tangent plus any requested heading
        offset. It is the most convenient way to initialize the simulator when
        the aircraft starts near the procedure rather than at an arbitrary
        latitude/longitude.

        Parameters
        ----------
        t_s, s_m, h_m, v_tas_mps:
            Initial time, path coordinate, altitude, and true airspeed.
        reference_path:
            The path to anchor against.
        bank_rad:
            Initial bank angle. Defaults to wings-level.
        heading_offset_rad:
            Offset applied to the local path tangent, in radians.
        cross_track_m:
            Lateral displacement from the path, measured along the path normal.

        Returns
        -------
        State
            A state positioned near the reference path.

        Examples
        --------
        If the path position at ``s_m=10_000`` is ``(east=2_000, north=3_000)``
        and the local path normal points due north, then
        ``cross_track_m=50`` places the aircraft at ``(2_000, 3_050)``.
        With zero ``heading_offset_rad`` and zero bank, the returned state is
        aligned with the path tangent and wings level.

        Notes
        -----
        The returned coordinates are in the same local north/east frame used by
        the rest of the simulator, not geodetic latitude/longitude.
        """
        east_m, north_m = reference_path.position_ne(s_m)
        normal_hat = reference_path.normal_hat(s_m)
        psi_rad = wrap_angle_rad(reference_path.track_angle_rad(s_m) + heading_offset_rad)
        return cls(
            t_s=t_s,
            s_m=s_m,
            h_m=h_m,
            v_tas_mps=v_tas_mps,
            east_m=east_m + cross_track_m * normal_hat[0],
            north_m=north_m + cross_track_m * normal_hat[1],
            psi_rad=psi_rad,
            phi_rad=bank_rad,
        )


@dataclass(frozen=True)
class Trajectory:
    t_s: np.ndarray
    s_m: np.ndarray
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    v_cas_mps: np.ndarray
    gs_mps: np.ndarray
    h_ref_m: np.ndarray
    v_ref_cas_mps: np.ndarray
    mode: tuple[str, ...]
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    heading_rad: np.ndarray
    bank_rad: np.ndarray
    cross_track_m: np.ndarray

    def __len__(self) -> int:
        """Return the number of samples in the trajectory.

        Examples
        --------
        A trajectory with 121 time samples reports a length of ``121``.

        >>> len(Trajectory(
        ...     t_s=np.arange(3.0),
        ...     s_m=np.arange(3.0),
        ...     h_m=np.arange(3.0),
        ...     v_tas_mps=np.arange(3.0),
        ...     v_cas_mps=np.arange(3.0),
        ...     gs_mps=np.arange(3.0),
        ...     h_ref_m=np.arange(3.0),
        ...     v_ref_cas_mps=np.arange(3.0),
        ...     mode=("clean", "clean", "approach"),
        ...     lat_deg=np.arange(3.0),
        ...     lon_deg=np.arange(3.0),
        ...     heading_rad=np.arange(3.0),
        ...     bank_rad=np.arange(3.0),
        ...     cross_track_m=np.arange(3.0),
        ... ))
        3

        Notes
        -----
        The length is driven by ``t_s``; all trajectory arrays are expected to
        have the same shape because they are populated sample-by-sample in
        lockstep.
        """
        return int(len(self.t_s))

    def to_pandas(self) -> pd.DataFrame:
        """Convert the trajectory to a tabular ``pandas.DataFrame``.

        This is mainly a convenience for analysis, plotting, and exporting.
        Each field in the trajectory becomes a column with the same name.

        Returns
        -------
        pandas.DataFrame
            A dataframe with one row per simulation sample.

        Examples
        --------
        A 3-sample trajectory becomes a 3-row dataframe with columns such as
        ``t_s``, ``h_m``, ``mode``, and ``cross_track_m``. For example, the
        first row might contain ``t_s=0.0``, ``mode='clean'``, and
        ``cross_track_m=12.5``.

        Notes
        -----
        ``mode`` is stored as an object array so that string values survive the
        conversion cleanly when written to CSV or inspected interactively.
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
                "lat_deg": self.lat_deg,
                "lon_deg": self.lon_deg,
                "heading_rad": self.heading_rad,
                "bank_rad": self.bank_rad,
                "cross_track_m": self.cross_track_m,
            }
        )


def coupled_rhs(
    state: State,
    cfg: AircraftConfig,
    perf: PerformanceBackend,
    scenario: Scenario,
    feasible_speed_schedule_cas: ScalarProfile,
) -> np.ndarray:
    """Compute the coupled state derivatives for the longitudinal and lateral model.

    This function is the continuous-time right-hand side used by the simulator's
    integrator. It combines the current state with the lateral guidance law and
    the longitudinal dynamics to produce derivatives for:

    - path coordinate ``s``
    - altitude ``h``
    - true airspeed ``v_tas``
    - east / north position
    - heading ``psi``
    - bank angle ``phi``

    Parameters
    ----------
    state:
        Current simulator state.
    cfg:
        Aircraft configuration.
    perf:
        Performance backend used by the longitudinal model.
    scenario:
        Altitude, speed, weather, and path inputs.
    feasible_speed_schedule_cas:
        The clipped CAS schedule actually used by the longitudinal model.

    Returns
    -------
    numpy.ndarray
        A 7-element derivative vector in the same order used by
        :meth:`ApproachSimulator.step`.

    Examples
    --------
    With a level wings-level state and a straight path, the output is a vector
    of rates such as:

    - ``s_dot``: negative when the aircraft is moving forward along the path
    - ``h_dot``: climb or descent rate from the longitudinal model
    - ``psi_dot``: near zero in coordinated straight flight
    - ``phi_dot``: near zero when the requested bank matches the current bank

    The exact numbers depend on the aircraft performance model and the current
    altitude / speed schedule.

    Notes
    -----
    ``s_dot`` is returned with the simulator's sign convention, where decreasing
    ``s_m`` means progress toward the end of the path.
    """
    mode = mode_for_s(cfg, state.s_m)
    command = compute_lateral_command(
        s_m=state.s_m,
        east_m=state.east_m,
        north_m=state.north_m,
        h_m=state.h_m,
        t_s=state.t_s,
        psi_rad=state.psi_rad,
        v_tas_mps=state.v_tas_mps,
        cfg=cfg,
        mode=mode,
        reference_path=scenario.reference_path,
        weather=scenario.weather,
        guidance=scenario.lateral_guidance,
    )
    s_dot_mps = -command.alongtrack_speed_mps
    long_rates = longitudinal_rhs(
        state=LongitudinalState(
            t_s=state.t_s,
            s_m=state.s_m,
            h_m=state.h_m,
            v_tas_mps=state.v_tas_mps,
        ),
        cfg=cfg,
        perf=perf,
        altitude_profile=scenario.altitude_profile,
        speed_schedule_cas=feasible_speed_schedule_cas,
        weather=scenario.weather,
        track_angle_rad=scenario.reference_path.track_angle_rad(state.s_m),
        bank_rad=state.phi_rad,
        s_dot_mps=s_dot_mps,
    )
    psi_dot_rps, phi_dot_rps = lateral_rates(
        phi_rad=state.phi_rad,
        phi_req_rad=command.phi_req_rad,
        tau_phi_s=mode.tau_phi_s,
        p_max_rps=mode.p_max_rps,
        v_tas_mps=state.v_tas_mps,
    )
    return np.asarray(
        [
            long_rates[0],
            long_rates[1],
            long_rates[2],
            command.east_dot_mps,
            command.north_dot_mps,
            psi_dot_rps,
            phi_dot_rps,
        ],
        dtype=float,
    )


class ApproachSimulator:
    def __init__(
        self,
        cfg: AircraftConfig,
        perf: PerformanceBackend,
        scenario: Scenario,
    ) -> None:
        """Build a simulator for one approach scenario.

        The constructor precomputes the feasible CAS schedule once so repeated
        calls to :meth:`run` or :meth:`step` can reuse it. That matters because
        the longitudinal dynamics query both the raw schedule and the feasibility
        constraints on every integration step.

        Parameters
        ----------
        cfg:
            Aircraft definition.
        perf:
            Performance model backend.
        scenario:
            The path, weather, altitude profile, and requested speed schedule.

        Examples
        --------
        A typical simulator is constructed once and then reused:

        - input: aircraft config, performance backend, scenario
        - output: a simulator instance with cached feasible speed schedule

        Notes
        -----
        This object is stateful only through the cached schedule; the simulation
        state itself is always carried explicitly in :class:`State`.
        """
        self.cfg = cfg
        self.perf = perf
        self.scenario = scenario
        self.altitude_profile = scenario.altitude_profile
        self.raw_speed_schedule_cas = scenario.raw_speed_schedule_cas
        self.weather = scenario.weather
        self.reference_path = scenario.reference_path
        self.feasible_speed_schedule_cas = build_feasible_cas_schedule(
            raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
            altitude_profile=scenario.altitude_profile,
            cfg=cfg,
            perf=perf,
            feasibility=scenario.feasibility,
        )

    def step(self, state: State, dt_s: float) -> State:
        """Advance the simulator state by one Runge-Kutta integration step.

        The method integrates the coupled longitudinal and lateral dynamics
        over ``dt_s`` using a classical RK4 scheme. It is the low-level
        primitive used by :meth:`run`.

        Parameters
        ----------
        state:
            State at the beginning of the step.
        dt_s:
            Integration step size in seconds.

        Returns
        -------
        State
            The propagated state at ``state.t_s + dt_s``.

        Examples
        --------
        If the current state is at ``t_s=12`` and ``dt_s=1``, the returned
        state has ``t_s=13``. In straight, steady flight the position and bank
        angle may change only slightly over one second; during a turn the
        heading and bank angle will evolve more noticeably.

        Notes
        -----
        The method clamps a few physically meaningful quantities after
        integration:

        - ``s_m`` and ``h_m`` are prevented from going below zero
        - ``v_tas_mps`` is prevented from dropping below ``1 m/s``
        - ``psi_rad`` is wrapped back to ``[-pi, pi]``
        """
        y0 = np.asarray(
            [
                state.s_m,
                state.h_m,
                state.v_tas_mps,
                state.east_m,
                state.north_m,
                state.psi_rad,
                state.phi_rad,
            ],
            dtype=float,
        )

        def f(y: np.ndarray, t_s: float) -> np.ndarray:
            """Evaluate the derivative at a temporary RK4 sub-step state."""
            step_state = State(
                t_s=t_s,
                s_m=float(y[0]),
                h_m=float(y[1]),
                v_tas_mps=float(y[2]),
                east_m=float(y[3]),
                north_m=float(y[4]),
                psi_rad=float(y[5]),
                phi_rad=float(y[6]),
            )
            return coupled_rhs(
                state=step_state,
                cfg=self.cfg,
                perf=self.perf,
                scenario=self.scenario,
                feasible_speed_schedule_cas=self.feasible_speed_schedule_cas,
            )

        k1 = f(y0, state.t_s)
        k2 = f(y0 + 0.5 * dt_s * k1, state.t_s + 0.5 * dt_s)
        k3 = f(y0 + 0.5 * dt_s * k2, state.t_s + 0.5 * dt_s)
        k4 = f(y0 + dt_s * k3, state.t_s + dt_s)
        y1 = y0 + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return State(
            t_s=state.t_s + dt_s,
            s_m=max(0.0, float(y1[0])),
            h_m=max(0.0, float(y1[1])),
            v_tas_mps=max(1.0, float(y1[2])),
            east_m=float(y1[3]),
            north_m=float(y1[4]),
            psi_rad=wrap_angle_rad(float(y1[5])),
            phi_rad=float(y1[6]),
        )

    def run(
        self,
        initial: State,
        *,
        dt_s: float = 1.0,
        t_max_s: float = 4_000.0,
    ) -> Trajectory:
        """Run a full simulation until the path is nearly complete or time expires.

        The simulator repeatedly records the current state, then advances it
        with :meth:`step` until either:

        - the elapsed time reaches ``t_max_s``, or
        - ``s_m`` drops to ``1.0`` or below, which means the aircraft has
          effectively reached the end of the reference path.

        Parameters
        ----------
        initial:
            Initial simulator state.
        dt_s:
            Integration step size in seconds.
        t_max_s:
            Hard time limit for the simulation.

        Returns
        -------
        Trajectory
            The recorded simulation history.

        Examples
        --------
        Starting from a state placed on the reference path, ``run`` returns a
        trajectory whose first row matches the initial state and whose last row
        is the final simulated state before termination. Typical output fields
        include:

        - ``t_s``: monotonically increasing sample times
        - ``s_m``: decreasing path coordinate toward the runway threshold
        - ``cross_track_m``: lateral deviation from the centerline

        Notes
        -----
        The method records the state before advancing each step, so the initial
        state is always included in the returned trajectory.
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
            "lat_deg": [],
            "lon_deg": [],
            "heading_rad": [],
            "bank_rad": [],
            "cross_track_m": [],
        }

        state = initial
        while state.t_s <= t_max_s and state.s_m > 1.0:
            mode = mode_for_s(self.cfg, state.s_m)
            command = compute_lateral_command(
                s_m=state.s_m,
                east_m=state.east_m,
                north_m=state.north_m,
                h_m=state.h_m,
                t_s=state.t_s,
                psi_rad=state.psi_rad,
                v_tas_mps=state.v_tas_mps,
                cfg=self.cfg,
                mode=mode,
                reference_path=self.reference_path,
                weather=self.weather,
                guidance=self.scenario.lateral_guidance,
            )
            delta_isa_K = self.weather.delta_isa_K(state.s_m, state.h_m, state.t_s)
            v_cas_mps = float(aero.tas2cas(state.v_tas_mps, state.h_m, dT=delta_isa_K))
            h_ref_m = self.altitude_profile.value(state.s_m)
            v_ref_cas_mps = self.feasible_speed_schedule_cas.value(state.s_m)
            lat_deg, lon_deg = self.reference_path.latlon_from_ne(state.east_m, state.north_m)

            rows["t_s"].append(state.t_s)
            rows["s_m"].append(state.s_m)
            rows["h_m"].append(state.h_m)
            rows["v_tas_mps"].append(state.v_tas_mps)
            rows["v_cas_mps"].append(v_cas_mps)
            rows["gs_mps"].append(command.ground_speed_mps)
            rows["h_ref_m"].append(h_ref_m)
            rows["v_ref_cas_mps"].append(v_ref_cas_mps)
            rows["mode"].append(mode.name)
            rows["lat_deg"].append(lat_deg)
            rows["lon_deg"].append(lon_deg)
            rows["heading_rad"].append(state.psi_rad)
            rows["bank_rad"].append(state.phi_rad)
            rows["cross_track_m"].append(command.cross_track_m)

            state = self.step(state, dt_s)

        return Trajectory(
            t_s=np.asarray(rows["t_s"], dtype=float),
            s_m=np.asarray(rows["s_m"], dtype=float),
            h_m=np.asarray(rows["h_m"], dtype=float),
            v_tas_mps=np.asarray(rows["v_tas_mps"], dtype=float),
            v_cas_mps=np.asarray(rows["v_cas_mps"], dtype=float),
            gs_mps=np.asarray(rows["gs_mps"], dtype=float),
            h_ref_m=np.asarray(rows["h_ref_m"], dtype=float),
            v_ref_cas_mps=np.asarray(rows["v_ref_cas_mps"], dtype=float),
            mode=tuple(str(value) for value in rows["mode"]),
            lat_deg=np.asarray(rows["lat_deg"], dtype=float),
            lon_deg=np.asarray(rows["lon_deg"], dtype=float),
            heading_rad=np.asarray(rows["heading_rad"], dtype=float),
            bank_rad=np.asarray(rows["bank_rad"], dtype=float),
            cross_track_m=np.asarray(rows["cross_track_m"], dtype=float),
        )
