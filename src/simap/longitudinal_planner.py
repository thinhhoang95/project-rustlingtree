from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from openap import aero
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.sparse import csr_matrix, lil_matrix

from .backends import PerformanceBackend
from .config import AircraftConfig, mode_for_s, planned_cas_bounds_mps
from .longitudinal_dynamics import distance_state_derivatives, quasi_steady_cl
from .longitudinal_profiles import ConstraintEnvelope
from .openap_adapter import openap_dT
from .weather import ConstantWeather, WeatherProvider, alongtrack_wind_mps


@dataclass(frozen=True)
class ThresholdBoundary:
    h_m: float
    cas_mps: float
    gamma_rad: float


@dataclass(frozen=True)
class UpstreamBoundary:
    h_m: float
    cas_window_mps: tuple[float, float]
    gamma_rad: float = 0.0

    def __post_init__(self) -> None:
        lower, upper = self.cas_window_mps
        if lower <= 0.0:
            raise ValueError("upstream CAS lower bound must be positive")
        if lower > upper:
            raise ValueError("upstream CAS lower bound must not exceed upper bound")

    @property
    def cas_lower_mps(self) -> float:
        return float(self.cas_window_mps[0])

    @property
    def cas_upper_mps(self) -> float:
        return float(self.cas_window_mps[1])


@dataclass(frozen=True)
class OptimizerConfig:
    num_nodes: int = 41
    maxiter: int = 400
    tod_reward_weight: float = 0.2
    thrust_penalty_weight: float = 0.5
    gamma_smoothness_weight: float = 0.05
    slack_penalty_weight: float = 1_000_000.0
    initial_tod_guess_m: float | None = None
    initial_slack: float = 0.0
    verbose: int = 0
    constraint_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.num_nodes < 4:
            raise ValueError("num_nodes must be at least 4")
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")


@dataclass(frozen=True)
class LongitudinalPlanRequest:
    cfg: AircraftConfig
    perf: PerformanceBackend
    threshold: ThresholdBoundary
    upstream: UpstreamBoundary
    constraints: ConstraintEnvelope
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    reference_track_rad: float = 0.0
    initial_tod_guess_m: float | None = None

    def __post_init__(self) -> None:
        if self.constraints.s_m[0] > 0.0:
            raise ValueError("constraint envelope must start at or before s=0")
        if self.constraints.s_m[-1] <= 0.0:
            raise ValueError("constraint envelope must extend upstream of the threshold")


@dataclass(frozen=True)
class LongitudinalPlanResult:
    s_m: np.ndarray
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    v_cas_mps: np.ndarray
    t_s: np.ndarray
    gamma_rad: np.ndarray
    thrust_n: np.ndarray
    mode: tuple[str, ...]
    solver_success: bool
    solver_status: int
    solver_message: str
    objective_value: float
    tod_m: float
    collocation_residual_max: float
    replay_h_error_m: float
    replay_v_error_mps: float
    replay_t_error_s: float
    replay_residual_max: float
    constraint_slack: float
    solve_profile: "LongitudinalSolveProfile"

    def __len__(self) -> int:
        return int(len(self.s_m))

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "s_m": self.s_m,
                "h_m": self.h_m,
                "v_tas_mps": self.v_tas_mps,
                "v_cas_mps": self.v_cas_mps,
                "t_s": self.t_s,
                "gamma_rad": self.gamma_rad,
                "thrust_n": self.thrust_n,
                "mode": np.asarray(self.mode, dtype=object),
            }
        )


@dataclass(frozen=True)
class LongitudinalSolveProfile:
    total_wall_time_s: float
    postprocess_wall_time_s: float
    objective_calls: int
    objective_time_s: float
    equality_calls: int
    equality_time_s: float
    inequality_calls: int
    inequality_time_s: float
    trajectory_evaluations: int
    trajectory_eval_time_s: float
    trajectory_cache_hits: int


@dataclass
class _SolverProfilingState:
    objective_calls: int = 0
    objective_time_s: float = 0.0
    equality_calls: int = 0
    equality_time_s: float = 0.0
    inequality_calls: int = 0
    inequality_time_s: float = 0.0
    trajectory_evaluations: int = 0
    trajectory_eval_time_s: float = 0.0
    trajectory_cache_hits: int = 0

    def snapshot(self, *, total_wall_time_s: float, postprocess_wall_time_s: float) -> LongitudinalSolveProfile:
        return LongitudinalSolveProfile(
            total_wall_time_s=float(total_wall_time_s),
            postprocess_wall_time_s=float(postprocess_wall_time_s),
            objective_calls=int(self.objective_calls),
            objective_time_s=float(self.objective_time_s),
            equality_calls=int(self.equality_calls),
            equality_time_s=float(self.equality_time_s),
            inequality_calls=int(self.inequality_calls),
            inequality_time_s=float(self.inequality_time_s),
            trajectory_evaluations=int(self.trajectory_evaluations),
            trajectory_eval_time_s=float(self.trajectory_eval_time_s),
            trajectory_cache_hits=int(self.trajectory_cache_hits),
        )


@dataclass
class _TrajectoryEvaluation:
    request: LongitudinalPlanRequest
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    gamma_rad: np.ndarray
    thrust_n: np.ndarray
    tod_m: float
    constraint_slack: float
    _s_m: np.ndarray | None = None
    _t_s: np.ndarray | None = None
    _delta_isa_K: np.ndarray | None = None
    _mode_by_node: tuple[Any, ...] | None = None
    _state_derivatives: np.ndarray | None = None
    _v_cas_mps: np.ndarray | None = None
    _idle_thrust_n: np.ndarray | None = None
    _thrust_bounds_backend: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def num_nodes(self) -> int:
        return int(len(self.h_m))

    @property
    def constant_weather(self) -> ConstantWeather | None:
        weather = self.request.weather
        return weather if isinstance(weather, ConstantWeather) else None

    @property
    def s_m(self) -> np.ndarray:
        if self._s_m is None:
            self._s_m = np.linspace(0.0, self.tod_m, self.num_nodes, dtype=float)
        return self._s_m

    @property
    def t_s(self) -> np.ndarray:
        if self._t_s is None:
            self._t_s = _integrate_time_profile(
                self.request,
                s_m=self.s_m,
                h_m=self.h_m,
                v_tas_mps=self.v_tas_mps,
            )
        return self._t_s

    @property
    def delta_isa_K(self) -> np.ndarray:
        if self._delta_isa_K is None:
            weather = self.constant_weather
            if weather is not None:
                self._delta_isa_K = np.full(self.num_nodes, weather.delta_isa_offset_K, dtype=float)
            else:
                self._delta_isa_K = np.asarray(
                    [
                        float(self.request.weather.delta_isa_K(float(s), float(h), float(t)))
                        for s, h, t in zip(self.s_m, self.h_m, self.t_s, strict=True)
                    ],
                    dtype=float,
                )
        return self._delta_isa_K

    @property
    def mode_by_node(self) -> tuple[Any, ...]:
        if self._mode_by_node is None:
            cfg = self.request.cfg
            s_m = self.s_m
            final_mask = s_m <= cfg.final_gate_m
            approach_mask = (s_m > cfg.final_gate_m) & (s_m <= cfg.approach_gate_m)
            modes = np.empty(self.num_nodes, dtype=object)
            modes[final_mask] = cfg.final
            modes[approach_mask] = cfg.approach
            modes[~(final_mask | approach_mask)] = cfg.clean
            self._mode_by_node = tuple(modes.tolist())
        return self._mode_by_node

    @property
    def state_derivatives(self) -> np.ndarray:
        if self._state_derivatives is None:
            cfg = self.request.cfg
            derivatives = np.empty((self.num_nodes, 3), dtype=float)
            weather = self.constant_weather
            constant_wind = None
            if weather is not None:
                constant_wind = float(
                    weather.wind_east_mps * np.cos(self.request.reference_track_rad)
                    + weather.wind_north_mps * np.sin(self.request.reference_track_rad)
                )
            for idx, (s_val, h_val, v_val, t_val, gamma_val, thrust_val, mode, delta_isa_K) in enumerate(
                zip(
                    self.s_m,
                    self.h_m,
                    self.v_tas_mps,
                    self.t_s,
                    self.gamma_rad,
                    self.thrust_n,
                    self.mode_by_node,
                    self.delta_isa_K,
                    strict=True,
                )
            ):
                drag_n = self.request.perf.drag_newtons(
                    mode=mode,
                    mass_kg=cfg.mass_kg,
                    wing_area_m2=cfg.wing_area_m2,
                    v_tas_mps=float(v_val),
                    h_m=float(h_val),
                    gamma_rad=float(gamma_val),
                    delta_isa_K=float(delta_isa_K),
                )
                cos_gamma = float(np.clip(np.cos(gamma_val), 0.05, None))
                v_tas = max(1.0, float(v_val))
                alongtrack_wind = (
                    float(constant_wind)
                    if constant_wind is not None
                    else alongtrack_wind_mps(
                        self.request.weather,
                        self.request.reference_track_rad,
                        float(s_val),
                        float(h_val),
                        float(t_val),
                    )
                )
                ground_speed = max(1.0, v_tas + alongtrack_wind)
                derivatives[idx, 0] = -float(np.tan(gamma_val))
                derivatives[idx, 1] = -(((float(thrust_val) - drag_n) / cfg.mass_kg) - aero.g0 * float(np.sin(gamma_val))) / (
                    v_tas * cos_gamma
                )
                derivatives[idx, 2] = 1.0 / ground_speed
            self._state_derivatives = derivatives
        return self._state_derivatives

    @property
    def v_cas_mps(self) -> np.ndarray:
        if self._v_cas_mps is None:
            weather = self.constant_weather
            if weather is not None:
                self._v_cas_mps = np.asarray(
                    aero.tas2cas(self.v_tas_mps, self.h_m, dT=openap_dT(weather.delta_isa_offset_K)),
                    dtype=float,
                )
            else:
                self._v_cas_mps = np.asarray(
                    [
                        float(aero.tas2cas(float(v_tas), float(h), dT=openap_dT(float(delta_isa_K))))
                        for v_tas, h, delta_isa_K in zip(self.v_tas_mps, self.h_m, self.delta_isa_K, strict=True)
                    ],
                    dtype=float,
                )
        return self._v_cas_mps

    @property
    def idle_thrust_n(self) -> np.ndarray:
        if self._idle_thrust_n is None:
            self._idle_thrust_n = np.asarray(
                [
                    float(
                        self.request.perf.idle_thrust_newtons(
                            v_tas_mps=float(v_tas),
                            h_m=float(h),
                            delta_isa_K=float(delta_isa_K),
                        )
                    )
                    for v_tas, h, delta_isa_K in zip(self.v_tas_mps, self.h_m, self.delta_isa_K, strict=True)
                ],
                dtype=float,
            )
        return self._idle_thrust_n

    @property
    def thrust_bounds_backend(self) -> tuple[np.ndarray, np.ndarray]:
        if self._thrust_bounds_backend is None:
            lower = np.empty(self.num_nodes, dtype=float)
            upper = np.empty(self.num_nodes, dtype=float)
            for idx, (mode, v_tas, h, delta_isa_K) in enumerate(
                zip(self.mode_by_node, self.v_tas_mps, self.h_m, self.delta_isa_K, strict=True)
            ):
                lower[idx], upper[idx] = self.request.perf.thrust_bounds_newtons(
                    mode=mode,
                    v_tas_mps=float(v_tas),
                    h_m=float(h),
                    delta_isa_K=float(delta_isa_K),
                )
            self._thrust_bounds_backend = (lower, upper)
        return self._thrust_bounds_backend


@dataclass
class _TrajectoryEvaluationCache:
    request: LongitudinalPlanRequest
    profiling: _SolverProfilingState
    _last_key: bytes | None = None
    _last_eval: _TrajectoryEvaluation | None = None

    def evaluate(self, z: np.ndarray) -> _TrajectoryEvaluation:
        z_arr = np.ascontiguousarray(np.asarray(z, dtype=float))
        key = z_arr.tobytes()
        if self._last_key == key and self._last_eval is not None:
            self.profiling.trajectory_cache_hits += 1
            return self._last_eval

        started_at = perf_counter()
        h_m, v_tas_mps, gamma_rad, thrust_n, tod_m, constraint_slack = _unpack(z_arr, self.request.optimizer.num_nodes)
        evaluation = _TrajectoryEvaluation(
            request=self.request,
            h_m=np.array(h_m, dtype=float, copy=True),
            v_tas_mps=np.array(v_tas_mps, dtype=float, copy=True),
            gamma_rad=np.array(gamma_rad, dtype=float, copy=True),
            thrust_n=np.array(thrust_n, dtype=float, copy=True),
            tod_m=float(tod_m),
            constraint_slack=float(constraint_slack),
        )
        self._last_key = key
        self._last_eval = evaluation
        self.profiling.trajectory_evaluations += 1
        self.profiling.trajectory_eval_time_s += perf_counter() - started_at
        return evaluation


def plan_longitudinal_descent(request: LongitudinalPlanRequest) -> LongitudinalPlanResult:
    solve_started_at = perf_counter()
    optimizer = request.optimizer
    num_nodes = optimizer.num_nodes
    max_tod_m = float(request.constraints.s_m[-1])
    threshold_delta_isa = request.weather.delta_isa_K(0.0, request.threshold.h_m, 0.0)
    threshold_v_tas = float(aero.cas2tas(request.threshold.cas_mps, request.threshold.h_m, dT=openap_dT(threshold_delta_isa)))
    initial_tod_guess_m = _initial_tod_guess(request, threshold_v_tas=threshold_v_tas, max_tod_m=max_tod_m)

    scale = _PlannerScale.from_request(request)
    z0 = _initial_guess(
        request=request,
        threshold_v_tas=threshold_v_tas,
        initial_tod_guess_m=initial_tod_guess_m,
        scale=scale,
    )
    bounds = _decision_bounds(
        request=request,
        threshold_v_tas=threshold_v_tas,
        max_tod_m=max_tod_m,
        scale=scale,
    )
    profiling = _SolverProfilingState()
    evaluation_cache = _TrajectoryEvaluationCache(request=request, profiling=profiling)
    equality_jac_sparsity, inequality_jac_sparsity = _constraint_jacobian_sparsity(request)

    def objective(z: np.ndarray) -> float:
        started_at = perf_counter()
        try:
            return _objective(z, request=request, scale=scale, evaluation_cache=evaluation_cache)
        finally:
            profiling.objective_calls += 1
            profiling.objective_time_s += perf_counter() - started_at

    def equality_fun(z: np.ndarray) -> np.ndarray:
        started_at = perf_counter()
        try:
            return _equality_constraints(
                z,
                request=request,
                threshold_v_tas=threshold_v_tas,
                evaluation_cache=evaluation_cache,
            )
        finally:
            profiling.equality_calls += 1
            profiling.equality_time_s += perf_counter() - started_at

    def inequality_fun(z: np.ndarray) -> np.ndarray:
        started_at = perf_counter()
        try:
            return _inequality_constraints(
                z,
                request=request,
                scale=scale,
                evaluation_cache=evaluation_cache,
            )
        finally:
            profiling.inequality_calls += 1
            profiling.inequality_time_s += perf_counter() - started_at

    equality = NonlinearConstraint(
        equality_fun,
        0.0,
        0.0,
        finite_diff_jac_sparsity=equality_jac_sparsity,
    )
    inequality = NonlinearConstraint(
        inequality_fun,
        0.0,
        np.inf,
        finite_diff_jac_sparsity=inequality_jac_sparsity,
    )

    solver_options = {
        "maxiter": optimizer.maxiter,
        "verbose": optimizer.verbose,
    }
    if equality_jac_sparsity is not None and inequality_jac_sparsity is not None:
        solver_options["sparse_jacobian"] = True
    result = minimize(
        objective,
        z0,
        method="trust-constr",
        bounds=bounds,
        constraints=[equality, inequality],
        options=solver_options,
    )

    postprocess_started_at = perf_counter()
    h_m, v_tas_mps, gamma_rad, thrust_n, tod_m, constraint_slack = _unpack(result.x, optimizer.num_nodes)
    h_m = np.array(h_m, dtype=float, copy=True)
    v_tas_mps = np.array(v_tas_mps, dtype=float, copy=True)
    gamma_rad = np.array(gamma_rad, dtype=float, copy=True)
    thrust_n = np.array(thrust_n, dtype=float, copy=True)
    h_m[0] = request.threshold.h_m
    v_tas_mps[0] = threshold_v_tas
    gamma_rad[0] = request.threshold.gamma_rad
    h_m[-1] = request.upstream.h_m
    gamma_rad[-1] = request.upstream.gamma_rad
    final_evaluation = _TrajectoryEvaluation(
        request=request,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        tod_m=float(tod_m),
        constraint_slack=float(constraint_slack),
    )
    s_m = np.array(final_evaluation.s_m, dtype=float, copy=True)
    t_s = np.array(final_evaluation.t_s, dtype=float, copy=True)
    v_cas_mps = np.array(final_evaluation.v_cas_mps, dtype=float, copy=True)
    collocation_residual_max = float(
        np.max(
            np.abs(
                _equality_constraints(
                    result.x,
                    request=request,
                    threshold_v_tas=threshold_v_tas,
                    evaluation_cache=evaluation_cache,
                )
            )
        )
    )
    replay = _replay_solution(
        request=request,
        s_m=s_m,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        t_s=t_s,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
    )
    mode = tuple(mode_for_s(request.cfg, float(s)).name for s in s_m)
    solve_profile = profiling.snapshot(
        total_wall_time_s=perf_counter() - solve_started_at,
        postprocess_wall_time_s=perf_counter() - postprocess_started_at,
    )
    return LongitudinalPlanResult(
        s_m=s_m,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        v_cas_mps=v_cas_mps,
        t_s=t_s,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        mode=mode,
        solver_success=bool(result.success),
        solver_status=int(result.status),
        solver_message=str(result.message),
        objective_value=float(result.fun),
        tod_m=float(tod_m),
        collocation_residual_max=collocation_residual_max,
        replay_h_error_m=float(replay["h_error_m"]),
        replay_v_error_mps=float(replay["v_error_mps"]),
        replay_t_error_s=float(replay["t_error_s"]),
        replay_residual_max=float(replay["max_error"]),
        constraint_slack=float(constraint_slack),
        solve_profile=solve_profile,
    )


@dataclass(frozen=True)
class _PlannerScale:
    altitude_m: float
    speed_mps: float
    gamma_rad: float
    thrust_n: float

    @classmethod
    def from_request(cls, request: LongitudinalPlanRequest) -> "_PlannerScale":
        altitude_span = max(
            np.ptp(request.constraints.h_lower_m),
            np.ptp(request.constraints.h_upper_m),
            abs(request.upstream.h_m - request.threshold.h_m),
            100.0,
        )
        speed_span = max(
            np.ptp(request.constraints.cas_lower_mps),
            np.ptp(request.constraints.cas_upper_mps),
            request.upstream.cas_upper_mps - request.threshold.cas_mps,
            5.0,
        )
        thrust_candidates: list[float] = []
        for mode in (request.cfg.clean, request.cfg.approach, request.cfg.final):
            lower, upper = request.perf.thrust_bounds_newtons(
                mode=mode,
                v_tas_mps=max(1.0, request.threshold.cas_mps),
                h_m=max(request.threshold.h_m, request.upstream.h_m),
            )
            thrust_candidates.extend([lower, upper])
        thrust_scale = max(1_000.0, max(abs(value) for value in thrust_candidates))
        return cls(
            altitude_m=float(altitude_span),
            speed_mps=float(speed_span),
            gamma_rad=float(max(np.deg2rad(1.0), abs(request.threshold.gamma_rad - request.upstream.gamma_rad))),
            thrust_n=float(thrust_scale),
        )


def _initial_tod_guess(
    request: LongitudinalPlanRequest,
    *,
    threshold_v_tas: float,
    max_tod_m: float,
) -> float:
    del threshold_v_tas
    guess = request.initial_tod_guess_m
    if guess is None:
        guess = request.optimizer.initial_tod_guess_m
    if guess is None:
        descent_gamma = max(abs(request.threshold.gamma_rad), np.deg2rad(3.0))
        guess = (request.upstream.h_m - request.threshold.h_m) / max(np.tan(descent_gamma), 1e-3)
    return float(np.clip(guess, 1_000.0, max_tod_m))


def _initial_guess(
    *,
    request: LongitudinalPlanRequest,
    threshold_v_tas: float,
    initial_tod_guess_m: float,
    scale: _PlannerScale,
) -> np.ndarray:
    del threshold_v_tas, scale
    num_nodes = request.optimizer.num_nodes
    s_m = np.linspace(0.0, initial_tod_guess_m, num_nodes, dtype=float)
    h_m, gamma_rad = _cubic_altitude_guess(
        s_m=s_m,
        s_tod_m=initial_tod_guess_m,
        h0_m=request.threshold.h_m,
        h1_m=request.upstream.h_m,
        gamma0_rad=request.threshold.gamma_rad,
        gamma1_rad=request.upstream.gamma_rad,
    )
    cas_lower_end = max(request.upstream.cas_lower_mps, request.constraints.cas_bounds(initial_tod_guess_m)[0])
    cas_upper_end = min(request.upstream.cas_upper_mps, request.constraints.cas_bounds(initial_tod_guess_m)[1])
    cas_mid_end = 0.5 * (cas_lower_end + cas_upper_end)
    cas_guess = np.linspace(request.threshold.cas_mps, cas_mid_end, num_nodes, dtype=float)
    if isinstance(request.weather, ConstantWeather):
        v_tas_mps = np.asarray(
            aero.cas2tas(cas_guess, h_m, dT=openap_dT(request.weather.delta_isa_offset_K)),
            dtype=float,
        )
    else:
        v_tas_mps = np.empty_like(cas_guess)
        for idx, (s_val, h_val, cas_val) in enumerate(zip(s_m, h_m, cas_guess, strict=True)):
            delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_val), 0.0)
            v_tas_mps[idx] = float(aero.cas2tas(float(cas_val), float(h_val), dT=openap_dT(delta_isa_K)))

    t_s = _integrate_time_profile(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps)

    thrust_n = np.empty_like(s_m)
    v_slope = np.gradient(v_tas_mps, s_m, edge_order=1)
    for idx, (s_val, h_val, v_val, gamma_val, dvds) in enumerate(
        zip(s_m, h_m, v_tas_mps, gamma_rad, v_slope, strict=True)
    ):
        mode = mode_for_s(request.cfg, float(s_val))
        delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_val), float(t_s[idx]))
        drag_n = request.perf.drag_newtons(
            mode=mode,
            mass_kg=request.cfg.mass_kg,
            wing_area_m2=request.cfg.wing_area_m2,
            v_tas_mps=float(v_val),
            h_m=float(h_val),
            gamma_rad=float(gamma_val),
            delta_isa_K=delta_isa_K,
        )
        thrust_lower, thrust_upper = request.perf.thrust_bounds_newtons(
            mode=mode,
            v_tas_mps=float(v_val),
            h_m=float(h_val),
            delta_isa_K=delta_isa_K,
        )
        thrust_guess = drag_n + request.cfg.mass_kg * (
            aero.g0 * np.sin(gamma_val) - float(v_val) * np.cos(gamma_val) * float(dvds)
        )
        thrust_n[idx] = float(np.clip(thrust_guess, thrust_lower, thrust_upper))

    return _pack(
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        tod_m=initial_tod_guess_m,
        constraint_slack=request.optimizer.initial_slack,
    )


def _cubic_altitude_guess(
    *,
    s_m: np.ndarray,
    s_tod_m: float,
    h0_m: float,
    h1_m: float,
    gamma0_rad: float,
    gamma1_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(s_m, dtype=float) / max(float(s_tod_m), 1.0)
    slope0 = -np.tan(gamma0_rad)
    slope1 = -np.tan(gamma1_rad)
    h00 = 2.0 * x**3 - 3.0 * x**2 + 1.0
    h10 = x**3 - 2.0 * x**2 + x
    h01 = -2.0 * x**3 + 3.0 * x**2
    h11 = x**3 - x**2
    h_m = h00 * h0_m + h10 * s_tod_m * slope0 + h01 * h1_m + h11 * s_tod_m * slope1

    dh00 = (6.0 * x**2 - 6.0 * x) / max(float(s_tod_m), 1.0)
    dh10 = 3.0 * x**2 - 4.0 * x + 1.0
    dh01 = (-6.0 * x**2 + 6.0 * x) / max(float(s_tod_m), 1.0)
    dh11 = 3.0 * x**2 - 2.0 * x
    dhds = dh00 * h0_m + dh10 * slope0 + dh01 * h1_m + dh11 * slope1
    gamma_rad = -np.arctan(dhds)
    return np.asarray(h_m, dtype=float), np.asarray(gamma_rad, dtype=float)


def _decision_bounds(
    *,
    request: LongitudinalPlanRequest,
    threshold_v_tas: float,
    max_tod_m: float,
    scale: _PlannerScale,
) -> Bounds:
    num_nodes = request.optimizer.num_nodes
    upper_altitude = max(float(np.max(request.constraints.h_upper_m)), request.upstream.h_m) + 0.25 * scale.altitude_m
    lower_altitude = max(0.0, min(float(np.min(request.constraints.h_lower_m)), request.threshold.h_m) - 0.1 * scale.altitude_m)

    sample_upper_tas = []
    for s_val, h_upper, cas_upper in zip(
        request.constraints.s_m,
        request.constraints.h_upper_m,
        request.constraints.cas_upper_mps,
        strict=True,
    ):
        cas_clipped = min(float(cas_upper), request.cfg.vmo_kts * aero.kts)
        delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_upper), 0.0)
        sample_upper_tas.append(float(aero.cas2tas(cas_clipped, float(h_upper), dT=openap_dT(delta_isa_K))))
    upper_tas = max(max(sample_upper_tas, default=threshold_v_tas), threshold_v_tas) * 1.15

    thrust_upper = 1.25 * scale.thrust_n
    gamma_limit = np.deg2rad(20.0)
    lower = np.concatenate(
        [
            np.full(num_nodes, lower_altitude, dtype=float),
            np.full(num_nodes, 1.0, dtype=float),
            np.full(num_nodes, -gamma_limit, dtype=float),
            np.zeros(num_nodes, dtype=float),
            np.asarray([1_000.0, 0.0], dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.full(num_nodes, upper_altitude, dtype=float),
            np.full(num_nodes, upper_tas, dtype=float),
            np.full(num_nodes, np.deg2rad(5.0), dtype=float),
            np.full(num_nodes, thrust_upper, dtype=float),
            np.asarray([max_tod_m, 5.0], dtype=float),
        ]
    )
    return cast(Any, Bounds)(lower, upper)


def _objective(
    z: np.ndarray,
    *,
    request: LongitudinalPlanRequest,
    scale: _PlannerScale,
    evaluation_cache: _TrajectoryEvaluationCache,
) -> float:
    evaluation = evaluation_cache.evaluate(z)
    ds_m = max(evaluation.tod_m / max(request.optimizer.num_nodes - 1, 1), 1.0)
    thrust_delta = (evaluation.thrust_n - evaluation.idle_thrust_n) / max(1.0, scale.thrust_n)
    gamma_gradient = np.diff(evaluation.gamma_rad) / max(ds_m, 1.0)
    objective = (
        -request.optimizer.tod_reward_weight * (evaluation.tod_m / 1_000.0)
        + request.optimizer.thrust_penalty_weight
        * (evaluation.tod_m / 1_000.0)
        * float(np.mean(thrust_delta**2))
        + request.optimizer.gamma_smoothness_weight
        * float(np.mean((gamma_gradient / np.deg2rad(1.0) * 1_000.0) ** 2))
        + request.optimizer.slack_penalty_weight * float(evaluation.constraint_slack**2)
    )
    return float(objective)


def _equality_constraints(
    z: np.ndarray,
    *,
    request: LongitudinalPlanRequest,
    threshold_v_tas: float,
    evaluation_cache: _TrajectoryEvaluationCache,
) -> np.ndarray:
    evaluation = evaluation_cache.evaluate(z)
    ds_m = float(evaluation.tod_m / (request.optimizer.num_nodes - 1))
    trapezoid = 0.5 * ds_m * (evaluation.state_derivatives[:-1, :2] + evaluation.state_derivatives[1:, :2])
    state_delta = np.column_stack(
        [
            np.diff(evaluation.h_m),
            np.diff(evaluation.v_tas_mps),
        ]
    )
    defects = np.empty(2 * (request.optimizer.num_nodes - 1) + 5, dtype=float)
    defects[: 2 * (request.optimizer.num_nodes - 1)] = (state_delta - trapezoid).reshape(-1)
    defects[-5:] = np.asarray(
        [
            float(evaluation.h_m[0] - request.threshold.h_m),
            float(evaluation.v_tas_mps[0] - threshold_v_tas),
            float(evaluation.gamma_rad[0] - request.threshold.gamma_rad),
            float(evaluation.h_m[-1] - request.upstream.h_m),
            float(evaluation.gamma_rad[-1] - request.upstream.gamma_rad),
        ],
        dtype=float,
    )
    return defects


def _inequality_constraints(
    z: np.ndarray,
    *,
    request: LongitudinalPlanRequest,
    scale: _PlannerScale,
    evaluation_cache: _TrajectoryEvaluationCache,
) -> np.ndarray:
    evaluation = evaluation_cache.evaluate(z)
    slack_altitude = evaluation.constraint_slack * scale.altitude_m
    slack_speed = evaluation.constraint_slack * scale.speed_mps
    slack_gamma = evaluation.constraint_slack * scale.gamma_rad
    slack_thrust = evaluation.constraint_slack * scale.thrust_n

    h_lower, h_upper = request.constraints.h_bounds_many(evaluation.s_m)
    cas_lower_env, cas_upper_env = request.constraints.cas_bounds_many(evaluation.s_m)
    mode_lower, mode_upper = _planned_cas_bounds_many(request.cfg, evaluation.s_m)
    cas_lower_eff = np.maximum(cas_lower_env, mode_lower)
    cas_upper_eff = np.minimum(cas_upper_env, mode_upper)

    thrust_lower_backend, thrust_upper_backend = evaluation.thrust_bounds_backend
    thrust_lower_env, thrust_upper_env = request.constraints.thrust_bounds_many(evaluation.s_m)
    thrust_lower = thrust_lower_backend if thrust_lower_env is None else np.maximum(thrust_lower_backend, thrust_lower_env)
    thrust_upper = thrust_upper_backend if thrust_upper_env is None else np.minimum(thrust_upper_backend, thrust_upper_env)

    residual_parts = [
        evaluation.h_m - h_lower + slack_altitude,
        h_upper - evaluation.h_m + slack_altitude,
        evaluation.v_cas_mps - cas_lower_eff + slack_speed,
        cas_upper_eff - evaluation.v_cas_mps + slack_speed,
        evaluation.thrust_n - thrust_lower + slack_thrust,
        thrust_upper - evaluation.thrust_n + slack_thrust,
    ]

    gamma_lower, gamma_upper = request.constraints.gamma_bounds_many(evaluation.s_m)
    if gamma_lower is not None:
        residual_parts.append(evaluation.gamma_rad - gamma_lower + slack_gamma)
    if gamma_upper is not None:
        residual_parts.append(gamma_upper - evaluation.gamma_rad + slack_gamma)

    if request.constraints.cl_max is not None:
        cl_upper = request.constraints._interp_many_required(request.constraints.cl_max, evaluation.s_m)
        cl_values = np.asarray(
            [
                float(
                    quasi_steady_cl(
                        mass_kg=request.cfg.mass_kg,
                        wing_area_m2=request.cfg.wing_area_m2,
                        v_tas_mps=float(v_tas),
                        h_m=float(h),
                        gamma_rad=float(gamma),
                        delta_isa_K=float(delta_isa_K),
                    )
                )
                for v_tas, h, gamma, delta_isa_K in zip(
                    evaluation.v_tas_mps,
                    evaluation.h_m,
                    evaluation.gamma_rad,
                    evaluation.delta_isa_K,
                    strict=True,
                )
            ],
            dtype=float,
        )
        residual_parts.append(cl_upper - cl_values + evaluation.constraint_slack)

    residual_parts.append(
        np.asarray(
            [
                float(evaluation.v_cas_mps[-1] - request.upstream.cas_lower_mps + slack_speed),
                float(request.upstream.cas_upper_mps - evaluation.v_cas_mps[-1] + slack_speed),
            ],
            dtype=float,
        )
    )
    return np.concatenate(residual_parts)


def _cas_from_tas(
    request: LongitudinalPlanRequest,
    *,
    s_m: np.ndarray,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    t_s: np.ndarray,
) -> np.ndarray:
    weather = request.weather
    if isinstance(weather, ConstantWeather):
        return np.asarray(aero.tas2cas(v_tas_mps, h_m, dT=openap_dT(weather.delta_isa_offset_K)), dtype=float)

    result = np.empty_like(v_tas_mps)
    for idx, (s_val, h_val, v_tas, t_val) in enumerate(zip(s_m, h_m, v_tas_mps, t_s, strict=True)):
        delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_val), float(t_val))
        result[idx] = float(aero.tas2cas(float(v_tas), float(h_val), dT=openap_dT(delta_isa_K)))
    return result


def _replay_solution(
    *,
    request: LongitudinalPlanRequest,
    s_m: np.ndarray,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    t_s: np.ndarray,
    gamma_rad: np.ndarray,
    thrust_n: np.ndarray,
) -> dict[str, float]:
    def rhs(s_val: float, state: np.ndarray) -> np.ndarray:
        gamma_val = float(np.interp(s_val, s_m, gamma_rad))
        thrust_val = float(np.interp(s_val, s_m, thrust_n))
        derivatives = distance_state_derivatives(
            s_m=float(s_val),
            h_m=float(state[0]),
            v_tas_mps=float(state[1]),
            t_s=float(np.interp(s_val, s_m, t_s)),
            gamma_rad=gamma_val,
            thrust_n=thrust_val,
            cfg=request.cfg,
            perf=request.perf,
            weather=request.weather,
            reference_track_rad=request.reference_track_rad,
        )
        return derivatives[:2]

    replay = solve_ivp(
        rhs,
        (float(s_m[0]), float(s_m[-1])),
        y0=np.asarray([h_m[0], v_tas_mps[0]], dtype=float),
        t_eval=s_m,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    replay_t_s = _integrate_time_profile(request, s_m=s_m, h_m=replay.y[0], v_tas_mps=replay.y[1])
    h_error = float(np.max(np.abs(replay.y[0] - h_m)))
    v_error = float(np.max(np.abs(replay.y[1] - v_tas_mps)))
    t_error = float(np.max(np.abs(replay_t_s - t_s)))
    return {
        "h_error_m": h_error,
        "v_error_mps": v_error,
        "t_error_s": t_error,
        "max_error": max(h_error, v_error, t_error),
    }


def _pack(
    *,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    gamma_rad: np.ndarray,
    thrust_n: np.ndarray,
    tod_m: float,
    constraint_slack: float,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(h_m, dtype=float),
            np.asarray(v_tas_mps, dtype=float),
            np.asarray(gamma_rad, dtype=float),
            np.asarray(thrust_n, dtype=float),
            np.asarray([float(tod_m), float(constraint_slack)], dtype=float),
        ]
    )


def _unpack(
    z: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    h_end = num_nodes
    v_end = h_end + num_nodes
    gamma_end = v_end + num_nodes
    thrust_end = gamma_end + num_nodes
    h_m = np.asarray(z[:h_end], dtype=float)
    v_tas_mps = np.asarray(z[h_end:v_end], dtype=float)
    gamma_rad = np.asarray(z[v_end:gamma_end], dtype=float)
    thrust_n = np.asarray(z[gamma_end:thrust_end], dtype=float)
    tod_m, constraint_slack = np.asarray(z[thrust_end : thrust_end + 2], dtype=float)
    return h_m, v_tas_mps, gamma_rad, thrust_n, float(tod_m), float(constraint_slack)


def _integrate_time_profile(
    request: LongitudinalPlanRequest,
    *,
    s_m: np.ndarray,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
) -> np.ndarray:
    weather = request.weather
    if isinstance(weather, ConstantWeather):
        segment_ground_speed = 0.5 * (v_tas_mps[:-1] + v_tas_mps[1:]) + (
            weather.wind_east_mps * np.cos(request.reference_track_rad)
            + weather.wind_north_mps * np.sin(request.reference_track_rad)
        )
        segment_ground_speed = np.maximum(1.0, segment_ground_speed)
        increments = np.diff(s_m) / segment_ground_speed
        return np.concatenate([np.zeros(1, dtype=float), np.cumsum(increments, dtype=float)])

    time_s = np.zeros_like(s_m, dtype=float)
    for idx in range(1, len(s_m)):
        ground_speed = max(
            1.0,
            0.5 * (v_tas_mps[idx - 1] + v_tas_mps[idx])
            + alongtrack_wind_mps(
                request.weather,
                request.reference_track_rad,
                float(0.5 * (s_m[idx - 1] + s_m[idx])),
                float(0.5 * (h_m[idx - 1] + h_m[idx])),
                float(time_s[idx - 1]),
            ),
        )
        time_s[idx] = time_s[idx - 1] + (s_m[idx] - s_m[idx - 1]) / ground_speed
    return time_s


def _planned_cas_bounds_many(cfg: AircraftConfig, s_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(s_m, dtype=float)
    lower = np.empty_like(points)
    upper = np.empty_like(points)
    final_mask = points <= cfg.final_gate_m
    approach_mask = (points > cfg.final_gate_m) & (points <= cfg.approach_gate_m)
    clean_mask = ~(final_mask | approach_mask)

    for mask, mode in ((final_mask, cfg.final), (approach_mask, cfg.approach), (clean_mask, cfg.clean)):
        if not np.any(mask):
            continue
        lower_value, upper_value = planned_cas_bounds_mps(cfg, float(points[np.flatnonzero(mask)[0]]))
        lower[mask] = lower_value
        upper[mask] = upper_value
    return lower, upper


def _constraint_jacobian_sparsity(
    request: LongitudinalPlanRequest,
) -> tuple[csr_matrix | None, csr_matrix | None]:
    if not isinstance(request.weather, ConstantWeather):
        return None, None

    num_nodes = request.optimizer.num_nodes
    num_variables = 4 * num_nodes + 2
    tod_index = 4 * num_nodes
    slack_index = tod_index + 1

    equality_rows = 2 * (num_nodes - 1) + 5
    equality = lil_matrix((equality_rows, num_variables), dtype=int)
    for idx in range(num_nodes - 1):
        row = 2 * idx
        local_indices = [
            idx,
            idx + 1,
            num_nodes + idx,
            num_nodes + idx + 1,
            2 * num_nodes + idx,
            2 * num_nodes + idx + 1,
            3 * num_nodes + idx,
            3 * num_nodes + idx + 1,
            tod_index,
        ]
        for col in local_indices:
            equality[row, col] = 1
            equality[row + 1, col] = 1
    equality[2 * (num_nodes - 1) + 0, 0] = 1
    equality[2 * (num_nodes - 1) + 1, num_nodes] = 1
    equality[2 * (num_nodes - 1) + 2, 2 * num_nodes] = 1
    equality[2 * (num_nodes - 1) + 3, num_nodes - 1] = 1
    equality[2 * (num_nodes - 1) + 4, 3 * num_nodes - 1] = 1

    inequality_rows = 6 * num_nodes
    if request.constraints.gamma_lower_rad is not None:
        inequality_rows += num_nodes
    if request.constraints.gamma_upper_rad is not None:
        inequality_rows += num_nodes
    if request.constraints.cl_max is not None:
        inequality_rows += num_nodes
    inequality_rows += 2

    inequality = lil_matrix((inequality_rows, num_variables), dtype=int)
    row = 0
    for idx in range(num_nodes):
        inequality[row, idx] = 1
        inequality[row, tod_index] = 1
        inequality[row, slack_index] = 1
        row += 1
    for idx in range(num_nodes):
        inequality[row, idx] = 1
        inequality[row, tod_index] = 1
        inequality[row, slack_index] = 1
        row += 1
    for idx in range(num_nodes):
        inequality[row, idx] = 1
        inequality[row, num_nodes + idx] = 1
        inequality[row, tod_index] = 1
        inequality[row, slack_index] = 1
        row += 1
    for idx in range(num_nodes):
        inequality[row, idx] = 1
        inequality[row, num_nodes + idx] = 1
        inequality[row, tod_index] = 1
        inequality[row, slack_index] = 1
        row += 1
    for idx in range(num_nodes):
        inequality[row, idx] = 1
        inequality[row, num_nodes + idx] = 1
        inequality[row, 3 * num_nodes + idx] = 1
        inequality[row, tod_index] = 1
        inequality[row, slack_index] = 1
        row += 1
    for idx in range(num_nodes):
        inequality[row, idx] = 1
        inequality[row, num_nodes + idx] = 1
        inequality[row, 3 * num_nodes + idx] = 1
        inequality[row, tod_index] = 1
        inequality[row, slack_index] = 1
        row += 1
    if request.constraints.gamma_lower_rad is not None:
        for idx in range(num_nodes):
            inequality[row, 2 * num_nodes + idx] = 1
            inequality[row, tod_index] = 1
            inequality[row, slack_index] = 1
            row += 1
    if request.constraints.gamma_upper_rad is not None:
        for idx in range(num_nodes):
            inequality[row, 2 * num_nodes + idx] = 1
            inequality[row, tod_index] = 1
            inequality[row, slack_index] = 1
            row += 1
    if request.constraints.cl_max is not None:
        for idx in range(num_nodes):
            inequality[row, idx] = 1
            inequality[row, num_nodes + idx] = 1
            inequality[row, 2 * num_nodes + idx] = 1
            inequality[row, tod_index] = 1
            inequality[row, slack_index] = 1
            row += 1
    for _ in range(2):
        inequality[row, num_nodes - 1] = 1
        inequality[row, 2 * num_nodes - 1] = 1
        inequality[row, tod_index] = 1
        inequality[row, slack_index] = 1
        row += 1

    return equality.tocsr(), inequality.tocsr()
