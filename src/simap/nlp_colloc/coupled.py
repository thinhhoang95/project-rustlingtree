from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from openap import aero
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.sparse import coo_matrix, csr_array, csr_matrix

from ..backends import PerformanceBackend
from ..config import AircraftConfig, bank_limit_rad, mode_for_s, planned_cas_bounds_mps
from ..longitudinal_profiles import ConstraintEnvelope
from ..openap_adapter import openap_dT
from ..path_geometry import ReferencePath
from ..weather import ConstantWeather, WeatherProvider, alongtrack_wind_mps


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

    @property
    def cas_target_mps(self) -> float | None:
        lower, upper = self.cas_window_mps
        if np.isclose(lower, upper, rtol=0.0, atol=1e-9):
            return float(0.5 * (lower + upper))
        return None


@dataclass(frozen=True)
class LateralBoundary:
    cross_track_m: float = 0.0
    heading_error_rad: float = 0.0
    bank_rad: float = 0.0


@dataclass(frozen=True)
class OptimizerConfig:
    num_nodes: int = 41
    maxiter: int = 400
    tod_reward_weight: float = 0.2
    thrust_penalty_weight: float = 0.5
    idle_thrust_margin_fraction: float | None = None
    gamma_smoothness_weight: float = 0.05
    slack_penalty_weight: float = 1_000_000.0
    initial_tod_guess_m: float | None = None
    initial_slack: float = 0.0
    verbose: int = 0
    constraint_tolerance: float = 1e-6
    cross_track_penalty_weight: float = 0.1
    heading_error_penalty_weight: float = 0.05
    bank_penalty_weight: float = 0.02
    roll_rate_penalty_weight: float = 0.02
    min_alongtrack_speed_mps: float = 1.0
    enforce_monotonic_descent: bool = False
    gamma_gradient_limit_deg_per_km: float | None = None
    gamma_curvature_limit_deg_per_km2: float | None = None

    def __post_init__(self) -> None:
        if self.num_nodes < 4:
            raise ValueError("num_nodes must be at least 4")
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self.min_alongtrack_speed_mps <= 0.0:
            raise ValueError("min_alongtrack_speed_mps must be positive")
        if self.idle_thrust_margin_fraction is not None and self.idle_thrust_margin_fraction < 0.0:
            raise ValueError("idle_thrust_margin_fraction must be nonnegative")
        if self.gamma_gradient_limit_deg_per_km is not None and self.gamma_gradient_limit_deg_per_km <= 0.0:
            raise ValueError("gamma_gradient_limit_deg_per_km must be positive")
        if self.gamma_curvature_limit_deg_per_km2 is not None and self.gamma_curvature_limit_deg_per_km2 <= 0.0:
            raise ValueError("gamma_curvature_limit_deg_per_km2 must be positive")


@dataclass(frozen=True)
class CoupledDescentPlanRequest:
    cfg: AircraftConfig
    perf: PerformanceBackend
    threshold: ThresholdBoundary
    upstream: UpstreamBoundary
    constraints: ConstraintEnvelope
    reference_path: ReferencePath
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    threshold_lateral: LateralBoundary = field(default_factory=LateralBoundary)
    upstream_lateral: LateralBoundary = field(default_factory=LateralBoundary)
    initial_tod_guess_m: float | None = None

    def __post_init__(self) -> None:
        if self.constraints.s_m[0] > 0.0:
            raise ValueError("constraint envelope must start at or before s=0")
        if self.constraints.s_m[-1] <= 0.0:
            raise ValueError("constraint envelope must extend upstream of the threshold")
        if self.reference_path.total_length_m <= 0.0:
            raise ValueError("reference path must have positive length")


@dataclass(frozen=True)
class CoupledDescentPlanResult:
    s_m: np.ndarray
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    v_cas_mps: np.ndarray
    t_s: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    cross_track_m: np.ndarray
    heading_error_rad: np.ndarray
    psi_rad: np.ndarray
    phi_rad: np.ndarray
    roll_rate_rps: np.ndarray
    ground_speed_mps: np.ndarray
    alongtrack_speed_mps: np.ndarray
    crosstrack_speed_mps: np.ndarray
    track_error_rad: np.ndarray
    phi_max_rad: np.ndarray
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
    solve_profile: "CoupledDescentSolveProfile"

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
                "east_m": self.east_m,
                "north_m": self.north_m,
                "lat_deg": self.lat_deg,
                "lon_deg": self.lon_deg,
                "cross_track_m": self.cross_track_m,
                "heading_error_rad": self.heading_error_rad,
                "psi_rad": self.psi_rad,
                "phi_rad": self.phi_rad,
                "roll_rate_rps": self.roll_rate_rps,
                "ground_speed_mps": self.ground_speed_mps,
                "alongtrack_speed_mps": self.alongtrack_speed_mps,
                "crosstrack_speed_mps": self.crosstrack_speed_mps,
                "track_error_rad": self.track_error_rad,
                "phi_max_rad": self.phi_max_rad,
                "gamma_rad": self.gamma_rad,
                "thrust_n": self.thrust_n,
                "mode": np.asarray(self.mode, dtype=object),
            }
        )


@dataclass(frozen=True)
class CoupledDescentSolveProfile:
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

    def snapshot(self, *, total_wall_time_s: float, postprocess_wall_time_s: float) -> CoupledDescentSolveProfile:
        return CoupledDescentSolveProfile(
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
    request: CoupledDescentPlanRequest
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    t_s: np.ndarray
    cross_track_m: np.ndarray
    heading_error_rad: np.ndarray
    phi_rad: np.ndarray
    gamma_rad: np.ndarray
    thrust_n: np.ndarray
    roll_rate_rps: np.ndarray
    tod_m: float
    constraint_slack: float
    _s_m: np.ndarray | None = None
    _delta_isa_K: np.ndarray | None = None
    _mode_by_node: tuple[Any, ...] | None = None
    _state_derivatives: np.ndarray | None = None
    _v_cas_mps: np.ndarray | None = None
    _idle_thrust_n: np.ndarray | None = None
    _thrust_bounds_backend: tuple[np.ndarray, np.ndarray] | None = None
    _psi_rad: np.ndarray | None = None
    _ground_velocity_ne_mps: np.ndarray | None = None
    _alongtrack_speed_mps: np.ndarray | None = None
    _crosstrack_speed_mps: np.ndarray | None = None
    _track_error_rad: np.ndarray | None = None
    _phi_max_rad: np.ndarray | None = None
    _position_ne_m: np.ndarray | None = None
    _mode_roll_limit_rps: np.ndarray | None = None

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
            derivatives = np.empty((self.num_nodes, 6), dtype=float)
            track_rad = self.request.reference_path.track_angle_rad_many(self.s_m)
            tangent = np.column_stack([np.cos(track_rad), np.sin(track_rad)])
            normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
            curvature = self.request.reference_path.curvature_many(self.s_m)
            weather = self.constant_weather
            for idx, (s_val, h_val, v_val, t_val, _y_val, chi_val, phi_val, gamma_val, thrust_val, roll_rate, mode, delta_isa_K) in enumerate(
                zip(
                    self.s_m,
                    self.h_m,
                    self.v_tas_mps,
                    self.t_s,
                    self.cross_track_m,
                    self.heading_error_rad,
                    self.phi_rad,
                    self.gamma_rad,
                    self.thrust_n,
                    self.roll_rate_rps,
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
                    bank_rad=float(phi_val),
                    delta_isa_K=float(delta_isa_K),
                )
                cos_gamma = float(np.clip(np.cos(gamma_val), 0.05, None))
                v_tas = max(1.0, float(v_val))
                theta = float(track_rad[idx])
                psi = theta + float(chi_val)
                if weather is None:
                    wind_east, wind_north = self.request.weather.wind_ne_mps(float(s_val), float(h_val), float(t_val))
                else:
                    wind_east, wind_north = weather.wind_east_mps, weather.wind_north_mps
                east_dot = v_tas * float(np.cos(psi)) + float(wind_east)
                north_dot = v_tas * float(np.sin(psi)) + float(wind_north)
                alongtrack_speed = max(
                    float(self.request.optimizer.min_alongtrack_speed_mps),
                    float(east_dot * tangent[idx, 0] + north_dot * tangent[idx, 1]),
                )
                crosstrack_speed = float(east_dot * normal[idx, 0] + north_dot * normal[idx, 1])
                derivatives[idx, 0] = -float(np.tan(gamma_val))
                derivatives[idx, 1] = -(((float(thrust_val) - drag_n) / cfg.mass_kg) - aero.g0 * float(np.sin(gamma_val))) / (
                    v_tas * cos_gamma
                )
                derivatives[idx, 2] = 1.0 / alongtrack_speed
                derivatives[idx, 3] = -crosstrack_speed / alongtrack_speed
                derivatives[idx, 4] = -aero.g0 * float(np.tan(phi_val)) / (v_tas * alongtrack_speed) + float(curvature[idx])
                derivatives[idx, 5] = -float(roll_rate) / alongtrack_speed
            self._state_derivatives = derivatives
        return self._state_derivatives

    @property
    def psi_rad(self) -> np.ndarray:
        if self._psi_rad is None:
            self._psi_rad = np.asarray(
                [
                    self.request.reference_path.track_angle_rad(float(s_val)) + float(heading_error)
                    for s_val, heading_error in zip(self.s_m, self.heading_error_rad, strict=True)
                ],
                dtype=float,
            )
        return self._psi_rad

    @property
    def ground_velocity_ne_mps(self) -> np.ndarray:
        if self._ground_velocity_ne_mps is None:
            weather = self.constant_weather
            if weather is not None:
                velocities = np.column_stack(
                    [
                        self.v_tas_mps * np.cos(self.psi_rad) + weather.wind_east_mps,
                        self.v_tas_mps * np.sin(self.psi_rad) + weather.wind_north_mps,
                    ]
                )
            else:
                velocities = np.empty((self.num_nodes, 2), dtype=float)
                for idx, (s_val, h_val, t_val, v_tas, psi) in enumerate(
                    zip(self.s_m, self.h_m, self.t_s, self.v_tas_mps, self.psi_rad, strict=True)
                ):
                    wind_east, wind_north = self.request.weather.wind_ne_mps(float(s_val), float(h_val), float(t_val))
                    velocities[idx, 0] = float(v_tas) * float(np.cos(psi)) + float(wind_east)
                    velocities[idx, 1] = float(v_tas) * float(np.sin(psi)) + float(wind_north)
            self._ground_velocity_ne_mps = velocities
        return self._ground_velocity_ne_mps

    @property
    def ground_speed_mps(self) -> np.ndarray:
        velocities = self.ground_velocity_ne_mps
        return np.hypot(velocities[:, 0], velocities[:, 1])

    @property
    def ground_track_rad(self) -> np.ndarray:
        velocities = self.ground_velocity_ne_mps
        return np.arctan2(velocities[:, 1], velocities[:, 0])

    @property
    def alongtrack_speed_mps(self) -> np.ndarray:
        if self._alongtrack_speed_mps is None:
            tangent = self.request.reference_path.tangent_hat_many(self.s_m)
            self._alongtrack_speed_mps = np.einsum("ij,ij->i", self.ground_velocity_ne_mps, tangent)
        return self._alongtrack_speed_mps

    @property
    def crosstrack_speed_mps(self) -> np.ndarray:
        if self._crosstrack_speed_mps is None:
            normal = self.request.reference_path.normal_hat_many(self.s_m)
            self._crosstrack_speed_mps = np.einsum("ij,ij->i", self.ground_velocity_ne_mps, normal)
        return self._crosstrack_speed_mps

    @property
    def track_error_rad(self) -> np.ndarray:
        if self._track_error_rad is None:
            diff = self.ground_track_rad - self.request.reference_path.track_angle_rad_many(self.s_m)
            self._track_error_rad = np.arctan2(np.sin(diff), np.cos(diff))
        return self._track_error_rad

    @property
    def phi_max_rad(self) -> np.ndarray:
        if self._phi_max_rad is None:
            self._phi_max_rad = np.asarray(
                [
                    bank_limit_rad(self.request.cfg, mode, float(v_cas))
                    for mode, v_cas in zip(self.mode_by_node, self.v_cas_mps, strict=True)
                ],
                dtype=float,
            )
        return self._phi_max_rad

    @property
    def position_ne_m(self) -> np.ndarray:
        if self._position_ne_m is None:
            ref = self.request.reference_path.position_ne_many(self.s_m)
            normal = self.request.reference_path.normal_hat_many(self.s_m)
            self._position_ne_m = ref + self.cross_track_m[:, np.newaxis] * normal
        return self._position_ne_m

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

    @property
    def mode_roll_limit_rps(self) -> np.ndarray:
        if self._mode_roll_limit_rps is None:
            self._mode_roll_limit_rps = np.asarray([mode.p_max_rps for mode in self.mode_by_node], dtype=float)
        return self._mode_roll_limit_rps


@dataclass
class _TrajectoryEvaluationCache:
    request: CoupledDescentPlanRequest
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
        (
            h_m,
            v_tas_mps,
            t_s,
            cross_track_m,
            heading_error_rad,
            phi_rad,
            gamma_rad,
            thrust_n,
            roll_rate_rps,
            tod_m,
            constraint_slack,
        ) = _unpack(z_arr, self.request.optimizer.num_nodes)
        evaluation = _TrajectoryEvaluation(
            request=self.request,
            h_m=np.array(h_m, dtype=float, copy=True),
            v_tas_mps=np.array(v_tas_mps, dtype=float, copy=True),
            t_s=np.array(t_s, dtype=float, copy=True),
            cross_track_m=np.array(cross_track_m, dtype=float, copy=True),
            heading_error_rad=np.array(heading_error_rad, dtype=float, copy=True),
            phi_rad=np.array(phi_rad, dtype=float, copy=True),
            gamma_rad=np.array(gamma_rad, dtype=float, copy=True),
            thrust_n=np.array(thrust_n, dtype=float, copy=True),
            roll_rate_rps=np.array(roll_rate_rps, dtype=float, copy=True),
            tod_m=float(tod_m),
            constraint_slack=float(constraint_slack),
        )
        self._last_key = key
        self._last_eval = evaluation
        self.profiling.trajectory_evaluations += 1
        self.profiling.trajectory_eval_time_s += perf_counter() - started_at
        return evaluation


def plan_coupled_descent(request: CoupledDescentPlanRequest) -> CoupledDescentPlanResult:
    solve_started_at = perf_counter()
    optimizer = request.optimizer
    num_nodes = optimizer.num_nodes
    max_tod_m = float(min(request.constraints.s_m[-1], request.reference_path.total_length_m))
    threshold_delta_isa = request.weather.delta_isa_K(0.0, request.threshold.h_m, 0.0)
    threshold_v_tas = float(
        aero.cas2tas(request.threshold.cas_mps, request.threshold.h_m, dT=openap_dT(threshold_delta_isa))
    )
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

    def objective_jac(z: np.ndarray) -> np.ndarray:
        return _objective_jac(z, request=request, scale=scale, evaluation_cache=evaluation_cache)

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
        jac=objective_jac if isinstance(request.weather, ConstantWeather) else None,
        bounds=bounds,
        constraints=[equality, inequality],
        options=solver_options,
    )

    postprocess_started_at = perf_counter()
    (
        h_m,
        v_tas_mps,
        t_s,
        cross_track_m,
        heading_error_rad,
        phi_rad,
        gamma_rad,
        thrust_n,
        roll_rate_rps,
        tod_m,
        constraint_slack,
    ) = _unpack(result.x, optimizer.num_nodes)
    h_m = np.array(h_m, dtype=float, copy=True)
    v_tas_mps = np.array(v_tas_mps, dtype=float, copy=True)
    t_s = np.array(t_s, dtype=float, copy=True)
    cross_track_m = np.array(cross_track_m, dtype=float, copy=True)
    heading_error_rad = np.array(heading_error_rad, dtype=float, copy=True)
    phi_rad = np.array(phi_rad, dtype=float, copy=True)
    gamma_rad = np.array(gamma_rad, dtype=float, copy=True)
    thrust_n = np.array(thrust_n, dtype=float, copy=True)
    roll_rate_rps = np.array(roll_rate_rps, dtype=float, copy=True)
    h_m[0] = request.threshold.h_m
    v_tas_mps[0] = threshold_v_tas
    t_s[0] = 0.0
    cross_track_m[0] = request.threshold_lateral.cross_track_m
    heading_error_rad[0] = request.threshold_lateral.heading_error_rad
    phi_rad[0] = request.threshold_lateral.bank_rad
    gamma_rad[0] = request.threshold.gamma_rad
    h_m[-1] = request.upstream.h_m
    cross_track_m[-1] = request.upstream_lateral.cross_track_m
    heading_error_rad[-1] = request.upstream_lateral.heading_error_rad
    phi_rad[-1] = request.upstream_lateral.bank_rad
    gamma_rad[-1] = request.upstream.gamma_rad
    upstream_cas_target_mps = request.upstream.cas_target_mps
    if upstream_cas_target_mps is not None:
        upstream_delta_isa_K = request.weather.delta_isa_K(float(tod_m), request.upstream.h_m, float(t_s[-1]))
        v_tas_mps[-1] = float(
            aero.cas2tas(upstream_cas_target_mps, request.upstream.h_m, dT=openap_dT(upstream_delta_isa_K))
        )
    final_evaluation = _TrajectoryEvaluation(
        request=request,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        t_s=t_s,
        cross_track_m=cross_track_m,
        heading_error_rad=heading_error_rad,
        phi_rad=phi_rad,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        roll_rate_rps=roll_rate_rps,
        tod_m=float(tod_m),
        constraint_slack=float(constraint_slack),
    )
    s_m = np.array(final_evaluation.s_m, dtype=float, copy=True)
    v_cas_mps = np.array(final_evaluation.v_cas_mps, dtype=float, copy=True)
    position_ne = final_evaluation.position_ne_m
    latlon_arr = request.reference_path.latlon_from_ne_many(position_ne[:, 0], position_ne[:, 1])
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
        cross_track_m=cross_track_m,
        heading_error_rad=heading_error_rad,
        phi_rad=phi_rad,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        roll_rate_rps=roll_rate_rps,
    )
    mode = tuple(mode_for_s(request.cfg, float(s)).name for s in s_m)
    solve_profile = profiling.snapshot(
        total_wall_time_s=perf_counter() - solve_started_at,
        postprocess_wall_time_s=perf_counter() - postprocess_started_at,
    )
    return CoupledDescentPlanResult(
        s_m=s_m,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        v_cas_mps=v_cas_mps,
        t_s=t_s,
        east_m=position_ne[:, 0],
        north_m=position_ne[:, 1],
        lat_deg=latlon_arr[:, 0],
        lon_deg=latlon_arr[:, 1],
        cross_track_m=cross_track_m,
        heading_error_rad=heading_error_rad,
        psi_rad=np.array(final_evaluation.psi_rad, dtype=float, copy=True),
        phi_rad=phi_rad,
        roll_rate_rps=roll_rate_rps,
        ground_speed_mps=np.array(final_evaluation.ground_speed_mps, dtype=float, copy=True),
        alongtrack_speed_mps=np.array(final_evaluation.alongtrack_speed_mps, dtype=float, copy=True),
        crosstrack_speed_mps=np.array(final_evaluation.crosstrack_speed_mps, dtype=float, copy=True),
        track_error_rad=np.array(final_evaluation.track_error_rad, dtype=float, copy=True),
        phi_max_rad=np.array(final_evaluation.phi_max_rad, dtype=float, copy=True),
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
    def from_request(cls, request: CoupledDescentPlanRequest) -> "_PlannerScale":
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
    request: CoupledDescentPlanRequest,
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
    request: CoupledDescentPlanRequest,
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
    cross_track_m = np.linspace(
        request.threshold_lateral.cross_track_m,
        request.upstream_lateral.cross_track_m,
        num_nodes,
        dtype=float,
    )
    heading_error_rad = np.linspace(
        request.threshold_lateral.heading_error_rad,
        request.upstream_lateral.heading_error_rad,
        num_nodes,
        dtype=float,
    )
    phi_rad = np.linspace(
        request.threshold_lateral.bank_rad,
        request.upstream_lateral.bank_rad,
        num_nodes,
        dtype=float,
    )
    q_guess = np.maximum(request.optimizer.min_alongtrack_speed_mps, np.gradient(s_m, t_s, edge_order=1))
    roll_rate_rps = -np.gradient(phi_rad, s_m, edge_order=1) * q_guess
    for idx, s_val in enumerate(s_m):
        mode = mode_for_s(request.cfg, float(s_val))
        roll_rate_rps[idx] = float(np.clip(roll_rate_rps[idx], -mode.p_max_rps, mode.p_max_rps))

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
        idle_upper = thrust_upper
        idle_margin_fraction = request.optimizer.idle_thrust_margin_fraction
        if idle_margin_fraction is not None:
            idle_thrust = request.perf.idle_thrust_newtons(
                v_tas_mps=float(v_val),
                h_m=float(h_val),
                delta_isa_K=delta_isa_K,
            )
            thrust_lower = max(float(thrust_lower), float(idle_thrust))
            idle_upper = float(idle_thrust) + float(idle_margin_fraction) * max(
                float(thrust_upper) - float(idle_thrust),
                0.0,
            )
        thrust_guess = drag_n + request.cfg.mass_kg * (
            aero.g0 * np.sin(gamma_val) - float(v_val) * np.cos(gamma_val) * float(dvds)
        )
        thrust_n[idx] = float(np.clip(thrust_guess, thrust_lower, idle_upper))

    return _pack(
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        t_s=t_s,
        cross_track_m=cross_track_m,
        heading_error_rad=heading_error_rad,
        phi_rad=phi_rad,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        roll_rate_rps=roll_rate_rps,
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
    request: CoupledDescentPlanRequest,
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
    max_time_s = max_tod_m / max(1.0, request.optimizer.min_alongtrack_speed_mps)
    lateral_span = max(
        10_000.0,
        2.0 * abs(request.threshold_lateral.cross_track_m),
        2.0 * abs(request.upstream_lateral.cross_track_m),
    )
    lower = np.concatenate(
        [
            np.full(num_nodes, lower_altitude, dtype=float),
            np.full(num_nodes, 1.0, dtype=float),
            np.zeros(num_nodes, dtype=float),
            np.full(num_nodes, -lateral_span, dtype=float),
            np.full(num_nodes, -np.deg2rad(60.0), dtype=float),
            np.full(num_nodes, -np.deg2rad(45.0), dtype=float),
            np.full(num_nodes, -gamma_limit, dtype=float),
            np.zeros(num_nodes, dtype=float),
            np.full(num_nodes, -max(mode.p_max_rps for mode in (request.cfg.clean, request.cfg.approach, request.cfg.final)), dtype=float),
            np.asarray([1_000.0, 0.0], dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.full(num_nodes, upper_altitude, dtype=float),
            np.full(num_nodes, upper_tas, dtype=float),
            np.full(num_nodes, max_time_s, dtype=float),
            np.full(num_nodes, lateral_span, dtype=float),
            np.full(num_nodes, np.deg2rad(60.0), dtype=float),
            np.full(num_nodes, np.deg2rad(45.0), dtype=float),
            np.full(num_nodes, np.deg2rad(5.0), dtype=float),
            np.full(num_nodes, thrust_upper, dtype=float),
            np.full(num_nodes, max(mode.p_max_rps for mode in (request.cfg.clean, request.cfg.approach, request.cfg.final)), dtype=float),
            np.asarray([max_tod_m, 5.0], dtype=float),
        ]
    )
    return cast(Any, Bounds)(lower, upper)


def _objective(
    z: np.ndarray,
    *,
    request: CoupledDescentPlanRequest,
    scale: _PlannerScale,
    evaluation_cache: _TrajectoryEvaluationCache,
) -> float:
    evaluation = evaluation_cache.evaluate(z)
    ds_m = max(evaluation.tod_m / max(request.optimizer.num_nodes - 1, 1), 1.0)
    gamma_gradient = np.diff(evaluation.gamma_rad) / max(ds_m, 1.0)
    lateral_scale = max(100.0, float(np.max(np.abs(evaluation.cross_track_m))), 1.0)
    heading_scale = np.deg2rad(5.0)
    bank_scale = np.deg2rad(20.0)
    roll_scale = max(1e-3, max(mode.p_max_rps for mode in (request.cfg.clean, request.cfg.approach, request.cfg.final)))
    thrust_objective = 0.0
    if request.optimizer.idle_thrust_margin_fraction is None:
        thrust_delta = (evaluation.thrust_n - evaluation.idle_thrust_n) / max(1.0, scale.thrust_n)
        thrust_objective = (
            request.optimizer.thrust_penalty_weight
            * (evaluation.tod_m / 1_000.0)
            * float(np.mean(thrust_delta**2))
        )
    tod_objective = request.optimizer.tod_reward_weight * (evaluation.tod_m / 1_000.0)
    if request.optimizer.idle_thrust_margin_fraction is None:
        tod_objective = -tod_objective
    objective = (
        tod_objective
        + thrust_objective
        + request.optimizer.gamma_smoothness_weight
        * float(np.mean((gamma_gradient / np.deg2rad(1.0) * 1_000.0) ** 2))
        + request.optimizer.cross_track_penalty_weight * float(np.mean((evaluation.cross_track_m / lateral_scale) ** 2))
        + request.optimizer.heading_error_penalty_weight * float(np.mean((evaluation.heading_error_rad / heading_scale) ** 2))
        + request.optimizer.bank_penalty_weight * float(np.mean((evaluation.phi_rad / bank_scale) ** 2))
        + request.optimizer.roll_rate_penalty_weight * float(np.mean((evaluation.roll_rate_rps / roll_scale) ** 2))
        + request.optimizer.slack_penalty_weight * float(evaluation.constraint_slack**2)
    )
    return float(objective)


def _idle_thrust_partials(
    evaluation: _TrajectoryEvaluation,
    *,
    v_step_mps: float = 1e-3,
    h_step_m: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    d_idle_dv = np.empty(evaluation.num_nodes, dtype=float)
    d_idle_dh = np.empty(evaluation.num_nodes, dtype=float)
    perf = evaluation.request.perf
    for idx, (v_tas, h_m, delta_isa_K) in enumerate(
        zip(evaluation.v_tas_mps, evaluation.h_m, evaluation.delta_isa_K, strict=True)
    ):
        v_plus = float(v_tas) + v_step_mps
        v_minus = max(1.0, float(v_tas) - v_step_mps)
        h_plus = float(h_m) + h_step_m
        h_minus = float(h_m) - h_step_m
        idle_v_plus = perf.idle_thrust_newtons(v_tas_mps=v_plus, h_m=float(h_m), delta_isa_K=float(delta_isa_K))
        idle_v_minus = perf.idle_thrust_newtons(v_tas_mps=v_minus, h_m=float(h_m), delta_isa_K=float(delta_isa_K))
        idle_h_plus = perf.idle_thrust_newtons(v_tas_mps=float(v_tas), h_m=h_plus, delta_isa_K=float(delta_isa_K))
        idle_h_minus = perf.idle_thrust_newtons(v_tas_mps=float(v_tas), h_m=h_minus, delta_isa_K=float(delta_isa_K))
        d_idle_dv[idx] = (float(idle_v_plus) - float(idle_v_minus)) / max(v_plus - v_minus, 1e-12)
        d_idle_dh[idx] = (float(idle_h_plus) - float(idle_h_minus)) / max(h_plus - h_minus, 1e-12)
    return d_idle_dv, d_idle_dh


def _objective_jac(
    z: np.ndarray,
    *,
    request: CoupledDescentPlanRequest,
    scale: _PlannerScale,
    evaluation_cache: _TrajectoryEvaluationCache,
) -> np.ndarray:
    evaluation = evaluation_cache.evaluate(z)
    num_nodes = request.optimizer.num_nodes
    grad = np.zeros(9 * num_nodes + 2, dtype=float)

    h_start = 0
    v_start = h_start + num_nodes
    y_start = v_start + 2 * num_nodes
    chi_start = y_start + num_nodes
    phi_start = chi_start + num_nodes
    gamma_start = phi_start + num_nodes
    thrust_start = gamma_start + num_nodes
    roll_start = thrust_start + num_nodes
    tod_col = 9 * num_nodes
    slack_col = tod_col + 1

    if request.optimizer.idle_thrust_margin_fraction is None:
        grad[tod_col] -= request.optimizer.tod_reward_weight / 1_000.0
    else:
        grad[tod_col] += request.optimizer.tod_reward_weight / 1_000.0

    if request.optimizer.idle_thrust_margin_fraction is None:
        thrust_scale = max(1.0, scale.thrust_n)
        thrust_delta = (evaluation.thrust_n - evaluation.idle_thrust_n) / thrust_scale
        thrust_mean_sq = float(np.mean(thrust_delta**2))
        thrust_weight = request.optimizer.thrust_penalty_weight
        grad[tod_col] += thrust_weight * thrust_mean_sq / 1_000.0
        common_thrust_grad = (
            thrust_weight * (evaluation.tod_m / 1_000.0) * (2.0 / num_nodes) * thrust_delta / thrust_scale
        )
        grad[thrust_start : thrust_start + num_nodes] += common_thrust_grad
        d_idle_dv, d_idle_dh = _idle_thrust_partials(evaluation)
        grad[v_start : v_start + num_nodes] -= common_thrust_grad * d_idle_dv
        grad[h_start : h_start + num_nodes] -= common_thrust_grad * d_idle_dh

    ds_m = max(evaluation.tod_m / max(num_nodes - 1, 1), 1.0)
    diff_count = max(num_nodes - 1, 1)
    gamma_diff = np.diff(evaluation.gamma_rad)
    gamma_factor = 1_000.0 / (np.deg2rad(1.0) * ds_m)
    gamma_term = request.optimizer.gamma_smoothness_weight * float(np.mean((gamma_diff * gamma_factor) ** 2))
    gamma_grad_diff = (
        request.optimizer.gamma_smoothness_weight
        * (2.0 / diff_count)
        * gamma_factor**2
        * gamma_diff
    )
    grad[gamma_start : gamma_start + num_nodes - 1] -= gamma_grad_diff
    grad[gamma_start + 1 : gamma_start + num_nodes] += gamma_grad_diff
    if evaluation.tod_m / max(num_nodes - 1, 1) >= 1.0:
        grad[tod_col] -= 2.0 * gamma_term / max(evaluation.tod_m, 1e-12)

    lateral_scale = max(100.0, float(np.max(np.abs(evaluation.cross_track_m))), 1.0)
    heading_scale = np.deg2rad(5.0)
    bank_scale = np.deg2rad(20.0)
    roll_scale = max(1e-3, max(mode.p_max_rps for mode in (request.cfg.clean, request.cfg.approach, request.cfg.final)))
    grad[y_start : y_start + num_nodes] += (
        request.optimizer.cross_track_penalty_weight
        * (2.0 / num_nodes)
        * evaluation.cross_track_m
        / lateral_scale**2
    )
    grad[chi_start : chi_start + num_nodes] += (
        request.optimizer.heading_error_penalty_weight
        * (2.0 / num_nodes)
        * evaluation.heading_error_rad
        / heading_scale**2
    )
    grad[phi_start : phi_start + num_nodes] += (
        request.optimizer.bank_penalty_weight
        * (2.0 / num_nodes)
        * evaluation.phi_rad
        / bank_scale**2
    )
    grad[roll_start : roll_start + num_nodes] += (
        request.optimizer.roll_rate_penalty_weight
        * (2.0 / num_nodes)
        * evaluation.roll_rate_rps
        / roll_scale**2
    )
    grad[slack_col] += 2.0 * request.optimizer.slack_penalty_weight * evaluation.constraint_slack
    return grad


def _equality_constraints(
    z: np.ndarray,
    *,
    request: CoupledDescentPlanRequest,
    threshold_v_tas: float,
    evaluation_cache: _TrajectoryEvaluationCache,
) -> np.ndarray:
    evaluation = evaluation_cache.evaluate(z)
    ds_m = float(evaluation.tod_m / (request.optimizer.num_nodes - 1))
    trapezoid = 0.5 * ds_m * (evaluation.state_derivatives[:-1, :] + evaluation.state_derivatives[1:, :])
    state_delta = np.column_stack(
        [
            np.diff(evaluation.h_m),
            np.diff(evaluation.v_tas_mps),
            np.diff(evaluation.t_s),
            np.diff(evaluation.cross_track_m),
            np.diff(evaluation.heading_error_rad),
            np.diff(evaluation.phi_rad),
        ]
    )
    dynamic_rows = 6 * (request.optimizer.num_nodes - 1)
    endpoint_defects = [
        float(evaluation.h_m[0] - request.threshold.h_m),
        float(evaluation.v_tas_mps[0] - threshold_v_tas),
        float(evaluation.t_s[0]),
        float(evaluation.cross_track_m[0] - request.threshold_lateral.cross_track_m),
        float(evaluation.heading_error_rad[0] - request.threshold_lateral.heading_error_rad),
        float(evaluation.phi_rad[0] - request.threshold_lateral.bank_rad),
        float(evaluation.gamma_rad[0] - request.threshold.gamma_rad),
        float(evaluation.h_m[-1] - request.upstream.h_m),
        float(evaluation.gamma_rad[-1] - request.upstream.gamma_rad),
        float(evaluation.cross_track_m[-1] - request.upstream_lateral.cross_track_m),
        float(evaluation.heading_error_rad[-1] - request.upstream_lateral.heading_error_rad),
        float(evaluation.phi_rad[-1] - request.upstream_lateral.bank_rad),
    ]
    upstream_cas_target_mps = request.upstream.cas_target_mps
    if upstream_cas_target_mps is not None:
        endpoint_defects.append(float(evaluation.v_cas_mps[-1] - upstream_cas_target_mps))

    defects = np.empty(dynamic_rows + len(endpoint_defects), dtype=float)
    defects[:dynamic_rows] = (state_delta - trapezoid).reshape(-1)
    defects[dynamic_rows:] = np.asarray(endpoint_defects, dtype=float)
    return defects


def _inequality_constraints(
    z: np.ndarray,
    *,
    request: CoupledDescentPlanRequest,
    scale: _PlannerScale,
    evaluation_cache: _TrajectoryEvaluationCache,
) -> np.ndarray:
    evaluation = evaluation_cache.evaluate(z)
    slack_altitude = evaluation.constraint_slack * scale.altitude_m
    slack_speed = evaluation.constraint_slack * scale.speed_mps
    slack_gamma = evaluation.constraint_slack * scale.gamma_rad
    slack_thrust = evaluation.constraint_slack * scale.thrust_n
    slack_bank = evaluation.constraint_slack * np.deg2rad(10.0)
    slack_roll = evaluation.constraint_slack * max(
        1e-3,
        max(mode.p_max_rps for mode in (request.cfg.clean, request.cfg.approach, request.cfg.final)),
    )

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
        evaluation.phi_rad + evaluation.phi_max_rad + slack_bank,
        evaluation.phi_max_rad - evaluation.phi_rad + slack_bank,
        evaluation.roll_rate_rps + evaluation.mode_roll_limit_rps + slack_roll,
        evaluation.mode_roll_limit_rps - evaluation.roll_rate_rps + slack_roll,
        evaluation.alongtrack_speed_mps - request.optimizer.min_alongtrack_speed_mps + slack_speed,
    ]

    idle_margin_fraction = request.optimizer.idle_thrust_margin_fraction
    if idle_margin_fraction is not None:
        idle_band_upper = evaluation.idle_thrust_n + float(idle_margin_fraction) * np.maximum(
            thrust_upper_backend - evaluation.idle_thrust_n,
            0.0,
        )
        residual_parts.extend(
            [
                evaluation.thrust_n - evaluation.idle_thrust_n,
                idle_band_upper - evaluation.thrust_n,
            ]
        )

    gamma_lower, gamma_upper = request.constraints.gamma_bounds_many(evaluation.s_m)
    if gamma_lower is not None:
        residual_parts.append(evaluation.gamma_rad - gamma_lower + slack_gamma)
    if gamma_upper is not None:
        residual_parts.append(gamma_upper - evaluation.gamma_rad + slack_gamma)

    if request.constraints.cl_max is not None:
        cl_upper = request.constraints._interp_many_required(request.constraints.cl_max, evaluation.s_m)
        cl_values = np.asarray(
            [
                _banked_quasi_steady_cl(
                    mass_kg=request.cfg.mass_kg,
                    wing_area_m2=request.cfg.wing_area_m2,
                    v_tas_mps=float(v_tas),
                    h_m=float(h),
                    gamma_rad=float(gamma),
                    phi_rad=float(phi),
                    delta_isa_K=float(delta_isa_K),
                )
                for v_tas, h, gamma, phi, delta_isa_K in zip(
                    evaluation.v_tas_mps,
                    evaluation.h_m,
                    evaluation.gamma_rad,
                    evaluation.phi_rad,
                    evaluation.delta_isa_K,
                    strict=True,
                )
            ],
            dtype=float,
        )
        residual_parts.append(cl_upper - cl_values + evaluation.constraint_slack)

    if request.optimizer.enforce_monotonic_descent:
        residual_parts.append(np.diff(evaluation.h_m) + slack_altitude)

    ds_m = float(evaluation.tod_m / max(request.optimizer.num_nodes - 1, 1))
    gamma_gradient_limit = request.optimizer.gamma_gradient_limit_deg_per_km
    if gamma_gradient_limit is not None:
        gamma_step_limit = np.deg2rad(float(gamma_gradient_limit)) * max(ds_m, 1.0) / 1_000.0
        gamma_delta = np.diff(evaluation.gamma_rad)
        residual_parts.extend(
            [
                gamma_step_limit - gamma_delta + slack_gamma,
                gamma_step_limit + gamma_delta + slack_gamma,
            ]
        )

    gamma_curvature_limit = request.optimizer.gamma_curvature_limit_deg_per_km2
    if gamma_curvature_limit is not None and evaluation.num_nodes >= 3:
        gamma_second = evaluation.gamma_rad[2:] - 2.0 * evaluation.gamma_rad[1:-1] + evaluation.gamma_rad[:-2]
        gamma_second_limit = (
            np.deg2rad(float(gamma_curvature_limit)) * max(ds_m, 1.0) ** 2 / 1_000_000.0
        )
        residual_parts.extend(
            [
                gamma_second_limit - gamma_second + slack_gamma,
                gamma_second_limit + gamma_second + slack_gamma,
            ]
        )

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


def _banked_quasi_steady_cl(
    *,
    mass_kg: float,
    wing_area_m2: float,
    v_tas_mps: float,
    h_m: float,
    gamma_rad: float,
    phi_rad: float,
    delta_isa_K: float = 0.0,
) -> float:
    v_tas = max(1.0, float(v_tas_mps))
    _, rho, _ = aero.atmos(h_m, dT=openap_dT(delta_isa_K))
    dynamic_pressure = 0.5 * float(rho) * v_tas**2
    bank_cos = max(0.2, float(np.cos(phi_rad)))
    lift_newtons = mass_kg * aero.g0 * float(np.cos(gamma_rad)) / bank_cos
    return float(lift_newtons / max(dynamic_pressure * wing_area_m2, 1e-6))


def _cas_from_tas(
    request: CoupledDescentPlanRequest,
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
    request: CoupledDescentPlanRequest,
    s_m: np.ndarray,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    t_s: np.ndarray,
    cross_track_m: np.ndarray,
    heading_error_rad: np.ndarray,
    phi_rad: np.ndarray,
    gamma_rad: np.ndarray,
    thrust_n: np.ndarray,
    roll_rate_rps: np.ndarray,
) -> dict[str, float]:
    def rhs(s_val: float, state: np.ndarray) -> np.ndarray:
        gamma_val = float(np.interp(s_val, s_m, gamma_rad))
        thrust_val = float(np.interp(s_val, s_m, thrust_n))
        roll_rate_val = float(np.interp(s_val, s_m, roll_rate_rps))
        temp_eval = _TrajectoryEvaluation(
            request=request,
            h_m=np.asarray([state[0]], dtype=float),
            v_tas_mps=np.asarray([state[1]], dtype=float),
            t_s=np.asarray([state[2]], dtype=float),
            cross_track_m=np.asarray([state[3]], dtype=float),
            heading_error_rad=np.asarray([state[4]], dtype=float),
            phi_rad=np.asarray([state[5]], dtype=float),
            gamma_rad=np.asarray([gamma_val], dtype=float),
            thrust_n=np.asarray([thrust_val], dtype=float),
            roll_rate_rps=np.asarray([roll_rate_val], dtype=float),
            tod_m=max(float(s_val), 1.0),
            constraint_slack=0.0,
            _s_m=np.asarray([s_val], dtype=float),
        )
        return temp_eval.state_derivatives[0]

    replay = solve_ivp(
        rhs,
        (float(s_m[0]), float(s_m[-1])),
        y0=np.asarray([h_m[0], v_tas_mps[0], t_s[0], cross_track_m[0], heading_error_rad[0], phi_rad[0]], dtype=float),
        t_eval=s_m,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    if replay.y.shape[1] == len(s_m):
        expected_h_m = h_m
        expected_v_tas_mps = v_tas_mps
        expected_t_s = t_s
        expected_cross_track_m = cross_track_m
        expected_heading_error_rad = heading_error_rad
        expected_phi_rad = phi_rad
    else:
        replay_s_m = np.asarray(replay.t, dtype=float)
        expected_h_m = np.interp(replay_s_m, s_m, h_m)
        expected_v_tas_mps = np.interp(replay_s_m, s_m, v_tas_mps)
        expected_t_s = np.interp(replay_s_m, s_m, t_s)
        expected_cross_track_m = np.interp(replay_s_m, s_m, cross_track_m)
        expected_heading_error_rad = np.interp(replay_s_m, s_m, heading_error_rad)
        expected_phi_rad = np.interp(replay_s_m, s_m, phi_rad)

    h_error = float(np.max(np.abs(replay.y[0] - expected_h_m)))
    v_error = float(np.max(np.abs(replay.y[1] - expected_v_tas_mps)))
    t_error = float(np.max(np.abs(replay.y[2] - expected_t_s)))
    y_error = float(np.max(np.abs(replay.y[3] - expected_cross_track_m)))
    chi_error = float(np.max(np.abs(replay.y[4] - expected_heading_error_rad)))
    phi_error = float(np.max(np.abs(replay.y[5] - expected_phi_rad)))
    if not replay.success or replay.y.shape[1] != len(s_m):
        integration_error = float(np.finfo(float).max)
    else:
        integration_error = 0.0
    return {
        "h_error_m": h_error,
        "v_error_mps": v_error,
        "t_error_s": t_error,
        "max_error": max(h_error, v_error, t_error, y_error, chi_error, phi_error, integration_error),
    }


def _pack(
    *,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    t_s: np.ndarray,
    cross_track_m: np.ndarray,
    heading_error_rad: np.ndarray,
    phi_rad: np.ndarray,
    gamma_rad: np.ndarray,
    thrust_n: np.ndarray,
    roll_rate_rps: np.ndarray,
    tod_m: float,
    constraint_slack: float,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(h_m, dtype=float),
            np.asarray(v_tas_mps, dtype=float),
            np.asarray(t_s, dtype=float),
            np.asarray(cross_track_m, dtype=float),
            np.asarray(heading_error_rad, dtype=float),
            np.asarray(phi_rad, dtype=float),
            np.asarray(gamma_rad, dtype=float),
            np.asarray(thrust_n, dtype=float),
            np.asarray(roll_rate_rps, dtype=float),
            np.asarray([float(tod_m), float(constraint_slack)], dtype=float),
        ]
    )


def _unpack(
    z: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    h_end = num_nodes
    v_end = h_end + num_nodes
    t_end = v_end + num_nodes
    y_end = t_end + num_nodes
    chi_end = y_end + num_nodes
    phi_end = chi_end + num_nodes
    gamma_end = phi_end + num_nodes
    thrust_end = gamma_end + num_nodes
    roll_end = thrust_end + num_nodes
    h_m = np.asarray(z[:h_end], dtype=float)
    v_tas_mps = np.asarray(z[h_end:v_end], dtype=float)
    t_s = np.asarray(z[v_end:t_end], dtype=float)
    cross_track_m = np.asarray(z[t_end:y_end], dtype=float)
    heading_error_rad = np.asarray(z[y_end:chi_end], dtype=float)
    phi_rad = np.asarray(z[chi_end:phi_end], dtype=float)
    gamma_rad = np.asarray(z[phi_end:gamma_end], dtype=float)
    thrust_n = np.asarray(z[gamma_end:thrust_end], dtype=float)
    roll_rate_rps = np.asarray(z[thrust_end:roll_end], dtype=float)
    tod_m, constraint_slack = np.asarray(z[roll_end : roll_end + 2], dtype=float)
    return (
        h_m,
        v_tas_mps,
        t_s,
        cross_track_m,
        heading_error_rad,
        phi_rad,
        gamma_rad,
        thrust_n,
        roll_rate_rps,
        float(tod_m),
        float(constraint_slack),
    )


def _integrate_time_profile(
    request: CoupledDescentPlanRequest,
    *,
    s_m: np.ndarray,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
) -> np.ndarray:
    weather = request.weather
    if isinstance(weather, ConstantWeather):
        segment_mid_s = 0.5 * (s_m[:-1] + s_m[1:])
        segment_ground_speed = np.empty_like(segment_mid_s)
        for idx, s_val in enumerate(segment_mid_s):
            track = request.reference_path.track_angle_rad(float(s_val))
            segment_ground_speed[idx] = 0.5 * (v_tas_mps[idx] + v_tas_mps[idx + 1]) + (
                weather.wind_east_mps * np.cos(track)
                + weather.wind_north_mps * np.sin(track)
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
                request.reference_path.track_angle_rad(float(0.5 * (s_m[idx - 1] + s_m[idx]))),
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
    request: CoupledDescentPlanRequest,
) -> tuple[csr_matrix | csr_array | None, csr_matrix | csr_array | None]:
    num_nodes = request.optimizer.num_nodes
    num_vars = 9 * num_nodes + 2
    tod_col = 9 * num_nodes
    slack_col = 9 * num_nodes + 1

    h_start = 0
    v_start = h_start + num_nodes
    t_start = v_start + num_nodes
    y_start = t_start + num_nodes
    chi_start = y_start + num_nodes
    phi_start = chi_start + num_nodes
    gamma_start = phi_start + num_nodes
    thrust_start = gamma_start + num_nodes
    roll_start = thrust_start + num_nodes
    variable_weather = not isinstance(request.weather, ConstantWeather)
    state_starts = (h_start, v_start, t_start, y_start, chi_start, phi_start)
    var_starts = {
        "h": h_start,
        "v": v_start,
        "t": t_start,
        "y": y_start,
        "chi": chi_start,
        "phi": phi_start,
        "gamma": gamma_start,
        "thrust": thrust_start,
        "roll": roll_start,
    }

    equality_rows: list[int] = []
    equality_cols: list[int] = []

    def add_eq(row: int, cols: list[int]) -> None:
        equality_rows.extend([row] * len(cols))
        equality_cols.extend(cols)

    def node_var_cols(node_idx: int, names: tuple[str, ...]) -> list[int]:
        return [var_starts[name] + node_idx for name in names]

    row = 0
    derivative_dependencies: tuple[tuple[str, ...], ...] = (
        ("gamma",),
        ("h", "v", "phi", "gamma", "thrust", *(() if not variable_weather else ("t",))),
        ("v", "chi", *(() if not variable_weather else ("h", "t"))),
        ("v", "chi", *(() if not variable_weather else ("h", "t"))),
        ("v", "phi", *(() if not variable_weather else ("h", "t"))),
        ("v", "roll", *(() if not variable_weather else ("h", "t"))),
    )
    for idx in range(num_nodes - 1):
        for state_dim, state_start in enumerate(state_starts):
            cols = [state_start + idx, state_start + idx + 1, tod_col]
            for node_idx in (idx, idx + 1):
                cols.extend(node_var_cols(node_idx, derivative_dependencies[state_dim]))
            add_eq(row + state_dim, sorted(set(cols)))
        row += 6

    endpoint_cols = [
        h_start,
        v_start,
        t_start,
        y_start,
        chi_start,
        phi_start,
        gamma_start,
        h_start + num_nodes - 1,
        gamma_start + num_nodes - 1,
        y_start + num_nodes - 1,
        chi_start + num_nodes - 1,
        phi_start + num_nodes - 1,
    ]
    for offset, col in enumerate(endpoint_cols):
        add_eq(row + offset, [col])
    row += len(endpoint_cols)

    if request.upstream.cas_target_mps is not None:
        cols = [h_start + num_nodes - 1, v_start + num_nodes - 1]
        if variable_weather:
            cols.extend([t_start + num_nodes - 1, tod_col])
        add_eq(row, cols)
        row += 1

    equality = coo_matrix(
        (np.ones(len(equality_rows), dtype=bool), (equality_rows, equality_cols)),
        shape=(row, num_vars),
        dtype=bool,
    ).tocsr()

    inequality_rows: list[int] = []
    inequality_cols: list[int] = []

    def add_ineq(row: int, cols: list[int]) -> None:
        inequality_rows.extend([row] * len(cols))
        inequality_cols.extend(cols)

    def ineq_node_cols(node_idx: int, names: tuple[str, ...], *, depends_on_s: bool = True) -> list[int]:
        cols = node_var_cols(node_idx, names)
        if depends_on_s:
            cols.append(tod_col)
        cols.append(slack_col)
        return sorted(set(cols))

    row = 0
    inequality_blocks: tuple[tuple[tuple[str, ...], bool], ...] = (
        (("h",), True),
        (("h",), True),
        (("h", "v", *(() if not variable_weather else ("t",))), True),
        (("h", "v", *(() if not variable_weather else ("t",))), True),
        (("h", "v", "thrust", *(() if not variable_weather else ("t",))), True),
        (("h", "v", "thrust", *(() if not variable_weather else ("t",))), True),
        (("h", "v", "phi", *(() if not variable_weather else ("t",))), True),
        (("h", "v", "phi", *(() if not variable_weather else ("t",))), True),
        (("roll",), True),
        (("roll",), True),
        (("v", "chi", *(() if not variable_weather else ("h", "t"))), True),
    )
    for names, depends_on_s in inequality_blocks:
        for idx in range(num_nodes):
            add_ineq(row, ineq_node_cols(idx, names, depends_on_s=depends_on_s))
            row += 1

    if request.optimizer.idle_thrust_margin_fraction is not None:
        idle_thrust_block = ("h", "v", "thrust", *(() if not variable_weather else ("t",)))
        for _ in range(2):
            for idx in range(num_nodes):
                add_ineq(row, sorted(set(node_var_cols(idx, idle_thrust_block) + [tod_col])))
                row += 1

    gamma_lower, gamma_upper = request.constraints.gamma_bounds_many(np.asarray([0.0], dtype=float))
    if gamma_lower is not None:
        for idx in range(num_nodes):
            add_ineq(row, [gamma_start + idx, tod_col, slack_col])
            row += 1
    if gamma_upper is not None:
        for idx in range(num_nodes):
            add_ineq(row, [gamma_start + idx, tod_col, slack_col])
            row += 1

    if request.constraints.cl_max is not None:
        for idx in range(num_nodes):
            add_ineq(
                row,
                ineq_node_cols(
                    idx,
                    ("h", "v", "gamma", "phi", *(() if not variable_weather else ("t",))),
                    depends_on_s=True,
                ),
            )
            row += 1

    if request.optimizer.enforce_monotonic_descent:
        for idx in range(num_nodes - 1):
            add_ineq(row, sorted({h_start + idx, h_start + idx + 1, slack_col}))
            row += 1

    if request.optimizer.gamma_gradient_limit_deg_per_km is not None:
        for _ in range(2):
            for idx in range(num_nodes - 1):
                add_ineq(row, sorted({gamma_start + idx, gamma_start + idx + 1, tod_col, slack_col}))
                row += 1

    if request.optimizer.gamma_curvature_limit_deg_per_km2 is not None:
        for _ in range(2):
            for idx in range(num_nodes - 2):
                add_ineq(
                    row,
                    sorted(
                        {
                            gamma_start + idx,
                            gamma_start + idx + 1,
                            gamma_start + idx + 2,
                            tod_col,
                            slack_col,
                        }
                    ),
                )
                row += 1

    upstream_cas_cols = [h_start + num_nodes - 1, v_start + num_nodes - 1, slack_col]
    if variable_weather:
        upstream_cas_cols.extend([t_start + num_nodes - 1, tod_col])
    add_ineq(row, upstream_cas_cols)
    row += 1
    add_ineq(row, upstream_cas_cols)
    row += 1

    inequality = coo_matrix(
        (np.ones(len(inequality_rows), dtype=bool), (inequality_rows, inequality_cols)),
        shape=(row, num_vars),
        dtype=bool,
    ).tocsr()
    return equality, inequality

