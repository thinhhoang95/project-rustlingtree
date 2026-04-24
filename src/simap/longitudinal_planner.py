from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from openap import aero
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, NonlinearConstraint, minimize

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


def plan_longitudinal_descent(request: LongitudinalPlanRequest) -> LongitudinalPlanResult:
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

    equality = NonlinearConstraint(
        lambda z: _equality_constraints(z, request=request, threshold_v_tas=threshold_v_tas),
        0.0,
        0.0,
    )
    inequality = NonlinearConstraint(
        lambda z: _inequality_constraints(z, request=request, scale=scale),
        0.0,
        np.inf,
    )

    result = minimize(
        _objective,
        z0,
        args=(request, scale),
        method="trust-constr",
        bounds=bounds,
        constraints=[equality, inequality],
        options={
            "maxiter": optimizer.maxiter,
            "verbose": optimizer.verbose,
        },
    )

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
    s_m = np.linspace(0.0, tod_m, optimizer.num_nodes, dtype=float)
    t_s = _integrate_time_profile(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps)
    v_cas_mps = _cas_from_tas(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps, t_s=t_s)
    collocation_residual_max = float(
        np.max(np.abs(_equality_constraints(result.x, request=request, threshold_v_tas=threshold_v_tas)))
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


def _objective(z: np.ndarray, request: LongitudinalPlanRequest, scale: _PlannerScale) -> float:
    h_m, v_tas_mps, gamma_rad, thrust_n, tod_m, constraint_slack = _unpack(z, request.optimizer.num_nodes)
    s_m = np.linspace(0.0, tod_m, request.optimizer.num_nodes, dtype=float)
    t_s = _integrate_time_profile(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps)
    ds_m = max(tod_m / max(request.optimizer.num_nodes - 1, 1), 1.0)
    thrust_delta = np.empty_like(thrust_n)
    for idx, (s_val, h_val, v_val, t_val) in enumerate(zip(s_m, h_m, v_tas_mps, t_s, strict=True)):
        mode = mode_for_s(request.cfg, float(s_val))
        delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_val), float(t_val))
        idle = request.perf.idle_thrust_newtons(
            v_tas_mps=float(v_val),
            h_m=float(h_val),
            delta_isa_K=delta_isa_K,
        )
        thrust_delta[idx] = (float(thrust_n[idx]) - idle) / max(1.0, scale.thrust_n)
    gamma_gradient = np.diff(gamma_rad) / max(ds_m, 1.0)
    objective = (
        -request.optimizer.tod_reward_weight * (tod_m / 1_000.0)
        + request.optimizer.thrust_penalty_weight * (tod_m / 1_000.0) * float(np.mean(thrust_delta**2))
        + request.optimizer.gamma_smoothness_weight
        * float(np.mean((gamma_gradient / np.deg2rad(1.0) * 1_000.0) ** 2))
        + request.optimizer.slack_penalty_weight * float(constraint_slack**2)
    )
    return float(objective)


def _equality_constraints(
    z: np.ndarray,
    *,
    request: LongitudinalPlanRequest,
    threshold_v_tas: float,
) -> np.ndarray:
    h_m, v_tas_mps, gamma_rad, thrust_n, tod_m, constraint_slack = _unpack(z, request.optimizer.num_nodes)
    del constraint_slack
    s_m = np.linspace(0.0, tod_m, request.optimizer.num_nodes, dtype=float)
    t_s = _integrate_time_profile(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps)
    ds_m = float(tod_m / (request.optimizer.num_nodes - 1))
    defects: list[float] = []
    for idx in range(request.optimizer.num_nodes - 1):
        state_i = distance_state_derivatives(
            s_m=float(s_m[idx]),
            h_m=float(h_m[idx]),
            v_tas_mps=float(v_tas_mps[idx]),
            t_s=float(t_s[idx]),
            gamma_rad=float(gamma_rad[idx]),
            thrust_n=float(thrust_n[idx]),
            cfg=request.cfg,
            perf=request.perf,
            weather=request.weather,
            reference_track_rad=request.reference_track_rad,
        )
        state_j = distance_state_derivatives(
            s_m=float(s_m[idx + 1]),
            h_m=float(h_m[idx + 1]),
            v_tas_mps=float(v_tas_mps[idx + 1]),
            t_s=float(t_s[idx + 1]),
            gamma_rad=float(gamma_rad[idx + 1]),
            thrust_n=float(thrust_n[idx + 1]),
            cfg=request.cfg,
            perf=request.perf,
            weather=request.weather,
            reference_track_rad=request.reference_track_rad,
        )
        trapezoid = 0.5 * ds_m * (state_i + state_j)
        defects.extend(
            (
                np.asarray([h_m[idx + 1], v_tas_mps[idx + 1]])
                - np.asarray([h_m[idx], v_tas_mps[idx]])
                - trapezoid[:2]
            ).tolist()
        )

    defects.extend(
        [
            float(h_m[0] - request.threshold.h_m),
            float(v_tas_mps[0] - threshold_v_tas),
            float(gamma_rad[0] - request.threshold.gamma_rad),
            float(h_m[-1] - request.upstream.h_m),
            float(gamma_rad[-1] - request.upstream.gamma_rad),
        ]
    )
    return np.asarray(defects, dtype=float)


def _inequality_constraints(
    z: np.ndarray,
    *,
    request: LongitudinalPlanRequest,
    scale: _PlannerScale,
) -> np.ndarray:
    h_m, v_tas_mps, gamma_rad, thrust_n, tod_m, constraint_slack = _unpack(z, request.optimizer.num_nodes)
    s_m = np.linspace(0.0, tod_m, request.optimizer.num_nodes, dtype=float)
    t_s = _integrate_time_profile(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps)
    v_cas_mps = _cas_from_tas(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps, t_s=t_s)
    slack_altitude = constraint_slack * scale.altitude_m
    slack_speed = constraint_slack * scale.speed_mps
    slack_gamma = constraint_slack * scale.gamma_rad
    slack_thrust = constraint_slack * scale.thrust_n

    residuals: list[float] = []
    for idx, (s_val, h_val, v_tas, v_cas, t_val, gamma_val, thrust_val) in enumerate(
        zip(s_m, h_m, v_tas_mps, v_cas_mps, t_s, gamma_rad, thrust_n, strict=True)
    ):
        h_lower, h_upper = request.constraints.h_bounds(float(s_val))
        residuals.extend(
            [
                float(h_val - h_lower + slack_altitude),
                float(h_upper - h_val + slack_altitude),
            ]
        )

        cas_lower, cas_upper = request.constraints.cas_bounds(float(s_val))
        mode_lower, mode_upper = planned_cas_bounds_mps(request.cfg, float(s_val))
        cas_lower_eff = max(cas_lower, mode_lower)
        cas_upper_eff = min(cas_upper, mode_upper)
        residuals.extend(
            [
                float(v_cas - cas_lower_eff + slack_speed),
                float(cas_upper_eff - v_cas + slack_speed),
            ]
        )

        mode = mode_for_s(request.cfg, float(s_val))
        delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_val), float(t_val))
        thrust_lower_backend, thrust_upper_backend = request.perf.thrust_bounds_newtons(
            mode=mode,
            v_tas_mps=float(v_tas),
            h_m=float(h_val),
            delta_isa_K=delta_isa_K,
        )
        thrust_lower_env, thrust_upper_env = request.constraints.thrust_bounds(float(s_val))
        thrust_lower = thrust_lower_backend if thrust_lower_env is None else max(thrust_lower_backend, thrust_lower_env)
        thrust_upper = thrust_upper_backend if thrust_upper_env is None else min(thrust_upper_backend, thrust_upper_env)
        residuals.extend(
            [
                float(thrust_val - thrust_lower + slack_thrust),
                float(thrust_upper - thrust_val + slack_thrust),
            ]
        )

        gamma_lower, gamma_upper = request.constraints.gamma_bounds(float(s_val))
        if gamma_lower is not None:
            residuals.append(float(gamma_val - gamma_lower + slack_gamma))
        if gamma_upper is not None:
            residuals.append(float(gamma_upper - gamma_val + slack_gamma))

        cl_upper = request.constraints.cl_upper(float(s_val))
        if cl_upper is not None:
            cl = quasi_steady_cl(
                mass_kg=request.cfg.mass_kg,
                wing_area_m2=request.cfg.wing_area_m2,
                v_tas_mps=float(v_tas),
                h_m=float(h_val),
                gamma_rad=float(gamma_val),
                delta_isa_K=delta_isa_K,
            )
            residuals.append(float(cl_upper - cl + constraint_slack))

    residuals.extend(
        [
            float(v_cas_mps[-1] - request.upstream.cas_lower_mps + slack_speed),
            float(request.upstream.cas_upper_mps - v_cas_mps[-1] + slack_speed),
        ]
    )
    return np.asarray(residuals, dtype=float)


def _cas_from_tas(
    request: LongitudinalPlanRequest,
    *,
    s_m: np.ndarray,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    t_s: np.ndarray,
) -> np.ndarray:
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
