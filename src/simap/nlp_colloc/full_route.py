from __future__ import annotations

from time import perf_counter
from typing import Any, cast

import numpy as np
from openap import aero
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, minimize

from ..config import bank_limit_rad, mode_for_s
from .coupled import (
    CoupledDescentPlanRequest,
    CoupledDescentPlanResult,
    CoupledDescentSolveProfile,
    _PlannerScale,
    _planned_cas_bounds_many,
)
from ..longitudinal_dynamics import distance_state_derivatives, quasi_steady_cl
from ..openap_adapter import openap_dT

__all__ = ["plan_full_route_longitudinal_descent"]


def _s_grid(request: CoupledDescentPlanRequest) -> np.ndarray:
    total_s_m = float(request.reference_path.total_length_m)
    num_nodes = request.optimizer.num_nodes
    anchors = np.asarray(
        [
            0.0,
            min(8_000.0, total_s_m),
            min(float(request.cfg.final_gate_m), total_s_m),
            min(30_000.0, total_s_m),
            min(float(request.cfg.approach_gate_m), total_s_m),
            min(60_000.0, total_s_m),
            total_s_m,
        ],
        dtype=float,
    )
    anchors = np.unique(anchors)
    if len(anchors) >= num_nodes:
        return np.linspace(0.0, total_s_m, num_nodes, dtype=float)

    weights = np.sqrt(np.maximum(np.diff(anchors), 1.0))
    counts = np.maximum(2, np.rint(weights / np.sum(weights) * (num_nodes + len(anchors) - 1)).astype(int))
    while int(np.sum(counts) - len(counts) + 1) > num_nodes:
        idx = int(np.argmax(counts))
        if counts[idx] <= 2:
            break
        counts[idx] -= 1
    while int(np.sum(counts) - len(counts) + 1) < num_nodes:
        counts[int(np.argmax(weights))] += 1

    pieces = []
    for idx, count in enumerate(counts):
        segment = np.linspace(anchors[idx], anchors[idx + 1], int(count), dtype=float)
        if idx:
            segment = segment[1:]
        pieces.append(segment)
    return np.concatenate(pieces)


def _cas_from_tas(
    request: CoupledDescentPlanRequest,
    *,
    s_m: np.ndarray,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    t_s: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            aero.tas2cas(
                float(v_tas),
                float(h),
                dT=openap_dT(request.weather.delta_isa_K(float(s), float(h), float(t))),
            )
            for s, h, v_tas, t in zip(s_m, h_m, v_tas_mps, t_s, strict=True)
        ],
        dtype=float,
    )


def _tas_from_cas(
    request: CoupledDescentPlanRequest,
    *,
    s_m: np.ndarray,
    h_m: np.ndarray,
    cas_mps: np.ndarray,
    t_s: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            aero.cas2tas(
                float(cas),
                float(h),
                dT=openap_dT(request.weather.delta_isa_K(float(s), float(h), float(t))),
            )
            for s, h, cas, t in zip(s_m, h_m, cas_mps, t_s, strict=True)
        ],
        dtype=float,
    )


def _effective_cas_bounds(request: CoupledDescentPlanRequest, s_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cas_lower_env, cas_upper_env = request.constraints.cas_bounds_many(s_m)
    mode_lower, mode_upper = _planned_cas_bounds_many(request.cfg, s_m)
    return np.maximum(cas_lower_env, mode_lower), np.minimum(cas_upper_env, mode_upper)


def _pack(
    *,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    t_s: np.ndarray,
    gamma_rad: np.ndarray,
    thrust_n: np.ndarray,
    constraint_slack: float,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(h_m, dtype=float),
            np.asarray(v_tas_mps, dtype=float),
            np.asarray(t_s, dtype=float),
            np.asarray(gamma_rad, dtype=float),
            np.asarray(thrust_n, dtype=float),
            np.asarray([float(constraint_slack)], dtype=float),
        ]
    )


def _unpack(z: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    h_end = num_nodes
    v_end = h_end + num_nodes
    t_end = v_end + num_nodes
    gamma_end = t_end + num_nodes
    thrust_end = gamma_end + num_nodes
    return (
        np.asarray(z[:h_end], dtype=float),
        np.asarray(z[h_end:v_end], dtype=float),
        np.asarray(z[v_end:t_end], dtype=float),
        np.asarray(z[t_end:gamma_end], dtype=float),
        np.asarray(z[gamma_end:thrust_end], dtype=float),
        float(np.asarray(z[thrust_end], dtype=float)),
    )


def _initial_altitude_and_gamma(request: CoupledDescentPlanRequest, s_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total_s_m = max(float(s_m[-1]), 1.0)
    x = s_m / total_s_m
    slope0 = -np.tan(request.threshold.gamma_rad)
    slope1 = -np.tan(request.upstream.gamma_rad)
    h00 = 2.0 * x**3 - 3.0 * x**2 + 1.0
    h10 = x**3 - 2.0 * x**2 + x
    h01 = -2.0 * x**3 + 3.0 * x**2
    h11 = x**3 - x**2
    h_m = (
        h00 * request.threshold.h_m
        + h10 * total_s_m * slope0
        + h01 * request.upstream.h_m
        + h11 * total_s_m * slope1
    )
    h_lower, h_upper = request.constraints.h_bounds_many(s_m)
    h_m = np.clip(h_m, h_lower, h_upper)

    dhds = np.gradient(h_m, s_m, edge_order=1)
    gamma_rad = -np.arctan(dhds)
    gamma_rad[0] = request.threshold.gamma_rad
    gamma_rad[-1] = request.upstream.gamma_rad
    gamma_lower, gamma_upper = request.constraints.gamma_bounds_many(s_m)
    if gamma_lower is not None:
        gamma_rad = np.maximum(gamma_rad, gamma_lower)
    if gamma_upper is not None:
        gamma_rad = np.minimum(gamma_rad, gamma_upper)
    gamma_rad[0] = request.threshold.gamma_rad
    gamma_rad[-1] = request.upstream.gamma_rad
    return h_m, gamma_rad


def _initial_cas_guess(request: CoupledDescentPlanRequest, s_m: np.ndarray) -> np.ndarray:
    cas_lower, cas_upper = _effective_cas_bounds(request, s_m)
    upstream_target = request.upstream.cas_target_mps
    if upstream_target is None:
        upstream_target = 0.5 * (request.upstream.cas_lower_mps + request.upstream.cas_upper_mps)

    anchors_s = np.asarray(
        [
            0.0,
            min(float(request.cfg.final_gate_m), float(s_m[-1])),
            min(float(request.cfg.approach_gate_m), float(s_m[-1])),
            float(s_m[-1]),
        ],
        dtype=float,
    )
    anchors_cas = np.asarray(
        [
            request.threshold.cas_mps,
            min(cas_upper[np.searchsorted(s_m, anchors_s[1], side="left")], upstream_target),
            min(cas_upper[np.searchsorted(s_m, anchors_s[2], side="left")], upstream_target),
            upstream_target,
        ],
        dtype=float,
    )
    unique_s, unique_indices = np.unique(anchors_s, return_index=True)
    cas_guess = np.interp(s_m, unique_s, anchors_cas[unique_indices])
    cas_guess = np.clip(cas_guess, cas_lower, cas_upper)
    cas_guess[0] = request.threshold.cas_mps
    cas_guess[-1] = upstream_target
    return cas_guess


def _initial_guess(request: CoupledDescentPlanRequest, s_m: np.ndarray) -> np.ndarray:
    h_m, gamma_rad = _initial_altitude_and_gamma(request, s_m)
    cas_guess = _initial_cas_guess(request, s_m)
    t_s = np.zeros_like(s_m)
    v_tas_mps = _tas_from_cas(request, s_m=s_m, h_m=h_m, cas_mps=cas_guess, t_s=t_s)

    segment_speed = np.maximum(1.0, 0.5 * (v_tas_mps[:-1] + v_tas_mps[1:]))
    t_s = np.concatenate([np.zeros(1, dtype=float), np.cumsum(np.diff(s_m) / segment_speed)])
    v_tas_mps = _tas_from_cas(request, s_m=s_m, h_m=h_m, cas_mps=cas_guess, t_s=t_s)

    thrust_n = np.empty_like(s_m)
    dvds = np.gradient(v_tas_mps, s_m, edge_order=1)
    for idx, (s_val, h_val, v_val, t_val, gamma_val, dvds_val) in enumerate(
        zip(s_m, h_m, v_tas_mps, t_s, gamma_rad, dvds, strict=True)
    ):
        mode = mode_for_s(request.cfg, float(s_val))
        delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_val), float(t_val))
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
            aero.g0 * np.sin(gamma_val) - float(v_val) * np.cos(gamma_val) * float(dvds_val)
        )
        thrust_n[idx] = float(np.clip(thrust_guess, thrust_lower, thrust_upper))

    return _pack(
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        t_s=t_s,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        constraint_slack=request.optimizer.initial_slack,
    )


def _decision_bounds(request: CoupledDescentPlanRequest, s_m: np.ndarray, scale: _PlannerScale) -> Bounds:
    num_nodes = request.optimizer.num_nodes
    h_lower, h_upper = request.constraints.h_bounds_many(s_m)
    lower_altitude = np.maximum(0.0, h_lower - 0.1 * scale.altitude_m)
    upper_altitude = h_upper + 0.25 * scale.altitude_m

    cas_lower, cas_upper = _effective_cas_bounds(request, s_m)
    upper_tas = []
    for s_val, h_val, cas_val in zip(s_m, upper_altitude, cas_upper, strict=True):
        delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_val), 0.0)
        upper_tas.append(float(aero.cas2tas(min(float(cas_val), request.cfg.vmo_kts * aero.kts), float(h_val), dT=openap_dT(delta_isa_K))))

    max_time_s = float(s_m[-1]) / max(1.0, request.optimizer.min_alongtrack_speed_mps)
    thrust_upper = 1.25 * scale.thrust_n
    gamma_limit = np.deg2rad(20.0)
    lower = np.concatenate(
        [
            lower_altitude,
            np.full(num_nodes, 1.0, dtype=float),
            np.zeros(num_nodes, dtype=float),
            np.full(num_nodes, -gamma_limit, dtype=float),
            np.zeros(num_nodes, dtype=float),
            np.asarray([0.0], dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            upper_altitude,
            np.asarray(upper_tas, dtype=float) * 1.15,
            np.full(num_nodes, max_time_s, dtype=float),
            np.full(num_nodes, np.deg2rad(5.0), dtype=float),
            np.full(num_nodes, thrust_upper, dtype=float),
            np.asarray([5.0], dtype=float),
        ]
    )
    return cast(Any, Bounds)(lower, upper)


def _state_derivatives(
    request: CoupledDescentPlanRequest,
    *,
    s_m: np.ndarray,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    t_s: np.ndarray,
    gamma_rad: np.ndarray,
    thrust_n: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            distance_state_derivatives(
                s_m=float(s_val),
                h_m=float(h_val),
                v_tas_mps=float(v_val),
                t_s=float(t_val),
                gamma_rad=float(gamma_val),
                thrust_n=float(thrust_val),
                cfg=request.cfg,
                perf=request.perf,
                weather=request.weather,
                reference_track_rad=request.reference_path.track_angle_rad(float(s_val)),
            )
            for s_val, h_val, v_val, t_val, gamma_val, thrust_val in zip(
                s_m, h_m, v_tas_mps, t_s, gamma_rad, thrust_n, strict=True
            )
        ],
        dtype=float,
    )


def _equality_constraints_unscaled(
    z: np.ndarray,
    *,
    request: CoupledDescentPlanRequest,
    s_m: np.ndarray,
    threshold_v_tas: float,
) -> np.ndarray:
    h_m, v_tas_mps, t_s, gamma_rad, thrust_n, _slack = _unpack(z, request.optimizer.num_nodes)
    derivatives = _state_derivatives(
        request,
        s_m=s_m,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        t_s=t_s,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
    )
    ds = np.diff(s_m)
    trapezoid = 0.5 * ds[:, np.newaxis] * (derivatives[:-1, :] + derivatives[1:, :])
    state_delta = np.column_stack([np.diff(h_m), np.diff(v_tas_mps), np.diff(t_s)])
    endpoint_defects = [
        float(h_m[0] - request.threshold.h_m),
        float(v_tas_mps[0] - threshold_v_tas),
        float(t_s[0]),
        float(gamma_rad[0] - request.threshold.gamma_rad),
        float(h_m[-1] - request.upstream.h_m),
        float(gamma_rad[-1] - request.upstream.gamma_rad),
    ]
    upstream_cas_target_mps = request.upstream.cas_target_mps
    if upstream_cas_target_mps is not None:
        endpoint_defects.append(
            float(
                _cas_from_tas(
                    request,
                    s_m=np.asarray([s_m[-1]], dtype=float),
                    h_m=np.asarray([h_m[-1]], dtype=float),
                    v_tas_mps=np.asarray([v_tas_mps[-1]], dtype=float),
                    t_s=np.asarray([t_s[-1]], dtype=float),
                )[0]
                - upstream_cas_target_mps
            )
        )
    return np.concatenate([(state_delta - trapezoid).reshape(-1), np.asarray(endpoint_defects, dtype=float)])


def _equality_constraints(
    z: np.ndarray,
    *,
    request: CoupledDescentPlanRequest,
    s_m: np.ndarray,
    threshold_v_tas: float,
    scale: _PlannerScale,
) -> np.ndarray:
    raw = _equality_constraints_unscaled(z, request=request, s_m=s_m, threshold_v_tas=threshold_v_tas)
    dynamic_rows = 3 * (request.optimizer.num_nodes - 1)
    scales = np.concatenate(
        [
            np.tile(np.asarray([scale.altitude_m, scale.speed_mps, max(float(s_m[-1]) / 100.0, 1.0)], dtype=float), request.optimizer.num_nodes - 1),
            np.asarray(
                [
                    scale.altitude_m,
                    scale.speed_mps,
                    max(float(s_m[-1]) / 100.0, 1.0),
                    scale.gamma_rad,
                    scale.altitude_m,
                    scale.gamma_rad,
                    scale.speed_mps,
                ][: len(raw) - dynamic_rows],
                dtype=float,
            ),
        ]
    )
    return raw / np.maximum(scales, 1e-9)


def _inequality_constraints(
    z: np.ndarray,
    *,
    request: CoupledDescentPlanRequest,
    s_m: np.ndarray,
    scale: _PlannerScale,
) -> np.ndarray:
    h_m, v_tas_mps, t_s, gamma_rad, thrust_n, constraint_slack = _unpack(z, request.optimizer.num_nodes)
    v_cas_mps = _cas_from_tas(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps, t_s=t_s)
    h_lower, h_upper = request.constraints.h_bounds_many(s_m)
    cas_lower, cas_upper = _effective_cas_bounds(request, s_m)
    gamma_lower, gamma_upper = request.constraints.gamma_bounds_many(s_m)
    thrust_lower_backend = np.empty_like(s_m)
    thrust_upper_backend = np.empty_like(s_m)
    cl_values = np.empty_like(s_m)
    for idx, (s_val, h_val, v_val, t_val, gamma_val) in enumerate(zip(s_m, h_m, v_tas_mps, t_s, gamma_rad, strict=True)):
        mode = mode_for_s(request.cfg, float(s_val))
        delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_val), float(t_val))
        thrust_lower_backend[idx], thrust_upper_backend[idx] = request.perf.thrust_bounds_newtons(
            mode=mode,
            v_tas_mps=float(v_val),
            h_m=float(h_val),
            delta_isa_K=delta_isa_K,
        )
        cl_values[idx] = quasi_steady_cl(
            mass_kg=request.cfg.mass_kg,
            wing_area_m2=request.cfg.wing_area_m2,
            v_tas_mps=float(v_val),
            h_m=float(h_val),
            gamma_rad=float(gamma_val),
            delta_isa_K=delta_isa_K,
        )

    thrust_lower_env, thrust_upper_env = request.constraints.thrust_bounds_many(s_m)
    thrust_lower = thrust_lower_backend if thrust_lower_env is None else np.maximum(thrust_lower_backend, thrust_lower_env)
    thrust_upper = thrust_upper_backend if thrust_upper_env is None else np.minimum(thrust_upper_backend, thrust_upper_env)

    slack_altitude = constraint_slack * scale.altitude_m
    slack_speed = constraint_slack * scale.speed_mps
    slack_gamma = constraint_slack * scale.gamma_rad
    slack_thrust = constraint_slack * scale.thrust_n
    residuals = [
        h_m - h_lower + slack_altitude,
        h_upper - h_m + slack_altitude,
        v_cas_mps - cas_lower + slack_speed,
        cas_upper - v_cas_mps + slack_speed,
        thrust_n - thrust_lower + slack_thrust,
        thrust_upper - thrust_n + slack_thrust,
    ]
    if gamma_lower is not None:
        residuals.append(gamma_rad - gamma_lower + slack_gamma)
    if gamma_upper is not None:
        residuals.append(gamma_upper - gamma_rad + slack_gamma)
    if request.constraints.cl_max is not None:
        residuals.append(request.constraints._interp_many_required(request.constraints.cl_max, s_m) - cl_values + constraint_slack)
    residuals.append(np.diff(t_s))
    return np.concatenate(residuals)


def _objective(
    z: np.ndarray,
    *,
    request: CoupledDescentPlanRequest,
    s_m: np.ndarray,
    scale: _PlannerScale,
) -> float:
    h_m, v_tas_mps, t_s, gamma_rad, thrust_n, constraint_slack = _unpack(z, request.optimizer.num_nodes)
    idle_thrust = np.asarray(
        [
            request.perf.idle_thrust_newtons(
                v_tas_mps=float(v_val),
                h_m=float(h_val),
                delta_isa_K=request.weather.delta_isa_K(float(s_val), float(h_val), float(t_val)),
            )
            for s_val, h_val, v_val, t_val in zip(s_m, h_m, v_tas_mps, t_s, strict=True)
        ],
        dtype=float,
    )
    thrust_delta = (thrust_n - idle_thrust) / max(scale.thrust_n, 1.0)
    ds = max(float(np.mean(np.diff(s_m))), 1.0)
    gamma_diff = np.diff(gamma_rad)
    gamma_curvature = np.diff(gamma_rad, n=2) if len(gamma_rad) > 2 else np.zeros(1, dtype=float)
    thrust_diff = np.diff(thrust_n) / max(scale.thrust_n, 1.0)
    return float(
        request.optimizer.thrust_penalty_weight * float(np.mean(thrust_delta**2))
        + request.optimizer.gamma_smoothness_weight * float(np.mean((gamma_diff / np.deg2rad(0.25)) ** 2))
        + 0.25 * request.optimizer.gamma_smoothness_weight * float(np.mean((gamma_curvature / np.deg2rad(0.10)) ** 2))
        + 0.25 * request.optimizer.thrust_penalty_weight * float(np.mean(thrust_diff**2))
        + request.optimizer.slack_penalty_weight * float(constraint_slack**2)
        + 1e-8 * float(np.mean((np.diff(t_s) / max(ds / 100.0, 1.0)) ** 2))
    )


def _replay_longitudinal(
    *,
    request: CoupledDescentPlanRequest,
    s_m: np.ndarray,
    h_m: np.ndarray,
    v_tas_mps: np.ndarray,
    t_s: np.ndarray,
    gamma_rad: np.ndarray,
    thrust_n: np.ndarray,
) -> dict[str, float]:
    def rhs(s_val: float, state: np.ndarray) -> np.ndarray:
        return distance_state_derivatives(
            s_m=float(s_val),
            h_m=float(state[0]),
            v_tas_mps=float(state[1]),
            t_s=float(state[2]),
            gamma_rad=float(np.interp(s_val, s_m, gamma_rad)),
            thrust_n=float(np.interp(s_val, s_m, thrust_n)),
            cfg=request.cfg,
            perf=request.perf,
            weather=request.weather,
            reference_track_rad=request.reference_path.track_angle_rad(float(s_val)),
        )

    replay = solve_ivp(
        rhs,
        (float(s_m[0]), float(s_m[-1])),
        y0=np.asarray([h_m[0], v_tas_mps[0], t_s[0]], dtype=float),
        t_eval=s_m,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    if replay.success and replay.y.shape[1] == len(s_m):
        h_error = float(np.max(np.abs(replay.y[0] - h_m)))
        v_error = float(np.max(np.abs(replay.y[1] - v_tas_mps)))
        t_error = float(np.max(np.abs(replay.y[2] - t_s)))
        integration_error = 0.0
    else:
        h_error = v_error = t_error = float(np.finfo(float).max)
        integration_error = float(np.finfo(float).max)
    return {
        "h_error_m": h_error,
        "v_error_mps": v_error,
        "t_error_s": t_error,
        "max_error": max(h_error, v_error, t_error, integration_error),
    }


def plan_full_route_longitudinal_descent(request: CoupledDescentPlanRequest) -> CoupledDescentPlanResult:
    solve_started_at = perf_counter()
    s_m = _s_grid(request)
    scale = _PlannerScale.from_request(request)
    threshold_delta_isa = request.weather.delta_isa_K(0.0, request.threshold.h_m, 0.0)
    threshold_v_tas = float(aero.cas2tas(request.threshold.cas_mps, request.threshold.h_m, dT=openap_dT(threshold_delta_isa)))
    z0 = _initial_guess(request, s_m)
    bounds = _decision_bounds(request, s_m, scale)

    objective_calls = 0
    equality_calls = 0
    inequality_calls = 0
    objective_time_s = 0.0
    equality_time_s = 0.0
    inequality_time_s = 0.0

    def objective_fun(z: np.ndarray) -> float:
        nonlocal objective_calls, objective_time_s
        started_at = perf_counter()
        try:
            return _objective(z, request=request, s_m=s_m, scale=scale)
        finally:
            objective_calls += 1
            objective_time_s += perf_counter() - started_at

    def equality_fun(z: np.ndarray) -> np.ndarray:
        nonlocal equality_calls, equality_time_s
        started_at = perf_counter()
        try:
            return _equality_constraints(z, request=request, s_m=s_m, threshold_v_tas=threshold_v_tas, scale=scale)
        finally:
            equality_calls += 1
            equality_time_s += perf_counter() - started_at

    def inequality_fun(z: np.ndarray) -> np.ndarray:
        nonlocal inequality_calls, inequality_time_s
        started_at = perf_counter()
        try:
            return _inequality_constraints(z, request=request, s_m=s_m, scale=scale)
        finally:
            inequality_calls += 1
            inequality_time_s += perf_counter() - started_at

    result = minimize(
        objective_fun,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": equality_fun},
            {"type": "ineq", "fun": inequality_fun},
        ],
        options={
            "maxiter": request.optimizer.maxiter,
            "ftol": request.optimizer.constraint_tolerance,
            "disp": request.optimizer.verbose > 0,
        },
    )

    postprocess_started_at = perf_counter()
    h_m, v_tas_mps, t_s, gamma_rad, thrust_n, constraint_slack = _unpack(result.x, request.optimizer.num_nodes)
    h_m = np.array(h_m, dtype=float, copy=True)
    v_tas_mps = np.array(v_tas_mps, dtype=float, copy=True)
    t_s = np.array(t_s, dtype=float, copy=True)
    gamma_rad = np.array(gamma_rad, dtype=float, copy=True)
    thrust_n = np.array(thrust_n, dtype=float, copy=True)
    h_m[0] = request.threshold.h_m
    h_m[-1] = request.upstream.h_m
    gamma_rad[0] = request.threshold.gamma_rad
    gamma_rad[-1] = request.upstream.gamma_rad
    t_s[0] = 0.0
    if request.upstream.cas_target_mps is not None:
        delta_isa_K = request.weather.delta_isa_K(float(s_m[-1]), request.upstream.h_m, float(t_s[-1]))
        v_tas_mps[-1] = float(aero.cas2tas(request.upstream.cas_target_mps, request.upstream.h_m, dT=openap_dT(delta_isa_K)))
    v_tas_mps[0] = threshold_v_tas
    v_cas_mps = _cas_from_tas(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps, t_s=t_s)

    positions = request.reference_path.position_ne_many(s_m)
    latlon = request.reference_path.latlon_from_ne_many(positions[:, 0], positions[:, 1])
    psi_rad = request.reference_path.track_angle_rad_many(s_m)
    zero = np.zeros_like(s_m)
    mode = tuple(mode_for_s(request.cfg, float(s)).name for s in s_m)
    phi_max_rad = np.asarray([bank_limit_rad(request.cfg, mode_for_s(request.cfg, float(s)), float(cas)) for s, cas in zip(s_m, v_cas_mps, strict=True)])
    replay = _replay_longitudinal(
        request=request,
        s_m=s_m,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        t_s=t_s,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
    )
    collocation_residual_max = float(
        np.max(
            np.abs(
                _equality_constraints_unscaled(
                    result.x,
                    request=request,
                    s_m=s_m,
                    threshold_v_tas=threshold_v_tas,
                )
            )
        )
    )
    usable_solution = (
        bool(result.success)
        or (collocation_residual_max <= 1e-2 and float(constraint_slack) <= 2e-2)
    )
    solve_profile = CoupledDescentSolveProfile(
        total_wall_time_s=float(perf_counter() - solve_started_at),
        postprocess_wall_time_s=float(perf_counter() - postprocess_started_at),
        objective_calls=int(objective_calls),
        objective_time_s=float(objective_time_s),
        equality_calls=int(equality_calls),
        equality_time_s=float(equality_time_s),
        inequality_calls=int(inequality_calls),
        inequality_time_s=float(inequality_time_s),
        trajectory_evaluations=0,
        trajectory_eval_time_s=0.0,
        trajectory_cache_hits=0,
    )
    return CoupledDescentPlanResult(
        s_m=s_m,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        v_cas_mps=v_cas_mps,
        t_s=t_s,
        east_m=positions[:, 0],
        north_m=positions[:, 1],
        lat_deg=latlon[:, 0],
        lon_deg=latlon[:, 1],
        cross_track_m=zero.copy(),
        heading_error_rad=zero.copy(),
        psi_rad=psi_rad,
        phi_rad=zero.copy(),
        roll_rate_rps=zero.copy(),
        ground_speed_mps=v_tas_mps.copy(),
        alongtrack_speed_mps=v_tas_mps.copy(),
        crosstrack_speed_mps=zero.copy(),
        track_error_rad=zero.copy(),
        phi_max_rad=phi_max_rad,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        mode=mode,
        solver_success=bool(usable_solution),
        solver_status=int(result.status),
        solver_message=str(result.message),
        objective_value=float(result.fun),
        tod_m=float(s_m[-1]),
        collocation_residual_max=collocation_residual_max,
        replay_h_error_m=float(replay["h_error_m"]),
        replay_v_error_mps=float(replay["v_error_mps"]),
        replay_t_error_s=float(replay["t_error_s"]),
        replay_residual_max=float(replay["max_error"]),
        constraint_slack=float(constraint_slack),
        solve_profile=solve_profile,
    )
