from __future__ import annotations

import numpy as np
from openap import aero

from ..config import mode_for_s
from .datatypes import FMSPIConfig, FMSRequest, FMSResult, FMSSpeedTargets
from .helpers import (
    _cas_from_tas,
    _copy_request,
    _drag,
    _ground_speed_mps,
    _idle_thrust,
    _result_with_metadata,
    _simulate_level_segment,
    _stitch_results,
    _tas_from_cas,
    _tod_metric,
    infer_fms_speed_targets,
)


def simulate_fms_descent(request: FMSRequest) -> FMSResult:
    t_s = 0.0
    s_m = float(request.start_s_m)
    h_m = float(request.start_h_m)
    v_tas_mps = _tas_from_cas(
        weather=request.weather,
        s_m=s_m,
        h_m=h_m,
        t_s=t_s,
        v_cas_mps=float(request.start_cas_mps),
    )
    integral_error_mps_s = 0.0
    previous_mode_name: str | None = None

    t_hist: list[float] = []
    s_hist: list[float] = []
    distance_hist: list[float] = []
    h_hist: list[float] = []
    tas_hist: list[float] = []
    cas_hist: list[float] = []
    target_cas_hist: list[float] = []
    pitch_hist: list[float] = []
    gamma_hist: list[float] = []
    vertical_speed_hist: list[float] = []
    thrust_hist: list[float] = []
    drag_hist: list[float] = []
    ground_speed_hist: list[float] = []
    mode_hist: list[str] = []
    speed_error_hist: list[float] = []

    success = False
    message = f"simulation exceeded {request.max_time_s:.1f} s before reaching target altitude"

    while True:
        mode = mode_for_s(request.cfg, s_m)
        if previous_mode_name is not None and mode.name != previous_mode_name:
            integral_error_mps_s = 0.0
        previous_mode_name = mode.name
        cas_mps = _cas_from_tas(
            weather=request.weather,
            s_m=s_m,
            h_m=h_m,
            t_s=t_s,
            v_tas_mps=v_tas_mps,
        )
        target_cas_mps = request.speed_targets.for_mode(mode, h_m=h_m)
        speed_error_mps = float(cas_mps - target_cas_mps)
        raw_pitch_rad = float(
            request.controller.nominal_pitch_rad
            + request.controller.kp_rad_per_mps * speed_error_mps
            + request.controller.ki_rad_per_mps_s * integral_error_mps_s
        )
        raw_vertical_speed_mps = float(max(1.0, v_tas_mps) * np.sin(raw_pitch_rad))
        vertical_speed_mps = float(
            np.clip(
                raw_vertical_speed_mps,
                request.controller.min_vertical_speed_mps,
                request.controller.max_vertical_speed_mps,
            )
        )
        gamma_rad = float(np.arcsin(np.clip(vertical_speed_mps / max(1.0, v_tas_mps), -0.95, 0.95)))
        thrust_n = _idle_thrust(
            request=request,
            mode=mode,
            v_tas_mps=v_tas_mps,
            h_m=h_m,
            t_s=t_s,
            s_m=s_m,
        )
        drag_n = _drag(
            request=request,
            mode=mode,
            v_tas_mps=v_tas_mps,
            h_m=h_m,
            gamma_rad=gamma_rad,
            t_s=t_s,
            s_m=s_m,
        )
        ground_speed_mps = _ground_speed_mps(
            request=request,
            s_m=s_m,
            h_m=h_m,
            t_s=t_s,
            v_tas_mps=v_tas_mps,
        )

        t_hist.append(float(t_s))
        s_hist.append(float(s_m))
        distance_hist.append(float(request.start_s_m - s_m))
        h_hist.append(float(h_m))
        tas_hist.append(float(v_tas_mps))
        cas_hist.append(float(cas_mps))
        target_cas_hist.append(float(target_cas_mps))
        pitch_hist.append(float(gamma_rad))
        gamma_hist.append(float(gamma_rad))
        vertical_speed_hist.append(float(vertical_speed_mps))
        thrust_hist.append(float(thrust_n))
        drag_hist.append(float(drag_n))
        ground_speed_hist.append(float(ground_speed_mps))
        mode_hist.append(mode.name)
        speed_error_hist.append(float(speed_error_mps))

        if h_m <= request.target_h_m:
            success = True
            message = "reached target altitude"
            break
        if request.stop_at_reference_path_end and s_m <= 0.0:
            message = "reached end of reference path before target altitude"
            break
        if t_s >= request.max_time_s:
            break

        step_dt_s = float(min(request.dt_s, request.max_time_s - t_s))
        if request.stop_at_reference_path_end and ground_speed_mps > 1e-9:
            step_dt_s = float(min(step_dt_s, s_m / ground_speed_mps))
        if vertical_speed_mps < -1e-9 and h_m + vertical_speed_mps * step_dt_s < request.target_h_m:
            step_dt_s = float((request.target_h_m - h_m) / vertical_speed_mps)
        if step_dt_s <= 0.0:
            break

        saturated_level = (
            vertical_speed_mps >= request.controller.max_vertical_speed_mps - 1e-9
            and raw_vertical_speed_mps > request.controller.max_vertical_speed_mps
        )
        saturated_descent = (
            vertical_speed_mps <= request.controller.min_vertical_speed_mps + 1e-9
            and raw_vertical_speed_mps < request.controller.min_vertical_speed_mps
        )
        should_integrate = not (
            (saturated_level and speed_error_mps > 0.0)
            or (saturated_descent and speed_error_mps < 0.0)
        )
        if should_integrate:
            integral_error_mps_s = float(
                np.clip(
                    integral_error_mps_s + speed_error_mps * step_dt_s,
                    -request.controller.integral_limit_mps_s,
                    request.controller.integral_limit_mps_s,
                )
            )
        v_dot_mps2 = float((thrust_n - drag_n) / request.cfg.mass_kg - aero.g0 * np.sin(gamma_rad))
        t_s = float(t_s + step_dt_s)
        s_m = float(s_m - ground_speed_mps * step_dt_s)
        h_m = float(h_m + vertical_speed_mps * step_dt_s)
        v_tas_mps = float(max(1.0, v_tas_mps + v_dot_mps2 * step_dt_s))

    return FMSResult(
        t_s=np.asarray(t_hist, dtype=float),
        s_m=np.asarray(s_hist, dtype=float),
        distance_flown_m=np.asarray(distance_hist, dtype=float),
        h_m=np.asarray(h_hist, dtype=float),
        v_tas_mps=np.asarray(tas_hist, dtype=float),
        v_cas_mps=np.asarray(cas_hist, dtype=float),
        target_cas_mps=np.asarray(target_cas_hist, dtype=float),
        pitch_rad=np.asarray(pitch_hist, dtype=float),
        gamma_rad=np.asarray(gamma_hist, dtype=float),
        vertical_speed_mps=np.asarray(vertical_speed_hist, dtype=float),
        thrust_n=np.asarray(thrust_hist, dtype=float),
        drag_n=np.asarray(drag_hist, dtype=float),
        ground_speed_mps=np.asarray(ground_speed_hist, dtype=float),
        mode=tuple(mode_hist),
        speed_error_mps=np.asarray(speed_error_hist, dtype=float),
        success=success,
        message=message,
    )


def _tod_metric(result: FMSResult, *, target_h_m: float) -> float:
    if result.success:
        return float(max(result.s_m[-1], 0.0))
    return -float(max(result.h_m[-1] - target_h_m, 0.0))


def _simulate_descent_to_threshold(request: FMSRequest, *, start_s_m: float) -> FMSResult:
    return simulate_fms_descent(
        _copy_request(
            request,
            start_s_m=start_s_m,
            stop_at_reference_path_end=True,
        )
    )


def _find_tod_s_m(
    request: FMSRequest,
    *,
    tolerance_m: float,
    max_iterations: int,
) -> tuple[float | None, FMSResult]:
    available_s_m = float(request.start_s_m)
    fms_descent = _simulate_descent_to_threshold(request, start_s_m=available_s_m)
    if _tod_metric(fms_descent, target_h_m=request.target_h_m) < 0.0:
        return None, fms_descent

    low_s_m = 1e-3
    high_s_m = available_s_m
    for _ in range(max_iterations):
        mid_s_m = 0.5 * (low_s_m + high_s_m)
        candidate = _simulate_descent_to_threshold(request, start_s_m=mid_s_m)
        metric = _tod_metric(candidate, target_h_m=request.target_h_m)
        if candidate.success:
            high_s_m = mid_s_m
        else:
            low_s_m = mid_s_m
        if abs(metric) <= tolerance_m or high_s_m - low_s_m <= tolerance_m:
            break

    tod_s_m = float(high_s_m)
    descent = _simulate_descent_to_threshold(request, start_s_m=tod_s_m)
    for _ in range(8):
        if not descent.success:
            break
        adjusted_tod_s_m = float(descent.distance_flown_m[-1])
        if abs(adjusted_tod_s_m - tod_s_m) <= tolerance_m:
            break
        tod_s_m = adjusted_tod_s_m
        descent = _simulate_descent_to_threshold(request, start_s_m=tod_s_m)
    return tod_s_m, descent


def plan_fms_descent(
    request: FMSRequest,
    *,
    tod_tolerance_m: float = 5.0,
    max_tod_iterations: int = 40,
) -> FMSResult:
    tod_s_m, descent = _find_tod_s_m(
        request,
        tolerance_m=tod_tolerance_m,
        max_iterations=max_tod_iterations,
    )
    if tod_s_m is None:
        truncated = _result_with_metadata(
            descent,
            success=False,
            message=(
                "infeasible: not enough along-track distance to complete FMS profile before threshold; "
                "showing FMS response truncated at threshold"
            ),
            tod_s_m=None,
            level_distance_m=0.0,
            level_time_s=0.0,
            descent_segment_distance_m=float(descent.distance_flown_m[-1]),
            descent_segment_time_s=float(descent.t_s[-1]),
            phase=("descent",) * len(descent),
        )
        return truncated

    level = _simulate_level_segment(request, tod_s_m=tod_s_m)
    return _stitch_results(level, descent, tod_s_m=tod_s_m)


__all__ = [
    "FMSPIConfig",
    "FMSRequest",
    "FMSResult",
    "FMSSpeedTargets",
    "infer_fms_speed_targets",
    "plan_fms_descent",
    "simulate_fms_descent",
]
