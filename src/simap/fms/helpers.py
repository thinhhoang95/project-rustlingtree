from __future__ import annotations

import numpy as np
from openap import aero

from ..config import ModeConfig, mode_for_s
from ..nlp_colloc.coupled import CoupledDescentPlanRequest
from ..openap_adapter import openap_dT
from ..weather import WeatherProvider, alongtrack_wind_mps
from .datatypes import FMSRequest, FMSResult, FMSSpeedTargets


def _first_not_none(*values: float | None) -> float:
    for value in values:
        if value is not None:
            return float(value)
    raise ValueError("expected at least one non-None value")


def infer_fms_speed_targets(request: CoupledDescentPlanRequest) -> FMSSpeedTargets:
    return FMSSpeedTargets(
        clean_cas_mps=float(request.upstream.cas_upper_mps),
        approach_cas_mps=_first_not_none(request.cfg.approach.cas_min_mps, request.upstream.cas_lower_mps),
        final_cas_mps=_first_not_none(request.cfg.final.cas_min_mps, request.threshold.cas_mps),
    )


def _cas_from_tas(
    *,
    weather: WeatherProvider,
    s_m: float,
    h_m: float,
    t_s: float,
    v_tas_mps: float,
) -> float:
    delta_isa_K = float(weather.delta_isa_K(s_m, h_m, t_s))
    return float(aero.tas2cas(v_tas_mps, h_m, dT=openap_dT(delta_isa_K)))


def _tas_from_cas(
    *,
    weather: WeatherProvider,
    s_m: float,
    h_m: float,
    t_s: float,
    v_cas_mps: float,
) -> float:
    delta_isa_K = float(weather.delta_isa_K(s_m, h_m, t_s))
    return float(aero.cas2tas(v_cas_mps, h_m, dT=openap_dT(delta_isa_K)))


def _idle_thrust(
    *,
    request: FMSRequest,
    mode: ModeConfig,
    v_tas_mps: float,
    h_m: float,
    t_s: float,
    s_m: float,
) -> float:
    delta_isa_K = float(request.weather.delta_isa_K(s_m, h_m, t_s))
    lower, _upper = request.perf.thrust_bounds_newtons(
        mode=mode,
        v_tas_mps=v_tas_mps,
        h_m=h_m,
        delta_isa_K=delta_isa_K,
    )
    return float(lower)


def _drag(
    *,
    request: FMSRequest,
    mode: ModeConfig,
    v_tas_mps: float,
    h_m: float,
    gamma_rad: float,
    t_s: float,
    s_m: float,
) -> float:
    delta_isa_K = float(request.weather.delta_isa_K(s_m, h_m, t_s))
    return float(
        request.perf.drag_newtons(
            mode=mode,
            mass_kg=request.cfg.mass_kg,
            wing_area_m2=request.cfg.wing_area_m2,
            v_tas_mps=v_tas_mps,
            h_m=h_m,
            gamma_rad=gamma_rad,
            delta_isa_K=delta_isa_K,
        )
    )


def _ground_speed_mps(
    *,
    request: FMSRequest,
    s_m: float,
    h_m: float,
    t_s: float,
    v_tas_mps: float,
) -> float:
    track_rad = request.reference_path.track_angle_rad(s_m)
    wind_mps = alongtrack_wind_mps(request.weather, track_rad, s_m, h_m, t_s)
    return float(max(1.0, v_tas_mps + wind_mps))


def _copy_request(
    request: FMSRequest,
    *,
    start_s_m: float | None = None,
    stop_at_reference_path_end: bool | None = None,
) -> FMSRequest:
    return FMSRequest(
        cfg=request.cfg,
        perf=request.perf,
        reference_path=request.reference_path,
        start_s_m=float(request.start_s_m if start_s_m is None else start_s_m),
        start_h_m=request.start_h_m,
        start_cas_mps=request.start_cas_mps,
        target_h_m=request.target_h_m,
        speed_targets=request.speed_targets,
        weather=request.weather,
        controller=request.controller,
        dt_s=request.dt_s,
        max_time_s=request.max_time_s,
        stop_at_reference_path_end=(
            request.stop_at_reference_path_end if stop_at_reference_path_end is None else stop_at_reference_path_end
        ),
    )


def _result_with_metadata(
    result: FMSResult,
    *,
    success: bool,
    message: str,
    tod_s_m: float | None,
    level_distance_m: float,
    level_time_s: float,
    descent_segment_distance_m: float | None,
    descent_segment_time_s: float | None,
    phase: tuple[str, ...] | None = None,
) -> FMSResult:
    return FMSResult(
        t_s=result.t_s,
        s_m=result.s_m,
        distance_flown_m=result.distance_flown_m,
        h_m=result.h_m,
        v_tas_mps=result.v_tas_mps,
        v_cas_mps=result.v_cas_mps,
        target_cas_mps=result.target_cas_mps,
        pitch_rad=result.pitch_rad,
        gamma_rad=result.gamma_rad,
        vertical_speed_mps=result.vertical_speed_mps,
        thrust_n=result.thrust_n,
        drag_n=result.drag_n,
        ground_speed_mps=result.ground_speed_mps,
        mode=result.mode,
        speed_error_mps=result.speed_error_mps,
        success=success,
        message=message,
        tod_s_m=tod_s_m,
        level_distance_m=level_distance_m,
        level_time_s=level_time_s,
        descent_segment_distance_m=descent_segment_distance_m,
        descent_segment_time_s=descent_segment_time_s,
        phase=result.phase if phase is None else phase,
    )


def _tod_metric(result: FMSResult, *, target_h_m: float) -> float:
    if result.success:
        return float(max(result.s_m[-1], 0.0))
    return -float(max(result.h_m[-1] - target_h_m, 0.0))


def _simulate_level_segment(request: FMSRequest, *, tod_s_m: float) -> FMSResult:
    start_s_m = float(request.start_s_m)
    tod_s_m = float(tod_s_m)
    if tod_s_m >= start_s_m:
        empty = np.asarray([0.0], dtype=float)
        mode = mode_for_s(request.cfg, start_s_m)
        v_tas_mps = _tas_from_cas(
            weather=request.weather,
            s_m=start_s_m,
            h_m=request.start_h_m,
            t_s=0.0,
            v_cas_mps=request.start_cas_mps,
        )
        target_cas_mps = request.speed_targets.for_mode(mode, h_m=request.start_h_m)
        thrust_n = _drag(
            request=request,
            mode=mode,
            v_tas_mps=v_tas_mps,
            h_m=request.start_h_m,
            gamma_rad=0.0,
            t_s=0.0,
            s_m=start_s_m,
        )
        ground_speed = _ground_speed_mps(
            request=request,
            s_m=start_s_m,
            h_m=request.start_h_m,
            t_s=0.0,
            v_tas_mps=v_tas_mps,
        )
        return FMSResult(
            t_s=empty,
            s_m=np.asarray([start_s_m], dtype=float),
            distance_flown_m=empty,
            h_m=np.asarray([request.start_h_m], dtype=float),
            v_tas_mps=np.asarray([v_tas_mps], dtype=float),
            v_cas_mps=np.asarray([request.start_cas_mps], dtype=float),
            target_cas_mps=np.asarray([target_cas_mps], dtype=float),
            pitch_rad=empty,
            gamma_rad=empty,
            vertical_speed_mps=empty,
            thrust_n=np.asarray([thrust_n], dtype=float),
            drag_n=np.asarray([thrust_n], dtype=float),
            ground_speed_mps=np.asarray([ground_speed], dtype=float),
            mode=(mode.name,),
            speed_error_mps=np.asarray([request.start_cas_mps - target_cas_mps], dtype=float),
            success=True,
            message="no level segment required",
            phase=("level",),
        )

    t_hist: list[float] = []
    s_hist: list[float] = []
    distance_hist: list[float] = []
    h_hist: list[float] = []
    tas_hist: list[float] = []
    cas_hist: list[float] = []
    target_cas_hist: list[float] = []
    thrust_hist: list[float] = []
    drag_hist: list[float] = []
    ground_speed_hist: list[float] = []
    mode_hist: list[str] = []
    speed_error_hist: list[float] = []

    t_s = 0.0
    s_m = start_s_m
    h_m = float(request.start_h_m)
    v_cas_mps = float(request.start_cas_mps)
    v_tas_mps = _tas_from_cas(
        weather=request.weather,
        s_m=s_m,
        h_m=h_m,
        t_s=t_s,
        v_cas_mps=v_cas_mps,
    )

    while True:
        mode = mode_for_s(request.cfg, s_m)
        target_cas_mps = request.speed_targets.for_mode(mode, h_m=h_m)
        drag_n = _drag(
            request=request,
            mode=mode,
            v_tas_mps=v_tas_mps,
            h_m=h_m,
            gamma_rad=0.0,
            t_s=t_s,
            s_m=s_m,
        )
        ground_speed = _ground_speed_mps(
            request=request,
            s_m=s_m,
            h_m=h_m,
            t_s=t_s,
            v_tas_mps=v_tas_mps,
        )

        t_hist.append(float(t_s))
        s_hist.append(float(s_m))
        distance_hist.append(float(start_s_m - s_m))
        h_hist.append(float(h_m))
        tas_hist.append(float(v_tas_mps))
        cas_hist.append(float(v_cas_mps))
        target_cas_hist.append(float(target_cas_mps))
        thrust_hist.append(float(drag_n))
        drag_hist.append(float(drag_n))
        ground_speed_hist.append(float(ground_speed))
        mode_hist.append(mode.name)
        speed_error_hist.append(float(v_cas_mps - target_cas_mps))

        if s_m <= tod_s_m:
            break
        step_dt_s = float(request.dt_s)
        if ground_speed > 1e-9:
            step_dt_s = min(step_dt_s, (s_m - tod_s_m) / ground_speed)
        if step_dt_s <= 0.0:
            break
        t_s = float(t_s + step_dt_s)
        s_m = float(max(tod_s_m, s_m - ground_speed * step_dt_s))

    zeros = np.zeros(len(t_hist), dtype=float)
    return FMSResult(
        t_s=np.asarray(t_hist, dtype=float),
        s_m=np.asarray(s_hist, dtype=float),
        distance_flown_m=np.asarray(distance_hist, dtype=float),
        h_m=np.asarray(h_hist, dtype=float),
        v_tas_mps=np.asarray(tas_hist, dtype=float),
        v_cas_mps=np.asarray(cas_hist, dtype=float),
        target_cas_mps=np.asarray(target_cas_hist, dtype=float),
        pitch_rad=zeros,
        gamma_rad=zeros,
        vertical_speed_mps=zeros,
        thrust_n=np.asarray(thrust_hist, dtype=float),
        drag_n=np.asarray(drag_hist, dtype=float),
        ground_speed_mps=np.asarray(ground_speed_hist, dtype=float),
        mode=tuple(mode_hist),
        speed_error_mps=np.asarray(speed_error_hist, dtype=float),
        success=True,
        message="level segment complete",
        phase=("level",) * len(t_hist),
    )


def _stitch_results(level: FMSResult, descent: FMSResult, *, tod_s_m: float) -> FMSResult:
    descent_t_offset = float(level.t_s[-1])
    descent_distance_offset = float(level.distance_flown_m[-1])
    descent_slice = slice(1, None) if len(level) > 0 and len(descent) > 1 else slice(None)
    t_s = np.concatenate([level.t_s, descent.t_s[descent_slice] + descent_t_offset])
    distance_flown_m = np.concatenate(
        [level.distance_flown_m, descent.distance_flown_m[descent_slice] + descent_distance_offset]
    )
    phase = level.phase + ("descent",) * len(descent.t_s[descent_slice])
    return FMSResult(
        t_s=t_s,
        s_m=np.concatenate([level.s_m, descent.s_m[descent_slice]]),
        distance_flown_m=distance_flown_m,
        h_m=np.concatenate([level.h_m, descent.h_m[descent_slice]]),
        v_tas_mps=np.concatenate([level.v_tas_mps, descent.v_tas_mps[descent_slice]]),
        v_cas_mps=np.concatenate([level.v_cas_mps, descent.v_cas_mps[descent_slice]]),
        target_cas_mps=np.concatenate([level.target_cas_mps, descent.target_cas_mps[descent_slice]]),
        pitch_rad=np.concatenate([level.pitch_rad, descent.pitch_rad[descent_slice]]),
        gamma_rad=np.concatenate([level.gamma_rad, descent.gamma_rad[descent_slice]]),
        vertical_speed_mps=np.concatenate([level.vertical_speed_mps, descent.vertical_speed_mps[descent_slice]]),
        thrust_n=np.concatenate([level.thrust_n, descent.thrust_n[descent_slice]]),
        drag_n=np.concatenate([level.drag_n, descent.drag_n[descent_slice]]),
        ground_speed_mps=np.concatenate([level.ground_speed_mps, descent.ground_speed_mps[descent_slice]]),
        mode=level.mode + tuple(descent.mode[descent_slice]),
        speed_error_mps=np.concatenate([level.speed_error_mps, descent.speed_error_mps[descent_slice]]),
        success=True,
        message="stitched level cruise and descent reached target altitude at threshold",
        tod_s_m=tod_s_m,
        level_distance_m=float(level.distance_flown_m[-1]),
        level_time_s=float(level.t_s[-1]),
        descent_segment_distance_m=float(descent.distance_flown_m[-1]),
        descent_segment_time_s=float(descent.t_s[-1]),
        phase=phase,
    )


__all__ = [
    "infer_fms_speed_targets",
]
