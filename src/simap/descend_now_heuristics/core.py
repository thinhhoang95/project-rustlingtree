from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from openap import aero

from simap.backends import PerformanceBackend
from simap.config import AircraftConfig, ModeConfig, mode_for_s
from simap.coupled_descent_planner import CoupledDescentPlanRequest
from simap.openap_adapter import openap_dT
from simap.path_geometry import ReferencePath
from simap.units import fpm_to_mps, ft_to_m, kts_to_mps
from simap.weather import ConstantWeather, WeatherProvider, alongtrack_wind_mps


@dataclass(frozen=True)
class DescendNowSpeedTargets:
    clean_cas_mps: float
    approach_cas_mps: float
    final_cas_mps: float
    below_altitude_limit_h_m: float | None = ft_to_m(10_000.0)
    below_altitude_limit_cas_mps: float | None = kts_to_mps(250.0)

    def for_mode(self, mode: ModeConfig, *, h_m: float | None = None) -> float:
        target = float(
            {
                "clean": self.clean_cas_mps,
                "approach": self.approach_cas_mps,
                "final": self.final_cas_mps,
            }[mode.name]
        )
        if (
            h_m is not None
            and self.below_altitude_limit_h_m is not None
            and self.below_altitude_limit_cas_mps is not None
            and h_m <= self.below_altitude_limit_h_m
        ):
            target = min(target, float(self.below_altitude_limit_cas_mps))
        return target


@dataclass(frozen=True)
class DescendNowPIConfig:
    nominal_pitch_rad: float = -np.deg2rad(3.0)
    kp_rad_per_mps: float = np.deg2rad(0.10)
    ki_rad_per_mps_s: float = np.deg2rad(0.003)
    integral_limit_mps_s: float = 500.0
    min_vertical_speed_mps: float = fpm_to_mps(-3_000.0)
    max_vertical_speed_mps: float = 0.0

    def __post_init__(self) -> None:
        if self.min_vertical_speed_mps > self.max_vertical_speed_mps:
            raise ValueError("min_vertical_speed_mps must not exceed max_vertical_speed_mps")
        if self.integral_limit_mps_s <= 0.0:
            raise ValueError("integral_limit_mps_s must be positive")


@dataclass(frozen=True)
class DescendNowRequest:
    cfg: AircraftConfig
    perf: PerformanceBackend
    reference_path: ReferencePath
    start_s_m: float
    start_h_m: float
    start_cas_mps: float
    target_h_m: float
    speed_targets: DescendNowSpeedTargets
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    controller: DescendNowPIConfig = field(default_factory=DescendNowPIConfig)
    dt_s: float = 0.5
    max_time_s: float = 7_200.0
    stop_at_reference_path_end: bool = False

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.max_time_s <= 0.0:
            raise ValueError("max_time_s must be positive")
        if self.start_s_m <= 0.0:
            raise ValueError("start_s_m must be positive")
        if self.start_s_m > self.reference_path.total_length_m + 1e-9:
            raise ValueError("start_s_m must lie on the reference path")
        if self.start_cas_mps <= 0.0:
            raise ValueError("start_cas_mps must be positive")

    @classmethod
    def from_coupled_request(
        cls,
        request: CoupledDescentPlanRequest,
        *,
        speed_targets: DescendNowSpeedTargets | None = None,
        start_s_m: float | None = None,
        dt_s: float = 0.5,
        max_time_s: float = 7_200.0,
        controller: DescendNowPIConfig | None = None,
    ) -> "DescendNowRequest":
        return cls(
            cfg=request.cfg,
            perf=request.perf,
            reference_path=request.reference_path,
            start_s_m=float(request.reference_path.total_length_m if start_s_m is None else start_s_m),
            start_h_m=float(request.upstream.h_m),
            start_cas_mps=float(request.upstream.cas_upper_mps),
            target_h_m=float(request.threshold.h_m),
            speed_targets=infer_speed_targets(request) if speed_targets is None else speed_targets,
            weather=request.weather,
            controller=DescendNowPIConfig() if controller is None else controller,
            dt_s=dt_s,
            max_time_s=max_time_s,
        )


@dataclass(frozen=True)
class DescendNowResult:
    t_s: np.ndarray
    s_m: np.ndarray
    distance_flown_m: np.ndarray
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    v_cas_mps: np.ndarray
    target_cas_mps: np.ndarray
    pitch_rad: np.ndarray
    gamma_rad: np.ndarray
    vertical_speed_mps: np.ndarray
    thrust_n: np.ndarray
    drag_n: np.ndarray
    ground_speed_mps: np.ndarray
    mode: tuple[str, ...]
    speed_error_mps: np.ndarray
    success: bool
    message: str
    tod_s_m: float | None = None
    level_distance_m: float = 0.0
    level_time_s: float = 0.0
    descent_segment_distance_m: float | None = None
    descent_segment_time_s: float | None = None
    phase: tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(len(self.t_s))

    @property
    def descent_distance_m(self) -> float:
        return float(self.distance_flown_m[-1])

    @property
    def descent_time_s(self) -> float:
        return float(self.t_s[-1])

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t_s": self.t_s,
                "s_m": self.s_m,
                "distance_flown_m": self.distance_flown_m,
                "h_m": self.h_m,
                "v_tas_mps": self.v_tas_mps,
                "v_cas_mps": self.v_cas_mps,
                "target_cas_mps": self.target_cas_mps,
                "pitch_rad": self.pitch_rad,
                "gamma_rad": self.gamma_rad,
                "vertical_speed_mps": self.vertical_speed_mps,
                "thrust_n": self.thrust_n,
                "drag_n": self.drag_n,
                "ground_speed_mps": self.ground_speed_mps,
                "mode": np.asarray(self.mode, dtype=object),
                "speed_error_mps": self.speed_error_mps,
                "phase": np.asarray(self.phase if self.phase else ("descent",) * len(self), dtype=object),
            }
        )


def _first_not_none(*values: float | None) -> float:
    for value in values:
        if value is not None:
            return float(value)
    raise ValueError("expected at least one non-None value")


def infer_speed_targets(request: CoupledDescentPlanRequest) -> DescendNowSpeedTargets:
    return DescendNowSpeedTargets(
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
    request: DescendNowRequest,
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
    request: DescendNowRequest,
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
    request: DescendNowRequest,
    s_m: float,
    h_m: float,
    t_s: float,
    v_tas_mps: float,
) -> float:
    track_rad = request.reference_path.track_angle_rad(s_m)
    wind_mps = alongtrack_wind_mps(request.weather, track_rad, s_m, h_m, t_s)
    return float(max(1.0, v_tas_mps + wind_mps))


def _copy_request(
    request: DescendNowRequest,
    *,
    start_s_m: float | None = None,
    stop_at_reference_path_end: bool | None = None,
) -> DescendNowRequest:
    return DescendNowRequest(
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
    result: DescendNowResult,
    *,
    success: bool,
    message: str,
    tod_s_m: float | None,
    level_distance_m: float,
    level_time_s: float,
    descent_segment_distance_m: float | None,
    descent_segment_time_s: float | None,
    phase: tuple[str, ...] | None = None,
) -> DescendNowResult:
    return DescendNowResult(
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


def simulate_descend_now(request: DescendNowRequest) -> DescendNowResult:
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

    return DescendNowResult(
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


def _tod_metric(result: DescendNowResult, *, target_h_m: float) -> float:
    if result.success:
        return float(max(result.s_m[-1], 0.0))
    return -float(max(result.h_m[-1] - target_h_m, 0.0))


def _simulate_descent_to_threshold(request: DescendNowRequest, *, start_s_m: float) -> DescendNowResult:
    return simulate_descend_now(
        _copy_request(
            request,
            start_s_m=start_s_m,
            stop_at_reference_path_end=True,
        )
    )


def _find_tod_s_m(
    request: DescendNowRequest,
    *,
    tolerance_m: float,
    max_iterations: int,
) -> tuple[float | None, DescendNowResult]:
    available_s_m = float(request.start_s_m)
    descend_now = _simulate_descent_to_threshold(request, start_s_m=available_s_m)
    if _tod_metric(descend_now, target_h_m=request.target_h_m) < 0.0:
        return None, descend_now

    low_s_m = 1e-3
    high_s_m = available_s_m
    best = descend_now
    for _ in range(max_iterations):
        mid_s_m = 0.5 * (low_s_m + high_s_m)
        candidate = _simulate_descent_to_threshold(request, start_s_m=mid_s_m)
        metric = _tod_metric(candidate, target_h_m=request.target_h_m)
        if candidate.success:
            high_s_m = mid_s_m
            best = candidate
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


def _simulate_level_segment(request: DescendNowRequest, *, tod_s_m: float) -> DescendNowResult:
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
        return DescendNowResult(
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
    return DescendNowResult(
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


def _stitch_results(level: DescendNowResult, descent: DescendNowResult, *, tod_s_m: float) -> DescendNowResult:
    descent_t_offset = float(level.t_s[-1])
    descent_distance_offset = float(level.distance_flown_m[-1])
    descent_slice = slice(1, None) if len(level) > 0 and len(descent) > 1 else slice(None)
    t_s = np.concatenate([level.t_s, descent.t_s[descent_slice] + descent_t_offset])
    distance_flown_m = np.concatenate(
        [level.distance_flown_m, descent.distance_flown_m[descent_slice] + descent_distance_offset]
    )
    phase = level.phase + ("descent",) * len(descent.t_s[descent_slice])
    return DescendNowResult(
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


def simulate_stitched_level_then_descend(
    request: DescendNowRequest,
    *,
    tod_tolerance_m: float = 5.0,
    max_tod_iterations: int = 40,
) -> DescendNowResult:
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
                "infeasible: not enough along-track distance to complete descend-now profile before threshold; "
                "showing descend-now response truncated at threshold"
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
