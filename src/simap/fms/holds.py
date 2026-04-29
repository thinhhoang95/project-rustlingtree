from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from openap import aero

from ..config import ModeConfig, mode_for_s
from ..units import fpm_to_mps, ft_to_m, kts_to_mps

from .datatypes import FMSRequest, FMSResult
from .helpers import (
    _cas_from_tas,
    _copy_request,
    _drag,
    _ground_speed_mps,
    _idle_thrust,
    _simulate_level_segment,
    _tas_from_cas,
)


@dataclass(frozen=True)
class HoldInstruction:
    holding_altitude_ft: float
    holding_time_s: float
    holding_speed_kts: float | None = None

    def __post_init__(self) -> None:
        if self.holding_time_s < 0.0:
            raise ValueError("holding_time_s must be nonnegative")
        if self.holding_speed_kts is not None and self.holding_speed_kts <= 0.0:
            raise ValueError("holding_speed_kts must be positive when supplied")

    @property
    def holding_altitude_m(self) -> float:
        return ft_to_m(float(self.holding_altitude_ft))

    @property
    def holding_speed_mps(self) -> float | None:
        if self.holding_speed_kts is None:
            return None
        return kts_to_mps(float(self.holding_speed_kts))


@dataclass(frozen=True)
class HoldControllerConfig:
    altitude_kp_vs_per_m: float = 0.035
    altitude_ki_vs_per_m_s: float = 0.001
    altitude_integral_limit_m_s: float = 5_000.0
    hold_vs_min_mps: float = fpm_to_mps(-1_200.0)
    hold_vs_max_mps: float = fpm_to_mps(1_200.0)
    speed_kp_mps2_per_mps: float = 0.045
    speed_ki_mps2_per_mps_s: float = 0.002
    speed_integral_limit_mps_s: float = 300.0
    speed_accel_limit_mps2: float = 0.6
    speed_capture_tolerance_mps: float = kts_to_mps(1.0)

    def __post_init__(self) -> None:
        if self.hold_vs_min_mps > self.hold_vs_max_mps:
            raise ValueError("hold_vs_min_mps must not exceed hold_vs_max_mps")
        if self.altitude_integral_limit_m_s <= 0.0:
            raise ValueError("altitude_integral_limit_m_s must be positive")
        if self.speed_integral_limit_mps_s <= 0.0:
            raise ValueError("speed_integral_limit_mps_s must be positive")
        if self.speed_accel_limit_mps2 <= 0.0:
            raise ValueError("speed_accel_limit_mps2 must be positive")
        if self.speed_capture_tolerance_mps < 0.0:
            raise ValueError("speed_capture_tolerance_mps must be nonnegative")


@dataclass(frozen=True)
class HoldAwareFMSRequest:
    base_request: FMSRequest
    holds: tuple[HoldInstruction, ...] = field(default_factory=tuple)
    hold_controller: HoldControllerConfig = field(default_factory=HoldControllerConfig)

    def __post_init__(self) -> None:
        holds = tuple(sorted(self.holds, key=lambda hold: hold.holding_altitude_m, reverse=True))
        previous_h_m = float(self.base_request.start_h_m)
        for hold in holds:
            hold_h_m = hold.holding_altitude_m
            if hold_h_m >= previous_h_m:
                raise ValueError("hold altitudes must be strictly descending below start altitude")
            if hold_h_m <= self.base_request.target_h_m:
                raise ValueError("hold altitudes must be above target altitude")
            previous_h_m = hold_h_m
        object.__setattr__(self, "holds", holds)


@dataclass
class _HoldRuntime:
    instruction: HoldInstruction
    target_h_m: float
    target_cas_mps: float
    remaining_time_s: float
    speed_captured: bool
    altitude_integral_m_s: float = 0.0
    speed_integral_mps_s: float = 0.0


@dataclass(frozen=True)
class _StepCommand:
    phase: str
    target_cas_mps: float
    gamma_rad: float
    vertical_speed_mps: float
    thrust_n: float
    drag_n: float
    ground_speed_mps: float
    speed_error_mps: float


def _hold_target_speed_mps(instruction: HoldInstruction, *, captured_cas_mps: float) -> float:
    supplied = instruction.holding_speed_mps
    if supplied is None:
        return float(captured_cas_mps)
    return float(supplied)


def _managed_command(
    *,
    request: FMSRequest,
    mode: ModeConfig,
    s_m: float,
    h_m: float,
    t_s: float,
    v_tas_mps: float,
    cas_mps: float,
    speed_integral_mps_s: float,
) -> tuple[_StepCommand, float, float]:
    target_cas_mps = request.speed_targets.for_mode(mode, h_m=h_m)
    speed_error_mps = float(cas_mps - target_cas_mps)
    raw_pitch_rad = float(
        request.controller.nominal_pitch_rad
        + request.controller.kp_rad_per_mps * speed_error_mps
        + request.controller.ki_rad_per_mps_s * speed_integral_mps_s
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

    saturated_level = (
        vertical_speed_mps >= request.controller.max_vertical_speed_mps - 1e-9
        and raw_vertical_speed_mps > request.controller.max_vertical_speed_mps
    )
    saturated_descent = (
        vertical_speed_mps <= request.controller.min_vertical_speed_mps + 1e-9
        and raw_vertical_speed_mps < request.controller.min_vertical_speed_mps
    )
    speed_integral_rate_mps = 0.0
    if not ((saturated_level and speed_error_mps > 0.0) or (saturated_descent and speed_error_mps < 0.0)):
        speed_integral_rate_mps = speed_error_mps

    return (
        _StepCommand(
            phase="managed_descent",
            target_cas_mps=target_cas_mps,
            gamma_rad=gamma_rad,
            vertical_speed_mps=vertical_speed_mps,
            thrust_n=thrust_n,
            drag_n=drag_n,
            ground_speed_mps=ground_speed_mps,
            speed_error_mps=speed_error_mps,
        ),
        speed_integral_rate_mps,
        raw_vertical_speed_mps,
    )


def _hold_command(
    *,
    request: FMSRequest,
    hold_request: HoldAwareFMSRequest,
    hold: _HoldRuntime,
    mode: ModeConfig,
    s_m: float,
    h_m: float,
    t_s: float,
    v_tas_mps: float,
    cas_mps: float,
) -> _StepCommand:
    controller = hold_request.hold_controller
    altitude_error_m = float(hold.target_h_m - h_m)
    vertical_speed_mps = float(
        np.clip(
            controller.altitude_kp_vs_per_m * altitude_error_m
            + controller.altitude_ki_vs_per_m_s * hold.altitude_integral_m_s,
            controller.hold_vs_min_mps,
            controller.hold_vs_max_mps,
        )
    )
    gamma_rad = float(np.arcsin(np.clip(vertical_speed_mps / max(1.0, v_tas_mps), -0.95, 0.95)))
    drag_n = _drag(
        request=request,
        mode=mode,
        v_tas_mps=v_tas_mps,
        h_m=h_m,
        gamma_rad=gamma_rad,
        t_s=t_s,
        s_m=s_m,
    )
    speed_error_mps = float(hold.target_cas_mps - cas_mps)
    accel_cmd_mps2 = float(
        np.clip(
            controller.speed_kp_mps2_per_mps * speed_error_mps
            + controller.speed_ki_mps2_per_mps_s * hold.speed_integral_mps_s,
            -controller.speed_accel_limit_mps2,
            controller.speed_accel_limit_mps2,
        )
    )
    lower_thrust_n, upper_thrust_n = request.perf.thrust_bounds_newtons(
        mode=mode,
        v_tas_mps=v_tas_mps,
        h_m=h_m,
        delta_isa_K=float(request.weather.delta_isa_K(s_m, h_m, t_s)),
    )
    thrust_n = float(
        np.clip(
            drag_n + request.cfg.mass_kg * (accel_cmd_mps2 + aero.g0 * np.sin(gamma_rad)),
            lower_thrust_n,
            upper_thrust_n,
        )
    )
    ground_speed_mps = _ground_speed_mps(
        request=request,
        s_m=s_m,
        h_m=h_m,
        t_s=t_s,
        v_tas_mps=v_tas_mps,
    )
    phase = "hold" if hold.speed_captured else "hold_decelerate"
    return _StepCommand(
        phase=phase,
        target_cas_mps=hold.target_cas_mps,
        gamma_rad=gamma_rad,
        vertical_speed_mps=vertical_speed_mps,
        thrust_n=thrust_n,
        drag_n=drag_n,
        ground_speed_mps=ground_speed_mps,
        speed_error_mps=float(cas_mps - hold.target_cas_mps),
    )


def _append_sample(
    *,
    command: _StepCommand,
    mode: ModeConfig,
    t_s: float,
    s_m: float,
    start_s_m: float,
    h_m: float,
    v_tas_mps: float,
    cas_mps: float,
    histories: dict[str, list[float | str]],
) -> None:
    histories["t_s"].append(float(t_s))
    histories["s_m"].append(float(s_m))
    histories["distance_flown_m"].append(float(start_s_m - s_m))
    histories["h_m"].append(float(h_m))
    histories["v_tas_mps"].append(float(v_tas_mps))
    histories["v_cas_mps"].append(float(cas_mps))
    histories["target_cas_mps"].append(float(command.target_cas_mps))
    histories["pitch_rad"].append(float(command.gamma_rad))
    histories["gamma_rad"].append(float(command.gamma_rad))
    histories["vertical_speed_mps"].append(float(command.vertical_speed_mps))
    histories["thrust_n"].append(float(command.thrust_n))
    histories["drag_n"].append(float(command.drag_n))
    histories["ground_speed_mps"].append(float(command.ground_speed_mps))
    histories["mode"].append(mode.name)
    histories["speed_error_mps"].append(float(command.speed_error_mps))
    histories["phase"].append(command.phase)


def _result_from_histories(
    *,
    histories: dict[str, list[float | str]],
    success: bool,
    message: str,
) -> FMSResult:
    return FMSResult(
        t_s=np.asarray(histories["t_s"], dtype=float),
        s_m=np.asarray(histories["s_m"], dtype=float),
        distance_flown_m=np.asarray(histories["distance_flown_m"], dtype=float),
        h_m=np.asarray(histories["h_m"], dtype=float),
        v_tas_mps=np.asarray(histories["v_tas_mps"], dtype=float),
        v_cas_mps=np.asarray(histories["v_cas_mps"], dtype=float),
        target_cas_mps=np.asarray(histories["target_cas_mps"], dtype=float),
        pitch_rad=np.asarray(histories["pitch_rad"], dtype=float),
        gamma_rad=np.asarray(histories["gamma_rad"], dtype=float),
        vertical_speed_mps=np.asarray(histories["vertical_speed_mps"], dtype=float),
        thrust_n=np.asarray(histories["thrust_n"], dtype=float),
        drag_n=np.asarray(histories["drag_n"], dtype=float),
        ground_speed_mps=np.asarray(histories["ground_speed_mps"], dtype=float),
        mode=tuple(str(value) for value in histories["mode"]),
        speed_error_mps=np.asarray(histories["speed_error_mps"], dtype=float),
        success=success,
        message=message,
        phase=tuple(str(value) for value in histories["phase"]),
    )


def simulate_hold_aware_fms_descent(request: HoldAwareFMSRequest) -> FMSResult:
    base = request.base_request
    t_s = 0.0
    s_m = float(base.start_s_m)
    h_m = float(base.start_h_m)
    v_tas_mps = _tas_from_cas(
        weather=base.weather,
        s_m=s_m,
        h_m=h_m,
        t_s=t_s,
        v_cas_mps=float(base.start_cas_mps),
    )
    managed_speed_integral_mps_s = 0.0
    previous_mode_name: str | None = None
    next_hold_idx = 0
    active_hold: _HoldRuntime | None = None

    histories: dict[str, list[float | str]] = {
        "t_s": [],
        "s_m": [],
        "distance_flown_m": [],
        "h_m": [],
        "v_tas_mps": [],
        "v_cas_mps": [],
        "target_cas_mps": [],
        "pitch_rad": [],
        "gamma_rad": [],
        "vertical_speed_mps": [],
        "thrust_n": [],
        "drag_n": [],
        "ground_speed_mps": [],
        "mode": [],
        "speed_error_mps": [],
        "phase": [],
    }

    success = False
    message = f"simulation exceeded {base.max_time_s:.1f} s before reaching target altitude"

    while True:
        mode = mode_for_s(base.cfg, s_m)
        if previous_mode_name is not None and mode.name != previous_mode_name:
            managed_speed_integral_mps_s = 0.0
        previous_mode_name = mode.name
        cas_mps = _cas_from_tas(
            weather=base.weather,
            s_m=s_m,
            h_m=h_m,
            t_s=t_s,
            v_tas_mps=v_tas_mps,
        )

        if active_hold is None and next_hold_idx < len(request.holds):
            next_hold = request.holds[next_hold_idx]
            if h_m <= next_hold.holding_altitude_m + 1e-9:
                target_speed = _hold_target_speed_mps(next_hold, captured_cas_mps=cas_mps)
                active_hold = _HoldRuntime(
                    instruction=next_hold,
                    target_h_m=next_hold.holding_altitude_m,
                    target_cas_mps=target_speed,
                    remaining_time_s=float(next_hold.holding_time_s),
                    speed_captured=next_hold.holding_speed_mps is None,
                )
                h_m = active_hold.target_h_m
                managed_speed_integral_mps_s = 0.0

        if active_hold is None:
            command, speed_integral_rate_mps, _raw_vertical_speed_mps = _managed_command(
                request=base,
                mode=mode,
                s_m=s_m,
                h_m=h_m,
                t_s=t_s,
                v_tas_mps=v_tas_mps,
                cas_mps=cas_mps,
                speed_integral_mps_s=managed_speed_integral_mps_s,
            )
        else:
            command = _hold_command(
                request=base,
                hold_request=request,
                hold=active_hold,
                mode=mode,
                s_m=s_m,
                h_m=h_m,
                t_s=t_s,
                v_tas_mps=v_tas_mps,
                cas_mps=cas_mps,
            )
            speed_integral_rate_mps = 0.0

        _append_sample(
            command=command,
            mode=mode,
            t_s=t_s,
            s_m=s_m,
            start_s_m=base.start_s_m,
            h_m=h_m,
            v_tas_mps=v_tas_mps,
            cas_mps=cas_mps,
            histories=histories,
        )

        if active_hold is None and h_m <= base.target_h_m:
            success = True
            message = "reached target altitude"
            break
        if base.stop_at_reference_path_end and s_m <= 0.0:
            message = "reached end of reference path before target altitude"
            break
        if t_s >= base.max_time_s:
            break

        step_dt_s = float(min(base.dt_s, base.max_time_s - t_s))
        if base.stop_at_reference_path_end and command.ground_speed_mps > 1e-9:
            step_dt_s = float(min(step_dt_s, s_m / command.ground_speed_mps))
        if active_hold is None:
            if command.vertical_speed_mps < -1e-9 and h_m + command.vertical_speed_mps * step_dt_s < base.target_h_m:
                step_dt_s = float((base.target_h_m - h_m) / command.vertical_speed_mps)
            if next_hold_idx < len(request.holds):
                hold_h_m = request.holds[next_hold_idx].holding_altitude_m
                if command.vertical_speed_mps < -1e-9 and h_m + command.vertical_speed_mps * step_dt_s < hold_h_m:
                    step_dt_s = float((hold_h_m - h_m) / command.vertical_speed_mps)
        elif active_hold.speed_captured and active_hold.remaining_time_s > 0.0:
            step_dt_s = min(step_dt_s, active_hold.remaining_time_s)
        if step_dt_s <= 0.0:
            break

        if active_hold is None:
            managed_speed_integral_mps_s = float(
                np.clip(
                    managed_speed_integral_mps_s + speed_integral_rate_mps * step_dt_s,
                    -base.controller.integral_limit_mps_s,
                    base.controller.integral_limit_mps_s,
                )
            )
        else:
            altitude_error_m = float(active_hold.target_h_m - h_m)
            active_hold.altitude_integral_m_s = float(
                np.clip(
                    active_hold.altitude_integral_m_s + altitude_error_m * step_dt_s,
                    -request.hold_controller.altitude_integral_limit_m_s,
                    request.hold_controller.altitude_integral_limit_m_s,
                )
            )
            speed_error_for_control_mps = float(active_hold.target_cas_mps - cas_mps)
            active_hold.speed_integral_mps_s = float(
                np.clip(
                    active_hold.speed_integral_mps_s + speed_error_for_control_mps * step_dt_s,
                    -request.hold_controller.speed_integral_limit_mps_s,
                    request.hold_controller.speed_integral_limit_mps_s,
                )
            )
            if not active_hold.speed_captured and abs(speed_error_for_control_mps) <= request.hold_controller.speed_capture_tolerance_mps:
                active_hold.speed_captured = True
            if active_hold.speed_captured:
                active_hold.remaining_time_s = max(0.0, active_hold.remaining_time_s - step_dt_s)

        v_dot_mps2 = float((command.thrust_n - command.drag_n) / base.cfg.mass_kg - aero.g0 * np.sin(command.gamma_rad))
        t_s = float(t_s + step_dt_s)
        s_m = float(s_m - command.ground_speed_mps * step_dt_s)
        h_m = float(h_m + command.vertical_speed_mps * step_dt_s)
        v_tas_mps = float(max(1.0, v_tas_mps + v_dot_mps2 * step_dt_s))

        if active_hold is not None and active_hold.speed_captured and active_hold.remaining_time_s <= 1e-9:
            next_hold_idx += 1
            active_hold = None
            managed_speed_integral_mps_s = 0.0

    return _result_from_histories(histories=histories, success=success, message=message)


def _copy_hold_request(
    request: HoldAwareFMSRequest,
    *,
    start_s_m: float | None = None,
    stop_at_reference_path_end: bool | None = None,
) -> HoldAwareFMSRequest:
    return HoldAwareFMSRequest(
        base_request=_copy_request(
            request.base_request,
            start_s_m=start_s_m,
            stop_at_reference_path_end=stop_at_reference_path_end,
        ),
        holds=request.holds,
        hold_controller=request.hold_controller,
    )


def _tod_metric(result: FMSResult, *, target_h_m: float) -> float:
    if result.success:
        return float(max(result.s_m[-1], 0.0))
    return -float(max(result.h_m[-1] - target_h_m, 0.0))


def _simulate_to_threshold(request: HoldAwareFMSRequest, *, start_s_m: float) -> FMSResult:
    return simulate_hold_aware_fms_descent(
        _copy_hold_request(
            request,
            start_s_m=start_s_m,
            stop_at_reference_path_end=True,
        )
    )


def _find_tod_s_m(
    request: HoldAwareFMSRequest,
    *,
    tolerance_m: float,
    max_iterations: int,
) -> tuple[float | None, FMSResult]:
    available_s_m = float(request.base_request.start_s_m)
    fms_descent = _simulate_to_threshold(request, start_s_m=available_s_m)
    if _tod_metric(fms_descent, target_h_m=request.base_request.target_h_m) < 0.0:
        return None, fms_descent

    low_s_m = 1e-3
    high_s_m = available_s_m
    for _ in range(max_iterations):
        mid_s_m = 0.5 * (low_s_m + high_s_m)
        candidate = _simulate_to_threshold(request, start_s_m=mid_s_m)
        metric = _tod_metric(candidate, target_h_m=request.base_request.target_h_m)
        if candidate.success:
            high_s_m = mid_s_m
        else:
            low_s_m = mid_s_m
        if abs(metric) <= tolerance_m or high_s_m - low_s_m <= tolerance_m:
            break

    tod_s_m = float(high_s_m)
    descent = _simulate_to_threshold(request, start_s_m=tod_s_m)
    for _ in range(8):
        if not descent.success:
            break
        adjusted_tod_s_m = float(descent.distance_flown_m[-1])
        if abs(adjusted_tod_s_m - tod_s_m) <= tolerance_m:
            break
        tod_s_m = adjusted_tod_s_m
        descent = _simulate_to_threshold(request, start_s_m=tod_s_m)
    return tod_s_m, descent


def _with_metadata(
    result: FMSResult,
    *,
    success: bool,
    message: str,
    tod_s_m: float | None,
    level_distance_m: float,
    level_time_s: float,
    descent_segment_distance_m: float | None,
    descent_segment_time_s: float | None,
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
        phase=result.phase,
    )


def _stitch_with_level(level: FMSResult, descent: FMSResult, *, tod_s_m: float) -> FMSResult:
    descent_slice = slice(1, None) if len(level) > 0 and len(descent) > 1 else slice(None)
    t_offset_s = float(level.t_s[-1])
    distance_offset_m = float(level.distance_flown_m[-1])
    return FMSResult(
        t_s=np.concatenate([level.t_s, descent.t_s[descent_slice] + t_offset_s]),
        s_m=np.concatenate([level.s_m, descent.s_m[descent_slice]]),
        distance_flown_m=np.concatenate(
            [level.distance_flown_m, descent.distance_flown_m[descent_slice] + distance_offset_m]
        ),
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
        message="stitched level cruise and hold-aware descent reached target altitude at threshold",
        tod_s_m=tod_s_m,
        level_distance_m=float(level.distance_flown_m[-1]),
        level_time_s=float(level.t_s[-1]),
        descent_segment_distance_m=float(descent.distance_flown_m[-1]),
        descent_segment_time_s=float(descent.t_s[-1]),
        phase=level.phase + tuple(descent.phase[descent_slice]),
    )


def plan_hold_aware_fms_descent(
    request: HoldAwareFMSRequest,
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
        return _with_metadata(
            descent,
            success=False,
            message=(
                "infeasible: not enough along-track distance to complete hold-aware FMS profile before "
                "threshold; showing immediate descent with holds truncated at threshold"
            ),
            tod_s_m=None,
            level_distance_m=0.0,
            level_time_s=0.0,
            descent_segment_distance_m=float(descent.distance_flown_m[-1]),
            descent_segment_time_s=float(descent.t_s[-1]),
        )

    level = _simulate_level_segment(request.base_request, tod_s_m=tod_s_m)
    return _stitch_with_level(level, descent, tod_s_m=tod_s_m)


__all__ = [
    "HoldAwareFMSRequest",
    "HoldControllerConfig",
    "HoldInstruction",
    "plan_hold_aware_fms_descent",
    "simulate_hold_aware_fms_descent",
]
