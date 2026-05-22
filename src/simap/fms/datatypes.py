from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from ..backends import PerformanceBackend
from ..config import AircraftConfig, ModeConfig, mode_for_s, planned_cas_bounds_mps
from ..nlp_colloc.coupled import CoupledDescentPlanRequest
from ..path_geometry import ReferencePath
from ..units import fpm_to_mps, ft_to_m, kts_to_mps
from ..weather import ConstantWeather, WeatherProvider


@dataclass(frozen=True)
class FMSSpeedTargets:
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
class ATCSpeedSegment:
    s_from_m: float
    cas_mps: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.s_from_m):
            raise ValueError("ATCSpeedSegment.s_from_m must be finite")
        if not np.isfinite(self.cas_mps) or self.cas_mps <= 0.0:
            raise ValueError(
                f"ATCSpeedSegment at s_from_m={self.s_from_m:.3f} has invalid cas_mps={self.cas_mps:.3f}"
            )


ATCSpeedSegmentInput = ATCSpeedSegment | tuple[float, float]


def _coerce_atc_speed_segment(segment: ATCSpeedSegmentInput) -> ATCSpeedSegment:
    if isinstance(segment, ATCSpeedSegment):
        return segment
    try:
        s_from_m, cas_mps = segment
    except (TypeError, ValueError) as exc:
        raise ValueError("ATC speed segments must be ATCSpeedSegment instances or (s_from_m, cas_mps) pairs") from exc
    return ATCSpeedSegment(s_from_m=float(s_from_m), cas_mps=float(cas_mps))


@dataclass(frozen=True)
class FMSPIConfig:
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
class FMSRequest:
    cfg: AircraftConfig
    perf: PerformanceBackend
    reference_path: ReferencePath
    start_s_m: float
    start_h_m: float
    start_cas_mps: float
    target_h_m: float
    speed_targets: FMSSpeedTargets
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    controller: FMSPIConfig = field(default_factory=FMSPIConfig)
    dt_s: float = 0.5
    max_time_s: float = 7_200.0
    stop_at_reference_path_end: bool = False
    atc_speed_segments: tuple[ATCSpeedSegmentInput, ...] = field(default_factory=tuple)
    atc_speed_reference_start_s_m: float | None = None

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
        atc_speed_reference_start_s_m = float(
            self.start_s_m if self.atc_speed_reference_start_s_m is None else self.atc_speed_reference_start_s_m
        )
        if not np.isfinite(atc_speed_reference_start_s_m):
            raise ValueError("atc_speed_reference_start_s_m must be finite")
        if atc_speed_reference_start_s_m < self.start_s_m - 1e-9:
            raise ValueError("atc_speed_reference_start_s_m must be greater than or equal to start_s_m")
        if atc_speed_reference_start_s_m > self.reference_path.total_length_m + 1e-9:
            raise ValueError("atc_speed_reference_start_s_m must lie on the reference path")
        object.__setattr__(self, "atc_speed_reference_start_s_m", atc_speed_reference_start_s_m)
        segments = tuple(_coerce_atc_speed_segment(segment) for segment in self.atc_speed_segments)
        object.__setattr__(self, "atc_speed_segments", segments)
        self._validate_atc_speed_segments()

    def _base_target_cas_mps(self, s_m: float) -> float:
        return float(self.speed_targets.for_mode(mode_for_s(self.cfg, s_m), h_m=None))

    def _validate_atc_speed_segments(self) -> None:
        seen_s_from: set[float] = set()
        reference_start_s_m = float(
            self.start_s_m if self.atc_speed_reference_start_s_m is None else self.atc_speed_reference_start_s_m
        )
        for segment in cast(tuple[ATCSpeedSegment, ...], self.atc_speed_segments):
            s_from_m = float(segment.s_from_m)
            cas_mps = float(segment.cas_mps)
            if s_from_m in seen_s_from:
                raise ValueError(
                    f"ATC speed segment at s_from_m={s_from_m:.3f}, cas_mps={cas_mps:.3f} duplicates s_from_m"
                )
            seen_s_from.add(s_from_m)
            if s_from_m < 0.0 or s_from_m > reference_start_s_m:
                raise ValueError(
                    f"ATC speed segment at s_from_m={s_from_m:.3f}, cas_mps={cas_mps:.3f} "
                    "must lie within [0, atc_speed_reference_start_s_m]"
                )

            base_at_acceptance = self._base_target_cas_mps(s_from_m)
            if cas_mps >= base_at_acceptance - 1e-9:
                raise ValueError(
                    f"ATC speed segment at s_from_m={s_from_m:.3f}, cas_mps={cas_mps:.3f} "
                    f"is not lower than base profile cas_mps={base_at_acceptance:.3f}"
                )

            validation_s_m = {0.0, s_from_m}
            for gate_m in (self.cfg.final_gate_m, self.cfg.approach_gate_m):
                if 0.0 <= gate_m <= s_from_m:
                    validation_s_m.add(float(gate_m))
                    validation_s_m.add(float(min(s_from_m, np.nextafter(gate_m, np.inf))))
                    validation_s_m.add(float(max(0.0, np.nextafter(gate_m, -np.inf))))

            for sample_s_m in validation_s_m:
                if sample_s_m > s_from_m:
                    continue
                base_target_mps = self._base_target_cas_mps(sample_s_m)
                if base_target_mps <= cas_mps + 1e-9:
                    continue
                lower_mps, upper_mps = planned_cas_bounds_mps(self.cfg, sample_s_m)
                if cas_mps < lower_mps - 1e-9:
                    raise ValueError(
                        f"ATC speed segment at s_from_m={s_from_m:.3f}, cas_mps={cas_mps:.3f} "
                        f"is below planned lower CAS bound {lower_mps:.3f} at s_m={sample_s_m:.3f}"
                    )
                if cas_mps > upper_mps + 1e-9:
                    raise ValueError(
                        f"ATC speed segment at s_from_m={s_from_m:.3f}, cas_mps={cas_mps:.3f} "
                        f"is above planned upper CAS bound {upper_mps:.3f} at s_m={sample_s_m:.3f}"
                    )

    @classmethod
    def from_coupled_request(
        cls,
        request: CoupledDescentPlanRequest,
        *,
        speed_targets: FMSSpeedTargets | None = None,
        start_s_m: float | None = None,
        dt_s: float = 0.5,
        max_time_s: float = 7_200.0,
        controller: FMSPIConfig | None = None,
        atc_speed_segments: tuple[ATCSpeedSegmentInput, ...] = (),
        atc_speed_reference_start_s_m: float | None = None,
    ) -> "FMSRequest":
        from .helpers import infer_fms_speed_targets

        return cls(
            cfg=request.cfg,
            perf=request.perf,
            reference_path=request.reference_path,
            start_s_m=float(request.reference_path.total_length_m if start_s_m is None else start_s_m),
            start_h_m=float(request.upstream.h_m),
            start_cas_mps=float(request.upstream.cas_upper_mps),
            target_h_m=float(request.threshold.h_m),
            speed_targets=infer_fms_speed_targets(request) if speed_targets is None else speed_targets,
            weather=request.weather,
            controller=FMSPIConfig() if controller is None else controller,
            dt_s=dt_s,
            max_time_s=max_time_s,
            atc_speed_segments=atc_speed_segments,
            atc_speed_reference_start_s_m=atc_speed_reference_start_s_m,
        )


@dataclass(frozen=True)
class FMSResult:
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


__all__ = [
    "ATCSpeedSegment",
    "ATCSpeedSegmentInput",
    "FMSPIConfig",
    "FMSRequest",
    "FMSResult",
    "FMSSpeedTargets",
]
