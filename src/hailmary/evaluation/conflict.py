"""Native continuous-time conflict detection over immutable 4-D tracks.

The hot evaluator consumes dense local-coordinate trajectories directly.  It
does not depend on scenario-manager payloads, compressed JSON, or Pydantic
models.  Each pair of overlapping piecewise-linear segments is solved
analytically for the intersection of a lateral circle and a vertical band.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Iterable, Mapping

import numpy as np


M_PER_NM = 1_852.0
M_PER_FT = 0.3048
DEFAULT_LATERAL_SEPARATION_M = 5.0 * M_PER_NM
DEFAULT_VERTICAL_SEPARATION_M = 1_000.0 * M_PER_FT
_EPS = 1.0e-9


def _readonly_float64(value: Any, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TimedTrajectory:
    """One flight's absolute-time piecewise-linear local trajectory."""

    flight_id: str
    time_s: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    altitude_m: np.ndarray

    def __post_init__(self) -> None:
        if not self.flight_id:
            raise ValueError("flight_id must be non-empty")
        for name in ("time_s", "east_m", "north_m", "altitude_m"):
            object.__setattr__(self, name, _readonly_float64(getattr(self, name), name=name))
        lengths = {len(self.time_s), len(self.east_m), len(self.north_m), len(self.altitude_m)}
        if len(lengths) != 1 or len(self.time_s) < 2:
            raise ValueError("trajectory arrays must have the same length of at least two")
        if np.any(np.diff(self.time_s) <= 0.0):
            raise ValueError("trajectory times must be strictly increasing")

    @property
    def start_time_s(self) -> float:
        return float(self.time_s[0])

    @property
    def end_time_s(self) -> float:
        return float(self.time_s[-1])

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @classmethod
    def from_variant(
        cls,
        variant: object,
        *,
        flight_id: str,
        release_time_s: float,
    ) -> "TimedTrajectory":
        """Adapt a threshold-to-upstream variant to absolute flight-time order."""

        release = float(release_time_s)
        if not math.isfinite(release):
            raise ValueError("release_time_s must be finite")
        elapsed = _variant_array(
            variant,
            ("elapsed_time_s", "relative_elapsed_time_s", "t_s"),
        )
        east = _variant_array(variant, ("east_m",))
        north = _variant_array(variant, ("north_m",))
        altitude = _variant_array(variant, ("altitude_m", "h_m", "geoaltitude_m"))
        lengths = {len(elapsed), len(east), len(north), len(altitude)}
        if len(lengths) != 1 or len(elapsed) < 2:
            raise ValueError("variant time/position arrays must have equal length of at least two")

        delta = np.diff(elapsed)
        if np.all(delta > 0.0):
            order = np.arange(len(elapsed), dtype=np.int64)
        elif np.all(delta < 0.0):
            order = np.arange(len(elapsed) - 1, -1, -1, dtype=np.int64)
        else:
            raise ValueError("variant elapsed time must be strictly monotone")
        ordered_elapsed = elapsed[order]
        absolute_time = release + ordered_elapsed - float(ordered_elapsed[0])
        return cls(
            flight_id=flight_id,
            time_s=absolute_time,
            east_m=east[order],
            north_m=north[order],
            altitude_m=altitude[order],
        )


def timed_trajectory_from_variant(
    variant: object,
    *,
    flight_id: str,
    release_time_s: float,
) -> TimedTrajectory:
    return TimedTrajectory.from_variant(
        variant,
        flight_id=flight_id,
        release_time_s=release_time_s,
    )


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    flight_a_id: str
    flight_b_id: str
    start_time_s: float
    end_time_s: float
    resource_id: str | None = None
    minimum_lateral_separation_m: float | None = None
    minimum_vertical_separation_m: float | None = None

    def __post_init__(self) -> None:
        if not self.flight_a_id or not self.flight_b_id:
            raise ValueError("conflict flight IDs cannot be empty")
        if self.flight_a_id == self.flight_b_id:
            raise ValueError("a flight cannot conflict with itself")
        if self.flight_b_id < self.flight_a_id:
            first, second = self.flight_b_id, self.flight_a_id
            object.__setattr__(self, "flight_a_id", first)
            object.__setattr__(self, "flight_b_id", second)
        if not np.isfinite(self.start_time_s) or not np.isfinite(self.end_time_s):
            raise ValueError("conflict times must be finite")
        if self.end_time_s < self.start_time_s - _EPS:
            raise ValueError("conflict interval is reversed")
        for name in ("minimum_lateral_separation_m", "minimum_vertical_separation_m"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when present")

    @property
    def ordered_pair(self) -> tuple[str, str]:
        return self.flight_a_id, self.flight_b_id

    @property
    def pair_ids(self) -> tuple[str, str]:
        return self.ordered_pair

    @property
    def identity(self) -> tuple[str, str, str | None, float, float]:
        return (
            self.flight_a_id,
            self.flight_b_id,
            self.resource_id,
            float(self.start_time_s),
            float(self.end_time_s),
        )


@dataclass(frozen=True, slots=True)
class ConflictSummary:
    conflicts: tuple[ConflictRecord, ...]
    new_conflicts: tuple[ConflictRecord, ...]

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)

    @property
    def new_conflict_count(self) -> int:
        return len(self.new_conflicts)


@dataclass(frozen=True, slots=True)
class _Segment:
    flight_id: str
    index: int
    t0: float
    t1: float
    x0: float
    y0: float
    z0: float
    vx: float
    vy: float
    vz: float

    def position_at(self, time_s: float) -> tuple[float, float, float]:
        offset = float(time_s - self.t0)
        return (
            self.x0 + self.vx * offset,
            self.y0 + self.vy * offset,
            self.z0 + self.vz * offset,
        )


def _segments(trajectory: TimedTrajectory) -> tuple[_Segment, ...]:
    result: list[_Segment] = []
    for index in range(len(trajectory.time_s) - 1):
        t0 = float(trajectory.time_s[index])
        t1 = float(trajectory.time_s[index + 1])
        duration = t1 - t0
        x0 = float(trajectory.east_m[index])
        y0 = float(trajectory.north_m[index])
        z0 = float(trajectory.altitude_m[index])
        result.append(
            _Segment(
                flight_id=trajectory.flight_id,
                index=index,
                t0=t0,
                t1=t1,
                x0=x0,
                y0=y0,
                z0=z0,
                vx=(float(trajectory.east_m[index + 1]) - x0) / duration,
                vy=(float(trajectory.north_m[index + 1]) - y0) / duration,
                vz=(float(trajectory.altitude_m[index + 1]) - z0) / duration,
            )
        )
    return tuple(result)


def detect_conflicts(
    trajectories: Iterable[TimedTrajectory],
    *,
    lateral_separation_m: float = DEFAULT_LATERAL_SEPARATION_M,
    vertical_separation_m: float = DEFAULT_VERTICAL_SEPARATION_M,
    merge_gap_s: float = 0.0,
) -> tuple[ConflictRecord, ...]:
    """Detect and merge all pairwise continuous-time conflicts."""

    tracks = tuple(sorted(trajectories, key=lambda item: item.flight_id))
    identifiers = [track.flight_id for track in tracks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("trajectory flight IDs must be unique")
    lateral, vertical, gap = _validated_thresholds(
        lateral_separation_m,
        vertical_separation_m,
        merge_gap_s,
    )
    records: list[ConflictRecord] = []
    for first, second in combinations(tracks, 2):
        records.extend(
            detect_pair_conflicts(
                first,
                second,
                lateral_separation_m=lateral,
                vertical_separation_m=vertical,
                merge_gap_s=gap,
            )
        )
    return normalize_conflicts(records)


def detect_pair_conflicts(
    first: TimedTrajectory,
    second: TimedTrajectory,
    *,
    lateral_separation_m: float = DEFAULT_LATERAL_SEPARATION_M,
    vertical_separation_m: float = DEFAULT_VERTICAL_SEPARATION_M,
    merge_gap_s: float = 0.0,
) -> tuple[ConflictRecord, ...]:
    """Solve one trajectory pair without time stepping."""

    if first.flight_id == second.flight_id:
        raise ValueError("pair trajectories must represent different flights")
    lateral, vertical, gap = _validated_thresholds(
        lateral_separation_m,
        vertical_separation_m,
        merge_gap_s,
    )
    left = _segments(first)
    right = _segments(second)
    left_index = 0
    right_index = 0
    hits: list[ConflictRecord] = []
    while left_index < len(left) and right_index < len(right):
        left_segment = left[left_index]
        right_segment = right[right_index]
        overlap_start = max(left_segment.t0, right_segment.t0)
        overlap_end = min(left_segment.t1, right_segment.t1)
        if overlap_start <= overlap_end + _EPS:
            hit = _segment_conflict(
                left_segment,
                right_segment,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                lateral_separation_m=lateral,
                vertical_separation_m=vertical,
            )
            if hit is not None:
                hits.append(hit)

        if left_segment.t1 < right_segment.t1 - _EPS:
            left_index += 1
        elif right_segment.t1 < left_segment.t1 - _EPS:
            right_index += 1
        else:
            left_index += 1
            right_index += 1
    return _merge_adjacent_hits(hits, merge_gap_s=gap)


def _segment_conflict(
    first: _Segment,
    second: _Segment,
    *,
    overlap_start: float,
    overlap_end: float,
    lateral_separation_m: float,
    vertical_separation_m: float,
) -> ConflictRecord | None:
    duration = max(0.0, overlap_end - overlap_start)
    first_start = first.position_at(overlap_start)
    second_start = second.position_at(overlap_start)
    dx = first_start[0] - second_start[0]
    dy = first_start[1] - second_start[1]
    dz = first_start[2] - second_start[2]
    dvx = first.vx - second.vx
    dvy = first.vy - second.vy
    dvz = first.vz - second.vz

    lateral_interval = _quadratic_leq_interval(
        a=dvx * dvx + dvy * dvy,
        b=2.0 * (dx * dvx + dy * dvy),
        c=dx * dx + dy * dy - lateral_separation_m * lateral_separation_m,
        low=0.0,
        high=duration,
    )
    if lateral_interval is None:
        return None
    vertical_interval = _linear_abs_leq_interval(
        start_value=dz,
        slope=dvz,
        threshold=vertical_separation_m,
        low=0.0,
        high=duration,
    )
    if vertical_interval is None:
        return None
    relative_start = max(lateral_interval[0], vertical_interval[0])
    relative_end = min(lateral_interval[1], vertical_interval[1])
    if relative_start > relative_end + _EPS:
        return None

    lateral_minimum = _minimum_lateral_separation(
        dx,
        dy,
        dvx,
        dvy,
        low=relative_start,
        high=relative_end,
    )
    vertical_minimum = _minimum_vertical_separation(
        dz,
        dvz,
        low=relative_start,
        high=relative_end,
    )
    return ConflictRecord(
        flight_a_id=first.flight_id,
        flight_b_id=second.flight_id,
        start_time_s=overlap_start + relative_start,
        end_time_s=overlap_start + relative_end,
        minimum_lateral_separation_m=lateral_minimum,
        minimum_vertical_separation_m=vertical_minimum,
    )


def _quadratic_leq_interval(
    *,
    a: float,
    b: float,
    c: float,
    low: float,
    high: float,
) -> tuple[float, float] | None:
    if a <= _EPS:
        return (low, high) if c <= _EPS else None
    discriminant = b * b - 4.0 * a * c
    if discriminant < -_EPS:
        return None
    root_delta = math.sqrt(max(0.0, discriminant))
    left = (-b - root_delta) / (2.0 * a)
    right = (-b + root_delta) / (2.0 * a)
    start = max(low, left)
    end = min(high, right)
    return (start, end) if start <= end + _EPS else None


def _linear_abs_leq_interval(
    *,
    start_value: float,
    slope: float,
    threshold: float,
    low: float,
    high: float,
) -> tuple[float, float] | None:
    if abs(slope) <= _EPS:
        return (low, high) if abs(start_value) <= threshold + _EPS else None
    first_root = (-threshold - start_value) / slope
    second_root = (threshold - start_value) / slope
    start = max(low, min(first_root, second_root))
    end = min(high, max(first_root, second_root))
    return (start, end) if start <= end + _EPS else None


def _minimum_lateral_separation(
    dx: float,
    dy: float,
    dvx: float,
    dvy: float,
    *,
    low: float,
    high: float,
) -> float:
    denominator = dvx * dvx + dvy * dvy
    time = low if denominator <= _EPS else float(np.clip(-(dx * dvx + dy * dvy) / denominator, low, high))
    return float(math.hypot(dx + dvx * time, dy + dvy * time))


def _minimum_vertical_separation(
    dz: float,
    dvz: float,
    *,
    low: float,
    high: float,
) -> float:
    time = low if abs(dvz) <= _EPS else float(np.clip(-dz / dvz, low, high))
    return float(abs(dz + dvz * time))


def _merge_adjacent_hits(
    records: Iterable[ConflictRecord],
    *,
    merge_gap_s: float,
) -> tuple[ConflictRecord, ...]:
    ordered = sorted(
        records,
        key=lambda item: (
            item.flight_a_id,
            item.flight_b_id,
            item.start_time_s,
            item.end_time_s,
        ),
    )
    merged: list[ConflictRecord] = []
    for record in ordered:
        if (
            not merged
            or record.ordered_pair != merged[-1].ordered_pair
            or record.resource_id != merged[-1].resource_id
            or record.start_time_s > merged[-1].end_time_s + merge_gap_s + _EPS
        ):
            merged.append(record)
            continue
        current = merged[-1]
        merged[-1] = ConflictRecord(
            flight_a_id=current.flight_a_id,
            flight_b_id=current.flight_b_id,
            start_time_s=min(current.start_time_s, record.start_time_s),
            end_time_s=max(current.end_time_s, record.end_time_s),
            resource_id=current.resource_id,
            minimum_lateral_separation_m=_optional_minimum(
                current.minimum_lateral_separation_m,
                record.minimum_lateral_separation_m,
            ),
            minimum_vertical_separation_m=_optional_minimum(
                current.minimum_vertical_separation_m,
                record.minimum_vertical_separation_m,
            ),
        )
    return tuple(merged)


def _optional_minimum(first: float | None, second: float | None) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def normalize_conflicts(records: Iterable[ConflictRecord]) -> tuple[ConflictRecord, ...]:
    unique = {record.identity: record for record in records}
    ordered_keys = sorted(
        unique,
        key=lambda key: (key[0], key[1], "" if key[2] is None else key[2], key[3], key[4]),
    )
    return tuple(unique[key] for key in ordered_keys)


def summarize_conflicts(
    candidate: Iterable[ConflictRecord],
    *,
    baseline: Iterable[ConflictRecord] = (),
) -> ConflictSummary:
    normalized = normalize_conflicts(candidate)
    baseline_ids = {record.identity for record in normalize_conflicts(baseline)}
    new = tuple(record for record in normalized if record.identity not in baseline_ids)
    return ConflictSummary(conflicts=normalized, new_conflicts=new)


def _validated_thresholds(
    lateral_separation_m: float,
    vertical_separation_m: float,
    merge_gap_s: float,
) -> tuple[float, float, float]:
    values = tuple(float(item) for item in (lateral_separation_m, vertical_separation_m, merge_gap_s))
    if not all(math.isfinite(item) for item in values):
        raise ValueError("conflict thresholds and merge gap must be finite")
    if values[0] <= 0.0 or values[1] <= 0.0 or values[2] < 0.0:
        raise ValueError("separation thresholds must be positive and merge gap non-negative")
    return values[0], values[1], values[2]


def _variant_array(variant: object, names: tuple[str, ...]) -> np.ndarray:
    value: Any | None = None
    if isinstance(variant, Mapping):
        for name in names:
            if name in variant:
                value = variant[name]
                break
    else:
        for name in names:
            if hasattr(variant, name):
                value = getattr(variant, name)
                break
    if value is None:
        raise ValueError(f"variant is missing one of arrays {names!r}")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"variant array {names!r} must be finite and one-dimensional")
    return array


__all__ = [
    "DEFAULT_LATERAL_SEPARATION_M",
    "DEFAULT_VERTICAL_SEPARATION_M",
    "ConflictRecord",
    "ConflictSummary",
    "TimedTrajectory",
    "detect_conflicts",
    "detect_pair_conflicts",
    "normalize_conflicts",
    "summarize_conflicts",
    "timed_trajectory_from_variant",
]
