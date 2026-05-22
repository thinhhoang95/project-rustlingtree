from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
from typing import Any, Iterable, Protocol

import numpy as np

CONFLICT_ENVELOPE_RADIUS_NM = 5.0
CONFLICT_ENVELOPE_HEIGHT_FL = 100.0

_METERS_PER_NM = 1_852.0
_METERS_PER_FT = 0.3048
_FEET_PER_CONFLICT_HEIGHT_FL = 10.0
_CONFLICT_ENVELOPE_HEIGHT_FT = CONFLICT_ENVELOPE_HEIGHT_FL * _FEET_PER_CONFLICT_HEIGHT_FL
_DEFAULT_MERGE_GAP_S = 5.0
_EPS = 1.0e-9


class ArrivalScheduleProvider(Protocol):
    def arrival_schedule(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ConflictFlight:
    flight_number: str
    icao24: str
    flight_id: str
    runway: str


@dataclass(frozen=True)
class ConflictEvent:
    flight_a: ConflictFlight
    flight_b: ConflictFlight
    start_time: int
    end_time: int
    closest_time: int
    closest_time_utc: str
    latitude: float
    longitude: float
    lateral_distance_nmi: float
    vertical_separation_ft: float
    lateral_threshold_nmi: float
    vertical_threshold_ft: float
    severity: float
    confidence: str


@dataclass(frozen=True)
class ConflictEvaluator:
    manager: ArrivalScheduleProvider

    def evaluate(self) -> list[ConflictEvent]:
        arrivals = self.manager.arrival_schedule()
        if len(arrivals) < 2:
            return []

        projection = _LocalProjection.from_arrivals(arrivals)
        trajectories = [_Trajectory.from_arrival(arrival, projection) for arrival in arrivals]
        segments = _segments_from_trajectories(trajectories)
        hits = _find_conflict_hits(segments, projection)
        return _merge_hits(hits)


@dataclass(frozen=True)
class _LocalProjection:
    lat0_rad: float
    lon0_rad: float

    @classmethod
    def from_arrivals(cls, arrivals: Iterable[dict[str, Any]]) -> "_LocalProjection":
        latitudes: list[float] = []
        longitudes: list[float] = []
        for arrival in arrivals:
            columns = _columns(arrival)
            lat_index = _column_index(columns, "lat", _flight_label(arrival))
            lon_index = _column_index(columns, "lon", _flight_label(arrival))
            points = _points(arrival)
            for row_index, point in enumerate(points):
                _validate_point_shape(point, max(lat_index, lon_index) + 1, row_index, arrival)
                latitudes.append(_finite_number(point[lat_index], f"points[{row_index}].lat", arrival))
                longitudes.append(_finite_number(point[lon_index], f"points[{row_index}].lon", arrival))

        if not latitudes:
            raise ValueError("arrival trajectories contain no points")

        lat0_rad = math.radians(math.fsum(latitudes) / len(latitudes))
        lon0_rad = math.radians(math.fsum(longitudes) / len(longitudes))
        return cls(lat0_rad=lat0_rad, lon0_rad=lon0_rad)

    def project(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        earth_radius_m = 6_371_000.0
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)
        x_m = earth_radius_m * (lon_rad - self.lon0_rad) * math.cos(self.lat0_rad)
        y_m = earth_radius_m * (lat_rad - self.lat0_rad)
        return x_m, y_m

    def inverse(self, x_m: float, y_m: float) -> tuple[float, float]:
        earth_radius_m = 6_371_000.0
        lat_rad = y_m / earth_radius_m + self.lat0_rad
        lon_rad = x_m / (earth_radius_m * math.cos(self.lat0_rad)) + self.lon0_rad
        return math.degrees(lat_rad), math.degrees(lon_rad)


@dataclass(frozen=True)
class _Trajectory:
    flight: ConflictFlight
    times: tuple[float, ...]
    xs_m: tuple[float, ...]
    ys_m: tuple[float, ...]
    zs_m: tuple[float, ...]
    lateral_tolerance_m: float
    altitude_tolerance_m: float

    @classmethod
    def from_arrival(cls, arrival: dict[str, Any], projection: _LocalProjection) -> "_Trajectory":
        label = _flight_label(arrival)
        columns = _columns(arrival)
        time_index = _column_index(columns, "time", label)
        lat_index = _column_index(columns, "lat", label)
        lon_index = _column_index(columns, "lon", label)
        altitude_index = _column_index(columns, "geoaltitude_m", label)
        points = _points(arrival)
        if len(points) < 2:
            raise ValueError(f"{label} must contain at least two trajectory points")

        times: list[float] = []
        xs_m: list[float] = []
        ys_m: list[float] = []
        zs_m: list[float] = []
        for row_index, point in enumerate(points):
            minimum_length = max(time_index, lat_index, lon_index, altitude_index) + 1
            _validate_point_shape(point, minimum_length, row_index, arrival)

            time_s = _finite_number(point[time_index], f"points[{row_index}].time", arrival)
            lat_deg = _finite_number(point[lat_index], f"points[{row_index}].lat", arrival)
            lon_deg = _finite_number(point[lon_index], f"points[{row_index}].lon", arrival)
            altitude_m = _finite_number(point[altitude_index], f"points[{row_index}].geoaltitude_m", arrival)
            x_m, y_m = projection.project(lat_deg, lon_deg)
            times.append(time_s)
            xs_m.append(x_m)
            ys_m.append(y_m)
            zs_m.append(altitude_m)

        times, xs_m, ys_m, zs_m, lateral_duplicate_tolerance_m, altitude_duplicate_tolerance_m = (
            _collapse_duplicate_times(times, xs_m, ys_m, zs_m)
        )
        if len(times) < 2:
            raise ValueError(f"{label} must contain at least two unique trajectory times")

        for previous, current in zip(times, times[1:]):
            if current <= previous:
                raise ValueError(f"{label} trajectory times must be strictly increasing")

        flight = ConflictFlight(
            flight_number=str(arrival.get("callsign", "")),
            icao24=str(arrival.get("icao24", "")),
            flight_id=str(arrival.get("flight_id", "")),
            runway=str(arrival.get("runway", "")),
        )
        return cls(
            flight=flight,
            times=tuple(times),
            xs_m=tuple(xs_m),
            ys_m=tuple(ys_m),
            zs_m=tuple(zs_m),
            lateral_tolerance_m=_optional_nonnegative_number(
                arrival.get("lateral_tolerance_m"),
                "lateral_tolerance_m",
                arrival,
            )
            + lateral_duplicate_tolerance_m,
            altitude_tolerance_m=_optional_nonnegative_number(
                arrival.get("altitude_tolerance_m"),
                "altitude_tolerance_m",
                arrival,
            )
            + altitude_duplicate_tolerance_m,
        )


@dataclass(frozen=True)
class _Segment:
    id: int
    trajectory: _Trajectory
    index: int
    t0: float
    t1: float
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @classmethod
    def from_trajectory(cls, segment_id: int, trajectory: _Trajectory, index: int) -> "_Segment":
        x0 = trajectory.xs_m[index]
        x1 = trajectory.xs_m[index + 1]
        y0 = trajectory.ys_m[index]
        y1 = trajectory.ys_m[index + 1]
        z0 = trajectory.zs_m[index]
        z1 = trajectory.zs_m[index + 1]
        return cls(
            id=segment_id,
            trajectory=trajectory,
            index=index,
            t0=trajectory.times[index],
            t1=trajectory.times[index + 1],
            x0=x0,
            y0=y0,
            z0=z0,
            x1=x1,
            y1=y1,
            z1=z1,
            min_x=min(x0, x1),
            max_x=max(x0, x1),
            min_y=min(y0, y1),
            max_y=max(y0, y1),
            min_z=min(z0, z1),
            max_z=max(z0, z1),
        )

    @property
    def vx(self) -> float:
        return (self.x1 - self.x0) / (self.t1 - self.t0)

    @property
    def vy(self) -> float:
        return (self.y1 - self.y0) / (self.t1 - self.t0)

    @property
    def vz(self) -> float:
        return (self.z1 - self.z0) / (self.t1 - self.t0)

    def position_at(self, time_s: float) -> tuple[float, float, float]:
        fraction = (time_s - self.t0) / (self.t1 - self.t0)
        x_m = self.x0 + (self.x1 - self.x0) * fraction
        y_m = self.y0 + (self.y1 - self.y0) * fraction
        z_m = self.z0 + (self.z1 - self.z0) * fraction
        return x_m, y_m, z_m


@dataclass(frozen=True)
class _ConflictHit:
    flight_a: ConflictFlight
    flight_b: ConflictFlight
    start_time: float
    end_time: float
    closest_time: float
    latitude: float
    longitude: float
    lateral_distance_m: float
    vertical_separation_m: float
    severity: float
    confidence: str

    @property
    def pair_key(self) -> tuple[str, str]:
        return self.flight_a.flight_id, self.flight_b.flight_id


def _segments_from_trajectories(trajectories: list[_Trajectory]) -> list[_Segment]:
    segments: list[_Segment] = []
    segment_id = 0
    for trajectory in trajectories:
        for index in range(len(trajectory.times) - 1):
            segments.append(_Segment.from_trajectory(segment_id, trajectory, index))
            segment_id += 1
    return segments


def _collapse_duplicate_times(
    times: list[float],
    xs_m: list[float],
    ys_m: list[float],
    zs_m: list[float],
) -> tuple[list[float], list[float], list[float], list[float], float, float]:
    collapsed_times: list[float] = []
    collapsed_xs_m: list[float] = []
    collapsed_ys_m: list[float] = []
    collapsed_zs_m: list[float] = []
    lateral_tolerance_m = 0.0
    altitude_tolerance_m = 0.0
    index = 0
    while index < len(times):
        group_end = index + 1
        while group_end < len(times) and times[group_end] == times[index]:
            group_end += 1

        representative = group_end - 1
        rep_x = xs_m[representative]
        rep_y = ys_m[representative]
        rep_z = zs_m[representative]
        for duplicate_index in range(index, group_end):
            lateral_tolerance_m = max(
                lateral_tolerance_m,
                math.hypot(xs_m[duplicate_index] - rep_x, ys_m[duplicate_index] - rep_y),
            )
            altitude_tolerance_m = max(altitude_tolerance_m, abs(zs_m[duplicate_index] - rep_z))

        collapsed_times.append(times[representative])
        collapsed_xs_m.append(rep_x)
        collapsed_ys_m.append(rep_y)
        collapsed_zs_m.append(rep_z)
        index = group_end

    return (
        collapsed_times,
        collapsed_xs_m,
        collapsed_ys_m,
        collapsed_zs_m,
        lateral_tolerance_m,
        altitude_tolerance_m,
    )


def _find_conflict_hits(segments: list[_Segment], projection: _LocalProjection) -> list[_ConflictHit]:
    hits: list[_ConflictHit] = []
    if not segments:
        return hits

    lateral_threshold_m = 2.0 * CONFLICT_ENVELOPE_RADIUS_NM * _METERS_PER_NM
    vertical_threshold_m = _CONFLICT_ENVELOPE_HEIGHT_FT * _METERS_PER_FT
    arrays = _SegmentArrays.from_segments(segments)
    left_indices, right_indices = _candidate_pair_indices(
        arrays,
        lateral_threshold_m=lateral_threshold_m,
        vertical_threshold_m=vertical_threshold_m,
    )
    if len(left_indices) == 0:
        return hits

    confirmed_mask = _segment_pair_conflict_mask(
        arrays,
        left_indices,
        right_indices,
        np.full(len(left_indices), lateral_threshold_m, dtype=float),
        np.full(len(left_indices), vertical_threshold_m, dtype=float),
    )
    confirmed_positions = np.flatnonzero(confirmed_mask)
    possible_positions = _possible_conflict_positions(
        arrays,
        left_indices,
        right_indices,
        confirmed_mask,
        lateral_threshold_m=lateral_threshold_m,
        vertical_threshold_m=vertical_threshold_m,
    )

    for pair_position in confirmed_positions:
        hit = _hit_for_pair_position(
            segments,
            projection,
            left_indices,
            right_indices,
            int(pair_position),
            lateral_threshold_m,
            vertical_threshold_m,
            confidence="confirmed",
        )
        if hit is not None:
            hits.append(hit)

    for pair_position in possible_positions:
        left_index = int(left_indices[int(pair_position)])
        right_index = int(right_indices[int(pair_position)])
        expanded_lateral_m = (
            lateral_threshold_m
            + arrays.lateral_tolerance_m[left_index]
            + arrays.lateral_tolerance_m[right_index]
        )
        expanded_vertical_m = (
            vertical_threshold_m
            + arrays.altitude_tolerance_m[left_index]
            + arrays.altitude_tolerance_m[right_index]
        )
        hit = _hit_for_pair_position(
            segments,
            projection,
            left_indices,
            right_indices,
            int(pair_position),
            float(expanded_lateral_m),
            float(expanded_vertical_m),
            confidence="possible",
        )
        if hit is not None:
            hits.append(hit)

    return hits


@dataclass(frozen=True)
class _SegmentArrays:
    t0: np.ndarray
    t1: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    z0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    z1: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    min_x: np.ndarray
    max_x: np.ndarray
    min_y: np.ndarray
    max_y: np.ndarray
    min_z: np.ndarray
    max_z: np.ndarray
    flight_index: np.ndarray
    lateral_tolerance_m: np.ndarray
    altitude_tolerance_m: np.ndarray
    order_by_start_time: np.ndarray

    @classmethod
    def from_segments(cls, segments: list[_Segment]) -> "_SegmentArrays":
        flight_indices: dict[str, int] = {}
        segment_flight_indices: list[int] = []
        for segment in segments:
            flight_id = segment.trajectory.flight.flight_id
            if flight_id not in flight_indices:
                flight_indices[flight_id] = len(flight_indices)
            segment_flight_indices.append(flight_indices[flight_id])

        t0 = np.asarray([segment.t0 for segment in segments], dtype=float)
        t1 = np.asarray([segment.t1 for segment in segments], dtype=float)
        x0 = np.asarray([segment.x0 for segment in segments], dtype=float)
        y0 = np.asarray([segment.y0 for segment in segments], dtype=float)
        z0 = np.asarray([segment.z0 for segment in segments], dtype=float)
        x1 = np.asarray([segment.x1 for segment in segments], dtype=float)
        y1 = np.asarray([segment.y1 for segment in segments], dtype=float)
        z1 = np.asarray([segment.z1 for segment in segments], dtype=float)
        duration_s = t1 - t0
        return cls(
            t0=t0,
            t1=t1,
            x0=x0,
            y0=y0,
            z0=z0,
            x1=x1,
            y1=y1,
            z1=z1,
            vx=(x1 - x0) / duration_s,
            vy=(y1 - y0) / duration_s,
            vz=(z1 - z0) / duration_s,
            min_x=np.minimum(x0, x1),
            max_x=np.maximum(x0, x1),
            min_y=np.minimum(y0, y1),
            max_y=np.maximum(y0, y1),
            min_z=np.minimum(z0, z1),
            max_z=np.maximum(z0, z1),
            flight_index=np.asarray(segment_flight_indices, dtype=np.int64),
            lateral_tolerance_m=np.asarray(
                [segment.trajectory.lateral_tolerance_m for segment in segments],
                dtype=float,
            ),
            altitude_tolerance_m=np.asarray(
                [segment.trajectory.altitude_tolerance_m for segment in segments],
                dtype=float,
            ),
            order_by_start_time=np.argsort(t0, kind="stable"),
        )


def _candidate_pair_indices(
    arrays: _SegmentArrays,
    *,
    lateral_threshold_m: float,
    vertical_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    ordered_t0 = arrays.t0[arrays.order_by_start_time]
    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []

    for order_position, left_index in enumerate(arrays.order_by_start_time):
        end_position = int(np.searchsorted(ordered_t0, arrays.t1[left_index], side="right"))
        if end_position <= order_position + 1:
            continue

        right_indices = arrays.order_by_start_time[order_position + 1 : end_position]
        expanded_lateral_m = lateral_threshold_m + arrays.lateral_tolerance_m[left_index] + arrays.lateral_tolerance_m[
            right_indices
        ]
        expanded_vertical_m = (
            vertical_threshold_m
            + arrays.altitude_tolerance_m[left_index]
            + arrays.altitude_tolerance_m[right_indices]
        )
        overlap_mask = (
            (arrays.flight_index[right_indices] != arrays.flight_index[left_index])
            & (arrays.max_x[left_index] + expanded_lateral_m >= arrays.min_x[right_indices])
            & (arrays.max_x[right_indices] + expanded_lateral_m >= arrays.min_x[left_index])
            & (arrays.max_y[left_index] + expanded_lateral_m >= arrays.min_y[right_indices])
            & (arrays.max_y[right_indices] + expanded_lateral_m >= arrays.min_y[left_index])
            & (arrays.max_z[left_index] + expanded_vertical_m >= arrays.min_z[right_indices])
            & (arrays.max_z[right_indices] + expanded_vertical_m >= arrays.min_z[left_index])
        )
        matched_right_indices = right_indices[overlap_mask]
        if len(matched_right_indices) == 0:
            continue

        left_chunks.append(np.full(len(matched_right_indices), left_index, dtype=np.int64))
        right_chunks.append(matched_right_indices.astype(np.int64, copy=False))

    if not left_chunks:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty

    return np.concatenate(left_chunks), np.concatenate(right_chunks)


def _segment_pair_conflict_mask(
    arrays: _SegmentArrays,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    lateral_thresholds_m: np.ndarray,
    vertical_thresholds_m: np.ndarray,
) -> np.ndarray:
    if len(left_indices) == 0:
        return np.asarray([], dtype=bool)

    overlap_start = np.maximum(arrays.t0[left_indices], arrays.t0[right_indices])
    overlap_end = np.minimum(arrays.t1[left_indices], arrays.t1[right_indices])
    duration_s = overlap_end - overlap_start

    first_x = arrays.x0[left_indices] + arrays.vx[left_indices] * (overlap_start - arrays.t0[left_indices])
    first_y = arrays.y0[left_indices] + arrays.vy[left_indices] * (overlap_start - arrays.t0[left_indices])
    first_z = arrays.z0[left_indices] + arrays.vz[left_indices] * (overlap_start - arrays.t0[left_indices])
    second_x = arrays.x0[right_indices] + arrays.vx[right_indices] * (overlap_start - arrays.t0[right_indices])
    second_y = arrays.y0[right_indices] + arrays.vy[right_indices] * (overlap_start - arrays.t0[right_indices])
    second_z = arrays.z0[right_indices] + arrays.vz[right_indices] * (overlap_start - arrays.t0[right_indices])

    dx0 = first_x - second_x
    dy0 = first_y - second_y
    dz0 = first_z - second_z
    dvx = arrays.vx[left_indices] - arrays.vx[right_indices]
    dvy = arrays.vy[left_indices] - arrays.vy[right_indices]
    dvz = arrays.vz[left_indices] - arrays.vz[right_indices]

    lateral_start, lateral_end, lateral_ok = _quadratic_leq_intervals(
        a=dvx * dvx + dvy * dvy,
        b=2.0 * (dx0 * dvx + dy0 * dvy),
        c=dx0 * dx0 + dy0 * dy0 - lateral_thresholds_m * lateral_thresholds_m,
        high=duration_s,
    )
    vertical_start, vertical_end, vertical_ok = _linear_abs_leq_intervals(
        start_value=dz0,
        slope=dvz,
        threshold=vertical_thresholds_m,
        high=duration_s,
    )

    interval_start = np.maximum(lateral_start, vertical_start)
    interval_end = np.minimum(lateral_end, vertical_end)
    return lateral_ok & vertical_ok & (interval_start <= interval_end + _EPS)


def _quadratic_leq_intervals(
    *,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interval_start = np.zeros_like(high, dtype=float)
    interval_end = high.copy()
    ok = np.zeros_like(high, dtype=bool)

    flat_mask = a <= _EPS
    ok[flat_mask] = c[flat_mask] <= _EPS

    curved_indices = np.flatnonzero(~flat_mask)
    if len(curved_indices) == 0:
        return interval_start, interval_end, ok

    discriminant = b[curved_indices] * b[curved_indices] - 4.0 * a[curved_indices] * c[curved_indices]
    good_discriminant_mask = discriminant >= -_EPS
    good_indices = curved_indices[good_discriminant_mask]
    if len(good_indices) == 0:
        return interval_start, interval_end, ok

    root_delta = np.sqrt(np.maximum(0.0, discriminant[good_discriminant_mask]))
    left_root = (-b[good_indices] - root_delta) / (2.0 * a[good_indices])
    right_root = (-b[good_indices] + root_delta) / (2.0 * a[good_indices])
    interval_start[good_indices] = np.maximum(0.0, left_root)
    interval_end[good_indices] = np.minimum(high[good_indices], right_root)
    ok[good_indices] = interval_start[good_indices] <= interval_end[good_indices] + _EPS
    return interval_start, interval_end, ok


def _linear_abs_leq_intervals(
    *,
    start_value: np.ndarray,
    slope: np.ndarray,
    threshold: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interval_start = np.zeros_like(high, dtype=float)
    interval_end = high.copy()
    ok = np.zeros_like(high, dtype=bool)

    flat_mask = np.abs(slope) <= _EPS
    ok[flat_mask] = np.abs(start_value[flat_mask]) <= threshold[flat_mask] + _EPS

    linear_indices = np.flatnonzero(~flat_mask)
    if len(linear_indices) == 0:
        return interval_start, interval_end, ok

    first_root = (-threshold[linear_indices] - start_value[linear_indices]) / slope[linear_indices]
    second_root = (threshold[linear_indices] - start_value[linear_indices]) / slope[linear_indices]
    left_root = np.minimum(first_root, second_root)
    right_root = np.maximum(first_root, second_root)
    interval_start[linear_indices] = np.maximum(0.0, left_root)
    interval_end[linear_indices] = np.minimum(high[linear_indices], right_root)
    ok[linear_indices] = interval_start[linear_indices] <= interval_end[linear_indices] + _EPS
    return interval_start, interval_end, ok


def _possible_conflict_positions(
    arrays: _SegmentArrays,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    confirmed_mask: np.ndarray,
    *,
    lateral_threshold_m: float,
    vertical_threshold_m: float,
) -> np.ndarray:
    unconfirmed_positions = np.flatnonzero(~confirmed_mask)
    if len(unconfirmed_positions) == 0:
        return np.asarray([], dtype=np.int64)

    unconfirmed_left = left_indices[unconfirmed_positions]
    unconfirmed_right = right_indices[unconfirmed_positions]
    expanded_lateral_m = (
        lateral_threshold_m
        + arrays.lateral_tolerance_m[unconfirmed_left]
        + arrays.lateral_tolerance_m[unconfirmed_right]
    )
    expanded_vertical_m = (
        vertical_threshold_m
        + arrays.altitude_tolerance_m[unconfirmed_left]
        + arrays.altitude_tolerance_m[unconfirmed_right]
    )
    expandable_mask = (expanded_lateral_m > lateral_threshold_m) | (expanded_vertical_m > vertical_threshold_m)
    if not bool(np.any(expandable_mask)):
        return np.asarray([], dtype=np.int64)

    expandable_positions = unconfirmed_positions[expandable_mask]
    possible_mask = _segment_pair_conflict_mask(
        arrays,
        left_indices[expandable_positions],
        right_indices[expandable_positions],
        expanded_lateral_m[expandable_mask],
        expanded_vertical_m[expandable_mask],
    )
    return expandable_positions[np.flatnonzero(possible_mask)]


def _hit_for_pair_position(
    segments: list[_Segment],
    projection: _LocalProjection,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    pair_position: int,
    lateral_threshold_m: float,
    vertical_threshold_m: float,
    *,
    confidence: str,
) -> _ConflictHit | None:
    first = segments[int(left_indices[pair_position])]
    second = segments[int(right_indices[pair_position])]
    return _segment_pair_hit(
        first,
        second,
        projection,
        max(first.t0, second.t0),
        min(first.t1, second.t1),
        lateral_threshold_m,
        vertical_threshold_m,
        confidence=confidence,
    )


def _segment_pair_hit(
    first: _Segment,
    second: _Segment,
    projection: _LocalProjection,
    overlap_start: float,
    overlap_end: float,
    lateral_threshold_m: float,
    vertical_threshold_m: float,
    *,
    confidence: str,
) -> _ConflictHit | None:
    interval = _conflict_interval(
        first,
        second,
        overlap_start,
        overlap_end,
        lateral_threshold_m,
        vertical_threshold_m,
    )
    if interval is None:
        return None

    event_start, event_end = interval
    closest_time = _closest_time(first, second, event_start, event_end)
    first_position = first.position_at(closest_time)
    second_position = second.position_at(closest_time)
    lateral_distance_m = math.hypot(
        first_position[0] - second_position[0],
        first_position[1] - second_position[1],
    )
    vertical_separation_m = abs(first_position[2] - second_position[2])
    midpoint_x_m = 0.5 * (first_position[0] + second_position[0])
    midpoint_y_m = 0.5 * (first_position[1] + second_position[1])
    latitude, longitude = projection.inverse(midpoint_x_m, midpoint_y_m)

    flight_a, flight_b = _ordered_flights(first.trajectory.flight, second.trajectory.flight)
    return _ConflictHit(
        flight_a=flight_a,
        flight_b=flight_b,
        start_time=event_start,
        end_time=event_end,
        closest_time=closest_time,
        latitude=latitude,
        longitude=longitude,
        lateral_distance_m=lateral_distance_m,
        vertical_separation_m=vertical_separation_m,
        severity=_severity(lateral_distance_m, vertical_separation_m),
        confidence=confidence,
    )


def _conflict_interval(
    first: _Segment,
    second: _Segment,
    overlap_start: float,
    overlap_end: float,
    lateral_threshold_m: float,
    vertical_threshold_m: float,
) -> tuple[float, float] | None:
    duration_s = overlap_end - overlap_start
    first_start = first.position_at(overlap_start)
    second_start = second.position_at(overlap_start)
    dx0 = first_start[0] - second_start[0]
    dy0 = first_start[1] - second_start[1]
    dz0 = first_start[2] - second_start[2]
    dvx = first.vx - second.vx
    dvy = first.vy - second.vy
    dvz = first.vz - second.vz

    lateral_interval = _quadratic_leq_interval(
        a=dvx * dvx + dvy * dvy,
        b=2.0 * (dx0 * dvx + dy0 * dvy),
        c=dx0 * dx0 + dy0 * dy0 - lateral_threshold_m * lateral_threshold_m,
        low=0.0,
        high=duration_s,
    )
    if lateral_interval is None:
        return None

    vertical_interval = _linear_abs_leq_interval(
        start_value=dz0,
        slope=dvz,
        threshold=vertical_threshold_m,
        low=0.0,
        high=duration_s,
    )
    if vertical_interval is None:
        return None

    interval_start = max(lateral_interval[0], vertical_interval[0])
    interval_end = min(lateral_interval[1], vertical_interval[1])
    if interval_start > interval_end + _EPS:
        return None
    return overlap_start + interval_start, overlap_start + interval_end


def _quadratic_leq_interval(
    *,
    a: float,
    b: float,
    c: float,
    low: float,
    high: float,
) -> tuple[float, float] | None:
    if a <= _EPS:
        if c <= _EPS:
            return low, high
        return None

    discriminant = b * b - 4.0 * a * c
    if discriminant < -_EPS:
        return None

    root_delta = math.sqrt(max(0.0, discriminant))
    left = (-b - root_delta) / (2.0 * a)
    right = (-b + root_delta) / (2.0 * a)
    interval_start = max(low, left)
    interval_end = min(high, right)
    if interval_start > interval_end + _EPS:
        return None
    return interval_start, interval_end


def _linear_abs_leq_interval(
    *,
    start_value: float,
    slope: float,
    threshold: float,
    low: float,
    high: float,
) -> tuple[float, float] | None:
    if abs(slope) <= _EPS:
        if abs(start_value) <= threshold + _EPS:
            return low, high
        return None

    first_root = (-threshold - start_value) / slope
    second_root = (threshold - start_value) / slope
    left = min(first_root, second_root)
    right = max(first_root, second_root)
    interval_start = max(low, left)
    interval_end = min(high, right)
    if interval_start > interval_end + _EPS:
        return None
    return interval_start, interval_end


def _closest_time(first: _Segment, second: _Segment, interval_start: float, interval_end: float) -> float:
    first_start = first.position_at(interval_start)
    second_start = second.position_at(interval_start)
    dx0 = first_start[0] - second_start[0]
    dy0 = first_start[1] - second_start[1]
    dz0 = first_start[2] - second_start[2]
    dvx = first.vx - second.vx
    dvy = first.vy - second.vy
    dvz = first.vz - second.vz

    lateral_threshold_m = 2.0 * CONFLICT_ENVELOPE_RADIUS_NM * _METERS_PER_NM
    vertical_threshold_m = _CONFLICT_ENVELOPE_HEIGHT_FT * _METERS_PER_FT
    sx0 = dx0 / lateral_threshold_m
    sy0 = dy0 / lateral_threshold_m
    sz0 = dz0 / vertical_threshold_m
    svx = dvx / lateral_threshold_m
    svy = dvy / lateral_threshold_m
    svz = dvz / vertical_threshold_m
    denominator = svx * svx + svy * svy + svz * svz
    if denominator <= _EPS:
        return interval_start

    offset = -(sx0 * svx + sy0 * svy + sz0 * svz) / denominator
    return min(max(interval_start + offset, interval_start), interval_end)


def _merge_hits(hits: list[_ConflictHit]) -> list[ConflictEvent]:
    ordered_hits = sorted(
        hits,
        key=lambda hit: (hit.pair_key, hit.start_time, hit.end_time, -hit.severity),
    )
    merged: list[_ConflictHit] = []
    for hit in ordered_hits:
        starts_new_event = (
            not merged
            or hit.pair_key != merged[-1].pair_key
            or hit.start_time > merged[-1].end_time + _DEFAULT_MERGE_GAP_S
        )
        if starts_new_event:
            merged.append(hit)
            continue

        current = merged[-1]
        best = hit if _better_hit(hit, current) else current
        merged[-1] = _ConflictHit(
            flight_a=current.flight_a,
            flight_b=current.flight_b,
            start_time=min(current.start_time, hit.start_time),
            end_time=max(current.end_time, hit.end_time),
            closest_time=best.closest_time,
            latitude=best.latitude,
            longitude=best.longitude,
            lateral_distance_m=best.lateral_distance_m,
            vertical_separation_m=best.vertical_separation_m,
            severity=best.severity,
            confidence=_merged_confidence(current.confidence, hit.confidence),
        )

    events = [_event_from_hit(hit) for hit in merged]
    return sorted(
        events,
        key=lambda event: (
            event.confidence != "confirmed",
            -event.severity,
            event.lateral_distance_nmi,
            event.vertical_separation_ft,
            event.start_time,
            event.flight_a.flight_id,
            event.flight_b.flight_id,
        ),
    )


def _event_from_hit(hit: _ConflictHit) -> ConflictEvent:
    lateral_threshold_nmi = 2.0 * CONFLICT_ENVELOPE_RADIUS_NM
    vertical_threshold_ft = _CONFLICT_ENVELOPE_HEIGHT_FT
    return ConflictEvent(
        flight_a=hit.flight_a,
        flight_b=hit.flight_b,
        start_time=int(round(hit.start_time)),
        end_time=int(round(hit.end_time)),
        closest_time=int(round(hit.closest_time)),
        closest_time_utc=_time_utc(hit.closest_time),
        latitude=hit.latitude,
        longitude=hit.longitude,
        lateral_distance_nmi=hit.lateral_distance_m / _METERS_PER_NM,
        vertical_separation_ft=hit.vertical_separation_m / _METERS_PER_FT,
        lateral_threshold_nmi=lateral_threshold_nmi,
        vertical_threshold_ft=vertical_threshold_ft,
        severity=hit.severity,
        confidence=hit.confidence,
    )


def _better_hit(candidate: _ConflictHit, current: _ConflictHit) -> bool:
    if candidate.confidence == "confirmed" and current.confidence != "confirmed":
        return True
    if candidate.confidence != "confirmed" and current.confidence == "confirmed":
        return False
    if candidate.severity != current.severity:
        return candidate.severity > current.severity
    if candidate.lateral_distance_m != current.lateral_distance_m:
        return candidate.lateral_distance_m < current.lateral_distance_m
    return candidate.vertical_separation_m < current.vertical_separation_m


def _merged_confidence(first: str, second: str) -> str:
    if first == "confirmed" or second == "confirmed":
        return "confirmed"
    return "possible"


def _ordered_flights(first: ConflictFlight, second: ConflictFlight) -> tuple[ConflictFlight, ConflictFlight]:
    if (first.flight_id, first.flight_number, first.icao24) <= (second.flight_id, second.flight_number, second.icao24):
        return first, second
    return second, first


def _severity(lateral_distance_m: float, vertical_separation_m: float) -> float:
    lateral_threshold_m = 2.0 * CONFLICT_ENVELOPE_RADIUS_NM * _METERS_PER_NM
    vertical_threshold_m = _CONFLICT_ENVELOPE_HEIGHT_FT * _METERS_PER_FT
    lateral_penetration = max(0.0, (lateral_threshold_m - lateral_distance_m) / lateral_threshold_m)
    vertical_penetration = max(0.0, (vertical_threshold_m - vertical_separation_m) / vertical_threshold_m)
    return min(lateral_penetration, vertical_penetration)


def _time_utc(time_s: float) -> str:
    return datetime.fromtimestamp(int(round(time_s)), tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _columns(arrival: dict[str, Any]) -> list[str]:
    columns = arrival.get("columns")
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise ValueError(f"{_flight_label(arrival)} has missing or malformed columns")
    return columns


def _points(arrival: dict[str, Any]) -> list[Any]:
    points = arrival.get("points")
    if not isinstance(points, list):
        raise ValueError(f"{_flight_label(arrival)} has missing or malformed points")
    return points


def _validate_point_shape(point: Any, minimum_length: int, row_index: int, arrival: dict[str, Any]) -> None:
    if not isinstance(point, list | tuple) or len(point) < minimum_length:
        raise ValueError(f"{_flight_label(arrival)} has malformed trajectory point at index {row_index}")


def _column_index(columns: list[str], name: str, label: str) -> int:
    try:
        return columns.index(name)
    except ValueError as exc:
        raise ValueError(f"{label} trajectory columns must include {name}") from exc


def _finite_number(value: Any, field_name: str, arrival: dict[str, Any]) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{_flight_label(arrival)} has missing or malformed {field_name}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{_flight_label(arrival)} has missing or malformed {field_name}")
    return number


def _nonnegative_number(value: Any, field_name: str, arrival: dict[str, Any]) -> float:
    number = _finite_number(value, field_name, arrival)
    if number < 0.0:
        raise ValueError(f"{_flight_label(arrival)} has missing or malformed {field_name}")
    return number


def _optional_nonnegative_number(value: Any, field_name: str, arrival: dict[str, Any]) -> float:
    if value is None:
        return 0.0
    return _nonnegative_number(value, field_name, arrival)


def _flight_label(arrival: dict[str, Any]) -> str:
    flight_id = str(arrival.get("flight_id", ""))
    callsign = str(arrival.get("callsign", ""))
    if flight_id and callsign:
        return f"flight_id={flight_id} callsign={callsign}"
    if flight_id:
        return f"flight_id={flight_id}"
    if callsign:
        return f"callsign={callsign}"
    return "arrival"
