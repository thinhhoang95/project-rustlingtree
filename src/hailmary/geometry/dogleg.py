"""Deterministic runway-away triangular dogleg construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, pi
from typing import Iterable

import numpy as np

from hailmary._arrays import readonly_float64
from hailmary.errors import InfeasibleActionError


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray, *, tolerance: float = 1e-8) -> int:
    value = _cross(b - a, c - a)
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> bool:
    o1 = _orientation(a, b, c, tolerance=tolerance)
    o2 = _orientation(a, b, d, tolerance=tolerance)
    o3 = _orientation(c, d, a, tolerance=tolerance)
    o4 = _orientation(c, d, b, tolerance=tolerance)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    def on_segment(start: np.ndarray, point: np.ndarray, end: np.ndarray) -> bool:
        return bool(
            min(start[0], end[0]) - tolerance <= point[0] <= max(start[0], end[0]) + tolerance
            and min(start[1], end[1]) - tolerance <= point[1] <= max(start[1], end[1]) + tolerance
        )

    return bool(
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    length_sq = float(np.dot(direction, direction))
    if length_sq <= 1e-18:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, direction) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * direction)))


def segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def polyline_min_distance(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != (2,) or right.shape[1:] != (2,):
        raise ValueError("polylines must have shape (n, 2)")
    if len(left) < 2 or len(right) < 2:
        raise ValueError("polylines require at least two points")
    best = np.inf
    for a, b in zip(left[:-1], left[1:], strict=True):
        for c, d in zip(right[:-1], right[1:], strict=True):
            best = min(best, segment_distance(a, b, c, d))
            if best <= 0.0:
                return 0.0
    return float(best)


def turn_angle_deg(previous: np.ndarray, vertex: np.ndarray, following: np.ndarray) -> float:
    incoming = vertex - previous
    outgoing = following - vertex
    denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if denominator <= 1e-12:
        return 180.0
    return float(np.degrees(acos(float(np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0)))))


@dataclass(frozen=True)
class DoglegGeometry:
    rejoin_index: int
    action_index: int
    rejoin_point_m: np.ndarray
    apex_point_m: np.ndarray
    action_point_m: np.ndarray
    target_added_distance_m: float
    realized_added_distance_m: float
    medoid_clearance_m: float
    runway_away_displacement_m: float
    boundary_clearance_m: float
    azimuth_rad: float
    corridor_points_m: np.ndarray | None = None
    corridor_parent_s_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("rejoin_point_m", "apex_point_m", "action_point_m"):
            object.__setattr__(self, name, readonly_float64(getattr(self, name), name=name))
        if self.corridor_points_m is not None:
            if self.corridor_parent_s_m is None:
                raise ValueError("dogleg corridor stations are required with corridor points")
            corridor = readonly_float64(
                self.corridor_points_m,
                name="corridor_points_m",
                ndim=2,
            )
            parent_s = readonly_float64(
                self.corridor_parent_s_m,
                name="corridor_parent_s_m",
            )
            if corridor.ndim != 2 or corridor.shape[1:] != (2,) or len(corridor) < 3:
                raise ValueError("dogleg corridor must have shape (n>=3, 2)")
            if parent_s.ndim != 1 or len(parent_s) != len(corridor):
                raise ValueError("dogleg corridor stations must match corridor points")
            if np.any(np.diff(parent_s) <= 0.0):
                raise ValueError("dogleg corridor parent stations must strictly increase")
            if not np.allclose(corridor[0], self.rejoin_point_m, atol=1e-7):
                raise ValueError("dogleg corridor must begin at the rejoin point")
            if not np.allclose(corridor[-1], self.action_point_m, atol=1e-7):
                raise ValueError("dogleg corridor must end at the action point")
            object.__setattr__(self, "corridor_points_m", corridor)
            object.__setattr__(self, "corridor_parent_s_m", parent_s)

    @property
    def points_m(self) -> np.ndarray:
        if self.corridor_points_m is not None:
            return self.corridor_points_m
        result = np.vstack((self.rejoin_point_m, self.apex_point_m, self.action_point_m))
        result.setflags(write=False)
        return result

    @property
    def parent_stations_m(self) -> np.ndarray | None:
        return self.corridor_parent_s_m


def _smooth_corridor(
    base: np.ndarray,
    stations: np.ndarray,
    *,
    rejoin_index: int,
    action_index: int,
    direction: np.ndarray,
    offset_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a tangent-continuous raised-cosine lane change.

    The previous quartic bump concentrated curvature near its shoulders.  It
    also became cusp-like when a clearance candidate had a large along-track
    component.  ``sin(pi*u)**2`` distributes heading change across the full
    corridor while retaining exact zero displacement and zero lateral slope at
    both splice endpoints.
    """

    start_s = float(stations[rejoin_index])
    end_s = float(stations[action_index])
    span_s = end_s - start_s
    if span_s <= 0.0:
        raise InfeasibleActionError("dogleg parent stations are not increasing")
    original_s = stations[rejoin_index : action_index + 1]
    sample_count = max(
        129,
        min(257, 2 * (action_index - rejoin_index) + 1),
    )
    parent_s = np.unique(
        np.round(
            np.concatenate((original_s, np.linspace(start_s, end_s, sample_count))),
            decimals=6,
        )
    )
    u = (parent_s - start_s) / span_s
    # The raised cosine reaches one at the center. Both displacement and its
    # first derivative are analytically zero at the live splice endpoints.
    bump = np.sin(pi * u) ** 2
    baseline = np.column_stack(
        (
            np.interp(parent_s, stations, base[:, 0]),
            np.interp(parent_s, stations, base[:, 1]),
        )
    )
    corridor = baseline + float(offset_m) * bump[:, np.newaxis] * direction
    # Eliminate trigonometric roundoff at the live splices; the simulator
    # requires byte-exact endpoint positions when installing a branch variant.
    corridor[0] = base[rejoin_index]
    corridor[-1] = base[action_index]
    length_m = float(np.linalg.norm(np.diff(corridor, axis=0), axis=1).sum())
    return corridor, parent_s, length_m


def _candidate_lateral_directions(
    start: np.ndarray,
    end: np.ndarray,
    *,
    count: int,
    maximum_tangent_bias_deg: float = 15.0,
) -> tuple[np.ndarray, ...]:
    """Return deterministic near-normal directions for runway-away search.

    Unconstrained absolute azimuths allow the optimizer to win the radial
    runway-away score by moving mostly along track, then doubling back to the
    action point.  Those geometries have artificial cusp-level curvature.
    Searching both local sides with only a small tangent bias preserves free
    space selection without permitting a reversal.
    """

    chord = np.asarray(end - start, dtype=np.float64)
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm <= 1e-9:
        raise InfeasibleActionError("dogleg splice endpoints are coincident")
    tangent = chord / chord_norm
    left = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    directions: list[np.ndarray] = []
    side_counts = ((1.0, (count + 1) // 2), (-1.0, count // 2))
    for side, per_side in side_counts:
        if per_side == 0:
            continue
        biases = (
            np.asarray([0.0])
            if per_side == 1
            else np.linspace(
                -np.deg2rad(maximum_tangent_bias_deg),
                np.deg2rad(maximum_tangent_bias_deg),
                per_side,
            )
        )
        normal = side * left
        for bias in biases:
            direction = np.cos(bias) * normal + np.sin(bias) * tangent
            direction /= np.linalg.norm(direction)
            directions.append(direction)
    return tuple(directions)


def _solve_smooth_offset(
    base: np.ndarray,
    stations: np.ndarray,
    *,
    rejoin_index: int,
    action_index: int,
    direction: np.ndarray,
    target_added_m: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    base_length_m = float(stations[action_index] - stations[rejoin_index])
    baseline, parent_s, _baseline_length_m = _smooth_corridor(
        base,
        stations,
        rejoin_index=rejoin_index,
        action_index=action_index,
        direction=direction,
        offset_m=0.0,
    )
    u = (parent_s - parent_s[0]) / (parent_s[-1] - parent_s[0])
    displacement = (np.sin(pi * u) ** 2)[:, np.newaxis] * direction

    def candidate(offset_m: float) -> tuple[np.ndarray, np.ndarray, float]:
        corridor = baseline + float(offset_m) * displacement
        corridor[0] = base[rejoin_index]
        corridor[-1] = base[action_index]
        length_m = float(np.linalg.norm(np.diff(corridor, axis=0), axis=1).sum())
        return corridor, parent_s, length_m - base_length_m

    start = base[rejoin_index]
    end = base[action_index]
    chord_m = float(np.linalg.norm(end - start))
    high = max(
        1.0,
        0.5
        * np.sqrt(
            max(0.0, target_added_m * (2.0 * max(chord_m, 1.0) + target_added_m))
        ),
    )
    high_candidate = candidate(high)
    while high_candidate[2] < target_added_m and high < 2_000_000.0:
        high *= 2.0
        high_candidate = candidate(high)
    if high_candidate[2] < target_added_m:
        raise InfeasibleActionError("cannot solve smooth dogleg offset for target added distance")
    low = 0.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        if candidate(middle)[2] < target_added_m:
            low = middle
        else:
            high = middle
    offset = 0.5 * (low + high)
    corridor, parent_s, realized_added_m = candidate(offset)
    return offset, corridor, parent_s, realized_added_m


def _corridor_self_intersects(
    base: np.ndarray,
    *,
    rejoin_index: int,
    action_index: int,
    corridor: np.ndarray,
) -> bool:
    # The corridor is a single smooth polynomial bump.  A bounded inspection
    # grid is sufficient to detect its only possible large-scale loop and
    # avoids quadratic work on the dense executable sampling grid.
    if len(corridor) > 65:
        indices = np.unique(np.linspace(0, len(corridor) - 1, 65, dtype=int))
        inspected = corridor[indices]
    else:
        inspected = corridor
    candidate_segments = tuple(zip(inspected[:-1], inspected[1:], strict=True))
    for first_index, (first_start, first_end) in enumerate(candidate_segments):
        for second_index in range(first_index + 2, len(candidate_segments)):
            if second_index == first_index + 1:
                continue
            second_start, second_end = candidate_segments[second_index]
            if segments_intersect(first_start, first_end, second_start, second_end):
                return True
    base_indices = np.concatenate(
        (
            np.arange(0, rejoin_index, dtype=int),
            np.arange(action_index, len(base) - 1, dtype=int),
        )
    )
    if len(base_indices):
        base_starts = base[base_indices]
        base_ends = base[base_indices + 1]
        base_min = np.minimum(base_starts, base_ends)
        base_max = np.maximum(base_starts, base_ends)
        for candidate_start, candidate_end in candidate_segments:
            candidate_min = np.minimum(candidate_start, candidate_end)
            candidate_max = np.maximum(candidate_start, candidate_end)
            overlapping = np.flatnonzero(
                np.all(base_max >= candidate_min - 1e-8, axis=1)
                & np.all(candidate_max >= base_min - 1e-8, axis=1)
            )
            for local_index in overlapping:
                start = base_starts[local_index]
                end = base_ends[local_index]
                shares_endpoint = (
                    np.linalg.norm(start - candidate_start) <= 1e-7
                    or np.linalg.norm(start - candidate_end) <= 1e-7
                    or np.linalg.norm(end - candidate_start) <= 1e-7
                    or np.linalg.norm(end - candidate_end) <= 1e-7
                )
                if not shares_endpoint and segments_intersect(
                    start,
                    end,
                    candidate_start,
                    candidate_end,
                ):
                    return True
    return False


def construct_runway_away_dogleg(
    base_points_m: np.ndarray,
    base_s_m: np.ndarray,
    *,
    action_index: int,
    rejoin_span_m: float,
    target_added_distance_m: float,
    other_medoid_polylines_m: Iterable[np.ndarray] = (),
    boundary_polylines_m: Iterable[np.ndarray] = (),
    candidate_azimuth_count: int = 24,
    max_turn_deg: float = 70.0,
    added_distance_tolerance_m: float = 0.25 * 1_852.0,
    minimum_medoid_clearance_m: float = 0.0,
    minimum_rejoin_span_m: float = 2.0 * 1_852.0,
    minimum_rejoin_station_m: float = 0.0,
) -> DoglegGeometry:
    """Choose a deterministic dogleg direction by library-wide free space."""

    base = np.asarray(base_points_m, dtype=float)
    stations = np.asarray(base_s_m, dtype=float)
    if base.ndim != 2 or base.shape[1] != 2 or len(base) != len(stations) or len(base) < 4:
        raise InfeasibleActionError("base geometry/stations are invalid")
    if action_index <= 1 or action_index >= len(base):
        raise InfeasibleActionError("path-stretch action index is outside the controllable path")
    if rejoin_span_m <= 0.0 or target_added_distance_m <= 0.0:
        raise InfeasibleActionError("rejoin span and added distance must be positive")

    target_rejoin_s = float(stations[action_index] - rejoin_span_m)
    candidates = np.flatnonzero(
        (stations[:action_index] <= target_rejoin_s + 1e-9)
        & (stations[:action_index] >= minimum_rejoin_station_m - 1e-9)
    )
    if len(candidates) == 0:
        raise InfeasibleActionError("insufficient downstream path for requested rejoin span")
    rejoin_index = int(candidates[-1])
    span = float(stations[action_index] - stations[rejoin_index])
    if span < minimum_rejoin_span_m:
        raise InfeasibleActionError("available rejoin span is too short")

    start = base[rejoin_index]
    end = base[action_index]
    midpoint = 0.5 * (start + end)
    medoids = tuple(np.asarray(polyline, dtype=float) for polyline in other_medoid_polylines_m)
    boundaries = tuple(np.asarray(polyline, dtype=float) for polyline in boundary_polylines_m)
    feasible: list[DoglegGeometry] = []
    for direction in _candidate_lateral_directions(
        start,
        end,
        count=candidate_azimuth_count,
    ):
        _offset, corridor, corridor_parent_s, realized = _solve_smooth_offset(
            base,
            stations,
            rejoin_index=rejoin_index,
            action_index=action_index,
            direction=direction,
            target_added_m=target_added_distance_m,
        )
        apex_index = int(
            np.argmin(
                np.abs(
                    corridor_parent_s
                    - 0.5 * (stations[rejoin_index] + stations[action_index])
                )
            )
        )
        apex = corridor[apex_index]
        if float(np.linalg.norm(apex)) <= float(np.linalg.norm(midpoint)) + 1e-7:
            continue
        if abs(realized - target_added_distance_m) > added_distance_tolerance_m:
            continue
        parent_baseline = np.column_stack(
            (
                np.interp(corridor_parent_s, stations, base[:, 0]),
                np.interp(corridor_parent_s, stations, base[:, 1]),
            )
        )
        corridor_delta = np.diff(corridor, axis=0)
        baseline_delta = np.diff(parent_baseline, axis=0)
        denominator = np.linalg.norm(corridor_delta, axis=1) * np.linalg.norm(
            baseline_delta,
            axis=1,
        )
        alongtrack_cosine = np.einsum(
            "ij,ij->i",
            corridor_delta,
            baseline_delta,
        ) / np.maximum(denominator, 1.0e-12)
        # A path stretch must remain a forward dogleg.  Without this guard a
        # nominally runway-away offset can manufacture distance by folding
        # backward along the arrival path, creating cusp-like curvature.
        if np.any(alongtrack_cosine < 0.45):
            continue
        if _corridor_self_intersects(
            base,
            rejoin_index=rejoin_index,
            action_index=action_index,
            corridor=corridor,
        ):
            continue

        previous = base[rejoin_index - 1]
        following = base[action_index + 1] if action_index + 1 < len(base) else end + (end - apex)
        turn_angles = (
            turn_angle_deg(previous, start, apex),
            turn_angle_deg(start, apex, end),
            turn_angle_deg(apex, end, following),
        )
        if any(angle > max_turn_deg + 1e-8 for angle in turn_angles):
            continue
        medoid_clearance = min(
            (polyline_min_distance(corridor, item) for item in medoids),
            default=np.inf,
        )
        if medoid_clearance < minimum_medoid_clearance_m - 1e-8:
            continue
        boundary_clearance = min(
            (polyline_min_distance(corridor, item) for item in boundaries),
            default=np.inf,
        )
        feasible.append(
            DoglegGeometry(
                rejoin_index=rejoin_index,
                action_index=action_index,
                rejoin_point_m=start,
                apex_point_m=apex,
                action_point_m=end,
                target_added_distance_m=float(target_added_distance_m),
                realized_added_distance_m=float(realized),
                medoid_clearance_m=float(medoid_clearance),
                runway_away_displacement_m=float(np.linalg.norm(apex) - np.linalg.norm(midpoint)),
                boundary_clearance_m=float(boundary_clearance),
                azimuth_rad=float(atan2(direction[1], direction[0]) % (2.0 * pi)),
                corridor_points_m=corridor,
                corridor_parent_s_m=corridor_parent_s,
            )
        )
    if not feasible:
        raise InfeasibleActionError("no runway-away dogleg direction satisfies geometry constraints")
    return max(
        feasible,
        key=lambda item: (
            item.medoid_clearance_m,
            item.runway_away_displacement_m,
            item.boundary_clearance_m,
            -item.azimuth_rad,
        ),
    )
