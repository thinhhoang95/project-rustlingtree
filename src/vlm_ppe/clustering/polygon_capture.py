from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence

import pandas as pd

from vlm_ppe.schemas import SubclusterPolygon

EPSILON = 1e-9

Point = tuple[float, float]


@dataclass(frozen=True)
class SubclusterCapture:
    subcluster_id: int
    label: str
    polygon: list[Point]
    track_ids: list[str]


@dataclass(frozen=True)
class PolygonCaptureResult:
    captures: list[SubclusterCapture]
    uncaptured_track_ids: list[str]
    overlapping_track_ids: dict[str, list[int]]


def assign_tracks_to_subcluster_polygons(
    resampled: pd.DataFrame,
    track_ids: Sequence[str],
    subclusters: Sequence[SubclusterPolygon],
) -> PolygonCaptureResult:
    ordered_track_ids = [str(track_id) for track_id in track_ids]
    track_set = set(ordered_track_ids)
    frame = resampled.loc[resampled["flight_id"].astype(str).isin(track_set)].copy()
    frame["flight_id"] = frame["flight_id"].astype(str)

    normalized_polygons = [
        (
            int(subcluster.subcluster_id),
            subcluster.label or f"Subcluster {int(subcluster.subcluster_id)}",
            convex_hull(subcluster.polygon),
        )
        for subcluster in subclusters
    ]

    captured: dict[int, list[str]] = {subcluster_id: [] for subcluster_id, _label, _polygon in normalized_polygons}
    overlapping: dict[str, list[int]] = {}
    uncaptured: list[str] = []

    points_by_track = _track_points(frame)
    for track_id in ordered_track_ids:
        points = points_by_track.get(track_id, [])
        matches = [
            subcluster_id
            for subcluster_id, _label, polygon in normalized_polygons
            if polyline_crosses_polygon(points, polygon)
        ]
        if not matches:
            uncaptured.append(track_id)
            continue
        if len(matches) > 1:
            overlapping[track_id] = matches
        captured[matches[0]].append(track_id)

    captures = [
        SubclusterCapture(
            subcluster_id=subcluster_id,
            label=label,
            polygon=polygon,
            track_ids=captured[subcluster_id],
        )
        for subcluster_id, label, polygon in normalized_polygons
    ]
    return PolygonCaptureResult(
        captures=captures,
        uncaptured_track_ids=uncaptured,
        overlapping_track_ids=overlapping,
    )


def convex_hull(points: Iterable[Sequence[float]]) -> list[Point]:
    normalized = sorted(set(_coerce_point(point) for point in points))
    if len(normalized) < 3:
        raise ValueError("a capture polygon needs at least three unique coordinate points")

    lower: list[Point] = []
    for point in normalized:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= EPSILON:
            lower.pop()
        lower.append(point)

    upper: list[Point] = []
    for point in reversed(normalized):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= EPSILON:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3 or abs(_polygon_area(hull)) <= EPSILON:
        raise ValueError("a capture polygon needs non-collinear coordinate points")
    return hull


def polyline_crosses_polygon(points: Sequence[Point], polygon: Sequence[Point]) -> bool:
    if not points or not polygon:
        return False
    if any(point_in_polygon(point, polygon) for point in points):
        return True

    edges = list(_polygon_edges(polygon))
    for start, end in zip(points[:-1], points[1:]):
        if any(segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in edges):
            return True
    return False


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if point_on_segment(point, previous, current):
            return True
        x0, y0 = previous
        x1, y1 = current
        crosses_y = (y0 > y) != (y1 > y)
        if crosses_y:
            x_intersection = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x <= x_intersection + EPSILON:
                inside = not inside
        previous = current
    return inside


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if o1 == 0 and point_on_segment(c, a, b):
        return True
    if o2 == 0 and point_on_segment(d, a, b):
        return True
    if o3 == 0 and point_on_segment(a, c, d):
        return True
    if o4 == 0 and point_on_segment(b, c, d):
        return True
    return o1 != o2 and o3 != o4


def point_on_segment(point: Point, start: Point, end: Point) -> bool:
    if abs(_cross(start, end, point)) > EPSILON:
        return False
    min_x, max_x = sorted((start[0], end[0]))
    min_y, max_y = sorted((start[1], end[1]))
    return min_x - EPSILON <= point[0] <= max_x + EPSILON and min_y - EPSILON <= point[1] <= max_y + EPSILON


def _track_points(frame: pd.DataFrame) -> dict[str, list[Point]]:
    points_by_track: dict[str, list[Point]] = {}
    for flight_id, group in frame.groupby("flight_id", sort=False):
        ordered = group.sort_values("station_index", kind="stable")
        points_by_track[str(flight_id)] = [
            (float(row.x_nm), float(row.y_nm)) for row in ordered.itertuples(index=False)
        ]
    return points_by_track


def _polygon_edges(polygon: Sequence[Point]) -> Iterable[tuple[Point, Point]]:
    for index, start in enumerate(polygon):
        yield start, polygon[(index + 1) % len(polygon)]


def _coerce_point(point: Sequence[float]) -> Point:
    if len(point) != 2:
        raise ValueError(f"polygon point must contain exactly two values: {point}")
    x = float(point[0])
    y = float(point[1])
    if not (isfinite(x) and isfinite(y)):
        raise ValueError(f"polygon point must contain finite coordinates: {point}")
    return x, y


def _cross(origin: Point, a: Point, b: Point) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = _cross(a, b, c)
    if abs(value) <= EPSILON:
        return 0
    return 1 if value > 0 else -1


def _polygon_area(polygon: Sequence[Point]) -> float:
    total = 0.0
    for start, end in _polygon_edges(polygon):
        total += start[0] * end[1] - end[0] * start[1]
    return total / 2.0
