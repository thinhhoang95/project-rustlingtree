from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import numpy as np

from ...path_geometry import ReferencePath

from .models import PathWaypoint, ResolvedTacticalPath

_COORD_RE = re.compile(
    r"^\(?\s*(?P<lat>[+-]?\d+(?:\.\d+)?)\s*[,/]\s*(?P<lon>[+-]?\d+(?:\.\d+)?)\s*\)?$"
)


def _tokenize_lateral_path(path: str | Iterable[str | tuple[float, float]]) -> list[str | tuple[float, float]]:
    if isinstance(path, str):
        return [token for token in path.split() if token]
    return list(path)


def _coordinate_waypoint(token: str | tuple[float, float], index: int) -> PathWaypoint | None:
    if isinstance(token, tuple):
        lat_deg, lon_deg = token
        return PathWaypoint(identifier=f"COORD{index:02d}", lat_deg=float(lat_deg), lon_deg=float(lon_deg), source="coordinate")
    match = _COORD_RE.match(token)
    if match is None:
        return None
    return PathWaypoint(
        identifier=f"COORD{index:02d}",
        lat_deg=float(match.group("lat")),
        lon_deg=float(match.group("lon")),
        source="coordinate",
    )


def resolve_lateral_path(
    lateral_path: str | Iterable[str | tuple[float, float]],
    fix_catalog: Mapping[str, PathWaypoint],
) -> ResolvedTacticalPath:
    waypoints: list[PathWaypoint] = []
    for index, token in enumerate(_tokenize_lateral_path(lateral_path), start=1):
        coordinate = _coordinate_waypoint(token, index)
        if coordinate is not None:
            waypoints.append(coordinate)
            continue

        identifier = str(token).strip().upper()
        if identifier not in fix_catalog:
            raise KeyError(f"unknown lateral-path fix: {identifier}")
        waypoints.append(fix_catalog[identifier])

    if len(waypoints) < 2:
        raise ValueError("a tactical lateral path requires at least two waypoints")
    return ResolvedTacticalPath(waypoints=tuple(waypoints))


def build_reference_path(path: ResolvedTacticalPath, *, samples_per_segment: int = 48) -> ReferencePath:
    return ReferencePath.from_geographic(
        lat_deg=np.asarray(path.lat_deg, dtype=float),
        lon_deg=np.asarray(path.lon_deg, dtype=float),
        samples_per_segment=samples_per_segment,
    )


def waypoint_distance_to_threshold_m(reference_path: ReferencePath, waypoint_index: int) -> float:
    if waypoint_index < 0 or waypoint_index >= len(reference_path.waypoint_lat_deg):
        raise IndexError("waypoint index is out of range")
    lat_deg = float(reference_path.waypoint_lat_deg[waypoint_index])
    lon_deg = float(reference_path.waypoint_lon_deg[waypoint_index])
    distances = np.hypot(reference_path.lat_deg - lat_deg, reference_path.lon_deg - lon_deg)
    sample_index = int(np.argmin(distances))
    return float(reference_path.s_m[sample_index])


def waypoint_s_by_identifier(path: ResolvedTacticalPath, reference_path: ReferencePath) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, waypoint in enumerate(path.waypoints):
        result[waypoint.identifier] = waypoint_distance_to_threshold_m(reference_path, index)
    return result
