from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from simap.nlp_colloc.tactical.models import PathWaypoint
from simap.nlp_colloc.tactical.path import resolve_lateral_path


DEFAULT_HEADING_TOLERANCE_DEG = 10.0

_RUNWAY_RE = re.compile(r"^(?:RW)?(?P<number>[0-9]{2})(?:[LCR])?$", re.IGNORECASE)


def parse_runway_final_course_deg(runway: str) -> float:
    match = _RUNWAY_RE.fullmatch(str(runway).strip().upper())
    if match is None:
        raise ValueError(f"invalid runway designator: {runway!r}")
    number = int(match.group("number"))
    if number < 1 or number > 36:
        raise ValueError(f"invalid runway designator: {runway!r}")
    return float((number * 10) % 360)


def detect_wait_atc_point(
    route: Sequence[str | tuple[float, float]],
    fix_catalog: Mapping[str, PathWaypoint],
    *,
    runway: str,
    heading_tolerance_deg: float = DEFAULT_HEADING_TOLERANCE_DEG,
) -> dict[str, Any] | None:
    resolved_path = resolve_lateral_path(route, fix_catalog)
    final_course_deg = parse_runway_final_course_deg(runway)
    downwind_course_deg = _normalize_course_deg(final_course_deg + 180.0)
    waypoints = resolved_path.waypoints

    for index in range(len(waypoints) - 2, -1, -1):
        waypoint = waypoints[index]
        next_waypoint = waypoints[index + 1]
        matched_course_deg = _initial_bearing_deg(
            waypoint.lat_deg,
            waypoint.lon_deg,
            next_waypoint.lat_deg,
            next_waypoint.lon_deg,
        )
        if _course_delta_deg(matched_course_deg, downwind_course_deg) > float(heading_tolerance_deg):
            continue
        if not _is_route_fix_candidate(next_waypoint):
            continue

        return {
            "source": "fix",
            "identifier": next_waypoint.identifier,
            "lat": float(next_waypoint.lat_deg),
            "lon": float(next_waypoint.lon_deg),
            "lateral_path_token": next_waypoint.identifier,
            "route_index": index + 1,
            "matched_course_deg": float(matched_course_deg),
            "final_course_deg": float(final_course_deg),
            "downwind_course_deg": float(downwind_course_deg),
        }

    return None


def _is_route_fix_candidate(waypoint: PathWaypoint) -> bool:
    if waypoint.identifier.upper().startswith("RW"):
        return False
    return waypoint.source.lower() != "coordinate"


def _normalize_course_deg(course_deg: float) -> float:
    return float(course_deg % 360.0)


def _course_delta_deg(a_deg: float, b_deg: float) -> float:
    return float(abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0))


def _initial_bearing_deg(lat_a_deg: float, lon_a_deg: float, lat_b_deg: float, lon_b_deg: float) -> float:
    lat_a_rad = math.radians(float(lat_a_deg))
    lat_b_rad = math.radians(float(lat_b_deg))
    dlon_rad = math.radians(float(lon_b_deg) - float(lon_a_deg))
    y = math.sin(dlon_rad) * math.cos(lat_b_rad)
    x = math.cos(lat_a_rad) * math.sin(lat_b_rad) - math.sin(lat_a_rad) * math.cos(lat_b_rad) * math.cos(dlon_rad)
    return _normalize_course_deg(math.degrees(math.atan2(y, x)))
