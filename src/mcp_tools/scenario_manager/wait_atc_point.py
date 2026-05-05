from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from simap.nlp_colloc.tactical.models import PathWaypoint
from simap.nlp_colloc.tactical.path import resolve_lateral_path
from simap.path_geometry import EARTH_RADIUS_M


NM_TO_M = 1_852.0
DEFAULT_MIN_DISTANCE_NM = 40.0
DEFAULT_MAX_DISTANCE_NM = 50.0
DEFAULT_HEADING_TOLERANCE_DEG = 45.0
GHOST_IDENTIFIER = "WAIT_ATC_GHOST"

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
    min_distance_nm: float = DEFAULT_MIN_DISTANCE_NM,
    max_distance_nm: float = DEFAULT_MAX_DISTANCE_NM,
    heading_tolerance_deg: float = DEFAULT_HEADING_TOLERANCE_DEG,
) -> dict[str, Any]:
    resolved_path = resolve_lateral_path(route, fix_catalog)
    final_course_deg = parse_runway_final_course_deg(runway)
    downwind_course_deg = _normalize_course_deg(final_course_deg + 180.0)
    airport_lat_deg, airport_lon_deg = _airport_reference_latlon(fix_catalog, runway=runway)

    candidates: list[dict[str, Any]] = []
    waypoints = resolved_path.waypoints
    for index, waypoint in enumerate(waypoints):
        if not _is_route_fix_candidate(waypoint):
            continue
        distance_nm = _haversine_distance_m(
            airport_lat_deg,
            airport_lon_deg,
            float(waypoint.lat_deg),
            float(waypoint.lon_deg),
        ) / NM_TO_M
        if distance_nm < min_distance_nm or distance_nm > max_distance_nm:
            continue

        matched_course_deg = _matched_downwind_course_deg(
            waypoints,
            index,
            downwind_course_deg=downwind_course_deg,
            tolerance_deg=heading_tolerance_deg,
        )
        if matched_course_deg is None:
            continue

        candidates.append(
            {
                "source": "fix",
                "identifier": waypoint.identifier,
                "lat": float(waypoint.lat_deg),
                "lon": float(waypoint.lon_deg),
                "lateral_path_token": waypoint.identifier,
                "distance_nm": float(distance_nm),
                "route_index": index,
                "matched_course_deg": float(matched_course_deg),
                "final_course_deg": float(final_course_deg),
                "downwind_course_deg": float(downwind_course_deg),
            }
        )

    if candidates:
        return candidates[-1]

    ghost_lat_deg, ghost_lon_deg = _destination_latlon(
        airport_lat_deg,
        airport_lon_deg,
        downwind_course_deg,
        min_distance_nm * NM_TO_M,
    )
    return {
        "source": "ghost",
        "identifier": GHOST_IDENTIFIER,
        "lat": float(ghost_lat_deg),
        "lon": float(ghost_lon_deg),
        "lateral_path_token": f"{ghost_lat_deg:.6f},{ghost_lon_deg:.6f}",
        "distance_nm": float(min_distance_nm),
        "route_index": None,
        "matched_course_deg": None,
        "final_course_deg": float(final_course_deg),
        "downwind_course_deg": float(downwind_course_deg),
    }


def _is_route_fix_candidate(waypoint: PathWaypoint) -> bool:
    if waypoint.identifier.upper().startswith("RW"):
        return False
    return waypoint.source.lower() != "coordinate"


def _airport_reference_latlon(
    fix_catalog: Mapping[str, PathWaypoint],
    *,
    runway: str,
) -> tuple[float, float]:
    runway_points = [
        waypoint
        for waypoint in fix_catalog.values()
        if waypoint.identifier.upper().startswith("RW")
        and np.isfinite(float(waypoint.lat_deg))
        and np.isfinite(float(waypoint.lon_deg))
    ]
    if runway_points:
        return (
            float(np.mean([float(waypoint.lat_deg) for waypoint in runway_points])),
            float(np.mean([float(waypoint.lon_deg) for waypoint in runway_points])),
        )

    runway_identifier = _runway_identifier(runway)
    runway_waypoint = fix_catalog.get(runway_identifier)
    if runway_waypoint is None:
        raise KeyError(f"unknown runway fix: {runway_identifier}")
    return float(runway_waypoint.lat_deg), float(runway_waypoint.lon_deg)


def _runway_identifier(runway: str) -> str:
    text = str(runway).strip().upper()
    return text if text.startswith("RW") else f"RW{text}"


def _matched_downwind_course_deg(
    waypoints: Sequence[PathWaypoint],
    index: int,
    *,
    downwind_course_deg: float,
    tolerance_deg: float,
) -> float | None:
    courses: list[float] = []
    waypoint = waypoints[index]
    if index > 0:
        previous = waypoints[index - 1]
        courses.append(_initial_bearing_deg(previous.lat_deg, previous.lon_deg, waypoint.lat_deg, waypoint.lon_deg))
    if index + 1 < len(waypoints):
        next_waypoint = waypoints[index + 1]
        courses.append(
            _initial_bearing_deg(waypoint.lat_deg, waypoint.lon_deg, next_waypoint.lat_deg, next_waypoint.lon_deg)
        )
    matching_courses = [
        course for course in courses if _course_delta_deg(course, downwind_course_deg) <= float(tolerance_deg)
    ]
    if not matching_courses:
        return None
    return min(matching_courses, key=lambda course: _course_delta_deg(course, downwind_course_deg))


def _normalize_course_deg(course_deg: float) -> float:
    return float(course_deg % 360.0)


def _course_delta_deg(a_deg: float, b_deg: float) -> float:
    return float(abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0))


def _haversine_distance_m(lat_a_deg: float, lon_a_deg: float, lat_b_deg: float, lon_b_deg: float) -> float:
    lat_a_rad = math.radians(float(lat_a_deg))
    lat_b_rad = math.radians(float(lat_b_deg))
    dlat_rad = lat_b_rad - lat_a_rad
    dlon_rad = math.radians(float(lon_b_deg) - float(lon_a_deg))
    hav = math.sin(dlat_rad / 2.0) ** 2 + math.cos(lat_a_rad) * math.cos(lat_b_rad) * math.sin(dlon_rad / 2.0) ** 2
    return float(2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(hav)))


def _initial_bearing_deg(lat_a_deg: float, lon_a_deg: float, lat_b_deg: float, lon_b_deg: float) -> float:
    lat_a_rad = math.radians(float(lat_a_deg))
    lat_b_rad = math.radians(float(lat_b_deg))
    dlon_rad = math.radians(float(lon_b_deg) - float(lon_a_deg))
    y = math.sin(dlon_rad) * math.cos(lat_b_rad)
    x = math.cos(lat_a_rad) * math.sin(lat_b_rad) - math.sin(lat_a_rad) * math.cos(lat_b_rad) * math.cos(dlon_rad)
    return _normalize_course_deg(math.degrees(math.atan2(y, x)))


def _destination_latlon(
    lat_deg: float,
    lon_deg: float,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    lat_rad = math.radians(float(lat_deg))
    lon_rad = math.radians(float(lon_deg))
    bearing_rad = math.radians(float(bearing_deg))
    angular_distance = float(distance_m) / EARTH_RADIUS_M

    dest_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(angular_distance)
        + math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    dest_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(dest_lat_rad),
    )
    dest_lon_rad = (dest_lon_rad + math.pi) % (2.0 * math.pi) - math.pi
    return float(math.degrees(dest_lat_rad)), float(math.degrees(dest_lon_rad))
