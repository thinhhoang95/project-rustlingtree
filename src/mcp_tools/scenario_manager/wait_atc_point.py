from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from simap.nlp_colloc.tactical.models import PathWaypoint
from simap.nlp_colloc.tactical.path import resolve_lateral_path


DEFAULT_RING_INNER_NM = 35.0
DEFAULT_RING_OUTER_NM = 40.0

EARTH_RADIUS_M = 6_371_000.0
METERS_PER_NM = 1_852.0


def detect_wait_atc_point(
    route: Sequence[str | tuple[float, float]],
    fix_catalog: Mapping[str, PathWaypoint],
    *,
    runway: str,
    ring_inner_nm: float = DEFAULT_RING_INNER_NM,
    ring_outer_nm: float = DEFAULT_RING_OUTER_NM,
) -> dict[str, Any] | None:
    resolved_path = resolve_lateral_path(route, fix_catalog)
    runway_waypoint = _runway_waypoint(resolved_path.waypoints, fix_catalog, runway)
    if runway_waypoint is None:
        return None

    inner_nm = float(ring_inner_nm)
    outer_nm = float(ring_outer_nm)
    if inner_nm > outer_nm:
        raise ValueError("ring_inner_nm must be less than or equal to ring_outer_nm")

    waypoints = resolved_path.waypoints

    for index, waypoint in enumerate(waypoints):
        if not _is_route_fix_candidate(waypoint):
            continue
        distance_nm = _distance_nm(
            waypoint.lat_deg,
            waypoint.lon_deg,
            runway_waypoint.lat_deg,
            runway_waypoint.lon_deg,
        )
        if not inner_nm <= distance_nm <= outer_nm:
            continue

        identifier = waypoint.identifier
        return {
            "source": "fix",
            "identifier": identifier,
            "lat": float(waypoint.lat_deg),
            "lon": float(waypoint.lon_deg),
            "lateral_path_token": identifier,
            "route_index": index,
            "distance_nm": float(distance_nm),
            "ring_inner_nm": inner_nm,
            "ring_outer_nm": outer_nm,
        }

    return None


def _runway_waypoint(
    waypoints: Sequence[PathWaypoint],
    fix_catalog: Mapping[str, PathWaypoint],
    runway: str,
) -> PathWaypoint | None:
    runway_identifier = _normalize_runway_identifier(runway)
    for waypoint in waypoints:
        if waypoint.identifier.upper() == runway_identifier:
            return waypoint
    return fix_catalog.get(runway_identifier)


def _normalize_runway_identifier(runway: str) -> str:
    runway_identifier = str(runway).strip().upper()
    if not runway_identifier.startswith("RW"):
        runway_identifier = f"RW{runway_identifier}"
    return runway_identifier


def _is_route_fix_candidate(waypoint: PathWaypoint) -> bool:
    if waypoint.identifier.upper().startswith("RW"):
        return False
    return waypoint.source.lower() != "coordinate"


def _distance_nm(lat_a_deg: float, lon_a_deg: float, lat_b_deg: float, lon_b_deg: float) -> float:
    lat_a_rad = math.radians(float(lat_a_deg))
    lon_a_rad = math.radians(float(lon_a_deg))
    lat_b_rad = math.radians(float(lat_b_deg))
    lon_b_rad = math.radians(float(lon_b_deg))
    dlat_rad = lat_b_rad - lat_a_rad
    dlon_rad = lon_b_rad - lon_a_rad
    haversine = (
        math.sin(dlat_rad / 2.0) ** 2
        + math.cos(lat_a_rad) * math.cos(lat_b_rad) * math.sin(dlon_rad / 2.0) ** 2
    )
    haversine = min(1.0, max(0.0, haversine))
    distance_m = EARTH_RADIUS_M * 2.0 * math.atan2(
        math.sqrt(haversine),
        math.sqrt(1.0 - haversine),
    )
    return float(distance_m / METERS_PER_NM)
