from __future__ import annotations

import logging
import math
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any

from simap.nlp_colloc.tactical.models import PathWaypoint
from simap.nlp_colloc.tactical.path import resolve_lateral_path


DEFAULT_GATE_RADIUS_NM = 50.0
DEFAULT_CAPTURE_MARGIN_NM = 3.0
DEFAULT_RING_INNER_NM = DEFAULT_GATE_RADIUS_NM
DEFAULT_RING_OUTER_NM = DEFAULT_GATE_RADIUS_NM

EARTH_RADIUS_M = 6_371_000.0
METERS_PER_NM = 1_852.0

_BOUNDARY_TOLERANCE_M = 30.0

_LOGGER = logging.getLogger(__name__)


CAPTURE_POLYGONS: dict[str, tuple[tuple[float, float], ...]] = {
    "NE": (
        (33.395008, -96.733883),
        (33.242253, -96.597791),
        (33.225492, -96.504043),
        (33.211758, -96.315705),
        (33.416741, -95.998957),
        (33.496088, -96.246511),
        (33.500451, -96.623405),
    ),
    "NW": (
        (33.589004, -98.567377),
        (33.330273, -97.638325),
        (33.430069, -97.375542),
        (33.668505, -97.576094),
        (33.768399, -97.715811),
        (33.777637, -98.089916),
    ),
    "SW": (
        (32.886483, -97.389219),
        (32.937610, -97.065853),
        (33.058993, -97.129553),
        (33.049660, -97.311745),
    ),
    "SE": (
        (33.136981, -97.007966),
        (33.001641, -96.980745),
        (32.971946, -96.968920),
        (32.927077, -96.942191),
        (32.902865, -96.739534),
        (33.119977, -96.696736),
    ),
}


def detect_wait_atc_point(
    route: Sequence[str | tuple[float, float]],
    fix_catalog: Mapping[str, PathWaypoint],
    *,
    runway: str,
    ring_inner_nm: float | None = None,
    ring_outer_nm: float | None = None,
    gate_radius_nm: float = DEFAULT_GATE_RADIUS_NM,
    diagnostics: MutableSequence[str] | None = None,
    trace_label: str | None = None,
) -> dict[str, Any] | None:
    resolved_path = resolve_lateral_path(route, fix_catalog)
    runway_waypoint = _runway_waypoint(resolved_path.waypoints, fix_catalog, runway)
    if runway_waypoint is None:
        _diagnose(diagnostics, trace_label, f"missing runway waypoint for runway {runway!r}")
        return None

    inner_nm = float(gate_radius_nm if ring_inner_nm is None else ring_inner_nm)
    outer_nm = float(gate_radius_nm if ring_outer_nm is None else ring_outer_nm)
    if inner_nm > outer_nm:
        raise ValueError("ring_inner_nm must be less than or equal to ring_outer_nm")
    gate_radius_nm = float(gate_radius_nm)
    if gate_radius_nm <= 0.0:
        raise ValueError("gate_radius_nm must be positive")

    gate = _gate_crossing(
        resolved_path.waypoints,
        runway_waypoint=runway_waypoint,
        gate_radius_nm=gate_radius_nm,
    )
    if gate is None:
        route_fix = _first_route_fix(resolved_path.waypoints)
        if route_fix is None:
            _diagnose(diagnostics, trace_label, "route has no non-coordinate fixes to classify")
            return None
        gate = _GateCrossing(
            lat_deg=float(route_fix.lat_deg),
            lon_deg=float(route_fix.lon_deg),
            east_m=_east_m(route_fix.lat_deg, route_fix.lon_deg, runway_waypoint),
            north_m=_north_m(route_fix.lat_deg, runway_waypoint),
            fallback=True,
        )
        _diagnose(
            diagnostics,
            trace_label,
            "no 50 nm gate crossing found; classified from first route fix "
            f"{route_fix.identifier}",
        )

    cluster = _cluster_from_ne(gate.north_m, gate.east_m)
    selected = _last_fix_inside_capture_polygon(resolved_path.waypoints, cluster)
    if selected is None:
        identifiers = " > ".join(waypoint.identifier for waypoint in resolved_path.waypoints)
        _diagnose(
            diagnostics,
            trace_label,
            f"gate classified {cluster}, but no route fix was inside the {cluster} capture polygon: "
            f"{identifiers}",
        )
        return None

    waypoint, route_index = selected
    distance_nm = _distance_nm(
        waypoint.lat_deg,
        waypoint.lon_deg,
        runway_waypoint.lat_deg,
        runway_waypoint.lon_deg,
    )
    return {
        "source": "fix",
        "identifier": waypoint.identifier,
        "lat": float(waypoint.lat_deg),
        "lon": float(waypoint.lon_deg),
        "lateral_path_token": waypoint.identifier,
        "route_index": int(route_index),
        "distance_nm": float(distance_nm),
        "ring_inner_nm": inner_nm,
        "ring_outer_nm": outer_nm,
        "selection_method": "cluster_capture_polygon",
        "arrival_cluster": cluster,
        "gate_cluster": cluster,
        "gate_radius_nm": gate_radius_nm,
        "gate_lat": float(gate.lat_deg),
        "gate_lon": float(gate.lon_deg),
        "gate_classification_fallback": bool(gate.fallback),
        "capture_margin_nm": DEFAULT_CAPTURE_MARGIN_NM,
    }


class _GateCrossing:
    def __init__(
        self,
        *,
        lat_deg: float,
        lon_deg: float,
        east_m: float,
        north_m: float,
        fallback: bool = False,
    ) -> None:
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.east_m = east_m
        self.north_m = north_m
        self.fallback = fallback


def _gate_crossing(
    waypoints: Sequence[PathWaypoint],
    *,
    runway_waypoint: PathWaypoint,
    gate_radius_nm: float,
) -> _GateCrossing | None:
    radius_m = gate_radius_nm * METERS_PER_NM
    projected = [
        (
            _east_m(waypoint.lat_deg, waypoint.lon_deg, runway_waypoint),
            _north_m(waypoint.lat_deg, runway_waypoint),
        )
        for waypoint in waypoints
    ]
    for (east_a_m, north_a_m), (east_b_m, north_b_m) in zip(
        projected,
        projected[1:],
        strict=False,
    ):
        crossing = _segment_circle_crossing(
            east_a_m,
            north_a_m,
            east_b_m,
            north_b_m,
            radius_m,
        )
        if crossing is None:
            continue
        east_m, north_m = crossing
        lat_deg, lon_deg = _latlon_from_ne(east_m, north_m, runway_waypoint)
        return _GateCrossing(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            east_m=east_m,
            north_m=north_m,
            fallback=False,
        )
    return None


def _segment_circle_crossing(
    east_a_m: float,
    north_a_m: float,
    east_b_m: float,
    north_b_m: float,
    radius_m: float,
) -> tuple[float, float] | None:
    dx_m = east_b_m - east_a_m
    dy_m = north_b_m - north_a_m
    a = dx_m * dx_m + dy_m * dy_m
    if a <= 0.0:
        return None
    b = 2.0 * (east_a_m * dx_m + north_a_m * dy_m)
    c = east_a_m * east_a_m + north_a_m * north_a_m - radius_m * radius_m
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    sqrt_discriminant = math.sqrt(max(0.0, discriminant))
    candidates = [
        (-b - sqrt_discriminant) / (2.0 * a),
        (-b + sqrt_discriminant) / (2.0 * a),
    ]
    valid = [t for t in candidates if 0.0 <= t <= 1.0]
    if not valid:
        return None
    t = min(valid)
    return east_a_m + t * dx_m, north_a_m + t * dy_m


def _cluster_from_ne(north_m: float, east_m: float) -> str:
    if north_m >= 0.0 and east_m >= 0.0:
        return "NE"
    if north_m >= 0.0 and east_m < 0.0:
        return "NW"
    if north_m < 0.0 and east_m >= 0.0:
        return "SE"
    return "SW"


def _last_fix_inside_capture_polygon(
    waypoints: Sequence[PathWaypoint],
    cluster: str,
) -> tuple[PathWaypoint, int] | None:
    polygon = CAPTURE_POLYGONS[cluster]
    selected: tuple[PathWaypoint, int] | None = None
    for index, waypoint in enumerate(waypoints):
        if not _is_route_fix_candidate(waypoint):
            continue
        if _point_in_polygon(waypoint.lat_deg, waypoint.lon_deg, polygon):
            selected = waypoint, index
    return selected


def _point_in_polygon(
    lat_deg: float,
    lon_deg: float,
    polygon_latlon: Sequence[tuple[float, float]],
) -> bool:
    points = _project_polygon(polygon_latlon)
    lat0_deg = sum(lat for lat, _lon in polygon_latlon) / len(polygon_latlon)
    lon0_deg = sum(lon for _lat, lon in polygon_latlon) / len(polygon_latlon)
    point = _project_latlon(lat_deg, lon_deg, lat0_deg=lat0_deg, lon0_deg=lon0_deg)
    if _point_on_polygon_boundary(point, points):
        return True

    inside = False
    x, y = point
    previous_x, previous_y = points[-1]
    for current_x, current_y in points:
        if (current_y > y) != (previous_y > y):
            intersection_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x <= intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _point_on_polygon_boundary(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    return any(
        _distance_to_segment_m(point, polygon[index - 1], polygon[index]) <= _BOUNDARY_TOLERANCE_M
        for index in range(len(polygon))
    )


def _distance_to_segment_m(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return float(math.hypot(px - sx, py - sy))
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return float(math.hypot(px - closest_x, py - closest_y))


def _project_polygon(
    polygon_latlon: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    lat0_deg = sum(lat for lat, _lon in polygon_latlon) / len(polygon_latlon)
    lon0_deg = sum(lon for _lat, lon in polygon_latlon) / len(polygon_latlon)
    return tuple(
        _project_latlon(lat_deg, lon_deg, lat0_deg=lat0_deg, lon0_deg=lon0_deg)
        for lat_deg, lon_deg in polygon_latlon
    )


def _project_latlon(
    lat_deg: float,
    lon_deg: float,
    *,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[float, float]:
    lat0_rad = math.radians(lat0_deg)
    east_m = EARTH_RADIUS_M * math.cos(lat0_rad) * math.radians(lon_deg - lon0_deg)
    north_m = EARTH_RADIUS_M * math.radians(lat_deg - lat0_deg)
    return float(east_m), float(north_m)


def _latlon_from_ne(
    east_m: float,
    north_m: float,
    runway_waypoint: PathWaypoint,
) -> tuple[float, float]:
    lat0_rad = math.radians(float(runway_waypoint.lat_deg))
    lat_deg = float(runway_waypoint.lat_deg) + math.degrees(north_m / EARTH_RADIUS_M)
    lon_deg = float(runway_waypoint.lon_deg) + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(lat0_rad)))
    return lat_deg, lon_deg


def _east_m(lat_deg: float, lon_deg: float, runway_waypoint: PathWaypoint) -> float:
    lat0_rad = math.radians(float(runway_waypoint.lat_deg))
    return float(EARTH_RADIUS_M * math.cos(lat0_rad) * math.radians(lon_deg - runway_waypoint.lon_deg))


def _north_m(lat_deg: float, runway_waypoint: PathWaypoint) -> float:
    return float(EARTH_RADIUS_M * math.radians(lat_deg - runway_waypoint.lat_deg))


def _first_route_fix(waypoints: Sequence[PathWaypoint]) -> PathWaypoint | None:
    for waypoint in waypoints:
        if _is_route_fix_candidate(waypoint):
            return waypoint
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


def _diagnose(
    diagnostics: MutableSequence[str] | None,
    trace_label: str | None,
    message: str,
) -> None:
    full_message = f"{trace_label}: {message}" if trace_label else message
    if diagnostics is not None:
        diagnostics.append(full_message)
        return
    _LOGGER.warning(full_message)
