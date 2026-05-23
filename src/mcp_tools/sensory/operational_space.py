from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from mcp_tools.scenario_manager.precompute_artifact import METERS_PER_NM
from simap.nlp_colloc.tactical.models import PathWaypoint


MaskName = Literal["north", "south"]

NORTH_MASK_FIXES = ("TTT", "WLLTR", "PRX")
SOUTH_MASK_FIXES = ("TTT", "BGTOE", "WAITT")
CLUSTER_TO_MASK: dict[str, MaskName] = {
    "NE": "north",
    "NW": "north",
    "SE": "south",
    "SW": "south",
}

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class OperationalSpaceMask:
    name: MaskName
    boundary_fix_identifiers: tuple[str, str, str]
    vertices: tuple[tuple[float, float], ...]

    def contains(self, lat: float, lon: float) -> bool:
        return point_in_polygon(float(lat), float(lon), self.vertices)

    def grid_points(self, spacing_nm: float) -> list[tuple[float, float]]:
        spacing_m = float(spacing_nm) * METERS_PER_NM
        if not math.isfinite(spacing_m) or spacing_m <= 0.0:
            raise ValueError("grid_spacing_nm must be finite and positive")

        lat0 = sum(lat for lat, _lon in self.vertices) / len(self.vertices)
        lon0 = sum(lon for _lat, lon in self.vertices) / len(self.vertices)
        projected = [_project_latlon(lat, lon, lat0=lat0, lon0=lon0) for lat, lon in self.vertices]
        min_x = min(x for x, _y in projected)
        max_x = max(x for x, _y in projected)
        min_y = min(y for _x, y in projected)
        max_y = max(y for _x, y in projected)

        points: list[tuple[float, float]] = []
        y = min_y
        while y <= max_y + 1e-9:
            x = min_x
            while x <= max_x + 1e-9:
                lat, lon = _latlon_from_xy(x, y, lat0=lat0, lon0=lon0)
                if self.contains(lat, lon):
                    points.append((lat, lon))
                x += spacing_m
            y += spacing_m
        return points


def mask_for_cluster(cluster: str, fix_catalog: dict[str, PathWaypoint]) -> OperationalSpaceMask:
    normalized_cluster = str(cluster).strip().upper()
    try:
        mask_name = CLUSTER_TO_MASK[normalized_cluster]
    except KeyError as exc:
        raise ValueError(f"unsupported arrival_cluster={cluster!r}") from exc
    return build_mask(mask_name, fix_catalog)


def build_mask(name: MaskName, fix_catalog: dict[str, PathWaypoint]) -> OperationalSpaceMask:
    boundary = NORTH_MASK_FIXES if name == "north" else SOUTH_MASK_FIXES
    vertices: list[tuple[float, float]] = []
    for identifier in boundary:
        waypoint = fix_catalog.get(identifier)
        if waypoint is None:
            raise ValueError(f"operational mask requires fix {identifier}")
        vertices.append((float(waypoint.lat_deg), float(waypoint.lon_deg)))
    return OperationalSpaceMask(
        name=name,
        boundary_fix_identifiers=boundary,
        vertices=tuple(vertices),
    )


def point_in_polygon(
    lat: float,
    lon: float,
    polygon_latlon: tuple[tuple[float, float], ...],
) -> bool:
    lat0 = sum(vertex_lat for vertex_lat, _vertex_lon in polygon_latlon) / len(polygon_latlon)
    lon0 = sum(vertex_lon for _vertex_lat, vertex_lon in polygon_latlon) / len(polygon_latlon)
    point = _project_latlon(lat, lon, lat0=lat0, lon0=lon0)
    polygon = tuple(_project_latlon(v_lat, v_lon, lat0=lat0, lon0=lon0) for v_lat, v_lon in polygon_latlon)
    if _point_on_boundary(point, polygon):
        return True

    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            intersection_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x <= intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _point_on_boundary(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    return any(
        _distance_to_segment_m(point, polygon[index - 1], polygon[index]) <= 30.0
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
    return float(math.hypot(px - (sx + t * dx), py - (sy + t * dy)))


def _project_latlon(lat: float, lon: float, *, lat0: float, lon0: float) -> tuple[float, float]:
    lat0_rad = math.radians(lat0)
    x = EARTH_RADIUS_M * math.cos(lat0_rad) * math.radians(float(lon) - lon0)
    y = EARTH_RADIUS_M * math.radians(float(lat) - lat0)
    return float(x), float(y)


def _latlon_from_xy(x: float, y: float, *, lat0: float, lon0: float) -> tuple[float, float]:
    lat0_rad = math.radians(lat0)
    lat = lat0 + math.degrees(float(y) / EARTH_RADIUS_M)
    lon = lon0 + math.degrees(float(x) / (EARTH_RADIUS_M * math.cos(lat0_rad)))
    return float(lat), float(lon)
