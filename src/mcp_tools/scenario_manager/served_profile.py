from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

from openap import aero

from mcp_tools.scenario_manager.precompute_artifact import (
    DEFAULT_FMS_DT_S,
    METERS_PER_NM,
    SeedState,
    _build_request,
    _upstream_identifier_for_route,
)
from simap.fms import ATCSpeedSegment
from simap.fms_bichannel import FMSBiChannelState
from simap.openap_adapter import openap_dT
from simap.units import kts_to_mps


FALLBACK_SPEED_USING_TWO_TRAJECTORY_POINTS = False


@dataclass(frozen=True)
class ServedSpeedAdvisory:
    s_m: float
    cas_kts: float
    lat: float | None = None
    lon: float | None = None

    @property
    def station_nm_to_runway(self) -> float:
        return float(self.s_m / METERS_PER_NM)

    @property
    def atc_segment(self) -> ATCSpeedSegment:
        return ATCSpeedSegment(s_from_m=self.s_m, cas_mps=kts_to_mps(self.cas_kts))

    def to_payload(self) -> dict[str, float | None]:
        return {
            "s_m": self.s_m,
            "station_nm_to_runway": self.station_nm_to_runway,
            "cas_kts": self.cas_kts,
            "lat": self.lat,
            "lon": self.lon,
        }


@dataclass(frozen=True)
class ServedFMSContext:
    arrival: dict[str, Any]
    route: list[str | tuple[float, float]]
    seed: SeedState
    fms_request: Any
    initial_state: FMSBiChannelState
    speed_advisories: tuple[ServedSpeedAdvisory, ...]

    @property
    def initial_ground_speed_mps(self) -> float:
        speed = float(getattr(self.initial_state, "v_tas_mps", 0.0) or 0.0)
        return speed if math.isfinite(speed) and speed > 0.0 else max(float(self.seed.ground_speed_mps), 1.0)


def build_served_fms_context(
    arrival: dict[str, Any],
    fixes_path: Path,
    *,
    route: list[str | tuple[float, float]] | None = None,
    fms_dt_s: float = DEFAULT_FMS_DT_S,
    include_speed_advisories: bool = True,
) -> ServedFMSContext:
    selected_route = arrival_lateral_path(arrival) if route is None else route
    seed = seed_from_served_arrival(arrival)
    fms_request, initial_state = _build_request(
        route=selected_route,
        upstream_identifier=_upstream_identifier_for_route(selected_route),
        seed=seed,
        fixes_csv=fixes_path,
        fms_dt_s=fms_dt_s,
    )
    speed_advisories = active_speed_advisories(arrival)
    if include_speed_advisories and speed_advisories:
        fms_request = replace(
            fms_request,
            atc_speed_segments=tuple(advisory.atc_segment for advisory in speed_advisories),
        )
    return ServedFMSContext(
        arrival=arrival,
        route=selected_route,
        seed=seed,
        fms_request=fms_request,
        initial_state=initial_state,
        speed_advisories=tuple(speed_advisories),
    )


def arrival_lateral_path(arrival: dict[str, Any]) -> list[str | tuple[float, float]]:
    base_route = arrival.get("base_route")
    if not isinstance(base_route, dict):
        raise ValueError("arrival has missing or malformed base_route")
    raw_route = base_route.get("lateral_path")
    if not isinstance(raw_route, list) or len(raw_route) < 2:
        raise ValueError("arrival has missing or malformed base_route.lateral_path")
    return [route_token(token, index) for index, token in enumerate(raw_route)]


def route_token(token: Any, index: int) -> str | tuple[float, float]:
    if isinstance(token, str):
        text = token.strip().upper()
        if not text:
            raise ValueError(f"base_route.lateral_path[{index}] is empty")
        return text
    if isinstance(token, (list, tuple)) and len(token) == 2:
        lat = finite_lat(token[0], f"base_route.lateral_path[{index}].lat")
        lon = finite_lon(token[1], f"base_route.lateral_path[{index}].lon")
        return (lat, lon)
    raise ValueError(f"base_route.lateral_path[{index}] must be a fix identifier or [lat, lon]")


def seed_from_served_arrival(arrival: dict[str, Any]) -> SeedState:
    columns = columns_for(arrival)
    points = arrival.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("arrival requires at least one trajectory point to seed SIMAP")
    first = trajectory_point(points[0], "points[0]")
    time_index = column_index(columns, "time")
    lat_index = column_index(columns, "lat")
    lon_index = column_index(columns, "lon")
    altitude_index = column_index(columns, "geoaltitude_m")

    time_s = finite_number(first[time_index], "points[0].time")
    lat = finite_lat(first[lat_index], "points[0].lat")
    lon = finite_lon(first[lon_index], "points[0].lon")
    altitude_m = finite_number(first[altitude_index], "points[0].geoaltitude_m")
    cas_mps = initial_cas_mps(arrival)
    if cas_mps is None and not FALLBACK_SPEED_USING_TWO_TRAJECTORY_POINTS:
        raise ValueError(
            "arrival requires cas_profile.points[0].cas_kts to seed SIMAP; "
            "two-point trajectory speed fallback is disabled"
        )
    ground_speed_mps = (
        float(aero.cas2tas(cas_mps, altitude_m, dT=openap_dT(0.0)))
        if cas_mps is not None
        else fallback_initial_ground_speed_mps(arrival)
    )
    if not math.isfinite(ground_speed_mps) or ground_speed_mps <= 1.0:
        raise ValueError("arrival initial trajectory points imply invalid ground speed")

    return SeedState(
        time_s=int(round(time_s)),
        lat_deg=lat,
        lon_deg=lon,
        geoaltitude_m=altitude_m,
        heading_deg=None,
        ground_speed_mps=float(ground_speed_mps),
        cas_mps=cas_mps,
    )


def initial_cas_mps(arrival: dict[str, Any]) -> float | None:
    cas_profile = arrival.get("cas_profile")
    if not isinstance(cas_profile, dict):
        return None
    columns = cas_profile.get("columns")
    points = cas_profile.get("points")
    if not isinstance(columns, list) or not isinstance(points, list) or not points:
        return None
    cas_index = column_index([str(column) for column in columns], "cas_kts")
    first_cas_point = points[0]
    if not isinstance(first_cas_point, list) or len(first_cas_point) <= cas_index:
        return None
    cas_kts = finite_number(first_cas_point[cas_index], "cas_profile.points[0].cas_kts")
    if cas_kts <= 0.0:
        return None
    return kts_to_mps(cas_kts)


def fallback_initial_ground_speed_mps(arrival: dict[str, Any]) -> float:
    columns = columns_for(arrival)
    points = arrival.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("arrival needs cas_profile or at least two trajectory points")
    first = trajectory_point(points[0], "points[0]")
    second = trajectory_point(points[1], "points[1]")
    time_index = column_index(columns, "time")
    lat_index = column_index(columns, "lat")
    lon_index = column_index(columns, "lon")

    first_time = finite_number(first[time_index], "points[0].time")
    second_time = finite_number(second[time_index], "points[1].time")
    dt_s = abs(second_time - first_time)
    if dt_s <= 0.0:
        raise ValueError("arrival initial trajectory points must have distinct times")

    distance_m = latlon_distance_m(
        finite_lat(first[lat_index], "points[0].lat"),
        finite_lon(first[lon_index], "points[0].lon"),
        finite_lat(second[lat_index], "points[1].lat"),
        finite_lon(second[lon_index], "points[1].lon"),
    )
    ground_speed_mps = distance_m / dt_s
    if not math.isfinite(ground_speed_mps) or ground_speed_mps <= 1.0:
        raise ValueError("arrival initial trajectory points imply invalid ground speed")
    return float(ground_speed_mps)


def active_speed_advisories(arrival: dict[str, Any]) -> tuple[ServedSpeedAdvisory, ...]:
    raw = None
    speed_intervention = arrival.get("speed_intervention")
    if isinstance(speed_intervention, dict):
        raw = speed_intervention.get("advisories")
    if raw is None:
        base_route = arrival.get("base_route")
        if isinstance(base_route, dict):
            raw = base_route.get("speed_advisories")
    if not isinstance(raw, list):
        return ()

    advisories: list[ServedSpeedAdvisory] = []
    for index, advisory in enumerate(raw):
        if not isinstance(advisory, dict):
            raise ValueError(f"speed_intervention.advisories[{index}] must be an object")
        s_m = finite_number(advisory.get("s_m"), f"speed_intervention.advisories[{index}].s_m")
        if s_m < 0.0:
            raise ValueError(f"speed_intervention.advisories[{index}].s_m must be nonnegative")
        cas_kts = finite_number(advisory.get("cas_kts"), f"speed_intervention.advisories[{index}].cas_kts")
        if cas_kts <= 0.0:
            raise ValueError(f"speed_intervention.advisories[{index}].cas_kts must be positive")
        lat = (
            finite_lat(advisory.get("lat"), f"speed_intervention.advisories[{index}].lat")
            if advisory.get("lat") is not None
            else None
        )
        lon = (
            finite_lon(advisory.get("lon"), f"speed_intervention.advisories[{index}].lon")
            if advisory.get("lon") is not None
            else None
        )
        advisories.append(ServedSpeedAdvisory(s_m=s_m, cas_kts=cas_kts, lat=lat, lon=lon))
    return tuple(sorted(advisories, key=lambda item: item.s_m, reverse=True))


def merge_speed_advisories(
    existing: tuple[ServedSpeedAdvisory, ...],
    requested: list[ServedSpeedAdvisory],
) -> list[ServedSpeedAdvisory]:
    by_station = {advisory.s_m: advisory for advisory in existing}
    for advisory in requested:
        by_station[advisory.s_m] = advisory
    return sorted(by_station.values(), key=lambda item: item.s_m, reverse=True)


def columns_for(arrival: dict[str, Any]) -> list[str]:
    columns = arrival.get("columns")
    if not isinstance(columns, list):
        raise ValueError("arrival has missing trajectory columns")
    return [str(column) for column in columns]


def column_index(columns: list[str], column: str) -> int:
    try:
        return columns.index(column)
    except ValueError as exc:
        raise ValueError(f"arrival trajectory columns must include {column}") from exc


def trajectory_point(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a trajectory point")
    return value


def finite_lat(value: Any, name: str) -> float:
    number = finite_number(value, name)
    if number < -90.0 or number > 90.0:
        raise ValueError(f"{name} must be between -90 and 90")
    return number


def finite_lon(value: Any, name: str) -> float:
    number = finite_number(value, name)
    if number < -180.0 or number > 180.0:
        raise ValueError(f"{name} must be between -180 and 180")
    return number


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def latlon_distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat0_rad = math.radians(0.5 * (lat_a + lat_b))
    dx_m = 6_371_000.0 * math.cos(lat0_rad) * math.radians(lon_b - lon_a)
    dy_m = 6_371_000.0 * math.radians(lat_b - lat_a)
    return float(math.hypot(dx_m, dy_m))
