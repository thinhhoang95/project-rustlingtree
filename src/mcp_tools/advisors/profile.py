from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from openap import aero

from mcp_tools.advisors.models import AdvisoryFlight
from mcp_tools.scenario_manager.models import project_root
from simap.fms import ATCSpeedSegmentInput, FMSRequest, FMSResult, plan_fms_descent
from simap.nlp_colloc.tactical.builder import build_tactical_plan_request
from simap.nlp_colloc.tactical.models import TacticalCommand, TacticalCondition
from simap.openap_adapter import openap_dT
from simap.path_geometry import EARTH_RADIUS_M, ReferencePath
from simap.units import kts_to_mps, m_to_ft, mps_to_kts

METERS_PER_NM = 1_852.0
DEFAULT_FIXES_PATH = Path("data/kdfw_procs/airport_related_fixes.csv")
DEFAULT_FMS_DT_S = 2.0
DEFAULT_TOD_TOLERANCE_M = 25.0
DEFAULT_MAX_TOD_ITERATIONS = 24


class ArrivalScheduleProvider(Protocol):
    def arrival_schedule(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ArrivalProfile:
    arrival: dict[str, Any]
    identity: AdvisoryFlight
    request: FMSRequest
    initial_ground_speed_mps: float


@dataclass(frozen=True)
class PlannedProfile:
    success: bool
    message: str
    total_time_s: float
    pre_tod_ground_speed_mps: float
    result: FMSResult


@dataclass(frozen=True)
class ProfilePlanner:
    fms_dt_s: float = DEFAULT_FMS_DT_S
    tod_tolerance_m: float = DEFAULT_TOD_TOLERANCE_M
    max_tod_iterations: int = DEFAULT_MAX_TOD_ITERATIONS

    def build(self, arrival: dict[str, Any], fixes_path: Path) -> ArrivalProfile:
        route, upstream_identifier = _route_from_arrival(arrival)
        first_point = _first_point(arrival)
        altitude_m = _point_value(arrival, first_point, "geoaltitude_m")
        initial_cas_mps = _initial_cas_mps(arrival, altitude_m=altitude_m)
        command = TacticalCommand(
            lateral_path=route,
            upstream=TacticalCondition(
                fix_identifier=upstream_identifier,
                cas_kts=max(80.0, mps_to_kts(initial_cas_mps)),
                altitude_ft=m_to_ft(altitude_m),
            ),
            altitude_constraints=(),
        )
        bundle = build_tactical_plan_request(command, fixes_csv=fixes_path)
        request = FMSRequest.from_coupled_request(
            bundle.request,
            start_s_m=float(bundle.request.reference_path.total_length_m),
            dt_s=self.fms_dt_s,
        )
        initial_ground_speed_mps = _ground_speed_from_cas(
            cas_mps=request.start_cas_mps,
            altitude_m=request.start_h_m,
        )
        return ArrivalProfile(
            arrival=arrival,
            identity=_identity(arrival),
            request=request,
            initial_ground_speed_mps=initial_ground_speed_mps,
        )

    def plan(
        self,
        profile: ArrivalProfile,
        *,
        extra_distance_m: float = 0.0,
        atc_speed_segments: tuple[ATCSpeedSegmentInput, ...] = (),
    ) -> PlannedProfile:
        extra_distance_m = _finite_nonnegative(extra_distance_m, "extra_distance_m")
        request = profile.request
        if extra_distance_m > 0.0:
            reference_path = extend_reference_path(request.reference_path, extra_distance_m)
            request = replace(
                request,
                reference_path=reference_path,
                start_s_m=float(request.start_s_m + extra_distance_m),
            )
        if atc_speed_segments:
            request = replace(request, atc_speed_segments=atc_speed_segments)

        result = plan_fms_descent(
            request,
            tod_tolerance_m=self.tod_tolerance_m,
            max_tod_iterations=self.max_tod_iterations,
        )
        return _planned_profile(result, initial_ground_speed_mps=profile.initial_ground_speed_mps)


def resolve_fixes_path(manager: object) -> Path:
    config = getattr(manager, "config", None)
    fixes_path = getattr(config, "fixes_path", None)
    if fixes_path is None:
        return project_root() / DEFAULT_FIXES_PATH
    return Path(fixes_path)


def arrivals_for(manager: ArrivalScheduleProvider, flight_id: str | None = None) -> list[dict[str, Any]]:
    arrivals = manager.arrival_schedule()
    if flight_id is None:
        return arrivals
    selected = [arrival for arrival in arrivals if str(arrival.get("flight_id", "")) == str(flight_id)]
    if not selected:
        raise ValueError(f"unknown arrival flight_id={flight_id}")
    return selected


def extend_reference_path(reference_path: ReferencePath, extra_distance_m: float) -> ReferencePath:
    extra_distance_m = _finite_nonnegative(extra_distance_m, "extra_distance_m")
    if extra_distance_m == 0.0:
        return reference_path

    tangent = reference_path.tangent_hat(reference_path.total_length_m)
    virtual_east_m = float(reference_path.east_m[0] - tangent[0] * extra_distance_m)
    virtual_north_m = float(reference_path.north_m[0] - tangent[1] * extra_distance_m)
    virtual_lat_deg, virtual_lon_deg = reference_path.latlon_from_ne(virtual_east_m, virtual_north_m)
    s_from_start_m = np.concatenate(
        (
            np.asarray([0.0], dtype=float),
            np.asarray(reference_path.s_from_start_m, dtype=float) + extra_distance_m,
        )
    )
    total_length_m = float(reference_path.total_length_m + extra_distance_m)
    return ReferencePath(
        origin_lat_deg=reference_path.origin_lat_deg,
        origin_lon_deg=reference_path.origin_lon_deg,
        waypoint_lat_deg=np.concatenate(
            (np.asarray([virtual_lat_deg], dtype=float), np.asarray(reference_path.waypoint_lat_deg, dtype=float))
        ),
        waypoint_lon_deg=np.concatenate(
            (np.asarray([virtual_lon_deg], dtype=float), np.asarray(reference_path.waypoint_lon_deg, dtype=float))
        ),
        s_from_start_m=s_from_start_m,
        s_m=total_length_m - s_from_start_m,
        east_m=np.concatenate(
            (
                np.asarray([virtual_east_m], dtype=float),
                np.asarray(reference_path.east_m, dtype=float),
            )
        ),
        north_m=np.concatenate(
            (np.asarray([virtual_north_m], dtype=float), np.asarray(reference_path.north_m, dtype=float))
        ),
        lat_deg=np.concatenate(
            (
                np.asarray([virtual_lat_deg], dtype=float),
                np.asarray(reference_path.lat_deg, dtype=float),
            )
        ),
        lon_deg=np.concatenate(
            (
                np.asarray([virtual_lon_deg], dtype=float),
                np.asarray(reference_path.lon_deg, dtype=float),
            )
        ),
        track_rad=np.concatenate(
            (
                np.asarray([reference_path.track_rad[0]], dtype=float),
                np.asarray(reference_path.track_rad, dtype=float),
            )
        ),
        curvature_inv_m=np.concatenate(
            (np.asarray([0.0], dtype=float), np.asarray(reference_path.curvature_inv_m, dtype=float))
        ),
        total_length_m=total_length_m,
    )


def _planned_profile(result: FMSResult, *, initial_ground_speed_mps: float) -> PlannedProfile:
    total_time_s = float(result.t_s[-1]) if len(result.t_s) else 0.0
    level_distance_m = float(getattr(result, "level_distance_m", 0.0) or 0.0)
    level_time_s = float(getattr(result, "level_time_s", 0.0) or 0.0)
    if level_distance_m > 0.0 and level_time_s > 0.0:
        pre_tod_ground_speed_mps = float(level_distance_m / level_time_s)
    elif len(result.ground_speed_mps):
        pre_tod_ground_speed_mps = float(result.ground_speed_mps[0])
    else:
        pre_tod_ground_speed_mps = float(initial_ground_speed_mps)
    if not math.isfinite(pre_tod_ground_speed_mps) or pre_tod_ground_speed_mps <= 0.0:
        pre_tod_ground_speed_mps = max(float(initial_ground_speed_mps), 1.0)
    return PlannedProfile(
        success=bool(result.success),
        message=str(result.message),
        total_time_s=total_time_s,
        pre_tod_ground_speed_mps=pre_tod_ground_speed_mps,
        result=result,
    )


def _identity(arrival: dict[str, Any]) -> AdvisoryFlight:
    return AdvisoryFlight(
        flight_number=str(arrival.get("callsign", "")),
        icao24=str(arrival.get("icao24", "")),
        flight_id=str(arrival.get("flight_id", "")),
        runway=str(arrival.get("runway", "")),
    )


def _route_from_arrival(arrival: dict[str, Any]) -> tuple[list[str | tuple[float, float]], str]:
    base_route = arrival.get("base_route")
    if not isinstance(base_route, dict):
        raise ValueError(f"{_flight_label(arrival)} has missing or malformed base_route")
    raw_route = base_route.get("lateral_path")
    if not isinstance(raw_route, list) or len(raw_route) < 2:
        raise ValueError(f"{_flight_label(arrival)} has missing or malformed base_route.lateral_path")
    route = [_route_token(token, index) for index, token in enumerate(raw_route, start=1)]
    upstream_identifier = base_route.get("upstream_identifier")
    if isinstance(upstream_identifier, str) and upstream_identifier.strip():
        upstream = upstream_identifier.strip().upper()
    else:
        first = route[0]
        upstream = "COORD01" if isinstance(first, tuple) else str(first).upper()
    return route, upstream


def _route_token(token: Any, index: int) -> str | tuple[float, float]:
    if isinstance(token, str):
        return token.strip().upper()
    if isinstance(token, (list, tuple)) and len(token) == 2:
        lat_deg = _finite_number(token[0], f"base_route.lateral_path[{index - 1}].lat")
        lon_deg = _finite_number(token[1], f"base_route.lateral_path[{index - 1}].lon")
        return (lat_deg, lon_deg)
    raise ValueError(f"base_route.lateral_path[{index - 1}] must be a fix identifier or [lat, lon]")


def _first_point(arrival: dict[str, Any]) -> list[int | float]:
    points = arrival.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"{_flight_label(arrival)} has missing or malformed points")
    point = points[0]
    if not isinstance(point, list):
        raise ValueError(f"{_flight_label(arrival)} has malformed points[0]")
    return point


def _point_value(arrival: dict[str, Any], point: list[int | float], column: str) -> float:
    columns = _columns(arrival)
    index = _column_index(columns, column, _flight_label(arrival))
    if len(point) <= index:
        raise ValueError(f"{_flight_label(arrival)} has malformed points[0].{column}")
    return _finite_number(point[index], f"points[0].{column}")


def _columns(arrival: dict[str, Any]) -> list[str]:
    columns = arrival.get("columns")
    if not isinstance(columns, list):
        raise ValueError(f"{_flight_label(arrival)} has missing or malformed columns")
    return [str(column) for column in columns]


def _column_index(columns: list[str], column: str, label: str) -> int:
    try:
        return columns.index(column)
    except ValueError as exc:
        raise ValueError(f"{label} has missing column {column}") from exc


def _initial_cas_mps(arrival: dict[str, Any], *, altitude_m: float) -> float:
    cas_profile = arrival.get("cas_profile")
    if isinstance(cas_profile, dict):
        columns = cas_profile.get("columns")
        points = cas_profile.get("points")
        if isinstance(columns, list) and isinstance(points, list) and points:
            cas_index = _column_index([str(column) for column in columns], "cas_kts", _flight_label(arrival))
            first_cas_point = points[0]
            if isinstance(first_cas_point, list) and len(first_cas_point) > cas_index:
                cas_kts = _finite_number(first_cas_point[cas_index], "cas_profile.points[0].cas_kts")
                if cas_kts > 0.0:
                    return kts_to_mps(cas_kts)

    ground_speed_mps = _fallback_initial_ground_speed_mps(arrival)
    return float(aero.tas2cas(ground_speed_mps, altitude_m, dT=openap_dT(0.0)))


def _fallback_initial_ground_speed_mps(arrival: dict[str, Any]) -> float:
    points = arrival.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError(f"{_flight_label(arrival)} needs cas_profile or at least two trajectory points")
    first = points[0]
    second = points[1]
    if not isinstance(first, list) or not isinstance(second, list):
        raise ValueError(f"{_flight_label(arrival)} has malformed trajectory points")
    columns = _columns(arrival)
    time_index = _column_index(columns, "time", _flight_label(arrival))
    lat_index = _column_index(columns, "lat", _flight_label(arrival))
    lon_index = _column_index(columns, "lon", _flight_label(arrival))
    minimum_length = max(time_index, lat_index, lon_index) + 1
    if len(first) < minimum_length or len(second) < minimum_length:
        raise ValueError(f"{_flight_label(arrival)} has malformed trajectory points")

    dt_s = abs(
        _finite_number(second[time_index], "points[1].time")
        - _finite_number(first[time_index], "points[0].time")
    )
    if dt_s <= 0.0:
        raise ValueError(f"{_flight_label(arrival)} has duplicate initial trajectory times")
    distance_m = _latlon_distance_m(
        _finite_number(first[lat_index], "points[0].lat"),
        _finite_number(first[lon_index], "points[0].lon"),
        _finite_number(second[lat_index], "points[1].lat"),
        _finite_number(second[lon_index], "points[1].lon"),
    )
    ground_speed_mps = distance_m / dt_s
    if not math.isfinite(ground_speed_mps) or ground_speed_mps <= 0.0:
        raise ValueError(f"{_flight_label(arrival)} has invalid fallback initial ground speed")
    return float(ground_speed_mps)


def _ground_speed_from_cas(*, cas_mps: float, altitude_m: float) -> float:
    return float(aero.cas2tas(cas_mps, altitude_m, dT=openap_dT(0.0)))


def _latlon_distance_m(lat_a_deg: float, lon_a_deg: float, lat_b_deg: float, lon_b_deg: float) -> float:
    lat0_rad = math.radians(0.5 * (lat_a_deg + lat_b_deg))
    dx_m = EARTH_RADIUS_M * math.cos(lat0_rad) * math.radians(lon_b_deg - lon_a_deg)
    dy_m = EARTH_RADIUS_M * math.radians(lat_b_deg - lat_a_deg)
    return float(math.hypot(dx_m, dy_m))


def _finite_nonnegative(value: float, name: str) -> float:
    value = _finite_number(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


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
