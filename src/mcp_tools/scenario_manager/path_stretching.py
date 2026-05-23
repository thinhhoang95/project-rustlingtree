from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import math
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

from mcp_tools.scenario_manager.precompute_artifact import (
    DEFAULT_ALTITUDE_TOLERANCE_M,
    DEFAULT_FMS_DT_S,
    DEFAULT_LATERAL_TOLERANCE_M,
    DEFAULT_MAX_TOD_ITERATIONS,
    DEFAULT_TOD_TOLERANCE_M,
    METERS_PER_NM,
    BaseRoute,
    FinalFixSelection,
    SeedState,
    _dedupe_consecutive_route_tokens,
    _default_lateral_guidance,
    _normalize_runway_identifier,
    _payload_from_result,
    _route_token_payload,
    _route_token_text,
    _upstream_identifier_for_route,
)
from mcp_tools.scenario_manager.served_profile import (
    ServedSpeedAdvisory,
    arrival_lateral_path as _shared_arrival_lateral_path,
    build_served_fms_context,
    seed_from_served_arrival,
)
from simap.fms_bichannel import FMSBiChannelRequest, plan_fms_bichannel
from simap.nlp_colloc.tactical.models import PathWaypoint
from simap.nlp_colloc.tactical.navdata import load_fix_catalog


class PathStretchHandleRequest(BaseModel):
    insert_after_index: int
    token_type: Literal["fix", "coordinate"]
    lat: float
    lon: float
    fix_identifier: str | None = None


class PathStretchRouteTokenRequest(BaseModel):
    token_type: Literal["fix", "coordinate"]
    lat: float
    lon: float
    fix_identifier: str | None = None


class PathStretchVectorAssistMetadata(BaseModel):
    variant: Literal["sandwiched_dogleg", "replaced_dogleg"]
    candidate_kind: Literal["identified", "free"]
    target_time_gain_s: float
    projected_segment_index: int
    lat: float
    lon: float
    fix_identifier: str | None = None


class PathStretchSimulationRequest(BaseModel):
    flight_id: str
    handles: list[PathStretchHandleRequest] = Field(default_factory=list)
    route: list[PathStretchRouteTokenRequest] | None = None
    vector_assist: PathStretchVectorAssistMetadata | None = None


class PathStretchSaveRequest(BaseModel):
    draft_id: str


@dataclass(frozen=True)
class _NormalizedHandle:
    insert_after_index: int
    token_type: Literal["fix", "coordinate"]
    lat: float
    lon: float
    fix_identifier: str | None = None

    @property
    def route_token(self) -> str | tuple[float, float]:
        if self.token_type == "fix":
            if not self.fix_identifier:
                raise ValueError("fix handle requires fix_identifier")
            return self.fix_identifier
        return (self.lat, self.lon)

    @property
    def display_token(self) -> str:
        if self.token_type == "fix" and self.fix_identifier:
            return self.fix_identifier
        return _coordinate_label(self.lat, self.lon)


@dataclass(frozen=True)
class _NormalizedRoutePoint:
    token_type: Literal["fix", "coordinate"]
    lat: float
    lon: float
    fix_identifier: str | None = None

    @property
    def route_token(self) -> str | tuple[float, float]:
        if self.token_type == "fix":
            if not self.fix_identifier:
                raise ValueError("fix route point requires fix_identifier")
            return self.fix_identifier
        return (self.lat, self.lon)


@dataclass(frozen=True)
class PathStretchRouteSimulation:
    artifact: Any
    payload: dict[str, Any]
    metrics: dict[str, float]
    old_route_tokens: list[str]
    new_route_tokens: list[str]
    simulation: dict[str, Any] | None


def simulate_path_stretch(
    manager: Any,
    request: PathStretchSimulationRequest,
) -> dict[str, Any]:
    """Answer: what trajectory results from editing this arrival's lateral route?"""
    flight_id = request.flight_id.strip()
    if not flight_id:
        raise ValueError("flight_id is required")

    arrival = _arrival_for(manager, flight_id)
    base_route = _arrival_lateral_path(arrival)
    fix_catalog = load_fix_catalog(manager.config.fixes_path)
    route_points: list[_NormalizedRoutePoint] | None = None
    if request.route is not None:
        route_points = _normalize_route_points(request.route, fix_catalog=fix_catalog)
        stretched_route = _dedupe_consecutive_route_tokens([point.route_token for point in route_points])
        handles: list[_NormalizedHandle] = []
    else:
        if not request.handles:
            raise ValueError("at least one path-stretch handle or edited route is required")
        handles = _normalize_handles(request.handles, route_length=len(base_route), fix_catalog=fix_catalog)
        stretched_route = _insert_handles(base_route, handles)

    if len(stretched_route) < 2:
        raise ValueError("path-stretch route requires at least two distinct points")
    if stretched_route == base_route:
        raise ValueError("path-stretch request did not change the lateral path")

    vector_assist_payload = _vector_assist_payload(request.vector_assist)
    _validate_vector_assist_attempt_limits(arrival, vector_assist_payload)
    route_simulation = simulate_path_stretch_route(
        manager,
        arrival,
        stretched_route,
        handles=handles,
        route_points=route_points,
        vector_assist=vector_assist_payload,
    )
    payload = route_simulation.payload
    metrics = route_simulation.metrics

    draft_id = f"path-stretch-{flight_id}-{uuid4().hex[:12]}"
    created_at_utc = _utc_now()
    diff_record = {
        "id": draft_id,
        "flight_id": flight_id,
        "created_at_utc": created_at_utc,
        "source": "path-stretching",
        "type": "path-stretch",
        "command": {
            "type": "path_stretch",
            "handles": [asdict(handle) for handle in handles],
            "route": [asdict(point) for point in route_points] if route_points is not None else None,
            "old_route": [_route_token_payload(token) for token in base_route],
            "new_route": [_route_token_payload(token) for token in stretched_route],
            "vector_assist": vector_assist_payload,
        },
        "overrides": payload,
        "base": {
            "route_type": arrival.get("route_type"),
            "fix_sequence": arrival.get("fix_sequence"),
            "fix_count": arrival.get("fix_count"),
            "base_route": arrival.get("base_route"),
            "final_fix": arrival.get("final_fix"),
            "baseline_final_fix": arrival.get("baseline_final_fix"),
            "simulation": arrival.get("simulation"),
        },
    }
    response = {
        "draft_id": draft_id,
        "flight_id": flight_id,
        "created_at_utc": created_at_utc,
        "trajectory": payload,
        "metrics": metrics,
        "old_route_tokens": route_simulation.old_route_tokens,
        "new_route_tokens": route_simulation.new_route_tokens,
        "simulation": route_simulation.simulation,
    }
    manager.path_stretch_drafts[draft_id] = {
        "flight_id": flight_id,
        "response": response,
        "diff_record": diff_record,
        "artifact": asdict(route_simulation.artifact),
    }
    return response


def simulate_path_stretch_route(
    manager: Any,
    arrival: dict[str, Any],
    stretched_route: list[str | tuple[float, float]],
    *,
    handles: list[_NormalizedHandle] | None = None,
    route_points: list[_NormalizedRoutePoint] | None = None,
    vector_assist: dict[str, Any] | None = None,
) -> PathStretchRouteSimulation:
    """Run SIMAP for a stretched route without creating a draft or mutating diffs."""
    base_route = _arrival_lateral_path(arrival)
    if len(stretched_route) < 2:
        raise ValueError("path-stretch route requires at least two distinct points")
    if stretched_route == base_route:
        raise ValueError("path-stretch request did not change the lateral path")

    context = build_served_fms_context(
        arrival,
        manager.config.fixes_path,
        route=stretched_route,
        fms_dt_s=DEFAULT_FMS_DT_S,
        include_speed_advisories=True,
    )
    seed = context.seed
    fms_request = context.fms_request
    initial_state = context.initial_state
    guidance = _default_lateral_guidance()
    result = plan_fms_bichannel(
        FMSBiChannelRequest(
            base_request=fms_request,
            guidance=guidance,
            initial_state=initial_state,
        ),
        tod_tolerance_m=DEFAULT_TOD_TOLERANCE_M,
        max_tod_iterations=DEFAULT_MAX_TOD_ITERATIONS,
    )

    wait_atc_point = _wait_atc_point_for_payload(arrival)
    artifact, payload = _payload_from_result(
        row=_arrival_row(arrival),
        seed=seed,
        wait_atc_point=wait_atc_point,
        base_route=_base_route_for_stretched_arrival(arrival, stretched_route),
        result=result,
        reference_path=getattr(fms_request, "reference_path", None),
        guidance=guidance,
        lateral_tolerance_m=float(arrival.get("lateral_tolerance_m", DEFAULT_LATERAL_TOLERANCE_M)),
        altitude_tolerance_m=float(arrival.get("altitude_tolerance_m", DEFAULT_ALTITUDE_TOLERANCE_M)),
    )
    _mark_payload_as_path_stretch(
        payload,
        arrival=arrival,
        handles=handles or [],
        route_points=route_points,
        speed_advisories=context.speed_advisories,
        vector_assist=vector_assist,
    )

    return PathStretchRouteSimulation(
        artifact=artifact,
        payload=payload,
        metrics=_metrics(old_arrival=arrival, new_arrival=payload),
        old_route_tokens=[_display_route_token(token) for token in base_route],
        new_route_tokens=[_display_route_token(token) for token in stretched_route],
        simulation=payload.get("simulation"),
    )


def save_path_stretch(
    manager: Any,
    flight_id: str,
    request: PathStretchSaveRequest,
) -> dict[str, Any]:
    """Answer: can this path-stretch draft become the active arrival trajectory?"""
    flight_id = flight_id.strip()
    draft = manager.path_stretch_drafts.get(request.draft_id)
    if draft is None:
        raise ValueError(f"unknown path-stretch draft_id={request.draft_id}")
    if draft["flight_id"] != flight_id:
        raise ValueError("path-stretch draft does not match requested flight_id")

    diff_record = dict(draft["diff_record"])
    command = diff_record.get("command")
    vector_assist = command.get("vector_assist") if isinstance(command, dict) else None
    _validate_vector_assist_attempt_limits(_arrival_for(manager, flight_id), vector_assist)
    manager.diff = [
        record
        for record in manager.diff
        if not (
            record.get("type") in {"path-stretch", "speed-intervention"}
            and str(record.get("flight_id", "")) == flight_id
        )
    ]
    manager.diff.append(diff_record)
    return {
        "diff": diff_record,
        "arrival": _arrival_for(manager, flight_id),
    }


def apply_path_stretch_diff(payload: dict[str, Any], diff: list[dict[str, Any]]) -> dict[str, Any]:
    flight_id = str(payload.get("flight_id", ""))
    applied = dict(payload)
    for record in diff:
        if record.get("type") not in {"path-stretch", "speed-intervention"} or str(record.get("flight_id", "")) != flight_id:
            continue
        overrides = record.get("overrides")
        if not isinstance(overrides, dict):
            continue
        for field in _TRAJECTORY_OVERRIDE_FIELDS:
            if field in overrides:
                applied[field] = overrides[field]
    return applied


_TRAJECTORY_OVERRIDE_FIELDS = {
    "route_type",
    "fix_sequence",
    "fix_count",
    "columns",
    "breakpoint_mask_bits",
    "points",
    "cas_profile",
    "lateral_breakpoint_times",
    "altitude_breakpoint_times",
    "first_time",
    "last_time",
    "raw_point_count",
    "compressed_point_count",
    "lateral_tolerance_m",
    "altitude_tolerance_m",
    "simulation",
    "final_fix",
    "baseline_final_fix",
    "base_route",
    "atc_wait_point",
    "wait_atc_point",
    "path_stretch",
    "speed_intervention",
}


def _arrival_for(manager: Any, flight_id: str) -> dict[str, Any]:
    selected = [
        arrival
        for arrival in manager.arrival_schedule()
        if str(arrival.get("flight_id", "")) == flight_id
    ]
    if not selected:
        raise ValueError(f"unknown arrival flight_id={flight_id}")
    return selected[0]


def _arrival_lateral_path(arrival: dict[str, Any]) -> list[str | tuple[float, float]]:
    return _shared_arrival_lateral_path(arrival)


def _route_token(token: Any, index: int) -> str | tuple[float, float]:
    if isinstance(token, str):
        text = token.strip().upper()
        if not text:
            raise ValueError(f"base_route.lateral_path[{index}] is empty")
        return text
    if isinstance(token, (list, tuple)) and len(token) == 2:
        lat = _finite_lat(token[0], f"base_route.lateral_path[{index}].lat")
        lon = _finite_lon(token[1], f"base_route.lateral_path[{index}].lon")
        return (lat, lon)
    raise ValueError(f"base_route.lateral_path[{index}] must be a fix identifier or [lat, lon]")


def _normalize_handles(
    handles: list[PathStretchHandleRequest],
    *,
    route_length: int,
    fix_catalog: dict[str, PathWaypoint],
) -> list[_NormalizedHandle]:
    normalized: list[_NormalizedHandle] = []
    for index, handle in enumerate(handles):
        insert_after_index = int(handle.insert_after_index)
        if insert_after_index < 0 or insert_after_index >= route_length - 1:
            raise ValueError(
                f"handles[{index}].insert_after_index must target an existing route segment"
            )
        lat = _finite_lat(handle.lat, f"handles[{index}].lat")
        lon = _finite_lon(handle.lon, f"handles[{index}].lon")
        if handle.token_type == "fix":
            fix_identifier = (handle.fix_identifier or "").strip().upper()
            if not fix_identifier:
                raise ValueError(f"handles[{index}].fix_identifier is required for fix handles")
            if fix_identifier not in fix_catalog:
                raise ValueError(f"unknown path-stretch fix: {fix_identifier}")
            normalized.append(
                _NormalizedHandle(
                    insert_after_index=insert_after_index,
                    token_type="fix",
                    lat=lat,
                    lon=lon,
                    fix_identifier=fix_identifier,
                )
            )
        else:
            normalized.append(
                _NormalizedHandle(
                    insert_after_index=insert_after_index,
                    token_type="coordinate",
                    lat=lat,
                    lon=lon,
                )
            )
    return normalized


def _normalize_route_points(
    route: list[PathStretchRouteTokenRequest],
    *,
    fix_catalog: dict[str, PathWaypoint],
) -> list[_NormalizedRoutePoint]:
    if len(route) < 2:
        raise ValueError("path-stretch route requires at least two points")

    normalized: list[_NormalizedRoutePoint] = []
    for index, point in enumerate(route):
        lat = _finite_lat(point.lat, f"route[{index}].lat")
        lon = _finite_lon(point.lon, f"route[{index}].lon")
        if point.token_type == "fix":
            fix_identifier = (point.fix_identifier or "").strip().upper()
            if not fix_identifier:
                raise ValueError(f"route[{index}].fix_identifier is required for fix route points")
            if fix_identifier not in fix_catalog:
                raise ValueError(f"unknown path-stretch fix: {fix_identifier}")
            normalized.append(
                _NormalizedRoutePoint(
                    token_type="fix",
                    lat=lat,
                    lon=lon,
                    fix_identifier=fix_identifier,
                )
            )
        else:
            normalized.append(
                _NormalizedRoutePoint(
                    token_type="coordinate",
                    lat=lat,
                    lon=lon,
                )
            )
    return normalized


def _insert_handles(
    base_route: list[str | tuple[float, float]],
    handles: list[_NormalizedHandle],
) -> list[str | tuple[float, float]]:
    handles_by_segment: dict[int, list[_NormalizedHandle]] = {}
    for handle in handles:
        handles_by_segment.setdefault(handle.insert_after_index, []).append(handle)

    route: list[str | tuple[float, float]] = []
    for index, token in enumerate(base_route):
        route.append(token)
        for handle in handles_by_segment.get(index, []):
            route.append(handle.route_token)
    return _dedupe_consecutive_route_tokens(route)


def _seed_from_arrival(arrival: dict[str, Any]) -> SeedState:
    return seed_from_served_arrival(arrival)


def _base_route_for_stretched_arrival(
    arrival: dict[str, Any],
    stretched_route: list[str | tuple[float, float]],
) -> BaseRoute:
    final_fix = _final_fix_selection(arrival)
    wait_atc_point = arrival.get("atc_wait_point") or arrival.get("wait_atc_point") or {}
    runway = _normalize_runway_identifier(
        arrival.get("runway") or arrival.get("base_route", {}).get("runway")
    )
    return BaseRoute(
        lateral_path=stretched_route,
        upstream_identifier=_upstream_identifier_for_route(stretched_route),
        runway_identifier=runway,
        final_fix=final_fix,
        atc_point=dict(wait_atc_point) if isinstance(wait_atc_point, dict) else {},
        target_final_fix_distance_nm=final_fix.distance_nm,
        final_fix_cross_track_tolerance_nm=abs(final_fix.cross_track_nm) or 0.15,
    )


def _wait_atc_point_for_payload(arrival: dict[str, Any]) -> dict[str, Any] | None:
    wait_atc_point = arrival.get("atc_wait_point") or arrival.get("wait_atc_point")
    if not isinstance(wait_atc_point, dict):
        return None
    payload = dict(wait_atc_point)
    payload.setdefault("arrival_cluster", "")
    return payload


def _final_fix_selection(arrival: dict[str, Any]) -> FinalFixSelection:
    raw_final_fix = arrival.get("final_fix") or arrival.get("baseline_final_fix")
    if not isinstance(raw_final_fix, dict):
        raw_final_fix = {}
    identifier = str(raw_final_fix.get("identifier") or "FINAL").strip().upper()
    lat = _finite_lat(raw_final_fix.get("lat", 0.0), "final_fix.lat")
    lon = _finite_lon(raw_final_fix.get("lon", 0.0), "final_fix.lon")
    raw_base_route = arrival.get("base_route")
    base_route = raw_base_route if isinstance(raw_base_route, dict) else {}
    return FinalFixSelection(
        waypoint=PathWaypoint(identifier=identifier, lat_deg=lat, lon_deg=lon),
        distance_nm=float(raw_final_fix.get("distance_nm") or 0.0),
        along_track_nm=float(raw_final_fix.get("along_track_nm") or 0.0),
        cross_track_nm=float(raw_final_fix.get("cross_track_nm") or 0.0),
        runway_true_heading_deg=float(base_route.get("runway_true_heading_deg") or 0.0),
    )


def _mark_payload_as_path_stretch(
    payload: dict[str, Any],
    *,
    arrival: dict[str, Any],
    handles: list[_NormalizedHandle],
    route_points: list[_NormalizedRoutePoint] | None,
    speed_advisories: tuple[ServedSpeedAdvisory, ...] = (),
    vector_assist: dict[str, Any] | None = None,
) -> None:
    speed_advisory_payload = [advisory.to_payload() for advisory in speed_advisories]
    base_route = payload.get("base_route")
    if isinstance(base_route, dict):
        base_route["type"] = "path-stretch"
        base_route["selection_method"] = "interactive_path_stretch"
        base_route["handles"] = [asdict(handle) for handle in handles]
        if route_points is not None:
            base_route["route_points"] = [asdict(point) for point in route_points]
        if speed_advisory_payload:
            base_route["speed_advisories"] = speed_advisory_payload
    payload["route_type"] = "path-stretch"
    payload["final_fix"] = payload.get("baseline_final_fix")
    payload["path_stretch"] = {
        "handles": [asdict(handle) for handle in handles],
        "handle_count": len(handles),
        "route_points": [asdict(point) for point in route_points] if route_points is not None else None,
        "route_point_count": len(route_points) if route_points is not None else None,
    }
    attempts = _vector_assist_attempts_after(arrival, vector_assist)
    if attempts:
        replaced_count = sum(1 for attempt in attempts if attempt.get("variant") == "replaced_dogleg")
        payload["path_stretch"]["vector_assist"] = {
            "attempts": attempts,
            "attempt_count": len(attempts),
            "replaced_dogleg_count": replaced_count,
        }
    if speed_advisory_payload:
        payload["speed_intervention"] = {
            "advisories": speed_advisory_payload,
            "advisory_count": len(speed_advisory_payload),
        }


def _vector_assist_payload(metadata: PathStretchVectorAssistMetadata | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    payload = metadata.model_dump()
    _finite_number(payload["target_time_gain_s"], "vector_assist.target_time_gain_s")
    _finite_lat(payload["lat"], "vector_assist.lat")
    _finite_lon(payload["lon"], "vector_assist.lon")
    segment_index = int(payload["projected_segment_index"])
    if segment_index < 0:
        raise ValueError("vector_assist.projected_segment_index must be nonnegative")
    payload["projected_segment_index"] = segment_index
    if payload.get("fix_identifier") is not None:
        fix_identifier = str(payload["fix_identifier"]).strip().upper()
        payload["fix_identifier"] = fix_identifier or None
    return payload


def _existing_vector_assist_attempts(arrival: dict[str, Any]) -> list[dict[str, Any]]:
    path_stretch = arrival.get("path_stretch")
    if not isinstance(path_stretch, dict):
        return []
    vector_assist = path_stretch.get("vector_assist")
    if not isinstance(vector_assist, dict):
        return []
    attempts = vector_assist.get("attempts")
    if not isinstance(attempts, list):
        return []
    return [dict(attempt) for attempt in attempts if isinstance(attempt, dict)]


def _validate_vector_assist_attempt_limits(
    arrival: dict[str, Any],
    vector_assist: dict[str, Any] | None,
) -> None:
    if vector_assist is None:
        return
    attempts = _existing_vector_assist_attempts(arrival)
    if len(attempts) >= 2:
        raise ValueError("maximum of 2 vector-assist attempts per flight has already been reached")
    if vector_assist.get("variant") == "replaced_dogleg" and any(
        attempt.get("variant") == "replaced_dogleg" for attempt in attempts
    ):
        raise ValueError("only one replaced dogleg vector-assist attempt is allowed per flight")


def _vector_assist_attempts_after(
    arrival: dict[str, Any],
    vector_assist: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    attempts = _existing_vector_assist_attempts(arrival)
    if vector_assist is not None:
        attempts.append(dict(vector_assist))
    return attempts


def _metrics(*, old_arrival: dict[str, Any], new_arrival: dict[str, Any]) -> dict[str, float]:
    old_distance_nm = _trajectory_distance_nm(old_arrival)
    new_distance_nm = _trajectory_distance_nm(new_arrival)
    old_elapsed_min = _elapsed_minutes(old_arrival)
    new_elapsed_min = _elapsed_minutes(new_arrival)
    return {
        "old_distance_nm": old_distance_nm,
        "new_distance_nm": new_distance_nm,
        "delta_distance_nm": new_distance_nm - old_distance_nm,
        "old_elapsed_min": old_elapsed_min,
        "new_elapsed_min": new_elapsed_min,
        "delta_elapsed_min": new_elapsed_min - old_elapsed_min,
    }


def _trajectory_distance_nm(arrival: dict[str, Any]) -> float:
    columns = _columns(arrival)
    lat_index = _column_index(columns, "lat")
    lon_index = _column_index(columns, "lon")
    points = arrival.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return 0.0
    distance_m = 0.0
    previous_lat: float | None = None
    previous_lon: float | None = None
    for row_index, raw_point in enumerate(points):
        point = _point(raw_point, f"points[{row_index}]")
        lat = _finite_lat(point[lat_index], f"points[{row_index}].lat")
        lon = _finite_lon(point[lon_index], f"points[{row_index}].lon")
        if previous_lat is not None and previous_lon is not None:
            distance_m += _latlon_distance_m(previous_lat, previous_lon, lat, lon)
        previous_lat = lat
        previous_lon = lon
    return float(distance_m / METERS_PER_NM)


def _elapsed_minutes(arrival: dict[str, Any]) -> float:
    first_time = arrival.get("first_time") or arrival.get("time_at_first_fix")
    last_time = arrival.get("last_time") or arrival.get("time_at_last_event")
    if first_time is None or last_time is None:
        return 0.0
    return float((_finite_number(last_time, "last_time") - _finite_number(first_time, "first_time")) / 60.0)


def _arrival_row(arrival: dict[str, Any]) -> pd.Series:
    return pd.Series(
        {
            "flight_id": arrival["flight_id"],
            "callsign": arrival.get("callsign", ""),
            "icao24": arrival.get("icao24", ""),
            "runway": arrival.get("runway", ""),
        }
    )


def _columns(arrival: dict[str, Any]) -> list[str]:
    columns = arrival.get("columns")
    if not isinstance(columns, list):
        raise ValueError("arrival has missing trajectory columns")
    return [str(column) for column in columns]


def _column_index(columns: list[str], column: str) -> int:
    try:
        return columns.index(column)
    except ValueError as exc:
        raise ValueError(f"arrival trajectory columns must include {column}") from exc


def _point(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a trajectory point")
    return value


def _finite_lat(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number < -90.0 or number > 90.0:
        raise ValueError(f"{name} must be between -90 and 90")
    return number


def _finite_lon(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number < -180.0 or number > 180.0:
        raise ValueError(f"{name} must be between -180 and 180")
    return number


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _latlon_distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat0_rad = math.radians(0.5 * (lat_a + lat_b))
    dx_m = 6_371_000.0 * math.cos(lat0_rad) * math.radians(lon_b - lon_a)
    dy_m = 6_371_000.0 * math.radians(lat_b - lat_a)
    return float(math.hypot(dx_m, dy_m))


def _bearing_deg(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat_a_rad = math.radians(lat_a)
    lat_b_rad = math.radians(lat_b)
    delta_lon_rad = math.radians(lon_b - lon_a)
    y = math.sin(delta_lon_rad) * math.cos(lat_b_rad)
    x = math.cos(lat_a_rad) * math.sin(lat_b_rad) - math.sin(lat_a_rad) * math.cos(lat_b_rad) * math.cos(delta_lon_rad)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return 0.0
    return float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)


def _display_route_token(token: str | tuple[float, float]) -> str:
    if isinstance(token, tuple):
        return _coordinate_label(token[0], token[1])
    return _route_token_text(token)


def _coordinate_label(lat: float, lon: float) -> str:
    return f"{lat:.5f},{lon:.5f}"


def _utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
