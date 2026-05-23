from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Literal

from mcp_tools.scenario_manager.path_stretching import simulate_path_stretch_route
from mcp_tools.scenario_manager.precompute_artifact import METERS_PER_NM
from mcp_tools.scenario_manager.served_profile import arrival_lateral_path
from mcp_tools.sensory.models import (
    VectorAssistCandidate,
    VectorAssistMapCell,
    VectorAssistRequest,
    VectorAssistResponse,
)
from mcp_tools.sensory.operational_space import OperationalSpaceMask, mask_for_cluster
from simap.nlp_colloc.tactical.models import PathWaypoint
from simap.nlp_colloc.tactical.navdata import load_fix_catalog


RouteToken = str | tuple[float, float]
DoglegVariant = Literal["sandwiched_dogleg", "replaced_dogleg"]
CandidateKind = Literal["identified", "free"]

EARTH_RADIUS_M = 6_371_000.0
MIN_INTERIOR_ANGLE_DEG = 25.0
DEFAULT_NOMINAL_SPEED_MPS = 120.0


@dataclass(frozen=True)
class _CandidateSeed:
    kind: CandidateKind
    lat: float
    lon: float
    fix_identifier: str | None = None


@dataclass(frozen=True)
class _CandidatePlan:
    seed: _CandidateSeed
    variant: DoglegVariant
    route: list[RouteToken]
    projected_segment_index: int
    f_a: str
    f_b: str
    estimated_time_gain_s: float
    estimated_error_s: float


def vector_assist(manager: Any, request: VectorAssistRequest) -> VectorAssistResponse:
    flight_id = request.flight_id.strip()
    if not flight_id:
        raise ValueError("flight_id is required")
    target_time_gain_s = _finite_nonnegative(request.target_time_gain_s, "target_time_gain_s")
    grid_spacing_nm = _finite_positive(request.grid_spacing_nm, "grid_spacing_nm")
    identified_threshold_s = _finite_nonnegative(request.identified_threshold_s, "identified_threshold_s")
    max_exact_candidates = _positive_int(request.max_exact_candidates, "max_exact_candidates")

    arrival = _arrival_for(manager, flight_id)
    route = arrival_lateral_path(arrival)
    _validate_seedable_arrival(arrival)
    fix_catalog = load_fix_catalog(manager.config.fixes_path)
    route_points = [_route_token_latlon(token, fix_catalog) for token in route]
    cluster = _arrival_cluster(arrival)
    mask = mask_for_cluster(cluster, fix_catalog)
    attempts = _existing_vector_assist_attempts(arrival)
    if len(attempts) >= 2:
        raise ValueError("maximum of 2 vector-assist attempts per flight has already been reached")
    replaced_dogleg_used = any(attempt.get("variant") == "replaced_dogleg" for attempt in attempts)

    base_route_length_m = _route_length_m(route_points)
    nominal_speed_mps = _nominal_speed_mps(arrival, base_route_length_m)
    rejected: Counter[str] = Counter()

    identified_seeds = _identified_seeds(fix_catalog, mask)
    free_grid = mask.grid_points(grid_spacing_nm)
    free_seeds = [_CandidateSeed(kind="free", lat=lat, lon=lon) for lat, lon in free_grid]
    free_seeds.extend(
        _refined_free_seeds(
            free_grid,
            target_time_gain_s=target_time_gain_s,
            spacing_nm=grid_spacing_nm,
            mask=mask,
            route=route,
            route_points=route_points,
            base_route_length_m=base_route_length_m,
            nominal_speed_mps=nominal_speed_mps,
        )
    )

    identified_plans = _candidate_plans(
        identified_seeds,
        route=route,
        route_points=route_points,
        fix_catalog=fix_catalog,
        base_route_length_m=base_route_length_m,
        nominal_speed_mps=nominal_speed_mps,
        target_time_gain_s=target_time_gain_s,
        replaced_dogleg_used=replaced_dogleg_used,
        rejected=rejected,
    )
    free_plans = _candidate_plans(
        free_seeds,
        route=route,
        route_points=route_points,
        fix_catalog=fix_catalog,
        base_route_length_m=base_route_length_m,
        nominal_speed_mps=nominal_speed_mps,
        target_time_gain_s=target_time_gain_s,
        replaced_dogleg_used=replaced_dogleg_used,
        rejected=rejected,
    )
    exact_plans = _select_exact_plans(identified_plans, free_plans, max_exact_candidates)
    exact_candidates = _score_exact_candidates(
        manager,
        arrival,
        exact_plans,
        fix_catalog=fix_catalog,
        target_time_gain_s=target_time_gain_s,
        rejected=rejected,
    )
    best_identified = _best_successful(exact_candidates, "identified")
    best_free = _best_successful(exact_candidates, "free")
    recommendation = _recommend(best_identified, best_free, identified_threshold_s)

    identity = {
        "flight_number": str(arrival.get("callsign", "")),
        "icao24": str(arrival.get("icao24", "")),
        "flight_id": str(arrival.get("flight_id", "")),
        "runway": str(arrival.get("runway", "")),
    }
    return VectorAssistResponse(
        **identity,
        arrival_cluster=cluster,
        operational_mask=mask.name,
        target_time_gain_s=target_time_gain_s,
        attempt_status={
            "previous_attempt_count": len(attempts),
            "remaining_attempts": max(0, 2 - len(attempts)),
            "replaced_dogleg_used": replaced_dogleg_used,
            "replaced_dogleg_available": not replaced_dogleg_used,
        },
        best_identified_candidate=best_identified,
        best_free_candidate=best_free,
        recommendation=recommendation,
        rejected_counts=dict(sorted(rejected.items())),
        evaluated_candidate_count=len(exact_candidates),
        map_cells=_map_cells(
            free_grid,
            route=route,
            route_points=route_points,
            fix_catalog=fix_catalog,
            base_route_length_m=base_route_length_m,
            nominal_speed_mps=nominal_speed_mps,
        )
        if request.include_map
        else None,
    )


def project_candidate_to_route(
    route_points: list[tuple[float, float]],
    lat: float,
    lon: float,
) -> int:
    projected = _project_route_points(route_points)
    point = _project_latlon(lat, lon, origin=route_points[0])
    best_index = 0
    best_distance_sq = math.inf
    for index, (start, end) in enumerate(zip(projected, projected[1:], strict=False)):
        closest = _closest_point_on_segment(point, start, end)
        distance_sq = (point[0] - closest[0]) ** 2 + (point[1] - closest[1]) ** 2
        if distance_sq < best_distance_sq:
            best_index = index
            best_distance_sq = distance_sq
    return best_index


def build_dogleg_route(
    route: list[RouteToken],
    candidate: RouteToken,
    *,
    projected_segment_index: int,
    variant: DoglegVariant,
) -> list[RouteToken]:
    if projected_segment_index < 0 or projected_segment_index >= len(route) - 1:
        raise ValueError("projected_segment_index must target an existing route segment")
    if variant == "sandwiched_dogleg":
        return [*route[: projected_segment_index + 1], candidate, *route[projected_segment_index + 1 :]]
    if projected_segment_index == 0:
        raise ValueError("replaced dogleg cannot target the first route segment")
    return [*route[:projected_segment_index], candidate, *route[projected_segment_index + 1 :]]


def has_tight_turn(
    route_points: list[tuple[float, float]],
    affected_indices: set[int] | None = None,
) -> bool:
    projected = _project_route_points(route_points)
    indices = affected_indices if affected_indices is not None else set(range(1, len(projected) - 1))
    for index in indices:
        if index <= 0 or index >= len(projected) - 1:
            continue
        angle = _interior_angle_deg(projected[index - 1], projected[index], projected[index + 1])
        if angle < MIN_INTERIOR_ANGLE_DEG:
            return True
    return False


def _arrival_for(manager: Any, flight_id: str) -> dict[str, Any]:
    selected = [arrival for arrival in manager.arrival_schedule() if str(arrival.get("flight_id", "")) == flight_id]
    if not selected:
        raise ValueError(f"unknown arrival flight_id={flight_id}")
    return selected[0]


def _validate_seedable_arrival(arrival: dict[str, Any]) -> None:
    arrival_lateral_path(arrival)
    if not isinstance(arrival.get("cas_profile"), dict):
        raise ValueError("arrival requires cas_profile for vector-assist scoring")
    points = arrival.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("arrival requires trajectory points for vector-assist scoring")


def _arrival_cluster(arrival: dict[str, Any]) -> str:
    wait_point = arrival.get("atc_wait_point") or arrival.get("wait_atc_point")
    if not isinstance(wait_point, dict):
        raise ValueError("arrival requires atc_wait_point.arrival_cluster for vector assist")
    cluster = str(wait_point.get("arrival_cluster") or "").strip().upper()
    if not cluster:
        raise ValueError("arrival requires atc_wait_point.arrival_cluster for vector assist")
    return cluster


def _identified_seeds(
    fix_catalog: dict[str, PathWaypoint],
    mask: OperationalSpaceMask,
) -> list[_CandidateSeed]:
    seeds: list[_CandidateSeed] = []
    for waypoint in fix_catalog.values():
        identifier = waypoint.identifier.strip().upper()
        if not identifier or identifier.startswith("RW"):
            continue
        if not mask.contains(waypoint.lat_deg, waypoint.lon_deg):
            continue
        seeds.append(
            _CandidateSeed(
                kind="identified",
                fix_identifier=identifier,
                lat=float(waypoint.lat_deg),
                lon=float(waypoint.lon_deg),
            )
        )
    return seeds


def _candidate_plans(
    seeds: list[_CandidateSeed],
    *,
    route: list[RouteToken],
    route_points: list[tuple[float, float]],
    fix_catalog: dict[str, PathWaypoint],
    base_route_length_m: float,
    nominal_speed_mps: float,
    target_time_gain_s: float,
    replaced_dogleg_used: bool,
    rejected: Counter[str],
) -> list[_CandidatePlan]:
    plans: list[_CandidatePlan] = []
    seen_routes: set[tuple[str, ...]] = set()
    for seed in seeds:
        projected_segment_index = project_candidate_to_route(route_points, seed.lat, seed.lon)
        candidate_token: RouteToken = seed.fix_identifier if seed.fix_identifier is not None else (seed.lat, seed.lon)
        for variant in ("sandwiched_dogleg", "replaced_dogleg"):
            if variant == "replaced_dogleg" and replaced_dogleg_used:
                rejected["replaced_dogleg_unavailable"] += 1
                continue
            try:
                plan_route = build_dogleg_route(
                    route,
                    candidate_token,
                    projected_segment_index=projected_segment_index,
                    variant=variant,
                )
            except ValueError:
                rejected["first_segment_replaced"] += 1
                continue
            if plan_route == route:
                rejected["unchanged_route"] += 1
                continue
            if _has_consecutive_duplicate(plan_route):
                rejected["duplicate_tokens"] += 1
                continue
            route_key = tuple(_route_token_display(token) for token in plan_route)
            if route_key in seen_routes:
                rejected["duplicate_routes"] += 1
                continue
            seen_routes.add(route_key)

            try:
                plan_points = [_route_token_latlon(token, fix_catalog) for token in plan_route]
            except KeyError:
                rejected["unknown_fix"] += 1
                continue
            if has_tight_turn(plan_points, _affected_indices(projected_segment_index, variant)):
                rejected["tight_turn"] += 1
                continue
            route_length_m = _route_length_m(plan_points)
            estimated_time_gain_s = (route_length_m - base_route_length_m) / nominal_speed_mps
            plans.append(
                _CandidatePlan(
                    seed=seed,
                    variant=variant,
                    route=plan_route,
                    projected_segment_index=projected_segment_index,
                    f_a=_route_token_display(route[projected_segment_index]),
                    f_b=_route_token_display(route[projected_segment_index + 1]),
                    estimated_time_gain_s=estimated_time_gain_s,
                    estimated_error_s=abs(estimated_time_gain_s - target_time_gain_s),
                )
            )
    return sorted(plans, key=lambda item: item.estimated_error_s)


def _select_exact_plans(
    identified_plans: list[_CandidatePlan],
    free_plans: list[_CandidatePlan],
    max_exact_candidates: int,
) -> list[_CandidatePlan]:
    identified_budget = min(len(identified_plans), max(1, max_exact_candidates // 2))
    free_budget = min(len(free_plans), max_exact_candidates - identified_budget)
    selected = [*identified_plans[:identified_budget], *free_plans[:free_budget]]
    remaining_budget = max_exact_candidates - len(selected)
    if remaining_budget > 0:
        selected_keys = {_plan_key(plan) for plan in selected}
        remaining = [
            plan
            for plan in sorted([*identified_plans[identified_budget:], *free_plans[free_budget:]], key=lambda item: item.estimated_error_s)
            if _plan_key(plan) not in selected_keys
        ]
        selected.extend(remaining[:remaining_budget])
    return sorted(selected, key=lambda item: item.estimated_error_s)


def _score_exact_candidates(
    manager: Any,
    arrival: dict[str, Any],
    plans: list[_CandidatePlan],
    *,
    fix_catalog: dict[str, PathWaypoint],
    target_time_gain_s: float,
    rejected: Counter[str],
) -> list[VectorAssistCandidate]:
    candidates: list[VectorAssistCandidate] = []
    for plan in plans:
        vector_metadata = _vector_metadata(plan, target_time_gain_s)
        try:
            route_simulation = simulate_path_stretch_route(
                manager,
                arrival,
                plan.route,
                vector_assist=vector_metadata,
            )
        except Exception:
            rejected["exact_simulation_error"] += 1
            continue
        metrics = route_simulation.metrics
        actual_time_gain_s = float(metrics["delta_elapsed_min"] * 60.0)
        simulation = route_simulation.simulation or {}
        simulation_success = bool(simulation.get("success", False))
        if not simulation_success:
            rejected["simap_failed"] += 1
        candidate = VectorAssistCandidate(
            candidate_kind=plan.seed.kind,
            variant=plan.variant,
            fix_identifier=plan.seed.fix_identifier,
            lat=plan.seed.lat,
            lon=plan.seed.lon,
            projected_segment_index=plan.projected_segment_index,
            f_a=plan.f_a,
            f_b=plan.f_b,
            target_time_gain_s=target_time_gain_s,
            actual_time_gain_s=actual_time_gain_s,
            error_s=abs(actual_time_gain_s - target_time_gain_s),
            estimated_time_gain_s=plan.estimated_time_gain_s,
            estimated_error_s=plan.estimated_error_s,
            simulation_success=simulation_success,
            simulation_message=str(simulation.get("message", "")),
            metrics=metrics,
            path_stretch_request=_path_stretch_request(arrival, plan, fix_catalog, vector_metadata),
        )
        candidates.append(candidate)
    return candidates


def _best_successful(
    candidates: list[VectorAssistCandidate],
    kind: CandidateKind,
) -> VectorAssistCandidate | None:
    matching = [candidate for candidate in candidates if candidate.candidate_kind == kind and candidate.simulation_success]
    if not matching:
        return None
    return min(matching, key=lambda item: item.error_s)


def _recommend(
    identified: VectorAssistCandidate | None,
    free: VectorAssistCandidate | None,
    identified_threshold_s: float,
) -> VectorAssistCandidate | None:
    if identified is None:
        return free
    if free is None:
        return identified
    if abs(identified.error_s - free.error_s) <= identified_threshold_s:
        return identified
    return min((identified, free), key=lambda item: item.error_s)


def _path_stretch_request(
    arrival: dict[str, Any],
    plan: _CandidatePlan,
    fix_catalog: dict[str, PathWaypoint],
    vector_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "flight_id": str(arrival.get("flight_id", "")),
        "route": [_route_point_payload(token, fix_catalog) for token in plan.route],
        "vector_assist": vector_metadata,
    }


def _route_point_payload(token: RouteToken, fix_catalog: dict[str, PathWaypoint]) -> dict[str, Any]:
    if isinstance(token, tuple):
        return {
            "token_type": "coordinate",
            "lat": float(token[0]),
            "lon": float(token[1]),
            "fix_identifier": None,
        }
    waypoint = fix_catalog[str(token).upper()]
    return {
        "token_type": "fix",
        "lat": float(waypoint.lat_deg),
        "lon": float(waypoint.lon_deg),
        "fix_identifier": waypoint.identifier.upper(),
    }


def _vector_metadata(plan: _CandidatePlan, target_time_gain_s: float) -> dict[str, Any]:
    return {
        "variant": plan.variant,
        "candidate_kind": plan.seed.kind,
        "target_time_gain_s": target_time_gain_s,
        "projected_segment_index": plan.projected_segment_index,
        "lat": plan.seed.lat,
        "lon": plan.seed.lon,
        "fix_identifier": plan.seed.fix_identifier,
    }


def _map_cells(
    grid: list[tuple[float, float]],
    *,
    route: list[RouteToken],
    route_points: list[tuple[float, float]],
    fix_catalog: dict[str, PathWaypoint],
    base_route_length_m: float,
    nominal_speed_mps: float,
) -> list[VectorAssistMapCell]:
    cells: list[VectorAssistMapCell] = []
    for lat, lon in grid:
        try:
            segment_index = project_candidate_to_route(route_points, lat, lon)
            plan_route = build_dogleg_route(route, (lat, lon), projected_segment_index=segment_index, variant="sandwiched_dogleg")
            plan_points = [_route_token_latlon(token, fix_catalog) for token in plan_route]
        except (KeyError, ValueError):
            continue
        estimated_time_gain_s = (_route_length_m(plan_points) - base_route_length_m) / nominal_speed_mps
        cells.append(VectorAssistMapCell(lat=lat, lon=lon, estimated_time_gain_s=estimated_time_gain_s))
    return cells


def _refined_free_seeds(
    free_grid: list[tuple[float, float]],
    *,
    target_time_gain_s: float,
    spacing_nm: float,
    mask: OperationalSpaceMask,
    route: list[RouteToken],
    route_points: list[tuple[float, float]],
    base_route_length_m: float,
    nominal_speed_mps: float,
) -> list[_CandidateSeed]:
    if len(free_grid) < 3:
        return []
    scored = []
    for lat, lon in free_grid:
        segment_index = project_candidate_to_route(route_points, lat, lon)
        try:
            plan_route = build_dogleg_route(route, (lat, lon), projected_segment_index=segment_index, variant="sandwiched_dogleg")
            projected_points = [_route_token_latlon_for_grid(token, route_points, route) for token in plan_route]
        except ValueError:
            continue
        gain = (_route_length_m(projected_points) - base_route_length_m) / nominal_speed_mps
        scored.append((abs(gain - target_time_gain_s), lat, lon, gain))

    scored.sort(key=lambda item: item[0])
    top = scored[:8]
    try:
        from scipy.interpolate import LinearNDInterpolator

        interpolator = LinearNDInterpolator([(lon, lat) for _err, lat, lon, _gain in scored], [gain for _err, _lat, _lon, gain in scored])
    except Exception:
        interpolator = None

    step_deg = spacing_nm / 60.0 * 0.5
    refined: list[_CandidateSeed] = []
    seen = {(round(lat, 6), round(lon, 6)) for lat, lon in free_grid}
    for _error, lat, lon, _gain in top:
        for d_lat in (-step_deg, 0.0, step_deg):
            for d_lon in (-step_deg, 0.0, step_deg):
                if d_lat == 0.0 and d_lon == 0.0:
                    continue
                candidate_lat = lat + d_lat
                candidate_lon = lon + d_lon
                key = (round(candidate_lat, 6), round(candidate_lon, 6))
                if key in seen or not mask.contains(candidate_lat, candidate_lon):
                    continue
                if interpolator is not None:
                    interpolated = interpolator(candidate_lon, candidate_lat)
                    try:
                        if not math.isfinite(float(interpolated)):
                            continue
                    except (TypeError, ValueError):
                        continue
                seen.add(key)
                refined.append(_CandidateSeed(kind="free", lat=candidate_lat, lon=candidate_lon))
    return refined


def _route_token_latlon_for_grid(
    token: RouteToken,
    original_route_points: list[tuple[float, float]],
    original_route: list[RouteToken],
) -> tuple[float, float]:
    if isinstance(token, tuple):
        return float(token[0]), float(token[1])
    for route_token, point in zip(original_route, original_route_points, strict=True):
        if route_token == token:
            return point
    raise ValueError(f"unknown route token {token}")


def _affected_indices(segment_index: int, variant: DoglegVariant) -> set[int]:
    candidate_index = segment_index + 1 if variant == "sandwiched_dogleg" else segment_index
    return {candidate_index - 1, candidate_index, candidate_index + 1}


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


def _route_token_latlon(token: RouteToken, fix_catalog: dict[str, PathWaypoint]) -> tuple[float, float]:
    if isinstance(token, tuple):
        return float(token[0]), float(token[1])
    waypoint = fix_catalog[str(token).upper()]
    return float(waypoint.lat_deg), float(waypoint.lon_deg)


def _route_length_m(points: list[tuple[float, float]]) -> float:
    return float(
        sum(
            _latlon_distance_m(lat_a, lon_a, lat_b, lon_b)
            for (lat_a, lon_a), (lat_b, lon_b) in zip(points, points[1:], strict=False)
        )
    )


def _nominal_speed_mps(arrival: dict[str, Any], base_route_length_m: float) -> float:
    elapsed_s = _elapsed_s(arrival)
    if elapsed_s > 0.0 and base_route_length_m > 0.0:
        speed = base_route_length_m / elapsed_s
        if math.isfinite(speed) and speed > 1.0:
            return speed
    return DEFAULT_NOMINAL_SPEED_MPS


def _elapsed_s(arrival: dict[str, Any]) -> float:
    first_time = arrival.get("first_time") or arrival.get("time_at_first_fix")
    last_time = arrival.get("last_time") or arrival.get("time_at_last_event")
    if not isinstance(first_time, int | float) or not isinstance(last_time, int | float):
        return 0.0
    return float(last_time - first_time)


def _has_consecutive_duplicate(route: list[RouteToken]) -> bool:
    return any(left == right for left, right in zip(route, route[1:], strict=False))


def _plan_key(plan: _CandidatePlan) -> tuple[str, ...]:
    return tuple(_route_token_display(token) for token in plan.route)


def _route_token_display(token: RouteToken) -> str:
    if isinstance(token, tuple):
        return f"{float(token[0]):.6f},{float(token[1]):.6f}"
    return str(token).upper()


def _project_route_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    origin = points[0]
    return [_project_latlon(lat, lon, origin=origin) for lat, lon in points]


def _project_latlon(
    lat: float,
    lon: float,
    *,
    origin: tuple[float, float],
) -> tuple[float, float]:
    origin_lat, origin_lon = origin
    lat0_rad = math.radians(origin_lat)
    x = EARTH_RADIUS_M * math.cos(lat0_rad) * math.radians(float(lon) - origin_lon)
    y = EARTH_RADIUS_M * math.radians(float(lat) - origin_lat)
    return float(x), float(y)


def _closest_point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return start
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    return sx + t * dx, sy + t * dy


def _interior_angle_deg(
    previous: tuple[float, float],
    current: tuple[float, float],
    next_point: tuple[float, float],
) -> float:
    a_x = previous[0] - current[0]
    a_y = previous[1] - current[1]
    b_x = next_point[0] - current[0]
    b_y = next_point[1] - current[1]
    a_norm = math.hypot(a_x, a_y)
    b_norm = math.hypot(b_x, b_y)
    if a_norm <= 0.0 or b_norm <= 0.0:
        return 0.0
    dot = max(-1.0, min(1.0, (a_x * b_x + a_y * b_y) / (a_norm * b_norm)))
    return float(math.degrees(math.acos(dot)))


def _latlon_distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat0_rad = math.radians(0.5 * (lat_a + lat_b))
    dx = EARTH_RADIUS_M * math.cos(lat0_rad) * math.radians(lon_b - lon_a)
    dy = EARTH_RADIUS_M * math.radians(lat_b - lat_a)
    return float(math.hypot(dx, dy))


def _finite_nonnegative(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _finite_positive(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
