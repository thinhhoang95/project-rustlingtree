from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from mcp_tools.scenario_manager.path_stretching import (
    _arrival_for,
    _arrival_row,
    _base_route_for_stretched_arrival,
    _finite_lat,
    _finite_lon,
    _finite_number,
    _metrics,
    _trajectory_distance_nm,
    _utc_now,
    _wait_atc_point_for_payload,
)
from mcp_tools.scenario_manager.precompute_artifact import (
    DEFAULT_ALTITUDE_TOLERANCE_M,
    DEFAULT_FMS_DT_S,
    DEFAULT_LATERAL_TOLERANCE_M,
    DEFAULT_MAX_TOD_ITERATIONS,
    DEFAULT_TOD_TOLERANCE_M,
    METERS_PER_NM,
    _default_lateral_guidance,
    _payload_from_result,
)
from mcp_tools.scenario_manager.served_profile import (
    ServedSpeedAdvisory,
    build_served_fms_context,
    merge_speed_advisories,
)
from simap.fms_bichannel import FMSBiChannelRequest, plan_fms_bichannel


class SpeedInterventionAdvisoryRequest(BaseModel):
    s_m: float
    cas_kts: float
    lat: float | None = None
    lon: float | None = None


class SpeedInterventionSimulationRequest(BaseModel):
    flight_id: str
    advisories: list[SpeedInterventionAdvisoryRequest] = Field(default_factory=list)


class SpeedInterventionSaveRequest(BaseModel):
    draft_id: str


def simulate_speed_intervention(
    manager: Any,
    request: SpeedInterventionSimulationRequest,
) -> dict[str, Any]:
    flight_id = request.flight_id.strip()
    if not flight_id:
        raise ValueError("flight_id is required")

    arrival = _arrival_for(manager, flight_id)
    context = build_served_fms_context(
        arrival,
        manager.config.fixes_path,
        fms_dt_s=DEFAULT_FMS_DT_S,
        include_speed_advisories=True,
    )
    base_route = context.route
    seed = context.seed
    fms_request = context.fms_request
    initial_state = context.initial_state
    requested_advisories = _normalize_advisories(
        request.advisories,
        max_s_m=float(fms_request.start_s_m),
    )
    advisories = merge_speed_advisories(context.speed_advisories, requested_advisories)
    guidance = _default_lateral_guidance()
    baseline_result = plan_fms_bichannel(
        FMSBiChannelRequest(
            base_request=fms_request,
            guidance=guidance,
            initial_state=initial_state,
        ),
        tod_tolerance_m=DEFAULT_TOD_TOLERANCE_M,
        max_tod_iterations=DEFAULT_MAX_TOD_ITERATIONS,
    )
    speed_request = replace(
        fms_request,
        atc_speed_segments=tuple(advisory.atc_segment for advisory in advisories),
    )
    speed_result = plan_fms_bichannel(
        FMSBiChannelRequest(
            base_request=speed_request,
            guidance=guidance,
            initial_state=initial_state,
        ),
        tod_tolerance_m=DEFAULT_TOD_TOLERANCE_M,
        max_tod_iterations=DEFAULT_MAX_TOD_ITERATIONS,
    )

    wait_atc_point = _wait_atc_point_for_payload(arrival)
    base_route_payload = _base_route_for_stretched_arrival(arrival, base_route)
    artifact, payload = _payload_from_result(
        row=_arrival_row(arrival),
        seed=seed,
        wait_atc_point=wait_atc_point,
        base_route=base_route_payload,
        result=speed_result,
        reference_path=getattr(fms_request, "reference_path", None),
        guidance=guidance,
        lateral_tolerance_m=float(arrival.get("lateral_tolerance_m", DEFAULT_LATERAL_TOLERANCE_M)),
        altitude_tolerance_m=float(arrival.get("altitude_tolerance_m", DEFAULT_ALTITUDE_TOLERANCE_M)),
    )
    _mark_payload_as_speed_intervention(payload, advisories=advisories)

    metrics = _speed_metrics(
        old_arrival=arrival,
        new_arrival=payload,
        baseline_result=baseline_result,
        speed_result=speed_result,
    )
    draft_id = f"speed-intervention-{flight_id}-{uuid4().hex[:12]}"
    created_at_utc = _utc_now()
    advisory_payload = [advisory.to_payload() for advisory in advisories]
    diff_record = {
        "id": draft_id,
        "flight_id": flight_id,
        "created_at_utc": created_at_utc,
        "source": "speed-intervention",
        "type": "speed-intervention",
        "command": {
            "type": "speed_intervention",
            "advisories": advisory_payload,
            "base_route": payload.get("base_route", {}).get("lateral_path"),
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
        "advisories": advisory_payload,
        "simulation": payload.get("simulation"),
        "baseline_simulation": _simulation_payload(baseline_result),
    }
    manager.speed_intervention_drafts[draft_id] = {
        "flight_id": flight_id,
        "response": response,
        "diff_record": diff_record,
        "artifact": asdict(artifact),
    }
    return response


def save_speed_intervention(
    manager: Any,
    flight_id: str,
    request: SpeedInterventionSaveRequest,
) -> dict[str, Any]:
    flight_id = flight_id.strip()
    draft = manager.speed_intervention_drafts.get(request.draft_id)
    if draft is None:
        raise ValueError(f"unknown speed-intervention draft_id={request.draft_id}")
    if draft["flight_id"] != flight_id:
        raise ValueError("speed-intervention draft does not match requested flight_id")

    diff_record = dict(draft["diff_record"])
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


def _normalize_advisories(
    advisories: list[SpeedInterventionAdvisoryRequest],
    *,
    max_s_m: float,
) -> list[ServedSpeedAdvisory]:
    if not advisories:
        raise ValueError("at least one speed-intervention advisory is required")

    normalized: list[ServedSpeedAdvisory] = []
    for index, advisory in enumerate(advisories):
        s_m = _finite_number(advisory.s_m, f"advisories[{index}].s_m")
        if s_m < 0.0 or s_m > max_s_m:
            raise ValueError(f"advisories[{index}].s_m must lie within the arrival reference path")
        cas_kts = _finite_number(advisory.cas_kts, f"advisories[{index}].cas_kts")
        if cas_kts <= 0.0:
            raise ValueError(f"advisories[{index}].cas_kts must be positive")
        lat = _finite_lat(advisory.lat, f"advisories[{index}].lat") if advisory.lat is not None else None
        lon = _finite_lon(advisory.lon, f"advisories[{index}].lon") if advisory.lon is not None else None
        normalized.append(
            ServedSpeedAdvisory(
                s_m=s_m,
                cas_kts=cas_kts,
                lat=lat,
                lon=lon,
            )
        )

    return sorted(normalized, key=lambda item: item.s_m, reverse=True)


def _mark_payload_as_speed_intervention(
    payload: dict[str, Any],
    *,
    advisories: list[ServedSpeedAdvisory],
) -> None:
    advisory_payload = [advisory.to_payload() for advisory in advisories]
    base_route = payload.get("base_route")
    if isinstance(base_route, dict):
        base_route["type"] = "speed-intervention"
        base_route["selection_method"] = "interactive_speed_intervention"
        base_route["speed_advisories"] = advisory_payload
    payload["route_type"] = "speed-intervention"
    payload["final_fix"] = payload.get("baseline_final_fix")
    payload["speed_intervention"] = {
        "advisories": advisory_payload,
        "advisory_count": len(advisory_payload),
    }


def _speed_metrics(
    *,
    old_arrival: dict[str, Any],
    new_arrival: dict[str, Any],
    baseline_result: Any,
    speed_result: Any,
) -> dict[str, float]:
    metrics = _metrics(old_arrival=old_arrival, new_arrival=new_arrival)
    baseline_elapsed_min = _result_elapsed_minutes(baseline_result)
    speed_elapsed_min = _result_elapsed_minutes(speed_result)
    delta_elapsed_min = speed_elapsed_min - baseline_elapsed_min
    equivalent_distance_nm = delta_elapsed_min * 60.0 * _pre_tod_ground_speed_mps(baseline_result) / METERS_PER_NM
    metrics.update(
        {
            "old_elapsed_min": baseline_elapsed_min,
            "new_elapsed_min": speed_elapsed_min,
            "delta_elapsed_min": delta_elapsed_min,
            "equivalent_distance_nm": float(equivalent_distance_nm),
            "baseline_distance_nm": _trajectory_distance_nm(old_arrival),
            "speed_intervention_distance_nm": _trajectory_distance_nm(new_arrival),
        }
    )
    return metrics


def _result_elapsed_minutes(result: Any) -> float:
    times = np.asarray(getattr(result, "t_s", []), dtype=float)
    if len(times) == 0:
        return 0.0
    return float((times[-1] - times[0]) / 60.0)


def _pre_tod_ground_speed_mps(result: Any) -> float:
    longitudinal = getattr(result, "longitudinal", None)
    level_distance_m = float(getattr(longitudinal, "level_distance_m", 0.0) or 0.0)
    level_time_s = float(getattr(longitudinal, "level_time_s", 0.0) or 0.0)
    if level_distance_m > 0.0 and level_time_s > 0.0:
        speed = level_distance_m / level_time_s
    else:
        ground_speed = np.asarray(getattr(result, "ground_speed_mps", []), dtype=float)
        speed = float(ground_speed[0]) if len(ground_speed) else 1.0
    return speed if np.isfinite(speed) and speed > 0.0 else 1.0


def _simulation_payload(result: Any) -> dict[str, Any]:
    return {
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "max_abs_cross_track_m": float(getattr(result, "max_abs_cross_track_m", 0.0) or 0.0),
        "max_abs_track_error_rad": float(getattr(result, "max_abs_track_error_rad", 0.0) or 0.0),
        "final_threshold_error_m": float(getattr(result, "final_threshold_error_m", 0.0) or 0.0),
    }
