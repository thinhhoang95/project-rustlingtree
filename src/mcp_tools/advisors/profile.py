from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from mcp_tools.advisors.models import AdvisoryFlight
from mcp_tools.scenario_manager.models import project_root
from mcp_tools.scenario_manager.served_profile import build_served_fms_context
from simap.fms import ATCSpeedSegmentInput, FMSRequest, FMSResult, plan_fms_descent
from simap.path_geometry import ReferencePath

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
        context = build_served_fms_context(
            arrival,
            fixes_path,
            fms_dt_s=self.fms_dt_s,
            include_speed_advisories=True,
        )
        return ArrivalProfile(
            arrival=arrival,
            identity=_identity(arrival),
            request=context.fms_request,
            initial_ground_speed_mps=context.initial_ground_speed_mps,
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
            extended_start_s_m = float(request.start_s_m + extra_distance_m)
            request = replace(
                request,
                reference_path=reference_path,
                start_s_m=extended_start_s_m,
                atc_speed_reference_start_s_m=extended_start_s_m,
            )
        if atc_speed_segments:
            request = replace(
                request,
                atc_speed_segments=tuple(request.atc_speed_segments) + tuple(atc_speed_segments),
            )

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
