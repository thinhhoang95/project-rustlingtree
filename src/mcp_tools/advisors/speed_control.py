from __future__ import annotations

from dataclasses import dataclass
import math

from mcp_tools.advisors.models import SpeedControlAdvisory
from mcp_tools.advisors.profile import (
    METERS_PER_NM,
    ArrivalScheduleProvider,
    ProfilePlanner,
    arrivals_for,
    resolve_fixes_path,
)
from simap.fms import ATCSpeedSegment
from simap.units import kts_to_mps


@dataclass(frozen=True)
class SpeedControlAdvisor:
    manager: ArrivalScheduleProvider
    planner: ProfilePlanner = ProfilePlanner()

    def advise(self, *, flight_id: str, s_m: float, cas_kts: float) -> SpeedControlAdvisory:
        """Answer: what happens if this arrival accepts one lower-CAS instruction?"""
        s_m = _finite_nonnegative(s_m, "s_m")
        cas_kts = _finite_positive(cas_kts, "cas_kts")
        arrival = arrivals_for(self.manager, flight_id=flight_id)[0]
        profile = self.planner.build(arrival, resolve_fixes_path(self.manager))
        if s_m > profile.request.start_s_m:
            raise ValueError("s_m must lie within the arrival reference path")
        baseline = self.planner.plan(profile)
        what_if = self.planner.plan(
            profile,
            atc_speed_segments=(ATCSpeedSegment(s_from_m=s_m, cas_mps=kts_to_mps(cas_kts)),),
        )
        time_gain_s = float(what_if.total_time_s - baseline.total_time_s)
        equivalent_vectoring_m = time_gain_s * baseline.pre_tod_ground_speed_mps
        identity = profile.identity
        return SpeedControlAdvisory(
            flight_number=identity.flight_number,
            icao24=identity.icao24,
            flight_id=identity.flight_id,
            runway=identity.runway,
            miles_to_gain_nmi=0.0,
            minutes_to_gain=float(time_gain_s / 60.0),
            baseline_success=baseline.success,
            baseline_message=baseline.message,
            what_if_success=what_if.success,
            what_if_message=what_if.message,
            baseline_time_s=baseline.total_time_s,
            what_if_time_s=what_if.total_time_s,
            s_m=s_m,
            cas_kts=cas_kts,
            equivalent_vectoring_miles_nmi=float(equivalent_vectoring_m / METERS_PER_NM),
            equivalent_vectoring_m=float(equivalent_vectoring_m),
        )


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite_positive(value: float, name: str) -> float:
    value = _finite_nonnegative(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value
