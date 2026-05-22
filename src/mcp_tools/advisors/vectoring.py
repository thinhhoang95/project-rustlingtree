from __future__ import annotations

from dataclasses import dataclass
import math

from mcp_tools.advisors.feasibility import FeasibilityAdvisor
from mcp_tools.advisors.models import VectoringAdvisory
from mcp_tools.advisors.profile import (
    METERS_PER_NM,
    ArrivalScheduleProvider,
    ProfilePlanner,
    arrivals_for,
    resolve_fixes_path,
)


@dataclass(frozen=True)
class VectoringAdvisor:
    manager: ArrivalScheduleProvider
    planner: ProfilePlanner = ProfilePlanner()

    def advise(self, *, flight_id: str, extra_distance_nmi: float) -> VectoringAdvisory:
        """Answer: what happens if this arrival gets N extra nautical miles?"""
        extra_distance_nmi = _finite_nonnegative(extra_distance_nmi, "extra_distance_nmi")
        arrival = arrivals_for(self.manager, flight_id=flight_id)[0]
        profile = self.planner.build(arrival, resolve_fixes_path(self.manager))
        baseline = self.planner.plan(profile)
        extra_distance_m = extra_distance_nmi * METERS_PER_NM
        what_if = (
            baseline
            if extra_distance_m == 0.0
            else self.planner.plan(profile, extra_distance_m=extra_distance_m)
        )
        feasibility = FeasibilityAdvisor(self.manager, planner=self.planner).advise_profile(profile, baseline=baseline)
        identity = profile.identity
        return VectoringAdvisory(
            flight_number=identity.flight_number,
            icao24=identity.icao24,
            flight_id=identity.flight_id,
            runway=identity.runway,
            miles_to_gain_nmi=extra_distance_nmi,
            minutes_to_gain=float((what_if.total_time_s - baseline.total_time_s) / 60.0),
            baseline_success=baseline.success,
            baseline_message=baseline.message,
            what_if_success=what_if.success,
            what_if_message=what_if.message,
            baseline_time_s=baseline.total_time_s,
            what_if_time_s=what_if.total_time_s,
            requested_extension_nmi=extra_distance_nmi,
            requested_extension_m=extra_distance_m,
            required_feasibility_miles_nmi=feasibility.miles_to_gain_nmi,
            required_feasibility_m=feasibility.miles_to_gain_m,
            feasibility_search_converged=feasibility.search_converged,
        )


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value
