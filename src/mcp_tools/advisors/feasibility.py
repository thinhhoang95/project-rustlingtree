from __future__ import annotations

from dataclasses import dataclass

from mcp_tools.advisors.models import FeasibilityAdvisory
from mcp_tools.advisors.profile import (
    DEFAULT_TOD_TOLERANCE_M,
    METERS_PER_NM,
    ArrivalProfile,
    ArrivalScheduleProvider,
    PlannedProfile,
    ProfilePlanner,
    arrivals_for,
    resolve_fixes_path,
)


@dataclass(frozen=True)
class FeasibilityAdvisor:
    manager: ArrivalScheduleProvider
    planner: ProfilePlanner = ProfilePlanner()
    initial_extension_nmi: float = 1.0
    max_extension_nmi: float = 256.0

    def evaluate(self, flight_id: str | None = None) -> list[FeasibilityAdvisory]:
        fixes_path = resolve_fixes_path(self.manager)
        advisories: list[FeasibilityAdvisory] = []
        for arrival in arrivals_for(self.manager, flight_id=flight_id):
            profile = self.planner.build(arrival, fixes_path)
            advisories.append(self.advise_profile(profile))
        return sorted(
            advisories,
            key=lambda item: (
                -item.miles_to_gain_nmi,
                item.flight_number,
                item.icao24,
                item.flight_id,
            ),
        )

    def advise_profile(
        self,
        profile: ArrivalProfile,
        *,
        baseline: PlannedProfile | None = None,
    ) -> FeasibilityAdvisory:
        baseline = self.planner.plan(profile) if baseline is None else baseline
        if baseline.success:
            return _advisory(
                profile=profile,
                miles_to_gain_m=0.0,
                baseline=baseline,
                what_if=baseline,
                search_converged=True,
            )

        high_m = self._initial_extension_m()
        max_m = self._max_extension_m()
        high_plan = self.planner.plan(profile, extra_distance_m=high_m)
        while not high_plan.success and high_m < max_m:
            next_high_m = min(high_m * 2.0, max_m)
            if next_high_m <= high_m:
                break
            high_m = next_high_m
            high_plan = self.planner.plan(profile, extra_distance_m=high_m)

        if not high_plan.success:
            return _advisory(
                profile=profile,
                miles_to_gain_m=high_m,
                baseline=baseline,
                what_if=high_plan,
                search_converged=False,
            )

        low_m = 0.0
        tolerance_m = _tod_tolerance_m(self.planner)
        while high_m - low_m > tolerance_m:
            mid_m = 0.5 * (low_m + high_m)
            mid_plan = self.planner.plan(profile, extra_distance_m=mid_m)
            if mid_plan.success:
                high_m = mid_m
                high_plan = mid_plan
            else:
                low_m = mid_m

        return _advisory(
            profile=profile,
            miles_to_gain_m=high_m,
            baseline=baseline,
            what_if=high_plan,
            search_converged=True,
        )

    def _initial_extension_m(self) -> float:
        if self.initial_extension_nmi <= 0.0:
            raise ValueError("initial_extension_nmi must be positive")
        return float(self.initial_extension_nmi * METERS_PER_NM)

    def _max_extension_m(self) -> float:
        if self.max_extension_nmi <= 0.0:
            raise ValueError("max_extension_nmi must be positive")
        return float(self.max_extension_nmi * METERS_PER_NM)


def _advisory(
    *,
    profile: ArrivalProfile,
    miles_to_gain_m: float,
    baseline: PlannedProfile,
    what_if: PlannedProfile,
    search_converged: bool,
) -> FeasibilityAdvisory:
    identity = profile.identity
    return FeasibilityAdvisory(
        flight_number=identity.flight_number,
        icao24=identity.icao24,
        flight_id=identity.flight_id,
        runway=identity.runway,
        miles_to_gain_nmi=float(miles_to_gain_m / METERS_PER_NM),
        miles_to_gain_m=float(miles_to_gain_m),
        minutes_to_gain=float((what_if.total_time_s - baseline.total_time_s) / 60.0),
        baseline_success=baseline.success,
        baseline_message=baseline.message,
        what_if_success=what_if.success,
        what_if_message=what_if.message,
        baseline_time_s=baseline.total_time_s,
        what_if_time_s=what_if.total_time_s,
        search_converged=search_converged,
    )


def _tod_tolerance_m(planner: object) -> float:
    value = getattr(planner, "tod_tolerance_m", DEFAULT_TOD_TOLERANCE_M)
    return float(value)
