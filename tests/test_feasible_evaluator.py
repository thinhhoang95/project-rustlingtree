from __future__ import annotations

from typing import Any

import pytest

from mcp_tools.evaluators import FeasibleEvaluator, FeasibleFlight


class ArrivalScheduleManager:
    def __init__(self, arrivals: list[dict[str, Any]]) -> None:
        self._arrivals = arrivals
        self.arrival_schedule_calls = 0

    @property
    def simap_arrival_flights(self) -> dict[str, Any]:
        raise AssertionError("FeasibleEvaluator must use the served arrival schedule")

    @property
    def diff(self) -> list[dict[str, Any]]:
        raise AssertionError("FeasibleEvaluator must use the served arrival schedule")

    @property
    def simap_arrival_artifact_manifest(self) -> dict[str, Any]:
        raise AssertionError("FeasibleEvaluator must use the served arrival schedule")

    def arrival_schedule(self) -> list[dict[str, Any]]:
        self.arrival_schedule_calls += 1
        return list(self._arrivals)


def _arrival(
    *,
    flight_id: str,
    callsign: str,
    icao24: str,
    runway: str = "RW35C",
    success: bool = False,
    final_threshold_error_m: object = 0.0,
    message: str = "infeasible",
) -> dict[str, Any]:
    return {
        "flight_id": flight_id,
        "callsign": callsign,
        "icao24": icao24,
        "runway": runway,
        "simulation": {
            "success": success,
            "message": message,
            "final_threshold_error_m": final_threshold_error_m,
        },
    }


def test_feasible_evaluator_returns_infeasible_flights_ranked_by_missing_distance() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival(
                flight_id="ARR_OK",
                callsign="OK100",
                icao24="ok100",
                success=True,
                final_threshold_error_m=500.0,
                message="ok",
            ),
            _arrival(
                flight_id="ARR_SMALL",
                callsign="SMALL200",
                icao24="sm200",
                runway="RW36L",
                final_threshold_error_m=1_852.0,
                message="short by one nautical mile",
            ),
            _arrival(
                flight_id="ARR_LARGE",
                callsign="LARGE300",
                icao24="lg300",
                runway="RW35C",
                final_threshold_error_m=3_704.0,
                message="short by two nautical miles",
            ),
        ]
    )

    result = FeasibleEvaluator(manager).evaluate()

    assert result == [
        FeasibleFlight(
            flight_number="LARGE300",
            icao24="lg300",
            flight_id="ARR_LARGE",
            runway="RW35C",
            missing_distance_nmi=2.0,
            missing_distance_m=3_704.0,
            simulation_message="short by two nautical miles",
        ),
        FeasibleFlight(
            flight_number="SMALL200",
            icao24="sm200",
            flight_id="ARR_SMALL",
            runway="RW36L",
            missing_distance_nmi=1.0,
            missing_distance_m=1_852.0,
            simulation_message="short by one nautical mile",
        ),
    ]


def test_feasible_evaluator_uses_served_arrival_schedule() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival(
                flight_id="DIFFED",
                callsign="DIFF400",
                icao24="df400",
                final_threshold_error_m=926.0,
                message="diff-applied arrival still infeasible",
            )
        ]
    )

    result = FeasibleEvaluator(manager).evaluate()

    assert manager.arrival_schedule_calls == 1
    assert len(result) == 1
    assert result[0].flight_number == "DIFF400"
    assert result[0].missing_distance_nmi == pytest.approx(0.5)


def test_feasible_evaluator_requires_simulation_metadata() -> None:
    manager = ArrivalScheduleManager(
        [
            {
                "flight_id": "ARR_MISSING",
                "callsign": "MISS500",
                "icao24": "ms500",
                "runway": "RW35C",
            }
        ]
    )

    with pytest.raises(ValueError, match="flight_id=ARR_MISSING callsign=MISS500"):
        FeasibleEvaluator(manager).evaluate()


@pytest.mark.parametrize("final_threshold_error_m", ["12.0", True, float("nan"), -1.0])
def test_feasible_evaluator_rejects_malformed_missing_distance(final_threshold_error_m: object) -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival(
                flight_id="ARR_BAD_DISTANCE",
                callsign="BAD600",
                icao24="bd600",
                final_threshold_error_m=final_threshold_error_m,
            )
        ]
    )

    with pytest.raises(ValueError, match="simulation.final_threshold_error_m"):
        FeasibleEvaluator(manager).evaluate()


def test_feasible_evaluator_requires_success_flag_on_every_arrival() -> None:
    arrival = _arrival(
        flight_id="ARR_BAD_SUCCESS",
        callsign="BAD700",
        icao24="bd700",
        success=False,
        final_threshold_error_m=100.0,
    )
    arrival["simulation"]["success"] = "false"
    manager = ArrivalScheduleManager([arrival])

    with pytest.raises(ValueError, match="simulation.success"):
        FeasibleEvaluator(manager).evaluate()
