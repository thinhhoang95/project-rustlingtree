from __future__ import annotations

from typing import Any

import pytest

from mcp_tools.evaluators import (
    RunwayOverlapEvaluator,
    RunwayOverlapEvent,
    RunwayUseFlight,
)


class ServedScheduleManager:
    def __init__(
        self,
        *,
        arrivals: list[dict[str, Any]] | None = None,
        departures: list[dict[str, Any]] | None = None,
    ) -> None:
        self._arrivals = arrivals or []
        self._departures = departures or []
        self.arrival_schedule_calls = 0
        self.departure_schedule_calls = 0

    @property
    def events(self) -> Any:
        raise AssertionError("RunwayOverlapEvaluator must use the served schedules")

    @property
    def simap_arrival_flights(self) -> dict[str, Any]:
        raise AssertionError("RunwayOverlapEvaluator must use the served schedules")

    @property
    def adsb_compressed_flights(self) -> dict[str, Any]:
        raise AssertionError("RunwayOverlapEvaluator must use the served schedules")

    @property
    def diff(self) -> list[dict[str, Any]]:
        raise AssertionError("RunwayOverlapEvaluator must use the served schedules")

    def arrival_schedule(self) -> list[dict[str, Any]]:
        self.arrival_schedule_calls += 1
        return list(self._arrivals)

    def departure_schedule(self) -> list[dict[str, Any]]:
        self.departure_schedule_calls += 1
        return list(self._departures)


def _arrival(
    flight_id: str,
    threshold_time: object,
    *,
    callsign: str | None = None,
    icao24: str | None = None,
    runway: str = "RW35C",
) -> dict[str, Any]:
    return {
        "flight_id": flight_id,
        "callsign": callsign or f"{flight_id}CALL",
        "icao24": icao24 or flight_id.lower(),
        "runway": runway,
        "time_at_last_event": threshold_time,
    }


def _departure(
    flight_id: str,
    departure_time: object,
    *,
    callsign: str | None = None,
    icao24: str | None = None,
    runway: str = "RW35C",
) -> dict[str, Any]:
    return {
        "flight_id": flight_id,
        "callsign": callsign or f"{flight_id}CALL",
        "icao24": icao24 or flight_id.lower(),
        "runway": runway,
        "departure_time": departure_time,
    }


def test_runway_overlap_evaluator_detects_arrival_departure_overlap_from_served_schedules() -> None:
    manager = ServedScheduleManager(
        arrivals=[_arrival("ARR_DIFFED", 1_000, callsign="ARR100", runway="RW35C")],
        departures=[_departure("DEP_DIFFED", 1_030, callsign="DEP200", runway="17C")],
    )

    result = RunwayOverlapEvaluator(manager).evaluate()

    assert manager.arrival_schedule_calls == 1
    assert manager.departure_schedule_calls == 1
    assert result == [
        RunwayOverlapEvent(
            runway="17C/35C",
            use_a=RunwayUseFlight(
                flight_number="ARR100",
                icao24="arr_diffed",
                flight_id="ARR_DIFFED",
                operation="arrival",
                runway="RW35C",
            ),
            use_b=RunwayUseFlight(
                flight_number="DEP200",
                icao24="dep_diffed",
                flight_id="DEP_DIFFED",
                operation="departure",
                runway="17C",
            ),
            start_time=1_030,
            end_time=1_060,
            overlapping_time=1_030,
            overlapping_time_utc="1970-01-01T00:17:10Z",
            overlapping_duration=30,
        )
    ]


def test_runway_overlap_evaluator_detects_departure_departure_overlap_on_reciprocal_ends() -> None:
    manager = ServedScheduleManager(
        departures=[
            _departure("DEP_A", 100, runway="RW18L"),
            _departure("DEP_B", 180, runway="36R"),
        ]
    )

    result = RunwayOverlapEvaluator(manager).evaluate()

    assert len(result) == 1
    event = result[0]
    assert event.runway == "18L/36R"
    assert event.use_a.flight_id == "DEP_A"
    assert event.use_b.flight_id == "DEP_B"
    assert event.overlapping_time == 180
    assert event.overlapping_duration == 10


def test_runway_overlap_evaluator_detects_arrival_arrival_overlap_on_same_runway() -> None:
    manager = ServedScheduleManager(
        arrivals=[
            _arrival("ARR_A", 300, runway="35C"),
            _arrival("ARR_B", 350, runway="RW35C"),
        ]
    )

    result = RunwayOverlapEvaluator(manager).evaluate()

    assert len(result) == 1
    event = result[0]
    assert event.runway == "17C/35C"
    assert event.use_a.operation == "arrival"
    assert event.use_b.operation == "arrival"
    assert event.overlapping_time == 350
    assert event.overlapping_duration == 10


def test_runway_overlap_evaluator_ignores_touching_intervals_and_different_runways() -> None:
    manager = ServedScheduleManager(
        arrivals=[
            _arrival("ARR_TOUCHING", 100, runway="RW35C"),
            _arrival("ARR_OTHER_RUNWAY", 120, runway="RW36L"),
        ],
        departures=[
            _departure("DEP_TOUCHING", 160, runway="RW17C"),
            _departure("DEP_OTHER_RUNWAY", 130, runway="RW13L"),
        ],
    )

    assert RunwayOverlapEvaluator(manager).evaluate() == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_arrival("ARR_BAD_TIME", "100"), "time_at_last_event"),
        (_departure("DEP_BAD_TIME", float("nan")), "departure_time"),
        (_arrival("ARR_BAD_RUNWAY", 100, runway=""), "runway"),
    ],
)
def test_runway_overlap_evaluator_rejects_malformed_served_schedule_fields(
    payload: dict[str, Any],
    message: str,
) -> None:
    manager = ServedScheduleManager(
        arrivals=[payload] if "time_at_last_event" in payload else [],
        departures=[payload] if "departure_time" in payload else [],
    )

    with pytest.raises(ValueError, match=message):
        RunwayOverlapEvaluator(manager).evaluate()
