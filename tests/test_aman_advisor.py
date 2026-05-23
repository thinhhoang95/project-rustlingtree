from __future__ import annotations

from typing import Any

import pytest

from mcp_tools.advisors import AmanAdvisor
from mcp_tools.evaluators import RunwayOverlapEvaluator


class ServedScheduleManager:
    def __init__(
        self,
        *,
        arrivals: list[dict[str, Any]] | None = None,
        departures: list[dict[str, Any]] | None = None,
    ) -> None:
        self._arrivals = arrivals or []
        self._departures = departures or []

    def arrival_schedule(self) -> list[dict[str, Any]]:
        return [dict(arrival) for arrival in self._arrivals]

    def departure_schedule(self) -> list[dict[str, Any]]:
        return [dict(departure) for departure in self._departures]


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


def test_aman_returns_empty_when_no_arrival_runway_overlaps() -> None:
    manager = ServedScheduleManager(
        arrivals=[
            _arrival("ARR_A", 100),
            _arrival("ARR_B", 200),
        ],
        departures=[_departure("DEP_A", 300)],
    )

    assert AmanAdvisor(manager).evaluate() == []


def test_aman_delays_later_arrival_for_arrival_arrival_overlap() -> None:
    manager = ServedScheduleManager(
        arrivals=[
            _arrival("ARR_A", 100),
            _arrival("ARR_B", 150, callsign="BARR"),
        ]
    )

    result = AmanAdvisor(manager).evaluate()

    assert len(result) == 1
    assert result[0].flight_id == "ARR_B"
    assert result[0].flight_number == "BARR"
    assert result[0].physical_runway == "17C/35C"
    assert result[0].original_time_at_last_event == 150
    assert result[0].advised_time_at_last_event == 160
    assert result[0].seconds_to_gain == 10
    assert result[0].minutes_to_gain == pytest.approx(10.0 / 60.0)


def test_aman_delays_arrival_after_fixed_departure_when_arrival_starts_first() -> None:
    manager = ServedScheduleManager(
        arrivals=[_arrival("ARR_A", 100)],
        departures=[_departure("DEP_A", 130)],
    )

    result = AmanAdvisor(manager).evaluate()

    assert len(result) == 1
    assert result[0].flight_id == "ARR_A"
    assert result[0].original_time_at_last_event == 100
    assert result[0].advised_time_at_last_event == 220
    assert result[0].seconds_to_gain == 120


def test_aman_cascades_delays_across_departures_and_allocated_arrivals() -> None:
    manager = ServedScheduleManager(
        arrivals=[
            _arrival("ARR_A", 100),
            _arrival("ARR_B", 120),
            _arrival("ARR_C", 220),
        ],
        departures=[_departure("DEP_A", 140)],
    )

    result = AmanAdvisor(manager).evaluate()

    assert [(item.flight_id, item.advised_time_at_last_event, item.seconds_to_gain) for item in result] == [
        ("ARR_A", 230, 130),
        ("ARR_B", 290, 170),
        ("ARR_C", 350, 130),
    ]


def test_aman_groups_reciprocal_runway_ends() -> None:
    manager = ServedScheduleManager(
        arrivals=[
            _arrival("ARR_A", 100, runway="RW35C"),
            _arrival("ARR_B", 150, runway="17C"),
        ]
    )

    result = AmanAdvisor(manager).evaluate()

    assert len(result) == 1
    assert result[0].flight_id == "ARR_B"
    assert result[0].physical_runway == "17C/35C"
    assert result[0].advised_time_at_last_event == 160


def test_aman_rejects_malformed_served_schedule_fields() -> None:
    manager = ServedScheduleManager(arrivals=[_arrival("ARR_BAD_TIME", "100")])

    with pytest.raises(ValueError, match="time_at_last_event"):
        AmanAdvisor(manager).evaluate()


def test_aman_advice_clears_all_arrival_involving_runway_overlaps() -> None:
    arrivals = [
        _arrival("ARR_A", 100),
        _arrival("ARR_B", 120),
        _arrival("ARR_C", 220),
        _arrival("ARR_OTHER_RUNWAY", 120, runway="RW36L"),
    ]
    departures = [
        _departure("DEP_A", 140),
        _departure("DEP_B", 150),
        _departure("DEP_OTHER_RUNWAY", 125, runway="RW18R"),
    ]
    manager = ServedScheduleManager(arrivals=arrivals, departures=departures)
    advice_by_flight_id = {item.flight_id: item.seconds_to_gain for item in AmanAdvisor(manager).evaluate()}
    advised_arrivals = []
    for arrival in arrivals:
        advised = dict(arrival)
        advised["time_at_last_event"] = int(advised["time_at_last_event"]) + advice_by_flight_id.get(
            str(advised["flight_id"]),
            0,
        )
        advised_arrivals.append(advised)

    remaining = RunwayOverlapEvaluator(
        ServedScheduleManager(arrivals=advised_arrivals, departures=departures)
    ).evaluate()
    arrival_involving = [
        event
        for event in remaining
        if event.use_a.operation == "arrival" or event.use_b.operation == "arrival"
    ]

    assert arrival_involving == []
