from __future__ import annotations

from typing import Any

import pytest

from mcp_tools.evaluators import ConflictEvaluator


class ArrivalScheduleManager:
    def __init__(self, arrivals: list[dict[str, Any]]) -> None:
        self._arrivals = arrivals
        self.arrival_schedule_calls = 0

    @property
    def simap_arrival_flights(self) -> dict[str, Any]:
        raise AssertionError("ConflictEvaluator must use the served arrival schedule")

    @property
    def diff(self) -> list[dict[str, Any]]:
        raise AssertionError("ConflictEvaluator must use the served arrival schedule")

    @property
    def simap_arrival_artifact_manifest(self) -> dict[str, Any]:
        raise AssertionError("ConflictEvaluator must use the served arrival schedule")

    def arrival_schedule(self) -> list[dict[str, Any]]:
        self.arrival_schedule_calls += 1
        return list(self._arrivals)


def _arrival(
    flight_id: str,
    callsign: str,
    points: list[list[float]],
    *,
    icao24: str | None = None,
    runway: str = "RW35C",
    lateral_tolerance_m: float = 0.0,
    altitude_tolerance_m: float = 0.0,
) -> dict[str, Any]:
    return {
        "flight_id": flight_id,
        "callsign": callsign,
        "icao24": icao24 or flight_id.lower(),
        "runway": runway,
        "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        "points": points,
        "lateral_tolerance_m": lateral_tolerance_m,
        "altitude_tolerance_m": altitude_tolerance_m,
    }


def test_conflict_evaluator_detects_overlapping_cylinders_from_served_arrivals() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival("ARR_A", "A100", [[0, 32.0, -97.0, 1_000.0, 3], [60, 32.01, -97.0, 1_000.0, 3]]),
            _arrival("ARR_B", "B200", [[0, 32.01, -97.0, 1_200.0, 3], [60, 32.02, -97.0, 1_200.0, 3]]),
        ]
    )

    result = ConflictEvaluator(manager).evaluate()

    assert manager.arrival_schedule_calls == 1
    assert len(result) == 1
    event = result[0]
    assert event.flight_a.flight_id == "ARR_A"
    assert event.flight_b.flight_id == "ARR_B"
    assert event.start_time == 0
    assert event.end_time == 60
    assert event.confidence == "confirmed"
    assert event.lateral_threshold_nmi == 10.0
    assert event.vertical_threshold_ft == 1_000.0
    assert event.lateral_distance_nmi < 1.0
    assert event.vertical_separation_ft == pytest.approx(656.168, rel=1.0e-3)
    assert event.severity > 0.0


def test_conflict_evaluator_rejects_lateral_only_overlap_when_vertical_clearance_is_safe() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival("ARR_A", "A100", [[0, 32.0, -97.0, 1_000.0, 3], [60, 32.01, -97.0, 1_000.0, 3]]),
            _arrival("ARR_B", "B200", [[0, 32.01, -97.0, 1_400.0, 3], [60, 32.02, -97.0, 1_400.0, 3]]),
        ]
    )

    assert ConflictEvaluator(manager).evaluate() == []


def test_conflict_evaluator_rejects_vertical_only_overlap_when_lateral_clearance_is_safe() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival("ARR_A", "A100", [[0, 32.0, -97.0, 1_000.0, 3], [60, 32.0, -97.0, 1_000.0, 3]]),
            _arrival("ARR_B", "B200", [[0, 32.19, -97.0, 1_000.0, 3], [60, 32.19, -97.0, 1_000.0, 3]]),
        ]
    )

    assert ConflictEvaluator(manager).evaluate() == []


def test_conflict_evaluator_uses_continuous_segment_closest_approach() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival("ARR_A", "A100", [[0, 32.0, -97.3, 1_000.0, 3], [60, 32.0, -96.7, 1_000.0, 3]]),
            _arrival("ARR_B", "B200", [[0, 32.0, -97.0, 1_000.0, 3], [60, 32.0, -97.0, 1_000.0, 3]]),
        ]
    )

    result = ConflictEvaluator(manager).evaluate()

    assert len(result) == 1
    event = result[0]
    assert 29 <= event.closest_time <= 31
    assert event.lateral_distance_nmi < 0.1
    assert event.vertical_separation_ft == 0.0


def test_conflict_evaluator_ignores_flights_without_temporal_overlap() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival("ARR_A", "A100", [[0, 32.0, -97.0, 1_000.0, 3], [60, 32.0, -97.0, 1_000.0, 3]]),
            _arrival("ARR_B", "B200", [[3_600, 32.0, -97.0, 1_000.0, 3], [3_660, 32.0, -97.0, 1_000.0, 3]]),
        ]
    )

    assert ConflictEvaluator(manager).evaluate() == []


def test_conflict_evaluator_reports_possible_conflict_when_compression_tolerance_bridges_gap() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival(
                "ARR_A",
                "A100",
                [[0, 32.0, -97.0, 1_000.0, 3], [60, 32.0, -97.0, 1_000.0, 3]],
                lateral_tolerance_m=100.0,
            ),
            _arrival(
                "ARR_B",
                "B200",
                [[0, 32.1675, -97.0, 1_000.0, 3], [60, 32.1675, -97.0, 1_000.0, 3]],
                lateral_tolerance_m=100.0,
            ),
        ]
    )

    result = ConflictEvaluator(manager).evaluate()

    assert len(result) == 1
    event = result[0]
    assert event.confidence == "possible"
    assert event.lateral_distance_nmi > event.lateral_threshold_nmi


def test_conflict_evaluator_merges_adjacent_segment_hits_for_same_flight_pair() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival(
                "ARR_A",
                "A100",
                [
                    [0, 32.0, -97.0, 1_000.0, 3],
                    [60, 32.01, -97.0, 1_000.0, 3],
                    [120, 32.02, -97.0, 1_000.0, 3],
                ],
            ),
            _arrival(
                "ARR_B",
                "B200",
                [
                    [0, 32.01, -97.0, 1_000.0, 3],
                    [60, 32.02, -97.0, 1_000.0, 3],
                    [120, 32.03, -97.0, 1_000.0, 3],
                ],
            ),
        ]
    )

    result = ConflictEvaluator(manager).evaluate()

    assert len(result) == 1
    assert result[0].start_time == 0
    assert result[0].end_time == 120


def test_conflict_evaluator_tolerates_duplicate_timestamps_in_artifact_points() -> None:
    manager = ArrivalScheduleManager(
        [
            _arrival(
                "ARR_A",
                "A100",
                [
                    [0, 32.0, -97.0, 1_000.0, 3],
                    [0, 32.0002, -97.0, 1_000.0, 1],
                    [60, 32.01, -97.0, 1_000.0, 3],
                ],
            ),
            _arrival("ARR_B", "B200", [[0, 32.01, -97.0, 1_000.0, 3], [60, 32.02, -97.0, 1_000.0, 3]]),
        ]
    )

    result = ConflictEvaluator(manager).evaluate()

    assert len(result) == 1
    assert result[0].confidence == "confirmed"


def test_conflict_evaluator_requires_geoaltitude_column() -> None:
    arrival = _arrival(
        "ARR_BAD",
        "BAD100",
        [[0, 32.0, -97.0, 1_000.0, 3], [60, 32.0, -97.0, 1_000.0, 3]],
    )
    arrival["columns"] = ["time", "lat", "lon", "altitude_m", "breakpoint_mask"]
    manager = ArrivalScheduleManager(
        [
            arrival,
            _arrival("ARR_OK", "OK200", [[0, 32.0, -97.0, 1_000.0, 3], [60, 32.0, -97.0, 1_000.0, 3]]),
        ]
    )

    with pytest.raises(ValueError, match="geoaltitude_m"):
        ConflictEvaluator(manager).evaluate()
