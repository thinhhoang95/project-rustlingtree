from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re
from typing import Any, Protocol

DEPARTURE_RUNWAY_OCCUPANCY_S = 90
ARRIVAL_RUNWAY_OCCUPANCY_S = 60

_RUNWAY_PATTERN = re.compile(r"^(\d{1,2})([LCR]?)$")
_RUNWAY_SIDE_RECIPROCAL = {"L": "R", "R": "L", "C": "C", "": ""}


class ScheduleProvider(Protocol):
    def arrival_schedule(self) -> list[dict[str, Any]]: ...

    def departure_schedule(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class RunwayUseFlight:
    flight_number: str
    icao24: str
    flight_id: str
    operation: str
    runway: str


@dataclass(frozen=True)
class RunwayUse:
    flight: RunwayUseFlight
    runway: str
    start_time: int
    end_time: int


@dataclass(frozen=True)
class RunwayOverlapEvent:
    runway: str
    use_a: RunwayUseFlight
    use_b: RunwayUseFlight
    start_time: int
    end_time: int
    overlapping_time: int
    overlapping_time_utc: str
    overlapping_duration: int


@dataclass(frozen=True)
class RunwayOverlapEvaluator:
    manager: ScheduleProvider

    def evaluate(self) -> list[RunwayOverlapEvent]:
        uses = [
            *(_runway_use_from_arrival(arrival) for arrival in self.manager.arrival_schedule()),
            *(_runway_use_from_departure(departure) for departure in self.manager.departure_schedule()),
        ]
        if len(uses) < 2:
            return []

        uses_by_runway: dict[str, list[RunwayUse]] = {}
        for use in uses:
            uses_by_runway.setdefault(use.runway, []).append(use)

        overlaps: list[RunwayOverlapEvent] = []
        for runway_uses in uses_by_runway.values():
            runway_uses.sort(key=_runway_use_sort_key)
            overlaps.extend(_overlaps_for_runway(runway_uses))

        return sorted(
            overlaps,
            key=lambda event: (
                -event.overlapping_duration,
                event.overlapping_time,
                event.runway,
                event.use_a.flight_id,
                event.use_b.flight_id,
            ),
        )


def _runway_use_from_arrival(arrival: dict[str, Any]) -> RunwayUse:
    threshold_time = _required_time(arrival.get("time_at_last_event"), "time_at_last_event", arrival)
    runway = _required_runway(arrival)
    return RunwayUse(
        flight=RunwayUseFlight(
            flight_number=str(arrival.get("callsign", "")),
            icao24=str(arrival.get("icao24", "")),
            flight_id=str(arrival.get("flight_id", "")),
            operation="arrival",
            runway=runway,
        ),
        runway=_physical_runway_key(runway),
        start_time=threshold_time,
        end_time=threshold_time + ARRIVAL_RUNWAY_OCCUPANCY_S,
    )


def _runway_use_from_departure(departure: dict[str, Any]) -> RunwayUse:
    departure_time = _required_time(departure.get("departure_time"), "departure_time", departure)
    runway = _required_runway(departure)
    return RunwayUse(
        flight=RunwayUseFlight(
            flight_number=str(departure.get("callsign", "")),
            icao24=str(departure.get("icao24", "")),
            flight_id=str(departure.get("flight_id", "")),
            operation="departure",
            runway=runway,
        ),
        runway=_physical_runway_key(runway),
        start_time=departure_time,
        end_time=departure_time + DEPARTURE_RUNWAY_OCCUPANCY_S,
    )


def _overlaps_for_runway(uses: list[RunwayUse]) -> list[RunwayOverlapEvent]:
    overlaps: list[RunwayOverlapEvent] = []
    for index, first in enumerate(uses):
        for second in uses[index + 1 :]:
            if second.start_time >= first.end_time:
                break

            overlap_start = max(first.start_time, second.start_time)
            overlap_end = min(first.end_time, second.end_time)
            if overlap_start >= overlap_end:
                continue

            overlaps.append(
                RunwayOverlapEvent(
                    runway=first.runway,
                    use_a=first.flight,
                    use_b=second.flight,
                    start_time=overlap_start,
                    end_time=overlap_end,
                    overlapping_time=overlap_start,
                    overlapping_time_utc=_time_utc(overlap_start),
                    overlapping_duration=overlap_end - overlap_start,
                )
            )
    return overlaps


def _runway_use_sort_key(use: RunwayUse) -> tuple[int, int, str, str, str]:
    return (
        use.start_time,
        use.end_time,
        use.flight.operation,
        use.flight.flight_id,
        use.flight.flight_number,
    )


def _required_time(value: Any, field_name: str, payload: dict[str, Any]) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{_flight_label(payload)} has missing or malformed {field_name}")
    time_s = float(value)
    if not math.isfinite(time_s):
        raise ValueError(f"{_flight_label(payload)} has missing or malformed {field_name}")
    return int(round(time_s))


def _required_runway(payload: dict[str, Any]) -> str:
    runway = payload.get("runway")
    if not isinstance(runway, str) or not runway.strip():
        raise ValueError(f"{_flight_label(payload)} has missing or malformed runway")
    return runway.strip().upper()


def _physical_runway_key(runway: str) -> str:
    normalized = runway.removeprefix("RWY").removeprefix("RW")
    match = _RUNWAY_PATTERN.fullmatch(normalized)
    if match is None:
        return normalized

    runway_number = int(match.group(1))
    side = match.group(2)
    if runway_number < 1 or runway_number > 36:
        return normalized

    reciprocal_number = ((runway_number + 17) % 36) + 1
    reciprocal_side = _RUNWAY_SIDE_RECIPROCAL[side]
    ends = sorted(
        (
            _format_runway_end(runway_number, side),
            _format_runway_end(reciprocal_number, reciprocal_side),
        )
    )
    return "/".join(ends)


def _format_runway_end(runway_number: int, side: str) -> str:
    return f"{runway_number:02d}{side}"


def _time_utc(time_s: int) -> str:
    return datetime.fromtimestamp(time_s, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flight_label(payload: dict[str, Any]) -> str:
    operation = str(payload.get("operation", "flight"))
    flight_id = str(payload.get("flight_id", ""))
    callsign = str(payload.get("callsign", ""))
    if flight_id and callsign:
        return f"{operation} flight_id={flight_id} callsign={callsign}"
    if flight_id:
        return f"{operation} flight_id={flight_id}"
    if callsign:
        return f"{operation} callsign={callsign}"
    return operation
