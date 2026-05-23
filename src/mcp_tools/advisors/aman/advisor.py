from __future__ import annotations

from dataclasses import dataclass

from mcp_tools.advisors.models import AmanAdvisory
from mcp_tools.evaluators.runway_overlap import (
    ARRIVAL_RUNWAY_OCCUPANCY_S,
    RunwayUse,
    ScheduleProvider,
    runway_use_from_arrival,
    runway_use_from_departure,
    time_utc,
)


@dataclass(frozen=True)
class AmanAdvisor:
    manager: ScheduleProvider

    def evaluate(self) -> list[AmanAdvisory]:
        """Return arrival delays that clear all runway overlaps involving arrivals."""
        arrivals_by_runway: dict[str, list[RunwayUse]] = {}
        departures_by_runway: dict[str, list[RunwayUse]] = {}

        for arrival in self.manager.arrival_schedule():
            use = runway_use_from_arrival(arrival)
            arrivals_by_runway.setdefault(use.runway, []).append(use)

        for departure in self.manager.departure_schedule():
            use = runway_use_from_departure(departure)
            departures_by_runway.setdefault(use.runway, []).append(use)

        advisories: list[AmanAdvisory] = []
        runway_keys = set(arrivals_by_runway) | set(departures_by_runway)
        for runway in sorted(runway_keys):
            allocated = sorted(departures_by_runway.get(runway, ()), key=_interval_sort_key)
            arrivals = sorted(arrivals_by_runway.get(runway, ()), key=_arrival_fcfs_sort_key)
            for arrival in arrivals:
                advised_start = _earliest_nonoverlapping_start(
                    arrival.start_time,
                    ARRIVAL_RUNWAY_OCCUPANCY_S,
                    allocated,
                )
                if advised_start > arrival.start_time:
                    advisories.append(_advisory(arrival, advised_start))
                allocated.append(
                    RunwayUse(
                        flight=arrival.flight,
                        runway=arrival.runway,
                        start_time=advised_start,
                        end_time=advised_start + ARRIVAL_RUNWAY_OCCUPANCY_S,
                    )
                )
                allocated.sort(key=_interval_sort_key)

        return sorted(
            advisories,
            key=lambda item: (
                item.original_time_at_last_event,
                item.physical_runway,
                item.flight_number,
                item.icao24,
                item.flight_id,
            ),
        )


def _earliest_nonoverlapping_start(
    requested_start: int,
    duration_s: int,
    blockers: list[RunwayUse],
) -> int:
    candidate = requested_start
    while True:
        for blocker in blockers:
            if blocker.end_time <= candidate:
                continue
            if candidate + duration_s <= blocker.start_time:
                return candidate
            candidate = blocker.end_time
            break
        else:
            return candidate


def _advisory(arrival: RunwayUse, advised_start: int) -> AmanAdvisory:
    delay_s = advised_start - arrival.start_time
    flight = arrival.flight
    return AmanAdvisory(
        flight_number=flight.flight_number,
        icao24=flight.icao24,
        flight_id=flight.flight_id,
        runway=flight.runway,
        physical_runway=arrival.runway,
        original_time_at_last_event=arrival.start_time,
        original_time_at_last_event_utc=time_utc(arrival.start_time),
        advised_time_at_last_event=advised_start,
        advised_time_at_last_event_utc=time_utc(advised_start),
        seconds_to_gain=delay_s,
        minutes_to_gain=float(delay_s / 60.0),
    )


def _arrival_fcfs_sort_key(use: RunwayUse) -> tuple[int, int, str, str]:
    return (
        use.start_time,
        use.end_time,
        use.flight.flight_id,
        use.flight.flight_number,
    )


def _interval_sort_key(use: RunwayUse) -> tuple[int, int, str, str, str]:
    return (
        use.start_time,
        use.end_time,
        use.flight.operation,
        use.flight.flight_id,
        use.flight.flight_number,
    )
