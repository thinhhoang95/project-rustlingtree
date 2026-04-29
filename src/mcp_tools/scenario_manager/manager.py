from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from mcp_tools.scenario_manager.models import ScenarioResourceConfig
from mcp_tools.scenario_manager.resources import (
    load_compressed_flights,
    load_events,
    load_fix_sequences,
)


class ScenarioManager:
    def __init__(self, config: ScenarioResourceConfig | None = None) -> None:
        self.config = config or ScenarioResourceConfig.default()
        self.events = load_events(self.config.events_path)
        self.fix_sequences = load_fix_sequences(self.config.fix_sequences_path)
        self.compressed_flights = load_compressed_flights(self.config.compressed_flights_path)
        self._fix_sequence_by_flight_id = self._build_fix_sequence_index(self.fix_sequences)

    @staticmethod
    def _build_fix_sequence_index(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for row in frame.to_dict("records"):
            index[str(row["flight_id"])] = row
        return index

    @property
    def arrivals(self) -> pd.DataFrame:
        return self.events.loc[self.events["operation"] == "arrival"].copy()

    @property
    def departures(self) -> pd.DataFrame:
        return self.events.loc[self.events["operation"] == "departure"].copy()

    def departure_schedule(self) -> list[dict[str, Any]]:
        schedule: list[dict[str, Any]] = []
        departures = self.departures.sort_values(["event_time", "flight_id"], kind="stable")
        for row in departures.to_dict("records"):
            schedule.append(
                {
                    "flight_id": str(row["flight_id"]),
                    "callsign": str(row["callsign"]),
                    "icao24": str(row["icao24"]),
                    "departure_time": int(row["event_time"]),
                    "departure_time_utc": str(row["event_time_utc"]),
                    "runway": str(row["runway"]),
                }
            )
        return schedule

    def arrival_schedule(self) -> list[dict[str, Any]]:
        arrivals: list[dict[str, Any]] = []
        for event in self.arrivals.to_dict("records"):
            flight_id = str(event["flight_id"])
            fix_sequence = self._fix_sequence_by_flight_id.get(flight_id)
            trajectory = self.compressed_flights.get(flight_id)
            if fix_sequence is None or trajectory is None:
                continue

            payload = dict(trajectory)
            payload.update(
                {
                    "arrival_time": int(fix_sequence["first_time"]),
                    "arrival_time_utc": self._arrival_time_utc(payload, int(fix_sequence["first_time"])),
                    "runway": str(event["runway"]),
                    "original_fix_sequence": str(fix_sequence["fix_sequence"]),
                    "original_fix_count": int(fix_sequence["fix_count"]),
                }
            )
            arrivals.append(payload)

        arrivals.sort(key=lambda item: (int(item["arrival_time"]), str(item["flight_id"])))
        return arrivals

    @staticmethod
    def _arrival_time_utc(_trajectory: dict[str, Any], arrival_time: int) -> str:
        return datetime.fromtimestamp(arrival_time, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def health(self) -> dict[str, Any]:
        arrival_ids = set(self.arrivals["flight_id"].astype(str))
        departure_ids = set(self.departures["flight_id"].astype(str))
        fix_ids = set(self._fix_sequence_by_flight_id)
        trajectory_ids = set(self.compressed_flights)

        return {
            "status": "ok",
            "events_count": int(len(self.events)),
            "arrivals_count": int(len(self.arrivals)),
            "departures_count": int(len(self.departures)),
            "fix_sequences_count": int(len(fix_ids)),
            "compressed_flights_count": int(len(trajectory_ids)),
            "arrivals_missing_fix_sequences_count": int(len(arrival_ids - fix_ids)),
            "arrivals_missing_trajectories_count": int(len(arrival_ids - trajectory_ids)),
            "departures_missing_trajectories_count": int(len(departure_ids - trajectory_ids)),
            "resource_paths": {
                "events": self.config.events_path.as_posix(),
                "fix_sequences": self.config.fix_sequences_path.as_posix(),
                "compressed_flights": self.config.compressed_flights_path.as_posix(),
            },
        }
