from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from mcp_tools.scenario_manager.models import ScenarioResourceConfig
from mcp_tools.scenario_manager.resources import (
    load_compressed_flights,
    load_events,
    load_fix_sequences,
    load_json_object,
)


class ScenarioManager:
    def __init__(self, config: ScenarioResourceConfig | None = None) -> None:
        self.config = config or ScenarioResourceConfig.default()
        self.events = load_events(self.config.events_path)
        self.fix_sequences = load_fix_sequences(self.config.fix_sequences_path)
        self.artifact_flights = load_compressed_flights(self.config.artifact_flights_path)
        self.artifact_manifest = load_json_object(self.config.artifact_manifest_path)
        self.diff: list[dict[str, Any]] = []
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
            trajectory = self.artifact_flights.get(flight_id)
            if fix_sequence is None or trajectory is None:
                continue

            payload = self._apply_diff(dict(trajectory))
            arrival_time = int(payload.get("first_time", fix_sequence["first_time"]))
            payload.update(
                {
                    "arrival_time": arrival_time,
                    "arrival_time_utc": self._arrival_time_utc(payload, arrival_time),
                    "runway": str(event["runway"]),
                    "original_fix_sequence": str(fix_sequence["fix_sequence"]),
                    "original_fix_count": int(fix_sequence["fix_count"]),
                }
            )
            arrivals.append(payload)

        arrivals.sort(key=lambda item: (int(item["arrival_time"]), str(item["flight_id"])))
        return arrivals

    def _apply_diff(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Intervention patching will be implemented later. Keep the hook wired
        # so API behavior is already centered on artifact + diff state.
        return payload

    def intervention_diff(self) -> list[dict[str, Any]]:
        return list(self.diff)

    @staticmethod
    def _arrival_time_utc(_trajectory: dict[str, Any], arrival_time: int) -> str:
        return datetime.fromtimestamp(arrival_time, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def health(self) -> dict[str, Any]:
        arrival_ids = set(self.arrivals["flight_id"].astype(str))
        departure_ids = set(self.departures["flight_id"].astype(str))
        fix_ids = set(self._fix_sequence_by_flight_id)
        trajectory_ids = set(self.artifact_flights)
        skipped_departures_count = int(self.artifact_manifest.get("skipped_departure_count", len(departure_ids)))

        return {
            "status": "ok",
            "events_count": int(len(self.events)),
            "arrivals_count": int(len(self.arrivals)),
            "departures_count": int(len(self.departures)),
            "fix_sequences_count": int(len(fix_ids)),
            "compressed_flights_count": int(len(trajectory_ids)),
            "artifact_flights_count": int(len(trajectory_ids)),
            "arrivals_missing_fix_sequences_count": int(len(arrival_ids - fix_ids)),
            "arrivals_missing_trajectories_count": int(len(arrival_ids - trajectory_ids)),
            "arrivals_missing_artifacts_count": int(len(arrival_ids - trajectory_ids)),
            "departures_missing_trajectories_count": int(len(departure_ids - trajectory_ids)),
            "skipped_departures_count": skipped_departures_count,
            "resource_paths": {
                "events": self.config.events_path.as_posix(),
                "fix_sequences": self.config.fix_sequences_path.as_posix(),
                "compressed_flights": self.config.artifact_flights_path.as_posix(),
                "artifact_flights": self.config.artifact_flights_path.as_posix(),
                "artifact_manifest": self.config.artifact_manifest_path.as_posix(),
            },
        }
