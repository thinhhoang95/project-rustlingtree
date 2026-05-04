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
        self.simap_arrival_flights = load_compressed_flights(self.config.simap_arrival_trajectories_path)
        self.adsb_compressed_flights = load_compressed_flights(self.config.adsb_compressed_trajectories_path)
        self.simap_arrival_artifact_manifest = (
            load_json_object(self.config.simap_arrival_artifact_manifest_path)
            if self.config.simap_arrival_artifact_manifest_path is not None
            else {}
        )
        self.adsb_compressed_metadata = (
            load_json_object(self.config.adsb_compressed_metadata_path)
            if self.config.adsb_compressed_metadata_path is not None
            else {}
        )
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
            flight_id = str(row["flight_id"])
            trajectory = self.adsb_compressed_flights.get(flight_id)
            if trajectory is None:
                continue

            payload = dict(trajectory)
            payload.update(
                {
                    "flight_id": flight_id,
                    "callsign": str(row["callsign"]),
                    "icao24": str(row["icao24"]),
                    "departure_time": int(row["event_time"]),
                    "departure_time_utc": str(row["event_time_utc"]),
                    "runway": str(row["runway"]),
                }
            )
            schedule.append(payload)
        return schedule

    def arrival_schedule(self) -> list[dict[str, Any]]:
        arrivals: list[dict[str, Any]] = []
        for event in self.arrivals.to_dict("records"):
            flight_id = str(event["flight_id"])
            fix_sequence = self._fix_sequence_by_flight_id.get(flight_id)
            trajectory = self.simap_arrival_flights.get(flight_id)
            if fix_sequence is None or trajectory is None:
                continue

            payload = self._apply_diff(dict(trajectory))
            time_at_first_fix = int(payload.get("first_time", fix_sequence["first_time"]))
            time_at_last_event = int(payload.get("last_time", fix_sequence["last_time"]))
            payload.update(
                {
                    "time_at_first_fix": time_at_first_fix,
                    "time_at_first_fix_utc": self._arrival_time_utc(payload, time_at_first_fix),
                    "time_at_last_event": time_at_last_event,
                    "time_at_last_event_utc": self._arrival_time_utc(payload, time_at_last_event),
                    "runway": str(event["runway"]),
                    "original_fix_sequence": str(fix_sequence["fix_sequence"]),
                    "original_fix_count": int(fix_sequence["fix_count"]),
                }
            )
            arrivals.append(payload)

        arrivals.sort(key=lambda item: (int(item["time_at_first_fix"]), str(item["flight_id"])))
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
        simap_arrival_trajectory_ids = set(self.simap_arrival_flights)
        adsb_compressed_trajectory_ids = set(self.adsb_compressed_flights)
        skipped_departures_count = int(
            self.simap_arrival_artifact_manifest.get("skipped_departure_count", len(departure_ids))
        )

        return {
            "status": "ok",
            "events_count": int(len(self.events)),
            "arrivals_count": int(len(self.arrivals)),
            "departures_count": int(len(self.departures)),
            "fix_sequences_count": int(len(fix_ids)),
            "compressed_flights_count": int(len(adsb_compressed_trajectory_ids)),
            "artifact_flights_count": int(len(simap_arrival_trajectory_ids)),
            "simap_arrival_flights_count": int(len(simap_arrival_trajectory_ids)),
            "adsb_compressed_flights_count": int(len(adsb_compressed_trajectory_ids)),
            "arrivals_missing_fix_sequences_count": int(len(arrival_ids - fix_ids)),
            "arrivals_missing_trajectories_count": int(len(arrival_ids - simap_arrival_trajectory_ids)),
            "arrivals_missing_artifacts_count": int(len(arrival_ids - simap_arrival_trajectory_ids)),
            "arrivals_missing_simap_trajectories_count": int(len(arrival_ids - simap_arrival_trajectory_ids)),
            "departures_missing_trajectories_count": int(len(departure_ids - adsb_compressed_trajectory_ids)),
            "departures_missing_adsb_trajectories_count": int(len(departure_ids - adsb_compressed_trajectory_ids)),
            "skipped_departures_count": skipped_departures_count,
            "resource_paths": {
                "data_manifest": self.config.data_manifest_path.as_posix()
                if self.config.data_manifest_path is not None
                else "",
                "data_date": self.config.data_date or "",
                "events": self.config.events_path.as_posix(),
                "landings_and_departures": self.config.events_path.as_posix(),
                "fix_sequences": self.config.fix_sequences_path.as_posix(),
                "simap_arrival_trajectories": self.config.simap_arrival_trajectories_path.as_posix(),
                "simap_arrival_artifact_manifest": self.config.simap_arrival_artifact_manifest_path.as_posix()
                if self.config.simap_arrival_artifact_manifest_path is not None
                else "",
                "adsb_compressed_trajectories": self.config.adsb_compressed_trajectories_path.as_posix(),
                "adsb_compressed_metadata": self.config.adsb_compressed_metadata_path.as_posix()
                if self.config.adsb_compressed_metadata_path is not None
                else "",
            },
        }
