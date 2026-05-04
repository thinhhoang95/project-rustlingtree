from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ScenarioResourceConfig:
    events_path: Path
    fix_sequences_path: Path
    simap_arrival_trajectories_path: Path
    adsb_compressed_trajectories_path: Path
    data_manifest_path: Path | None = None
    data_date: str | None = None
    simap_arrival_artifact_manifest_path: Path | None = None
    adsb_compressed_metadata_path: Path | None = None

    @classmethod
    def default(cls) -> "ScenarioResourceConfig":
        root = project_root()
        return cls.from_manifest(root / "data_manifest.json")

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "ScenarioResourceConfig":
        root = project_root()
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path} must contain a JSON object")

        date_label, entry = cls._default_manifest_entry(manifest, manifest_path)
        events_path = cls._manifest_path(entry, "landings_and_departures", root, manifest_path)
        fix_sequences_path = cls._manifest_path(entry, "fix_sequences", root, manifest_path)
        simap_arrival_trajectories_path = cls._manifest_path(
            entry,
            "simap_arrival_trajectories",
            root,
            manifest_path,
        )
        adsb_compressed_trajectories_path = cls._manifest_path(
            entry,
            "adsb_compressed_trajectories",
            root,
            manifest_path,
        )
        simap_arrival_artifact_manifest_path = cls._optional_manifest_path(
            entry,
            "simap_arrival_artifact_manifest",
            root,
            manifest_path,
        )
        adsb_compressed_metadata_path = cls._optional_manifest_path(
            entry,
            "adsb_compressed_metadata",
            root,
            manifest_path,
        )

        return cls(
            events_path=events_path,
            fix_sequences_path=fix_sequences_path,
            simap_arrival_trajectories_path=simap_arrival_trajectories_path,
            adsb_compressed_trajectories_path=adsb_compressed_trajectories_path,
            data_manifest_path=manifest_path,
            data_date=date_label,
            simap_arrival_artifact_manifest_path=simap_arrival_artifact_manifest_path,
            adsb_compressed_metadata_path=adsb_compressed_metadata_path,
        )

    @staticmethod
    def _default_manifest_entry(manifest: dict[str, Any], manifest_path: Path) -> tuple[str, dict[str, Any]]:
        default_entries = [
            (date_label, entry)
            for date_label, entry in manifest.items()
            if isinstance(entry, dict) and bool(entry.get("default"))
        ]
        if len(default_entries) != 1:
            raise ValueError(f"{manifest_path} must contain exactly one default data entry")

        date_label, entry = default_entries[0]
        return str(date_label), entry

    @staticmethod
    def _manifest_path(entry: dict[str, Any], key: str, root: Path, manifest_path: Path) -> Path:
        raw_path = entry.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{manifest_path} default entry is missing {key}")
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return root / path

    @classmethod
    def _optional_manifest_path(
        cls,
        entry: dict[str, Any],
        key: str,
        root: Path,
        manifest_path: Path,
    ) -> Path | None:
        raw_path = entry.get(key)
        if raw_path is None:
            return None
        return cls._manifest_path(entry, key, root, manifest_path)


class DepartureScheduleItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    flight_id: str
    callsign: str
    icao24: str
    columns: list[str]
    points: list[list[int | float]]
    departure_time: int
    departure_time_utc: str
    runway: str
    breakpoint_mask_bits: dict[str, int] | None = None
    lateral_breakpoint_times: list[int] = Field(default_factory=list)
    altitude_breakpoint_times: list[int] = Field(default_factory=list)
    first_time: int | None = None
    last_time: int | None = None
    raw_point_count: int | None = None
    compressed_point_count: int | None = None
    lateral_tolerance_m: float | None = None
    altitude_tolerance_m: float | None = None


class ArrivalScheduleItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    flight_id: str
    callsign: str
    icao24: str
    columns: list[str]
    points: list[list[int | float]]
    time_at_first_fix: int
    time_at_first_fix_utc: str | None = None
    time_at_last_event: int
    time_at_last_event_utc: str | None = None
    runway: str
    original_fix_sequence: str
    original_fix_count: int
    breakpoint_mask_bits: dict[str, int] | None = None
    lateral_breakpoint_times: list[int] = Field(default_factory=list)
    altitude_breakpoint_times: list[int] = Field(default_factory=list)
    first_time: int | None = None
    last_time: int | None = None
    raw_point_count: int | None = None
    compressed_point_count: int | None = None
    lateral_tolerance_m: float | None = None
    altitude_tolerance_m: float | None = None


class HealthResponse(BaseModel):
    status: str
    events_count: int
    arrivals_count: int
    departures_count: int
    fix_sequences_count: int
    compressed_flights_count: int
    artifact_flights_count: int
    simap_arrival_flights_count: int
    adsb_compressed_flights_count: int
    arrivals_missing_fix_sequences_count: int
    arrivals_missing_trajectories_count: int
    arrivals_missing_artifacts_count: int
    arrivals_missing_simap_trajectories_count: int
    departures_missing_trajectories_count: int
    departures_missing_adsb_trajectories_count: int
    skipped_departures_count: int
    resource_paths: dict[str, str]
