from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ScenarioResourceConfig:
    events_path: Path
    fix_sequences_path: Path
    compressed_flights_path: Path

    @classmethod
    def default(cls) -> "ScenarioResourceConfig":
        root = project_root()
        return cls(
            events_path=root / "data/adsb/catalogs/2025-04-01_landings_and_departures.csv",
            fix_sequences_path=root / "data/adsb/catalogs/2025-04-01_fix_sequences.csv",
            compressed_flights_path=root / "data/adsb/compressed/flights.jsonl",
        )


class DepartureScheduleItem(BaseModel):
    flight_id: str
    callsign: str
    icao24: str
    departure_time: int
    departure_time_utc: str
    runway: str


class ArrivalScheduleItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    flight_id: str
    callsign: str
    icao24: str
    columns: list[str]
    points: list[list[int | float]]
    arrival_time: int
    arrival_time_utc: str | None = None
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
    arrivals_missing_fix_sequences_count: int
    arrivals_missing_trajectories_count: int
    departures_missing_trajectories_count: int
    resource_paths: dict[str, str]
