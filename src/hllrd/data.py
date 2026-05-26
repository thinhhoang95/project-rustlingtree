from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


VALID_CLUSTERS = {"NE", "NW", "SE", "SW"}


@dataclass(frozen=True)
class ClusterFlight:
    flight_id: str
    callsign: str
    icao24: str
    runway: str
    cluster: str


def normalize_cluster(cluster: str) -> str:
    normalized = str(cluster).strip().upper()
    if normalized not in VALID_CLUSTERS:
        raise ValueError(f"cluster must be one of {sorted(VALID_CLUSTERS)}, got {cluster!r}")
    return normalized


def load_cluster_flights(artifacts_path: Path, cluster: str) -> list[ClusterFlight]:
    selected_cluster = normalize_cluster(cluster)
    flights: list[ClusterFlight] = []
    with artifacts_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{artifacts_path}:{line_number} is not valid JSON") from exc
            wait_point = payload.get("wait_atc_point") or {}
            flight_cluster = str(wait_point.get("arrival_cluster") or "").strip().upper()
            if flight_cluster != selected_cluster:
                continue
            flight_id = str(payload.get("flight_id") or "").strip()
            if not flight_id:
                continue
            flights.append(
                ClusterFlight(
                    flight_id=flight_id,
                    callsign=str(payload.get("callsign") or "").strip(),
                    icao24=str(payload.get("icao24") or "").strip(),
                    runway=str(payload.get("runway") or "").strip(),
                    cluster=selected_cluster,
                )
            )
    return flights


def filter_tracks_to_cluster(tracks: pd.DataFrame, flights: list[ClusterFlight]) -> pd.DataFrame:
    flight_ids = {flight.flight_id for flight in flights}
    filtered = tracks.loc[tracks["flight_id"].astype(str).isin(flight_ids)].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def cluster_metadata_by_flight(flights: list[ClusterFlight]) -> dict[str, ClusterFlight]:
    return {flight.flight_id: flight for flight in flights}
