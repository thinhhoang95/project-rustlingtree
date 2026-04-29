from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


EVENT_COLUMNS = {
    "flight_id",
    "callsign",
    "icao24",
    "operation",
    "runway",
    "event_time",
    "event_time_utc",
}

FIX_SEQUENCE_COLUMNS = {
    "flight_id",
    "first_time",
    "last_time",
    "fix_sequence",
    "fix_count",
}


def load_required_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scenario resource not found: {path}")

    frame = pd.read_csv(path)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"{path} is missing required columns: {joined}")

    return frame


def load_events(path: Path) -> pd.DataFrame:
    frame = load_required_csv(path, EVENT_COLUMNS).copy()
    frame["flight_id"] = frame["flight_id"].astype(str).str.strip()
    frame["callsign"] = frame["callsign"].astype(str).str.strip()
    frame["icao24"] = frame["icao24"].astype(str).str.strip()
    frame["operation"] = frame["operation"].astype(str).str.strip().str.lower()
    frame["runway"] = frame["runway"].astype(str).str.strip()
    frame["event_time"] = pd.to_numeric(frame["event_time"], errors="raise").astype("int64")
    return frame


def load_fix_sequences(path: Path) -> pd.DataFrame:
    frame = load_required_csv(path, FIX_SEQUENCE_COLUMNS).copy()
    frame["flight_id"] = frame["flight_id"].astype(str).str.strip()
    frame["fix_sequence"] = frame["fix_sequence"].fillna("").astype(str)
    frame["first_time"] = pd.to_numeric(frame["first_time"], errors="raise").astype("int64")
    frame["last_time"] = pd.to_numeric(frame["last_time"], errors="raise").astype("int64")
    frame["fix_count"] = pd.to_numeric(frame["fix_count"], errors="raise").astype("int64")
    return frame


def load_compressed_flights(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario resource not found: {path}")

    flights: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            flight_id = str(payload.get("flight_id", "")).strip()
            if not flight_id:
                raise ValueError(f"{path}:{line_number} is missing flight_id")
            flights[flight_id] = payload
    return flights
