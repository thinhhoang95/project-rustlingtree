"""Arrival-catalog ingestion independent of scenario-manager models."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def normalize_runway(value: object) -> str:
    token = str(value).strip().upper()
    if not token:
        raise ValueError("runway must be nonempty")
    return token if token.startswith("RW") else f"RW{token}"


def _required(row: dict[str, str], name: str, *, line_number: int) -> str:
    value = str(row.get(name, "")).strip()
    if not value:
        raise ValueError(f"catalog line {line_number}: {name} must be nonempty")
    return value


def _finite_float(row: dict[str, str], name: str, *, line_number: int) -> float:
    raw = _required(row, name, line_number=line_number)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"catalog line {line_number}: {name} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"catalog line {line_number}: {name} must be finite")
    return value


@dataclass(frozen=True, order=True)
class CatalogArrival:
    event_time_s: float
    flight_id: str
    callsign: str
    icao24: str
    runway: str
    threshold_lat_deg: float
    threshold_lon_deg: float
    airport: str | None = None
    event_lat_deg: float | None = None
    event_lon_deg: float | None = None
    event_time_utc: str | None = None

    def __post_init__(self) -> None:
        if not self.flight_id.strip():
            raise ValueError("flight_id must be nonempty")
        if not np.isfinite(self.event_time_s):
            raise ValueError("event_time_s must be finite")
        if not np.isfinite(self.threshold_lat_deg) or not -90.0 <= self.threshold_lat_deg <= 90.0:
            raise ValueError("threshold_lat_deg is invalid")
        if not np.isfinite(self.threshold_lon_deg) or not -180.0 <= self.threshold_lon_deg <= 180.0:
            raise ValueError("threshold_lon_deg is invalid")
        object.__setattr__(self, "flight_id", self.flight_id.strip())
        object.__setattr__(self, "callsign", self.callsign.strip())
        object.__setattr__(self, "icao24", self.icao24.strip())
        object.__setattr__(self, "runway", normalize_runway(self.runway))


def load_arrival_catalog(
    path: str | Path,
    *,
    runway: str | None = None,
    airport: str | None = None,
) -> tuple[CatalogArrival, ...]:
    """Load valid arrivals, optionally restricted to one runway partition."""

    wanted_runway = normalize_runway(runway) if runway is not None else None
    wanted_airport = airport.strip().upper() if airport is not None else None
    arrivals: list[CatalogArrival] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("arrival catalog has no header")
        required = {"flight_id", "operation", "runway", "event_time", "threshold_lat", "threshold_lon"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"arrival catalog is missing required columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            if str(row.get("operation", "")).strip().lower() != "arrival":
                continue
            row_runway = normalize_runway(row.get("runway", ""))
            # Current repository catalogs are single-airport and omit an
            # explicit airport column.  In that case the caller-supplied
            # partition is the declared airport provenance.
            row_airport = str(row.get("airport", "")).strip().upper() or wanted_airport
            if wanted_runway is not None and row_runway != wanted_runway:
                continue
            if wanted_airport is not None and row_airport is not None and row_airport != wanted_airport:
                continue
            event_lat = str(row.get("event_lat", "")).strip()
            event_lon = str(row.get("event_lon", "")).strip()
            arrivals.append(
                CatalogArrival(
                    event_time_s=_finite_float(row, "event_time", line_number=line_number),
                    flight_id=_required(row, "flight_id", line_number=line_number),
                    callsign=str(row.get("callsign", "")).strip(),
                    icao24=str(row.get("icao24", "")).strip(),
                    runway=row_runway,
                    threshold_lat_deg=_finite_float(row, "threshold_lat", line_number=line_number),
                    threshold_lon_deg=_finite_float(row, "threshold_lon", line_number=line_number),
                    airport=row_airport,
                    event_lat_deg=float(event_lat) if event_lat else None,
                    event_lon_deg=float(event_lon) if event_lon else None,
                    event_time_utc=str(row.get("event_time_utc", "")).strip() or None,
                )
            )
    arrivals.sort(key=lambda item: (item.event_time_s, item.flight_id))
    counts = Counter(item.flight_id for item in arrivals)
    duplicate_ids = sorted(item for item, count in counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"arrival catalog contains duplicate flight_id values: {duplicate_ids}")
    return tuple(arrivals)


load_catalog = load_arrival_catalog
