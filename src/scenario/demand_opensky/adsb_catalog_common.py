from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RAW_COLUMNS = ["time", "icao24", "lat", "lon", "heading", "callsign", "geoaltitude"]
FIX_SEQUENCE_DELIMITER = ">"
GROUND_RELEVANT_ALTITUDE_M = 500.0
DEFAULT_RUNWAY_RADIUS_M = 1_000.0
DEFAULT_FIX_RADIUS_M = 2_000.0
DEFAULT_LOOKAROUND_SECONDS = 300
DEFAULT_MIN_ALTITUDE_CHANGE_M = 75.0
ISO_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class ThresholdCandidate:
    operation: str
    runway: str
    event_index: int
    comparison_index: int
    event_distance_m: float
    altitude_delta_m: float


def haversine_distance_m(
    lat1_deg: np.ndarray | float,
    lon1_deg: np.ndarray | float,
    lat2_deg: np.ndarray | float,
    lon2_deg: np.ndarray | float,
) -> np.ndarray:
    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def nearest_index_in_slice(times: np.ndarray, target_time: float, start: int, stop: int) -> int:
    if start >= stop:
        return start
    subset = times[start:stop]
    relative_index = int(np.abs(subset - target_time).argmin())
    return start + relative_index
