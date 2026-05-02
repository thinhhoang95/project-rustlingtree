from __future__ import annotations

from collections import Counter
from typing import TypedDict, cast

import numpy as np
import pandas as pd

from scenario.demand_opensky.adsb_catalog_common import (
    DEFAULT_MIN_PROXIMITY_DISTANCE_CHANGE_M,
    FIX_SEQUENCE_DELIMITER,
    GROUND_RELEVANT_ALTITUDE_M,
    ISO_DATE_FORMAT,
    ThresholdCandidate,
    haversine_distance_m,
)


class ThresholdEventRecord(TypedDict):
    date: str
    flight_id: str
    callsign: str
    icao24: str
    operation: str
    runway: str
    event_time: int
    event_time_utc: str
    event_lat: float
    event_lon: float
    event_geoaltitude_m: float
    threshold_lat: float
    threshold_lon: float
    threshold_distance_m: float
    comparison_time: int
    comparison_time_utc: str
    comparison_geoaltitude_m: float
    altitude_delta_m: float
    distance_delta_m: float


class RunwayThresholdRecord(TypedDict):
    runway: str
    runway_pair: str
    threshold_lat: float
    threshold_lon: float


def build_event_record(
    flight: pd.DataFrame,
    classification: str,
    candidate: ThresholdCandidate,
    threshold_row: RunwayThresholdRecord,
    date_label: str,
) -> ThresholdEventRecord:
    event = flight.iloc[candidate.event_index]
    comparison = flight.iloc[candidate.comparison_index]
    event_time = int(event["time"])
    comparison_time = int(comparison["time"])
    return {
        "date": date_label,
        "flight_id": str(event["flight_id"]),
        "callsign": str(event["callsign"]),
        "icao24": str(event["icao24"]),
        "operation": classification,
        "runway": candidate.runway,
        "event_time": event_time,
        "event_time_utc": pd.to_datetime(event_time, unit="s", utc=True).strftime(ISO_DATE_FORMAT),
        "event_lat": float(event["lat"]),
        "event_lon": float(event["lon"]),
        "event_geoaltitude_m": float(event["geoaltitude"]),
        "threshold_lat": float(threshold_row["threshold_lat"]),
        "threshold_lon": float(threshold_row["threshold_lon"]),
        "threshold_distance_m": float(candidate.event_distance_m),
        "comparison_time": comparison_time,
        "comparison_time_utc": pd.to_datetime(comparison_time, unit="s", utc=True).strftime(ISO_DATE_FORMAT),
        "comparison_geoaltitude_m": float(comparison["geoaltitude"]),
        "altitude_delta_m": float(candidate.altitude_delta_m),
        "distance_delta_m": float(candidate.distance_delta_m),
    }


def is_better_candidate(candidate: ThresholdCandidate, current_best: ThresholdCandidate | None) -> bool:
    if current_best is None:
        return True

    candidate_score = (abs(candidate.distance_delta_m), -candidate.event_distance_m)
    current_score = (abs(current_best.distance_delta_m), -current_best.event_distance_m)
    return candidate_score > current_score


def classify_flight_track(
    flight: pd.DataFrame,
    thresholds: pd.DataFrame,
    runway_radius_m: float,
    lookaround_seconds: int,
    min_altitude_change_m: float,
    date_label: str,
    min_proximity_distance_change_m: float = DEFAULT_MIN_PROXIMITY_DISTANCE_CHANGE_M,
) -> tuple[str, ThresholdEventRecord | None]:
    _ = (lookaround_seconds, min_altitude_change_m)
    positions = flight.loc[
        flight["time"].notna()
        & flight["lat"].notna()
        & flight["lon"].notna()
    ].reset_index(drop=True)

    if positions.empty:
        return "unclassified", None

    lats = positions["lat"].to_numpy(dtype=float)
    lons = positions["lon"].to_numpy(dtype=float)
    altitudes = pd.to_numeric(positions["geoaltitude"], errors="coerce").to_numpy(dtype=float)

    best_candidate: ThresholdCandidate | None = None
    best_threshold_row: RunwayThresholdRecord | None = None
    saw_proximity_threshold = False

    threshold_rows = cast(list[RunwayThresholdRecord], thresholds.to_dict("records"))
    for threshold_row in threshold_rows:
        distances = haversine_distance_m(
            lats,
            lons,
            float(threshold_row["threshold_lat"]),
            float(threshold_row["threshold_lon"]),
        )
        proximity_indices = (distances <= runway_radius_m) & (altitudes <= GROUND_RELEVANT_ALTITUDE_M)
        if not proximity_indices.any():
            continue

        saw_proximity_threshold = True
        matching_indices = np.flatnonzero(proximity_indices)
        event_index = int(matching_indices[0])
        event_distance = float(distances[event_index])
        approach_evidence_m = float(distances[0] - event_distance)
        departure_evidence_m = float(distances[-1] - event_distance)

        if max(approach_evidence_m, departure_evidence_m) < min_proximity_distance_change_m:
            continue

        if approach_evidence_m >= departure_evidence_m:
            operation = "arrival"
            comparison_index = 0
            distance_delta = -approach_evidence_m
            altitude_delta = float(altitudes[event_index] - altitudes[comparison_index])
        else:
            operation = "departure"
            comparison_index = len(positions) - 1
            distance_delta = departure_evidence_m
            altitude_delta = float(altitudes[comparison_index] - altitudes[event_index])

        candidate = ThresholdCandidate(
            operation=operation,
            runway=str(threshold_row["runway"]),
            event_index=event_index,
            comparison_index=comparison_index,
            event_distance_m=event_distance,
            altitude_delta_m=altitude_delta,
            distance_delta_m=distance_delta,
        )
        if is_better_candidate(candidate, best_candidate):
            best_candidate = candidate
            best_threshold_row = threshold_row

    if best_candidate is None:
        if saw_proximity_threshold:
            return "unclassified", None
        return "overflight", None

    assert best_threshold_row is not None
    return best_candidate.operation, build_event_record(
        positions,
        best_candidate.operation,
        best_candidate,
        best_threshold_row,
        date_label,
    )


def extract_fix_sequence(
    flight: pd.DataFrame,
    fixes: pd.DataFrame,
    fix_radius_m: float,
    date_label: str,
) -> dict[str, object]:
    valid = flight.loc[flight["time"].notna() & flight["lat"].notna() & flight["lon"].notna()].reset_index(drop=True)
    first_row = flight.iloc[0]

    sequence: list[str] = []
    last_identifier: str | None = None
    if not valid.empty and not fixes.empty:
        fix_lats = fixes["latitude_deg"].to_numpy(dtype=float)
        fix_lons = fixes["longitude_deg"].to_numpy(dtype=float)
        fix_identifiers = fixes["identifier"].astype(str).to_numpy()

        for lat_deg, lon_deg in valid[["lat", "lon"]].itertuples(index=False, name=None):
            distances = haversine_distance_m(
                float(lat_deg),
                float(lon_deg),
                fix_lats,
                fix_lons,
            )
            best_index = int(distances.argmin())
            if float(distances[best_index]) <= fix_radius_m:
                identifier = str(fix_identifiers[best_index])
                if identifier != last_identifier:
                    sequence.append(identifier)
                    last_identifier = identifier
            else:
                last_identifier = None

    return {
        "date": date_label,
        "flight_id": str(first_row["flight_id"]),
        "callsign": str(first_row["callsign"]),
        "icao24": str(first_row["icao24"]),
        "first_time": int(valid.iloc[0]["time"]) if not valid.empty else pd.NA,
        "last_time": int(valid.iloc[-1]["time"]) if not valid.empty else pd.NA,
        "fix_sequence": FIX_SEQUENCE_DELIMITER.join(sequence),
        "fix_count": len(sequence),
    }


def split_flight_track(flight: pd.DataFrame, split_gap_seconds: int) -> list[pd.DataFrame]:
    if flight.empty:
        return [flight]

    times = flight["time"].to_numpy(dtype=float)
    split_points = [0]
    for index in range(1, len(flight)):
        previous_time = times[index - 1]
        current_time = times[index]
        if np.isnan(previous_time) or np.isnan(current_time):
            continue
        if current_time - previous_time > split_gap_seconds:
            split_points.append(index)
    split_points.append(len(flight))

    segments: list[pd.DataFrame] = []
    first_row = flight.iloc[0]
    base_callsign = str(first_row["callsign"])
    icao24 = str(first_row["icao24"])
    for segment_number, (start, stop) in enumerate(zip(split_points, split_points[1:], strict=False), start=1):
        segment = flight.iloc[start:stop].copy().reset_index(drop=True)
        segment_callsign = f"{base_callsign}M{segment_number}"
        segment["callsign"] = segment_callsign
        segment["flight_id"] = f"{segment_callsign}{icao24}"
        segments.append(segment)
    return segments


def process_tracks(
    tracks: pd.DataFrame,
    thresholds: pd.DataFrame,
    fixes: pd.DataFrame,
    runway_radius_m: float,
    fix_radius_m: float,
    lookaround_seconds: int,
    min_altitude_change_m: float,
    split_gap_seconds: int,
    date_label: str,
    min_proximity_distance_change_m: float = DEFAULT_MIN_PROXIMITY_DISTANCE_CHANGE_M,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str]]:
    event_records: list[ThresholdEventRecord] = []
    fix_records: list[dict[str, object]] = []
    classification_counts: Counter[str] = Counter()

    for _, flight in tracks.groupby("flight_id", sort=False):
        for segment in split_flight_track(flight, split_gap_seconds):
            classification, event_record = classify_flight_track(
                flight=segment,
                thresholds=thresholds,
                runway_radius_m=runway_radius_m,
                lookaround_seconds=lookaround_seconds,
                min_altitude_change_m=min_altitude_change_m,
                date_label=date_label,
                min_proximity_distance_change_m=min_proximity_distance_change_m,
            )
            classification_counts[classification] += 1
            if event_record is not None:
                event_records.append(event_record)

            fix_records.append(
                extract_fix_sequence(
                    flight=segment,
                    fixes=fixes,
                    fix_radius_m=fix_radius_m,
                    date_label=date_label,
                )
            )

    event_frame = (
        pd.DataFrame(event_records).sort_values(["event_time", "flight_id"], kind="stable")
        if event_records
        else pd.DataFrame()
    )
    fix_frame = (
        pd.DataFrame(fix_records).sort_values(
            ["first_time", "flight_id"],
            kind="stable",
            na_position="last",
        )
        if fix_records
        else pd.DataFrame()
    )
    return event_frame, fix_frame, classification_counts
