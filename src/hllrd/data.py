from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VALID_CLUSTERS = {"NE", "NW", "SE", "SW"}


@dataclass(frozen=True)
class ClusterFlight:
    flight_id: str
    callsign: str
    icao24: str
    runway: str
    cluster: str


@dataclass(frozen=True)
class AnchorTrimResult:
    tracks: pd.DataFrame
    metadata: pd.DataFrame
    refined_anchor_lat_deg: float
    refined_anchor_lon_deg: float
    dropped_flights: tuple[str, ...]


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
    flight_ids = [flight.flight_id for flight in flights]
    filtered = tracks.loc[tracks["flight_id"].astype(str).isin(flight_ids)].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def cluster_metadata_by_flight(flights: list[ClusterFlight]) -> dict[str, ClusterFlight]:
    return {flight.flight_id: flight for flight in flights}


def trim_tracks_from_anchor(
    tracks: pd.DataFrame,
    *,
    anchor_lat_deg: float,
    anchor_lon_deg: float,
    max_anchor_distance_nm: float = 15.0,
    min_points_after_anchor: int = 3,
    refine_anchor: bool = True,
    refinement_radius_nm: float = 25.0,
    min_refinement_flights: int = 5,
) -> AnchorTrimResult:
    """Trim every flight to start at its closest point to a shared merge anchor.

    The supplied anchor can be approximate. When ``refine_anchor`` is true, the
    function first finds each flight's closest point to the supplied anchor and
    uses the median of nearby closest points as a robust cluster-specific anchor.
    """

    required = {"flight_id", "time", "lat", "lon"}
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise ValueError(f"tracks missing required columns: {missing}")
    if min_points_after_anchor < 2:
        raise ValueError("min_points_after_anchor must be at least 2")
    if max_anchor_distance_nm <= 0.0:
        raise ValueError("max_anchor_distance_nm must be positive")
    if refinement_radius_nm <= 0.0:
        raise ValueError("refinement_radius_nm must be positive")

    clean = tracks.loc[
        tracks["flight_id"].notna()
        & tracks["time"].notna()
        & tracks["lat"].notna()
        & tracks["lon"].notna()
    ].copy()
    if clean.empty:
        return AnchorTrimResult(
            tracks=clean,
            metadata=_empty_trim_metadata(),
            refined_anchor_lat_deg=float(anchor_lat_deg),
            refined_anchor_lon_deg=float(anchor_lon_deg),
            dropped_flights=(),
        )
    clean["flight_id"] = clean["flight_id"].astype(str)
    clean.sort_values(["flight_id", "time"], inplace=True, kind="stable")

    first_pass = _anchor_closest_points(clean, float(anchor_lat_deg), float(anchor_lon_deg))
    refined_lat = float(anchor_lat_deg)
    refined_lon = float(anchor_lon_deg)
    if refine_anchor and not first_pass.empty:
        near = first_pass.loc[first_pass["min_anchor_distance_nm"] <= float(refinement_radius_nm)]
        if len(near) >= int(min_refinement_flights):
            refined_lat = float(near["anchor_sample_lat_deg"].median())
            refined_lon = float(near["anchor_sample_lon_deg"].median())

    closest = _anchor_closest_points(clean, refined_lat, refined_lon)
    trimmed_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    dropped: list[str] = []

    for row in closest.to_dict("records"):
        flight_id = str(row["flight_id"])
        anchor_index = int(row["anchor_index"])
        min_anchor_distance_nm = float(row["min_anchor_distance_nm"])
        trimmed_point_count = int(row["trimmed_point_count"])

        flight = clean.loc[clean["flight_id"] == flight_id].sort_values("time", kind="stable")
        keep = (
            min_anchor_distance_nm <= float(max_anchor_distance_nm)
            and trimmed_point_count >= int(min_points_after_anchor)
        )
        status = "kept" if keep else "dropped"
        if keep:
            trimmed_frames.append(flight.iloc[anchor_index:].copy())
        else:
            dropped.append(flight_id)
        metadata_rows.append(
            {
                "flight_id": flight_id,
                "status": status,
                "original_point_count": int(row["original_point_count"]),
                "anchor_index": anchor_index,
                "trimmed_point_count": trimmed_point_count if keep else 0,
                "min_anchor_distance_nm": min_anchor_distance_nm,
                "anchor_sample_lat_deg": float(row["anchor_sample_lat_deg"]),
                "anchor_sample_lon_deg": float(row["anchor_sample_lon_deg"]),
                "refined_anchor_lat_deg": refined_lat,
                "refined_anchor_lon_deg": refined_lon,
            }
        )

    trimmed = pd.concat(trimmed_frames, ignore_index=True) if trimmed_frames else clean.iloc[0:0].copy()
    trimmed.sort_values(["flight_id", "time"], inplace=True, kind="stable")
    trimmed.reset_index(drop=True, inplace=True)
    metadata = pd.DataFrame(metadata_rows, columns=_trim_metadata_columns())
    return AnchorTrimResult(
        tracks=trimmed,
        metadata=metadata,
        refined_anchor_lat_deg=refined_lat,
        refined_anchor_lon_deg=refined_lon,
        dropped_flights=tuple(dropped),
    )


def _anchor_closest_points(tracks: pd.DataFrame, anchor_lat_deg: float, anchor_lon_deg: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight_id, flight in tracks.groupby("flight_id", sort=False):
        ordered = flight.sort_values("time", kind="stable")
        lat = ordered["lat"].to_numpy(dtype=float)
        lon = ordered["lon"].to_numpy(dtype=float)
        distances_nm = _distance_to_anchor_nm(lat, lon, anchor_lat_deg, anchor_lon_deg)
        closest_index = int(np.nanargmin(distances_nm))
        rows.append(
            {
                "flight_id": str(flight_id),
                "original_point_count": int(len(ordered)),
                "anchor_index": closest_index,
                "trimmed_point_count": int(len(ordered) - closest_index),
                "min_anchor_distance_nm": float(distances_nm[closest_index]),
                "anchor_sample_lat_deg": float(lat[closest_index]),
                "anchor_sample_lon_deg": float(lon[closest_index]),
            }
        )
    return pd.DataFrame(rows)


def _distance_to_anchor_nm(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    anchor_lat_deg: float,
    anchor_lon_deg: float,
) -> np.ndarray:
    earth_radius_m = 6_371_000.0
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat0 = np.radians(float(anchor_lat_deg))
    lon0 = np.radians(float(anchor_lon_deg))
    x_m = earth_radius_m * np.cos(lat0) * (lon - lon0)
    y_m = earth_radius_m * (lat - lat0)
    return np.hypot(x_m, y_m) / 1852.0


def _trim_metadata_columns() -> list[str]:
    return [
        "flight_id",
        "status",
        "original_point_count",
        "anchor_index",
        "trimmed_point_count",
        "min_anchor_distance_nm",
        "anchor_sample_lat_deg",
        "anchor_sample_lon_deg",
        "refined_anchor_lat_deg",
        "refined_anchor_lon_deg",
    ]


def _empty_trim_metadata() -> pd.DataFrame:
    return pd.DataFrame(columns=_trim_metadata_columns())
