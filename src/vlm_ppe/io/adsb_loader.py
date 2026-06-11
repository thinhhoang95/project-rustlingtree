from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vlm_ppe.geo.polyline import cumulative_lengths, remove_duplicate_neighbors
from vlm_ppe.geo.projection import LocalProjection
from vlm_ppe.schemas import CoordinateSystem, PPEConfig


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _manifest_dataset(manifest: dict, dataset_id: str) -> dict:
    if dataset_id in manifest:
        return dict(manifest[dataset_id])
    for key, value in manifest.items():
        if isinstance(value, dict) and value.get("default"):
            return dict(value)
    raise KeyError(f"dataset_id={dataset_id!r} not found in manifest")


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _load_compressed_jsonl(path: Path, selected_ids: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            flight_id = str(payload.get("flight_id", ""))
            if flight_id not in selected_ids:
                continue
            columns = list(payload.get("columns", []))
            try:
                time_idx = columns.index("time")
                lat_idx = columns.index("lat")
                lon_idx = columns.index("lon")
                altitude_idx = columns.index("geoaltitude_m")
            except ValueError as exc:
                raise ValueError(f"compressed ADS-B payload has unsupported columns: {columns}") from exc

            for seq, point in enumerate(payload.get("points", [])):
                rows.append(
                    {
                        "flight_id": flight_id,
                        "callsign": str(payload.get("callsign", "")),
                        "icao24": str(payload.get("icao24", "")),
                        "seq": int(seq),
                        "time": int(point[time_idx]),
                        "lat": float(point[lat_idx]),
                        "lon": float(point[lon_idx]),
                        "geoaltitude_m": float(point[altitude_idx]),
                    }
                )
    if not rows:
        raise ValueError(f"no selected compressed ADS-B tracks found in {path}")
    frame = pd.DataFrame(rows)
    frame.sort_values(["flight_id", "time", "seq"], inplace=True, kind="stable")
    frame.reset_index(drop=True, inplace=True)
    return frame


def _choose_projection_origin(catalog: pd.DataFrame) -> tuple[float, float]:
    for lat_col, lon_col in (("threshold_lat", "threshold_lon"), ("event_lat", "event_lon")):
        if lat_col in catalog.columns and lon_col in catalog.columns:
            lat = pd.to_numeric(catalog[lat_col], errors="coerce").dropna()
            lon = pd.to_numeric(catalog[lon_col], errors="coerce").dropna()
            if not lat.empty and not lon.empty:
                return float(lat.median()), float(lon.median())
    raise ValueError("catalog does not contain usable origin latitude/longitude columns")


def ingest_adsb_tracks(config: PPEConfig) -> tuple[pd.DataFrame, pd.DataFrame, CoordinateSystem]:
    manifest = load_manifest(config.manifest_path)
    dataset = _manifest_dataset(manifest, config.dataset_id)
    catalog_path = _resolve_manifest_path(config.manifest_path, str(dataset["landings_and_departures"]))
    compressed_path = _resolve_manifest_path(config.manifest_path, str(dataset["adsb_compressed_trajectories"]))

    catalog = pd.read_csv(catalog_path)
    operations = catalog["operation"].astype(str).str.strip().str.lower()
    selected = catalog.loc[operations == config.operation].copy()
    if config.runway:
        selected = selected.loc[selected["runway"].astype(str).str.strip() == config.runway].copy()
    if selected.empty:
        runway = f" runway={config.runway}" if config.runway else ""
        raise ValueError(f"no {config.operation}{runway} records found in {catalog_path}")

    selected_ids = set(selected["flight_id"].astype(str))
    tracks = _load_compressed_jsonl(compressed_path, selected_ids)
    tracks = tracks.merge(
        selected[
            [
                "flight_id",
                "operation",
                "runway",
                "event_time",
                "event_lat",
                "event_lon",
                "threshold_lat",
                "threshold_lon",
            ]
        ],
        on="flight_id",
        how="inner",
    )

    origin_lat, origin_lon = _choose_projection_origin(selected)
    projection = LocalProjection.from_origin(origin_lat, origin_lon)
    x_nm, y_nm = projection.project_nm(tracks["lat"].to_numpy(dtype=float), tracks["lon"].to_numpy(dtype=float))
    tracks["x_nm"] = x_nm
    tracks["y_nm"] = y_nm

    valid_frames: list[pd.DataFrame] = []
    track_index_rows: list[dict] = []
    for flight_id, flight in tracks.groupby("flight_id", sort=False):
        points = remove_duplicate_neighbors(flight[["x_nm", "y_nm"]].to_numpy(dtype=float))
        if len(points) < config.min_track_points:
            continue
        lengths = cumulative_lengths(points)
        if len(lengths) < 2 or float(lengths[-1]) <= 0.0:
            continue
        kept = flight.iloc[: len(points)].copy()
        kept["s_nm"] = lengths
        kept["track_length_nm"] = float(lengths[-1])
        valid_frames.append(kept)
        first = kept.iloc[0]
        track_index_rows.append(
            {
                "flight_id": str(flight_id),
                "callsign": str(first["callsign"]),
                "icao24": str(first["icao24"]),
                "operation": str(first["operation"]),
                "runway": str(first["runway"]),
                "point_count": int(len(kept)),
                "track_length_nm": float(lengths[-1]),
                "first_time": int(kept["time"].min()),
                "last_time": int(kept["time"].max()),
            }
        )

    if not valid_frames:
        raise ValueError("all selected tracks were degenerate after projection")

    normalized = pd.concat(valid_frames, ignore_index=True)
    normalized.sort_values(["flight_id", "time", "seq"], inplace=True, kind="stable")
    normalized.reset_index(drop=True, inplace=True)
    track_index = pd.DataFrame(track_index_rows).sort_values("flight_id").reset_index(drop=True)
    return normalized, track_index, projection.coordinate_system()
