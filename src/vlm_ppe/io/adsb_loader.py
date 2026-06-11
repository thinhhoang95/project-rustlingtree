from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vlm_ppe.geo.polyline import cumulative_lengths
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


def _projection_origin(config: PPEConfig, catalog: pd.DataFrame) -> tuple[float, float]:
    if config.track_filter_center_lat is not None and config.track_filter_center_lon is not None:
        return float(config.track_filter_center_lat), float(config.track_filter_center_lon)
    return _choose_projection_origin(catalog)


def _segment_circle_intersection_fractions(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    center_xy: np.ndarray,
    radius_nm: float,
) -> list[float]:
    delta = end_xy - start_xy
    a = float(np.dot(delta, delta))
    if a <= 1e-12:
        return []
    shifted = start_xy - center_xy
    b = 2.0 * float(np.dot(shifted, delta))
    c = float(np.dot(shifted, shifted)) - float(radius_nm) ** 2
    discriminant = b * b - 4.0 * a * c
    if discriminant < -1e-9:
        return []
    root = float(np.sqrt(max(0.0, discriminant)))
    fractions = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    return sorted(
        {round(float(value), 12) for value in fractions if -1e-9 <= value <= 1.0 + 1e-9}
    )


def _row_value(row: pd.Series, column: str) -> Any:
    return row[column]


def _interpolate_track_row(start: pd.Series, end: pd.Series, fraction: float) -> dict:
    row = start.to_dict()
    numeric_columns = ("seq", "time", "lat", "lon", "geoaltitude_m", "x_nm", "y_nm")
    for column in numeric_columns:
        if column not in start.index or column not in end.index:
            continue
        start_value = float(_row_value(start, column))
        end_value = float(_row_value(end, column))
        value = start_value + float(fraction) * (end_value - start_value)
        row[column] = int(round(value)) if column == "time" else float(value)
    return row


def _append_distinct_row(rows: list[dict], row: dict) -> None:
    if rows:
        previous = rows[-1]
        previous_xy = np.asarray([previous["x_nm"], previous["y_nm"]], dtype=float)
        row_xy = np.asarray([row["x_nm"], row["y_nm"]], dtype=float)
        if float(np.linalg.norm(row_xy - previous_xy)) <= 1e-9:
            return
    rows.append(row)


def _segment_length_nm(rows: list[dict]) -> float:
    if len(rows) < 2:
        return 0.0
    points = np.asarray([[row["x_nm"], row["y_nm"]] for row in rows], dtype=float)
    return float(cumulative_lengths(points)[-1])


def _clip_track_to_radius(
    flight: pd.DataFrame,
    *,
    center_x_nm: float,
    center_y_nm: float,
    radius_nm: float,
) -> pd.DataFrame:
    ordered = flight.sort_values(["time", "seq"], kind="stable").reset_index(drop=True)
    points = ordered[["x_nm", "y_nm"]].to_numpy(dtype=float)
    center_xy = np.asarray([center_x_nm, center_y_nm], dtype=float)
    radius = float(radius_nm)
    distances = np.linalg.norm(points - center_xy, axis=1)
    inside = distances <= radius + 1e-9
    if bool(inside.all()):
        return ordered.copy()
    if len(ordered) < 2:
        return ordered.loc[inside].copy().reset_index(drop=True)

    segments: list[list[dict]] = []
    current: list[dict] = []

    for index in range(len(ordered) - 1):
        start = ordered.iloc[index]
        end = ordered.iloc[index + 1]
        start_inside = bool(inside[index])
        end_inside = bool(inside[index + 1])
        fractions = [
            value
            for value in _segment_circle_intersection_fractions(
                points[index],
                points[index + 1],
                center_xy=center_xy,
                radius_nm=radius,
            )
            if 1e-9 < value < 1.0 - 1e-9
        ]

        if start_inside and not current:
            _append_distinct_row(current, start.to_dict())

        if start_inside and end_inside:
            _append_distinct_row(current, end.to_dict())
            continue

        if start_inside and not end_inside:
            if fractions:
                _append_distinct_row(current, _interpolate_track_row(start, end, fractions[0]))
            if current:
                segments.append(current)
                current = []
            continue

        if not start_inside and end_inside:
            current = []
            if fractions:
                _append_distinct_row(current, _interpolate_track_row(start, end, fractions[-1]))
            _append_distinct_row(current, end.to_dict())
            continue

        if len(fractions) >= 2:
            inside_segment: list[dict] = []
            _append_distinct_row(inside_segment, _interpolate_track_row(start, end, fractions[0]))
            _append_distinct_row(inside_segment, _interpolate_track_row(start, end, fractions[-1]))
            if inside_segment:
                segments.append(inside_segment)

    if current:
        segments.append(current)
    if not segments:
        return ordered.iloc[0:0].copy()

    rows = max(segments, key=_segment_length_nm)
    clipped = pd.DataFrame(rows)
    clipped.sort_values(["time", "seq"], inplace=True, kind="stable")
    clipped.reset_index(drop=True, inplace=True)
    return clipped


def _apply_track_filter(tracks: pd.DataFrame, config: PPEConfig, projection: LocalProjection) -> pd.DataFrame:
    if config.track_filter_radius_nm is None:
        return tracks
    center_lat = config.track_filter_center_lat
    center_lon = config.track_filter_center_lon
    if center_lat is None or center_lon is None:
        raise ValueError("track_filter_center_lat/lon are required when track_filter_radius_nm is set")
    center_x_nm, center_y_nm = projection.project_nm(
        np.asarray([float(center_lat)]),
        np.asarray([float(center_lon)]),
    )
    filtered_frames: list[pd.DataFrame] = []
    for _, flight in tracks.groupby("flight_id", sort=False):
        clipped = _clip_track_to_radius(
            flight,
            center_x_nm=float(center_x_nm[0]),
            center_y_nm=float(center_y_nm[0]),
            radius_nm=float(config.track_filter_radius_nm),
        )
        if not clipped.empty:
            filtered_frames.append(clipped)
    if not filtered_frames:
        raise ValueError(
            f"all selected tracks were outside the configured {config.track_filter_radius_nm:.3f} NM track filter"
        )
    filtered = pd.concat(filtered_frames, ignore_index=True)
    filtered.sort_values(["flight_id", "time", "seq"], inplace=True, kind="stable")
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def _drop_duplicate_neighbor_rows(flight: pd.DataFrame) -> pd.DataFrame:
    points = flight[["x_nm", "y_nm"]].to_numpy(dtype=float)
    if len(points) <= 1:
        return flight.copy()
    keep = [0]
    for index in range(1, len(points)):
        if float(np.linalg.norm(points[index] - points[keep[-1]])) > 1e-9:
            keep.append(index)
    return flight.iloc[keep].copy()


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

    origin_lat, origin_lon = _projection_origin(config, selected)
    projection = LocalProjection.from_origin(origin_lat, origin_lon)
    x_nm, y_nm = projection.project_nm(tracks["lat"].to_numpy(dtype=float), tracks["lon"].to_numpy(dtype=float))
    tracks["x_nm"] = x_nm
    tracks["y_nm"] = y_nm
    tracks = _apply_track_filter(tracks, config, projection)

    valid_frames: list[pd.DataFrame] = []
    track_index_rows: list[dict] = []
    for flight_id, flight in tracks.groupby("flight_id", sort=False):
        kept = _drop_duplicate_neighbor_rows(flight)
        points = kept[["x_nm", "y_nm"]].to_numpy(dtype=float)
        if len(kept) < config.min_track_points:
            continue
        lengths = cumulative_lengths(points)
        if len(lengths) < 2 or float(lengths[-1]) <= 0.0:
            continue
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
