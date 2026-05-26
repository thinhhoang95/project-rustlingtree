from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hllrd.geometry import (
    LocalProjection,
    normal_deviation_m,
    projection_from_latlon,
    reference_tangent_normal,
    resample_polyline_by_fraction,
)


@dataclass(frozen=True)
class MatrixBuildConfig:
    station_count: int = 200
    min_points_per_flight: int = 3
    center_method: str = "median"

    def validate(self) -> None:
        if self.station_count < 3:
            raise ValueError("station_count must be at least 3")
        if self.min_points_per_flight < 2:
            raise ValueError("min_points_per_flight must be at least 2")
        if self.center_method not in {"median", "mean"}:
            raise ValueError("center_method must be 'median' or 'mean'")


@dataclass(frozen=True)
class MatrixArtifact:
    X: np.ndarray
    X_centered: np.ndarray
    column_center: np.ndarray
    flight_ids: tuple[str, ...]
    stations: np.ndarray
    reference_xy_m: np.ndarray
    normals_xy: np.ndarray
    origin_lat_deg: float
    origin_lon_deg: float
    cluster: str | None = None
    skipped_flights: tuple[str, ...] = ()
    sigma_hat: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def robust_center_columns(X: np.ndarray, method: str = "median") -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(X, dtype=float)
    if method == "median":
        center = np.median(matrix, axis=0)
    elif method == "mean":
        center = np.mean(matrix, axis=0)
    else:
        raise ValueError("method must be 'median' or 'mean'")
    return matrix - center, center


def estimate_noise_sigma(X: np.ndarray) -> float:
    matrix = np.asarray(X, dtype=float)
    if matrix.shape[1] < 2:
        return 0.0
    diffs = np.diff(matrix, axis=1).ravel()
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    if mad <= 0.0:
        return float(np.std(diffs) / np.sqrt(2.0))
    return mad / (0.6745 * np.sqrt(2.0))


def build_matrix_from_tracks(
    tracks: pd.DataFrame,
    *,
    config: MatrixBuildConfig | None = None,
    cluster: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MatrixArtifact:
    cfg = config or MatrixBuildConfig()
    cfg.validate()
    required = {"flight_id", "time", "lat", "lon"}
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise ValueError(f"tracks missing required columns: {missing}")

    clean = tracks.loc[
        tracks["flight_id"].notna()
        & tracks["time"].notna()
        & tracks["lat"].notna()
        & tracks["lon"].notna()
    ].copy()
    if clean.empty:
        raise ValueError("tracks contain no finite trajectory points")
    clean["flight_id"] = clean["flight_id"].astype(str)
    clean.sort_values(["flight_id", "time"], inplace=True, kind="stable")

    projection = projection_from_latlon(clean["lat"].to_numpy(dtype=float), clean["lon"].to_numpy(dtype=float))
    stations = np.linspace(0.0, 1.0, cfg.station_count)

    samples: list[np.ndarray] = []
    flight_ids: list[str] = []
    skipped: list[str] = []
    for flight_id, flight in clean.groupby("flight_id", sort=False):
        sampled = _sample_flight(flight, projection, stations, cfg.min_points_per_flight)
        if sampled is None:
            skipped.append(str(flight_id))
            continue
        flight_ids.append(str(flight_id))
        samples.append(sampled)

    if not samples:
        raise ValueError("no flights had enough valid trajectory points to build a matrix")

    sample_stack = np.stack(samples, axis=0)
    reference_xy = np.median(sample_stack, axis=0)
    _tangents, normals = reference_tangent_normal(reference_xy)
    X = np.vstack([normal_deviation_m(sample, reference_xy, normals) for sample in sample_stack])
    X_centered, column_center = robust_center_columns(X, method=cfg.center_method)
    sigma_hat = estimate_noise_sigma(X_centered)

    return MatrixArtifact(
        X=X,
        X_centered=X_centered,
        column_center=column_center,
        flight_ids=tuple(flight_ids),
        stations=stations,
        reference_xy_m=reference_xy,
        normals_xy=normals,
        origin_lat_deg=projection.origin_lat_deg,
        origin_lon_deg=projection.origin_lon_deg,
        cluster=cluster,
        skipped_flights=tuple(skipped),
        sigma_hat=float(sigma_hat),
        metadata={
            "config": asdict(cfg),
            **(metadata or {}),
        },
    )


def _sample_flight(
    flight: pd.DataFrame,
    projection: LocalProjection,
    stations: np.ndarray,
    min_points: int,
) -> np.ndarray | None:
    deduped = flight.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
    if len(deduped) < min_points:
        return None
    lat = deduped["lat"].to_numpy(dtype=float)
    lon = deduped["lon"].to_numpy(dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    if int(finite.sum()) < min_points:
        return None
    x_m, y_m = projection.project(lat[finite], lon[finite])
    try:
        sample_x, sample_y = resample_polyline_by_fraction(x_m, y_m, stations)
    except ValueError:
        return None
    return np.column_stack((sample_x, sample_y))


def save_matrix_artifact(path: Path, artifact: MatrixArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=artifact.X,
        X_centered=artifact.X_centered,
        column_center=artifact.column_center,
        flight_ids=np.asarray(artifact.flight_ids, dtype=str),
        stations=artifact.stations,
        reference_xy_m=artifact.reference_xy_m,
        normals_xy=artifact.normals_xy,
        origin=np.asarray([artifact.origin_lat_deg, artifact.origin_lon_deg], dtype=float),
        cluster=np.asarray("" if artifact.cluster is None else artifact.cluster, dtype=str),
        skipped_flights=np.asarray(artifact.skipped_flights, dtype=str),
        sigma_hat=np.asarray(artifact.sigma_hat, dtype=float),
        metadata=np.asarray(json.dumps(artifact.metadata, sort_keys=True), dtype=str),
    )


def load_matrix_artifact(path: Path) -> MatrixArtifact:
    with np.load(path, allow_pickle=False) as data:
        origin = np.asarray(data["origin"], dtype=float)
        cluster = str(np.asarray(data["cluster"]).item())
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        return MatrixArtifact(
            X=np.asarray(data["X"], dtype=float),
            X_centered=np.asarray(data["X_centered"], dtype=float),
            column_center=np.asarray(data["column_center"], dtype=float),
            flight_ids=tuple(str(item) for item in data["flight_ids"].tolist()),
            stations=np.asarray(data["stations"], dtype=float),
            reference_xy_m=np.asarray(data["reference_xy_m"], dtype=float),
            normals_xy=np.asarray(data["normals_xy"], dtype=float),
            origin_lat_deg=float(origin[0]),
            origin_lon_deg=float(origin[1]),
            cluster=cluster or None,
            skipped_flights=tuple(str(item) for item in data["skipped_flights"].tolist()),
            sigma_hat=float(np.asarray(data["sigma_hat"]).item()),
            metadata=metadata,
        )
