from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vlm_ppe.schemas import ClusterMedoid, InterventionWindow


def _cluster_track_tensor(resampled: pd.DataFrame, track_ids: list[str]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for track_id in track_ids:
        group = resampled.loc[resampled["flight_id"].astype(str) == str(track_id)].sort_values("station_index", kind="stable")
        arrays.append(group[["x_nm", "y_nm"]].to_numpy(dtype=float))
    if not arrays:
        raise ValueError("cluster must contain at least one track")
    return np.stack(arrays, axis=0)


def _template_station_frame(medoid: ClusterMedoid) -> pd.DataFrame:
    points = np.asarray(medoid.template_points, dtype=float)
    n_points = len(points)
    if n_points < 2:
        raise ValueError("medoid template must contain at least two points")
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    s_nm = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(s_nm[-1])
    if total <= 0.0:
        s_fraction = np.linspace(0.0, 1.0, n_points)
    else:
        s_fraction = s_nm / total
    return pd.DataFrame(
        {
            "cluster_id": medoid.cluster_id,
            "station_index": np.arange(n_points, dtype=int),
            "s_fraction": s_fraction,
            "s_nm": s_nm,
            "template_x_nm": points[:, 0],
            "template_y_nm": points[:, 1],
        }
    )


def compute_residual_energy(tracks_xy: np.ndarray, template_xy: np.ndarray) -> np.ndarray:
    """Median squared distance from cluster tracks to the template at each station."""
    tracks = np.asarray(tracks_xy, dtype=float)
    template = np.asarray(template_xy, dtype=float)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks_xy must have shape (n_tracks, n_stations, 2)")
    if template.ndim != 2 or template.shape[1] != 2:
        raise ValueError("template_xy must have shape (n_stations, 2)")
    if tracks.shape[1] != template.shape[0]:
        raise ValueError("tracks and template must have the same number of stations")
    residuals = tracks - template[np.newaxis, :, :]
    distances_sq = np.sum(residuals * residuals, axis=2)
    return np.median(distances_sq, axis=0)


def compute_heading_dispersion(tracks_xy: np.ndarray) -> np.ndarray:
    """Circular heading dispersion by station, in [0, 1]."""
    tracks = np.asarray(tracks_xy, dtype=float)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks_xy must have shape (n_tracks, n_stations, 2)")
    n_tracks, n_stations, _ = tracks.shape
    if n_stations < 2:
        raise ValueError("at least two stations are required")

    deltas = np.diff(tracks, axis=1)
    headings = np.arctan2(deltas[:, :, 1], deltas[:, :, 0])
    station_headings = np.empty((n_tracks, n_stations), dtype=float)
    station_headings[:, 0] = headings[:, 0]
    station_headings[:, -1] = headings[:, -1]
    if n_stations > 2:
        station_headings[:, 1:-1] = 0.5 * (headings[:, :-1] + headings[:, 1:])
    resultant = np.abs(np.exp(1j * station_headings).mean(axis=0))
    return np.clip(1.0 - resultant, 0.0, 1.0)


def compute_cluster_residual_profiles(
    resampled: pd.DataFrame,
    labels: pd.DataFrame,
    medoids: list[ClusterMedoid],
) -> pd.DataFrame:
    labels_frame = labels.copy()
    labels_frame["flight_id"] = labels_frame["flight_id"].astype(str)
    profiles: list[pd.DataFrame] = []

    for medoid in sorted(medoids, key=lambda item: item.cluster_id):
        track_ids = labels_frame.loc[
            labels_frame["cluster_id"].astype(int) == int(medoid.cluster_id), "flight_id"
        ].astype(str).tolist()
        tracks_xy = _cluster_track_tensor(resampled, track_ids)
        template_xy = np.asarray(medoid.template_points, dtype=float)
        station_frame = _template_station_frame(medoid)
        energy = compute_residual_energy(tracks_xy, template_xy)
        dispersion = compute_heading_dispersion(tracks_xy)

        profile = station_frame.copy()
        profile["residual_energy_nm2"] = energy
        profile["heading_dispersion"] = dispersion
        profiles.append(profile)

    if profiles:
        return pd.concat(profiles, ignore_index=True)
    return pd.DataFrame()


def write_residual_profiles(
    residual_profiles: pd.DataFrame,
    output_dir: str | Path,
) -> str:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    profiles_path = root / "residual_profiles.parquet"
    residual_profiles.to_parquet(profiles_path, index=False)
    return profiles_path.as_posix()


def write_intervention_windows(windows: list[InterventionWindow], output_dir: str | Path) -> str:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    windows_path = root / "intervention_windows.parquet"
    window_rows = [window.model_dump() for window in windows]
    pd.DataFrame(window_rows, columns=list(InterventionWindow.model_fields)).to_parquet(windows_path, index=False)
    return windows_path.as_posix()
