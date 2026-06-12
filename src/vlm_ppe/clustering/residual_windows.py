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


def robust_threshold(values: np.ndarray, lambda_value: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median + float(lambda_value) * mad


def detect_window_spans(
    residual_energy: np.ndarray,
    heading_dispersion: np.ndarray,
    *,
    residual_energy_lambda: float,
    heading_dispersion_threshold: float,
    station_s_nm: np.ndarray,
    min_window_length_nm: float,
    merge_windows_gap_nm: float,
) -> list[tuple[int, int, list[str]]]:
    energy = np.asarray(residual_energy, dtype=float)
    dispersion = np.asarray(heading_dispersion, dtype=float)
    s_nm = np.asarray(station_s_nm, dtype=float)
    if energy.shape != dispersion.shape or energy.shape != s_nm.shape:
        raise ValueError("energy, dispersion, and station_s_nm must have the same shape")
    if len(energy) < 2:
        return []

    energy_threshold = robust_threshold(energy, residual_energy_lambda)
    energy_mask = energy > energy_threshold
    heading_mask = dispersion > float(heading_dispersion_threshold)
    combined = energy_mask | heading_mask

    raw_spans: list[tuple[int, int, set[str]]] = []
    start: int | None = None
    for index, active in enumerate(combined):
        if bool(active) and start is None:
            start = index
        elif not bool(active) and start is not None:
            raw_spans.append((start, index - 1, _span_reasons(energy_mask, heading_mask, start, index - 1)))
            start = None
    if start is not None:
        raw_spans.append((start, len(combined) - 1, _span_reasons(energy_mask, heading_mask, start, len(combined) - 1)))

    if not raw_spans:
        return []

    merged: list[tuple[int, int, set[str]]] = []
    for span_start, span_end, reasons in raw_spans:
        if merged and float(s_nm[span_start] - s_nm[merged[-1][1]]) <= float(merge_windows_gap_nm):
            prev_start, _prev_end, prev_reasons = merged[-1]
            merged[-1] = (prev_start, span_end, prev_reasons | reasons)
        else:
            merged.append((span_start, span_end, set(reasons)))

    filtered: list[tuple[int, int, list[str]]] = []
    for span_start, span_end, reasons in merged:
        length_nm = float(s_nm[span_end] - s_nm[span_start])
        if length_nm >= float(min_window_length_nm):
            filtered.append((span_start, span_end, sorted(reasons)))
    return filtered


def _span_reasons(energy_mask: np.ndarray, heading_mask: np.ndarray, start: int, end: int) -> set[str]:
    reasons: set[str] = set()
    if bool(energy_mask[start : end + 1].any()):
        reasons.add("residual_energy")
    if bool(heading_mask[start : end + 1].any()):
        reasons.add("heading_dispersion")
    return reasons


def compute_cluster_residual_windows(
    resampled: pd.DataFrame,
    labels: pd.DataFrame,
    medoids: list[ClusterMedoid],
    *,
    residual_energy_lambda: float,
    min_window_length_nm: float,
    merge_windows_gap_nm: float,
    heading_dispersion_threshold: float,
) -> tuple[pd.DataFrame, list[InterventionWindow]]:
    labels_frame = labels.copy()
    labels_frame["flight_id"] = labels_frame["flight_id"].astype(str)
    profiles: list[pd.DataFrame] = []
    windows: list[InterventionWindow] = []

    for medoid in sorted(medoids, key=lambda item: item.cluster_id):
        track_ids = labels_frame.loc[
            labels_frame["cluster_id"].astype(int) == int(medoid.cluster_id), "flight_id"
        ].astype(str).tolist()
        tracks_xy = _cluster_track_tensor(resampled, track_ids)
        template_xy = np.asarray(medoid.template_points, dtype=float)
        station_frame = _template_station_frame(medoid)
        energy = compute_residual_energy(tracks_xy, template_xy)
        dispersion = compute_heading_dispersion(tracks_xy)
        energy_threshold = robust_threshold(energy, residual_energy_lambda)

        profile = station_frame.copy()
        profile["residual_energy_nm2"] = energy
        profile["residual_energy_threshold_nm2"] = energy_threshold
        profile["heading_dispersion"] = dispersion
        profile["heading_dispersion_threshold"] = float(heading_dispersion_threshold)
        profiles.append(profile)

        spans = detect_window_spans(
            energy,
            dispersion,
            residual_energy_lambda=residual_energy_lambda,
            heading_dispersion_threshold=heading_dispersion_threshold,
            station_s_nm=station_frame["s_nm"].to_numpy(dtype=float),
            min_window_length_nm=min_window_length_nm,
            merge_windows_gap_nm=merge_windows_gap_nm,
        )
        for ordinal, (start, end, reasons) in enumerate(spans, start=1):
            rows = profile.iloc[start : end + 1]
            windows.append(
                InterventionWindow(
                    cluster_id=int(medoid.cluster_id),
                    window_id=f"C{int(medoid.cluster_id)}_W{ordinal}",
                    start_station_index=int(start),
                    end_station_index=int(end),
                    start_s_fraction=float(profile.iloc[start]["s_fraction"]),
                    end_s_fraction=float(profile.iloc[end]["s_fraction"]),
                    start_s_nm=float(profile.iloc[start]["s_nm"]),
                    end_s_nm=float(profile.iloc[end]["s_nm"]),
                    length_nm=float(profile.iloc[end]["s_nm"] - profile.iloc[start]["s_nm"]),
                    peak_residual_energy_nm2=float(rows["residual_energy_nm2"].max()),
                    peak_heading_dispersion=float(rows["heading_dispersion"].max()),
                    trigger_reasons=reasons,
                    track_ids=track_ids,
                )
            )

    if profiles:
        residual_profiles = pd.concat(profiles, ignore_index=True)
    else:
        residual_profiles = pd.DataFrame()
    return residual_profiles, windows


def write_residual_windows(
    residual_profiles: pd.DataFrame,
    windows: list[InterventionWindow],
    output_dir: str | Path,
) -> tuple[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    profiles_path = root / "residual_profiles.parquet"
    residual_profiles.to_parquet(profiles_path, index=False)
    windows_path = root / "intervention_windows.parquet"
    window_rows = [window.model_dump() for window in windows]
    pd.DataFrame(window_rows, columns=list(InterventionWindow.model_fields)).to_parquet(windows_path, index=False)
    return profiles_path.as_posix(), windows_path.as_posix()
