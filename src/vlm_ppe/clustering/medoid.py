from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vlm_ppe.schemas import ClusterMedoid


def _track_tensor(resampled: pd.DataFrame, track_ids: list[str]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for track_id in track_ids:
        group = resampled.loc[resampled["flight_id"] == track_id].sort_values("station_index", kind="stable")
        arrays.append(group[["x_nm", "y_nm"]].to_numpy(dtype=float))
    return np.stack(arrays, axis=0)


def compute_cluster_medoids(resampled: pd.DataFrame, labels: pd.DataFrame) -> list[ClusterMedoid]:
    merged = labels.copy()
    merged["flight_id"] = merged["flight_id"].astype(str)
    medoids: list[ClusterMedoid] = []
    for cluster_id in sorted(merged["cluster_id"].astype(int).unique()):
        track_ids = merged.loc[merged["cluster_id"].astype(int) == cluster_id, "flight_id"].astype(str).tolist()
        tensor = _track_tensor(resampled, track_ids)
        diff = tensor[:, np.newaxis, :, :] - tensor[np.newaxis, :, :, :]
        pairwise = np.linalg.norm(diff, axis=3).mean(axis=2)
        sums = pairwise.sum(axis=1)
        medoid_index = int(np.argmin(sums))
        distances = pairwise[medoid_index]
        medoid_points = tensor[medoid_index]
        medoids.append(
            ClusterMedoid(
                cluster_id=int(cluster_id),
                medoid_track_id=track_ids[medoid_index],
                n_tracks=len(track_ids),
                mean_distance_nm=float(distances.mean()),
                max_distance_nm=float(distances.max()),
                template_points=[(float(x), float(y)) for x, y in medoid_points],
            )
        )
    return medoids


def write_medoids(medoids: list[ClusterMedoid], output_dir: str | Path) -> tuple[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summaries: list[dict] = []
    for medoid in medoids:
        summaries.append(medoid.model_dump(exclude={"template_points"}))
        for station_index, (x_nm, y_nm) in enumerate(medoid.template_points):
            rows.append(
                {
                    "cluster_id": medoid.cluster_id,
                    "medoid_track_id": medoid.medoid_track_id,
                    "station_index": station_index,
                    "x_nm": x_nm,
                    "y_nm": y_nm,
                }
            )
    medoids_path = root / "cluster_medoids.parquet"
    pd.DataFrame(rows).to_parquet(medoids_path, index=False)
    summary_path = root / "cluster_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump({"clusters": summaries}, stream, indent=2)
    return medoids_path.as_posix(), summary_path.as_posix()
