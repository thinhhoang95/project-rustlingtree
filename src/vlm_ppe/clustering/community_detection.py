from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score

from vlm_ppe.clustering.features import FeatureSet
from vlm_ppe.schemas import CommunityMetric


@dataclass(frozen=True)
class CommunityDetectionRun:
    candidate_id: int
    threshold_nm: float
    labels: np.ndarray
    metric: CommunityMetric


def run_candidate_community_detection(
    features: FeatureSet,
    *,
    threshold_min_nm: float,
    threshold_max_nm: float | None,
    threshold_steps: int,
    extra_thresholds_nm: list[float] | None = None,
) -> list[CommunityDetectionRun]:
    n_tracks = len(features.track_ids)
    if n_tracks < 1:
        raise ValueError("at least one track is required for clustering")

    distances = pairwise_trajectory_rms_nm(features)
    thresholds = candidate_thresholds_nm(
        distances,
        threshold_min_nm=threshold_min_nm,
        threshold_max_nm=threshold_max_nm,
        threshold_steps=threshold_steps,
        extra_thresholds_nm=extra_thresholds_nm,
    )
    runs = [
        _community_detection_run(
            distances,
            threshold_nm=threshold,
            candidate_id=candidate_id,
        )
        for candidate_id, threshold in enumerate(thresholds)
    ]
    if not runs:
        raise ValueError("no community-detection runs were produced")
    return runs


def pairwise_trajectory_rms_nm(features: FeatureSet) -> np.ndarray:
    coordinates = features.raw.reshape(len(features.track_ids), features.n_resample, 2)
    deltas = coordinates[:, np.newaxis, :, :] - coordinates[np.newaxis, :, :, :]
    squared_station_distances = np.sum(deltas * deltas, axis=3)
    distances = np.sqrt(np.mean(squared_station_distances, axis=2))
    np.fill_diagonal(distances, 0.0)
    return distances


def candidate_thresholds_nm(
    distances: np.ndarray,
    *,
    threshold_min_nm: float,
    threshold_max_nm: float | None,
    threshold_steps: int,
    extra_thresholds_nm: list[float] | None = None,
) -> list[float]:
    if threshold_steps < 1:
        raise ValueError("threshold_steps must be at least 1")
    if threshold_min_nm < 0.0:
        raise ValueError("threshold_min_nm must be non-negative")

    finite = distances[np.triu_indices_from(distances, k=1)]
    finite = finite[np.isfinite(finite)]
    observed_max = float(finite.max()) if finite.size else 0.0
    upper = observed_max if threshold_max_nm is None else float(threshold_max_nm)
    lower = float(threshold_min_nm)
    if upper < lower:
        raise ValueError("threshold_max_nm must be greater than or equal to threshold_min_nm")

    if threshold_steps == 1 or np.isclose(lower, upper, rtol=0.0, atol=1e-12):
        thresholds = [upper]
    else:
        thresholds = [float(value) for value in np.linspace(lower, upper, int(threshold_steps))]

    for threshold in extra_thresholds_nm or []:
        value = float(threshold)
        if value < 0.0:
            raise ValueError("extra threshold values must be non-negative")
        thresholds.append(value)

    return sorted(_dedupe_thresholds(thresholds))


def write_clustering_runs(
    runs: list[CommunityDetectionRun],
    track_ids: list[str],
    output_dir: str | Path,
) -> tuple[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame([run.metric.model_dump() for run in runs])
    metrics_path = root / "cd_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    for run in runs:
        labels = pd.DataFrame({"flight_id": track_ids, "cluster_id": run.labels})
        labels.to_csv(root / f"threshold_{run.candidate_id:02d}_labels.csv", index=False)
    return metrics_path.as_posix(), root.as_posix()


def load_labels(clustering_dir: str | Path, chosen_candidate_id: int) -> pd.DataFrame:
    path = Path(clustering_dir) / f"threshold_{int(chosen_candidate_id):02d}_labels.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _community_detection_run(
    distances: np.ndarray,
    *,
    threshold_nm: float,
    candidate_id: int,
) -> CommunityDetectionRun:
    adjacency = (distances <= float(threshold_nm)) & ~np.eye(distances.shape[0], dtype=bool)
    graph = csr_matrix(adjacency.astype(np.int8))
    community_count, labels = connected_components(graph, directed=False, return_labels=True)
    labels = labels.astype(int)
    counts = np.bincount(labels, minlength=int(community_count))
    edge_count = int(np.triu(adjacency, k=1).sum())
    possible_edges = distances.shape[0] * (distances.shape[0] - 1) / 2
    edge_density = float(edge_count / possible_edges) if possible_edges > 0 else 0.0
    silhouette = None
    if 1 < int(community_count) < distances.shape[0]:
        silhouette = float(silhouette_score(distances, labels, metric="precomputed"))

    intra_distances = _intra_community_distances(distances, labels)
    metric = CommunityMetric(
        candidate_id=int(candidate_id),
        threshold_nm=float(threshold_nm),
        community_count=int(community_count),
        edge_count=edge_count,
        edge_density=edge_density,
        silhouette=silhouette,
        community_count_min=int(counts.min()),
        community_count_max=int(counts.max()),
        community_count_mean=float(counts.mean()),
        singleton_count=int((counts == 1).sum()),
        mean_intra_community_distance_nm=float(np.mean(intra_distances)) if intra_distances.size else None,
        max_intra_community_distance_nm=float(np.max(intra_distances)) if intra_distances.size else None,
    )
    return CommunityDetectionRun(
        candidate_id=int(candidate_id),
        threshold_nm=float(threshold_nm),
        labels=labels,
        metric=metric,
    )


def _intra_community_distances(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    for cluster_id in sorted(int(item) for item in np.unique(labels)):
        rows = np.flatnonzero(labels == cluster_id)
        if rows.size < 2:
            continue
        block = distances[np.ix_(rows, rows)]
        values.append(block[np.triu_indices_from(block, k=1)])
    if not values:
        return np.asarray([], dtype=float)
    return np.concatenate(values).astype(float)


def _dedupe_thresholds(thresholds: list[float]) -> list[float]:
    unique: list[float] = []
    for threshold in thresholds:
        if not any(np.isclose(threshold, existing, rtol=0.0, atol=1e-9) for existing in unique):
            unique.append(float(threshold))
    return unique
