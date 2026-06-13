from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from vlm_ppe.clustering.features import FeatureSet
from vlm_ppe.schemas import KMetric


@dataclass(frozen=True)
class ClusteringRun:
    k: int
    labels: np.ndarray
    metric: KMetric


def run_candidate_kmeans(
    features: FeatureSet,
    *,
    k_min: int,
    k_max: int,
    n_init: int,
    random_state: int,
) -> list[ClusteringRun]:
    n_tracks = len(features.track_ids)
    if n_tracks < 1:
        raise ValueError("at least one track is required for clustering")
    effective_k_max = min(int(k_max), n_tracks)
    runs: list[ClusteringRun] = []
    for k in range(int(k_min), effective_k_max + 1):
        model = KMeans(n_clusters=k, n_init=int(n_init), random_state=int(random_state))
        labels = model.fit_predict(features.standardized)
        counts = np.bincount(labels, minlength=k)
        silhouette = None
        if 1 < k < n_tracks and len(np.unique(labels)) > 1:
            silhouette = float(silhouette_score(features.standardized, labels))
        metric = KMetric(
            k=k,
            inertia=float(model.inertia_),
            silhouette=silhouette,
            cluster_count_min=int(counts.min()),
            cluster_count_max=int(counts.max()),
            cluster_count_mean=float(counts.mean()),
        )
        runs.append(ClusteringRun(k=k, labels=labels.astype(int), metric=metric))
    if not runs:
        raise ValueError("no clustering runs were produced")
    return runs


def write_clustering_runs(runs: list[ClusteringRun], track_ids: list[str], output_dir: str | Path) -> tuple[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame([run.metric.model_dump() for run in runs])
    metrics_path = root / "k_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    for run in runs:
        labels = pd.DataFrame({"flight_id": track_ids, "cluster_id": run.labels})
        labels.to_csv(root / f"k_{run.k:02d}_labels.csv", index=False)
    return metrics_path.as_posix(), root.as_posix()


def load_labels(clustering_dir: str | Path, chosen_k: int) -> pd.DataFrame:
    path = Path(clustering_dir) / f"k_{int(chosen_k):02d}_labels.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
