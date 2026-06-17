"""Density and community clustering sweeps for the paper-June ablation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class DensityClusteringRun:
    algorithm: str
    run_id: str
    params: dict[str, Any]
    labels: np.ndarray
    metrics: dict[str, Any]


def run_density_sweeps(features: np.ndarray, *, random_state: int) -> list[DensityClusteringRun]:
    """Run the configured DBSCAN, HDBSCAN, and Leiden parameter sweeps."""
    return [
        *run_dbscan_sweep(features),
        *run_hdbscan_sweep(features),
        *run_leiden_sweep(features, random_state=random_state),
    ]


def run_dbscan_sweep(
    features: np.ndarray,
    *,
    min_samples_values: Iterable[int] = (3, 5, 8, 12),
    eps_quantiles: Iterable[float] = (0.55, 0.65, 0.75, 0.85, 0.92),
) -> list[DensityClusteringRun]:
    runs: list[DensityClusteringRun] = []
    for min_samples in min_samples_values:
        distances = _kth_neighbor_distances(features, int(min_samples))
        eps_values = _quantile_values(distances, eps_quantiles)
        for eps_quantile, eps in eps_values:
            params = {
                "eps": float(eps),
                "eps_quantile": float(eps_quantile),
                "min_samples": int(min_samples),
            }
            labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(features).astype(int)
            run_id = f"dbscan_ms{int(min_samples)}_q{int(round(float(eps_quantile) * 100)):02d}"
            runs.append(_make_run("dbscan", run_id, params, features, labels))
    return runs


def run_hdbscan_sweep(
    features: np.ndarray,
    *,
    min_cluster_size_values: Iterable[int] = (4, 6, 8, 12, 16, 24),
    min_samples_values: Iterable[int | None] = (None, 3, 5, 8),
    cluster_selection_methods: Iterable[str] = ("eom", "leaf"),
) -> list[DensityClusteringRun]:
    try:
        import hdbscan
    except ImportError as exc:  # pragma: no cover - exercised only in missing envs
        raise RuntimeError("hdbscan is required for the HDBSCAN clustering sweep") from exc

    runs: list[DensityClusteringRun] = []
    for min_cluster_size in min_cluster_size_values:
        for min_samples in min_samples_values:
            for cluster_selection_method in cluster_selection_methods:
                params = {
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": None if min_samples is None else int(min_samples),
                    "cluster_selection_method": str(cluster_selection_method),
                }
                model = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=None if min_samples is None else int(min_samples),
                    cluster_selection_method=str(cluster_selection_method),
                )
                labels = model.fit_predict(features).astype(int)
                sample_label = "auto" if min_samples is None else str(int(min_samples))
                run_id = (
                    f"hdbscan_mcs{int(min_cluster_size)}"
                    f"_ms{sample_label}_{str(cluster_selection_method)}"
                )
                runs.append(_make_run("hdbscan", run_id, params, features, labels))
    return runs


def run_leiden_sweep(
    features: np.ndarray,
    *,
    random_state: int,
    neighbor_values: Iterable[int] = (6, 8, 12, 16, 24, 32),
    resolution_values: Iterable[float] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8),
) -> list[DensityClusteringRun]:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover - exercised only in missing envs
        raise RuntimeError("igraph and leidenalg are required for the Leiden clustering sweep") from exc

    runs: list[DensityClusteringRun] = []
    n_samples = int(features.shape[0])
    for n_neighbors in neighbor_values:
        graph = _binary_knn_graph(features, int(n_neighbors), ig)
        effective_neighbors = min(max(1, int(n_neighbors)), max(1, n_samples - 1))
        for resolution in resolution_values:
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=float(resolution),
                seed=int(random_state),
            )
            labels = np.asarray(partition.membership, dtype=int)
            params = {
                "graph": "knn_binary",
                "n_neighbors": int(effective_neighbors),
                "resolution": float(resolution),
            }
            run_id = f"leiden_knn{int(effective_neighbors)}_r{_compact_float(float(resolution))}"
            runs.append(_make_run("leiden", run_id, params, features, labels))
    return runs


def _make_run(
    algorithm: str,
    run_id: str,
    params: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
) -> DensityClusteringRun:
    return DensityClusteringRun(
        algorithm=algorithm,
        run_id=run_id,
        params=params,
        labels=labels,
        metrics={
            "algorithm": algorithm,
            "run_id": run_id,
            **_summarize_labels(features, labels),
            **{f"param_{key}": value for key, value in params.items()},
        },
    )


def _summarize_labels(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    clustered_mask = labels >= 0
    clustered_labels = labels[clustered_mask]
    n_tracks = int(labels.size)
    n_noise = int((labels < 0).sum())
    unique_labels, counts = np.unique(clustered_labels, return_counts=True)
    metrics: dict[str, Any] = {
        "n_tracks": n_tracks,
        "n_clustered_tracks": int(clustered_mask.sum()),
        "noise_tracks": n_noise,
        "noise_fraction": float(n_noise / n_tracks) if n_tracks else math.nan,
        "pred_clusters": int(len(unique_labels)),
        "cluster_count_min": int(counts.min()) if len(counts) else 0,
        "cluster_count_max": int(counts.max()) if len(counts) else 0,
        "cluster_count_mean": float(counts.mean()) if len(counts) else math.nan,
        "silhouette": math.nan,
        "calinski_harabasz": math.nan,
        "davies_bouldin": math.nan,
    }
    if 1 < len(unique_labels) < int(clustered_mask.sum()):
        clustered_features = features[clustered_mask]
        metrics["silhouette"] = _metric_or_nan(silhouette_score, clustered_features, clustered_labels)
        metrics["calinski_harabasz"] = _metric_or_nan(
            calinski_harabasz_score,
            clustered_features,
            clustered_labels,
        )
        metrics["davies_bouldin"] = _metric_or_nan(davies_bouldin_score, clustered_features, clustered_labels)
    return metrics


def _metric_or_nan(func, features: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(func(features, labels))
    except ValueError:
        return math.nan


def _kth_neighbor_distances(features: np.ndarray, min_samples: int) -> np.ndarray:
    n_neighbors = min(max(2, int(min_samples) + 1), int(features.shape[0]))
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(features)
    distances, _ = model.kneighbors(features)
    return distances[:, -1]


def _quantile_values(values: np.ndarray, quantiles: Iterable[float]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    seen: set[float] = set()
    for quantile in quantiles:
        eps = float(np.quantile(values, float(quantile)))
        rounded = round(eps, 6)
        if rounded <= 0.0 or rounded in seen:
            continue
        seen.add(rounded)
        pairs.append((float(quantile), rounded))
    return pairs


def _binary_knn_graph(features: np.ndarray, n_neighbors: int, ig_module):
    n_samples = int(features.shape[0])
    effective_neighbors = min(max(1, int(n_neighbors)), max(1, n_samples - 1))
    model = NearestNeighbors(n_neighbors=effective_neighbors + 1)
    model.fit(features)
    _, indices = model.kneighbors(features)
    edges: set[tuple[int, int]] = set()
    for source, neighbors in enumerate(indices[:, 1:]):
        for target in neighbors:
            left, right = sorted((int(source), int(target)))
            if left != right:
                edges.add((left, right))
    return ig_module.Graph(n=n_samples, edges=sorted(edges), directed=False)


def _compact_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")
