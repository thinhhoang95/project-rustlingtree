from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vlm_ppe.clustering.kmeans_runner import ClusteringRun
from vlm_ppe.schemas import EvidenceImage


def _plot_tracks(ax, resampled: pd.DataFrame, labels: pd.DataFrame | None = None, cluster_id: int | None = None) -> None:
    if labels is not None and cluster_id is not None:
        ids = set(labels.loc[labels["cluster_id"].astype(int) == int(cluster_id), "flight_id"].astype(str))
        frame = resampled.loc[resampled["flight_id"].astype(str).isin(ids)]
    else:
        frame = resampled
    for _flight_id, group in frame.groupby("flight_id", sort=False):
        ordered = group.sort_values("station_index", kind="stable")
        ax.plot(ordered["x_nm"], ordered["y_nm"], linewidth=0.8, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (NM)")
    ax.set_ylabel("y (NM)")
    ax.grid(True, alpha=0.25)


def render_cluster_panels(
    resampled: pd.DataFrame,
    runs: list[ClusteringRun],
    track_ids: list[str],
    output_dir: str | Path,
) -> list[EvidenceImage]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    evidence: list[EvidenceImage] = []
    for run in runs:
        labels = pd.DataFrame({"flight_id": track_ids, "cluster_id": run.labels})
        cols = min(3, run.k)
        rows = int(np.ceil(run.k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows), squeeze=False)
        for cluster_id in range(run.k):
            ax = axes[cluster_id // cols][cluster_id % cols]
            _plot_tracks(ax, resampled, labels, cluster_id)
            count = int((run.labels == cluster_id).sum())
            ax.set_title(f"K={run.k} cluster {cluster_id} ({count} tracks)")
        for empty_index in range(run.k, rows * cols):
            axes[empty_index // cols][empty_index % cols].axis("off")
        fig.suptitle(f"Candidate K={run.k}: cluster overlays")
        fig.tight_layout()
        path = root / f"k_{run.k:02d}" / "cluster_panel.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        evidence.append(
            EvidenceImage(kind="cluster_panel", path=path.as_posix(), caption=f"Cluster overlay panel for K={run.k}")
        )
    return evidence


def render_metrics_chart(runs: list[ClusteringRun], output_dir: str | Path) -> EvidenceImage:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    k_values = [run.k for run in runs]
    inertia = [run.metric.inertia for run in runs]
    silhouette = [np.nan if run.metric.silhouette is None else run.metric.silhouette for run in runs]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(k_values, inertia, marker="o", color="#1f77b4")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(k_values, silhouette, marker="s", color="#d62728")
    ax2.set_ylabel("Silhouette", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.suptitle("KMeans candidate metrics")
    fig.tight_layout()
    path = root / "k_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return EvidenceImage(kind="metrics_chart", path=path.as_posix(), caption="Inertia and silhouette by K")
