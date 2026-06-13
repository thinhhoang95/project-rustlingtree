from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vlm_ppe.clustering.community_detection import CommunityDetectionRun
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
    runs: list[CommunityDetectionRun],
    track_ids: list[str],
    output_dir: str | Path,
) -> list[EvidenceImage]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    evidence: list[EvidenceImage] = []
    for run in runs:
        labels = pd.DataFrame({"flight_id": track_ids, "cluster_id": run.labels})
        community_count = int(run.metric.community_count)
        cols = min(3, community_count)
        rows = int(np.ceil(community_count / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows), squeeze=False)
        for cluster_id in range(community_count):
            ax = axes[cluster_id // cols][cluster_id % cols]
            _plot_tracks(ax, resampled, labels, cluster_id)
            count = int((run.labels == cluster_id).sum())
            ax.set_title(f"threshold={run.threshold_nm:.3f} NM community {cluster_id} ({count} tracks)")
        for empty_index in range(community_count, rows * cols):
            axes[empty_index // cols][empty_index % cols].axis("off")
        fig.suptitle(f"Candidate threshold={run.threshold_nm:.3f} NM: community overlays")
        fig.tight_layout()
        path = root / f"threshold_{run.candidate_id:02d}" / "cluster_panel.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        evidence.append(
            EvidenceImage(
                kind="cluster_panel",
                path=path.as_posix(),
                caption=f"Community overlay panel for threshold={run.threshold_nm:.3f} NM",
            )
        )
    return evidence


def render_metrics_chart(runs: list[CommunityDetectionRun], output_dir: str | Path) -> EvidenceImage:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    thresholds = [run.threshold_nm for run in runs]
    community_counts = [run.metric.community_count for run in runs]
    silhouette = [np.nan if run.metric.silhouette is None else run.metric.silhouette for run in runs]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(thresholds, community_counts, marker="o", color="#1f77b4")
    ax1.set_xlabel("Threshold (NM)")
    ax1.set_ylabel("Communities", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, silhouette, marker="s", color="#d62728")
    ax2.set_ylabel("Silhouette", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.suptitle("Community-detection candidate metrics")
    fig.tight_layout()
    path = root / "cd_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return EvidenceImage(
        kind="metrics_chart",
        path=path.as_posix(),
        caption="Community count and silhouette by threshold",
    )
