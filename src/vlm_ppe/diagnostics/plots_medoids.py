from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vlm_ppe.schemas import ClusterMedoid


def render_medoid_plot(resampled: pd.DataFrame, medoids: list[ClusterMedoid], output_dir: str | Path) -> str:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    for _flight_id, group in resampled.groupby("flight_id", sort=False):
        ordered = group.sort_values("station_index", kind="stable")
        ax.plot(ordered["x_nm"], ordered["y_nm"], color="#9aa0a6", linewidth=0.7, alpha=0.20)
    for medoid in medoids:
        xs = [point[0] for point in medoid.template_points]
        ys = [point[1] for point in medoid.template_points]
        ax.plot(xs, ys, linewidth=2.2, label=f"cluster {medoid.cluster_id}: {medoid.medoid_track_id}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (NM)")
    ax.set_ylabel("y (NM)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.suptitle("Cluster medoid trajectories")
    fig.tight_layout()
    path = root / "cluster_medoids.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.as_posix()
