from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vlm_ppe.schemas import ClusterMedoid, EvidenceImage, InterventionWindow, WindowProposal

WindowLike = InterventionWindow | WindowProposal


def render_residual_window_diagnostics(
    resampled: pd.DataFrame,
    labels: pd.DataFrame,
    medoids: list[ClusterMedoid],
    residual_profiles: pd.DataFrame,
    output_dir: str | Path,
    *,
    windows_by_cluster: dict[int, list[WindowLike]] | None = None,
    caption_suffix: str = "without window highlights",
) -> list[EvidenceImage]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    labels_frame = labels.copy()
    labels_frame["flight_id"] = labels_frame["flight_id"].astype(str)
    evidence: list[EvidenceImage] = []
    highlighted = windows_by_cluster or {}

    for medoid in sorted(medoids, key=lambda item: item.cluster_id):
        cluster_id = int(medoid.cluster_id)
        cluster_ids = set(
            labels_frame.loc[labels_frame["cluster_id"].astype(int) == cluster_id, "flight_id"].astype(str)
        )
        cluster_tracks = resampled.loc[resampled["flight_id"].astype(str).isin(cluster_ids)]
        cluster_profile = residual_profiles.loc[residual_profiles["cluster_id"].astype(int) == cluster_id].sort_values(
            "station_index", kind="stable"
        )
        cluster_windows = highlighted.get(cluster_id, [])

        fig, (ax_map, ax_energy, ax_heading) = plt.subplots(
            3,
            1,
            figsize=(9, 11),
            gridspec_kw={"height_ratios": [2.1, 1.0, 1.0]},
        )
        _plot_cluster_overlay(ax_map, cluster_tracks, medoid, cluster_profile, cluster_windows)
        _plot_residual_energy(ax_energy, cluster_profile, cluster_windows)
        _plot_heading_dispersion(ax_heading, cluster_profile, cluster_windows)
        fig.suptitle(f"Cluster {cluster_id}: residual diagnostics for VLM window proposal")
        fig.tight_layout()
        path = root / f"cluster_{cluster_id:02d}_residual_windows.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        evidence.append(
            EvidenceImage(
                kind="residual_windows",
                path=path.as_posix(),
                caption=f"Cluster {cluster_id} residual diagnostics {caption_suffix}",
            )
        )

    return evidence


def _plot_cluster_overlay(
    ax,
    cluster_tracks: pd.DataFrame,
    medoid: ClusterMedoid,
    profile: pd.DataFrame,
    windows: list[WindowLike],
) -> None:
    for _flight_id, group in cluster_tracks.groupby("flight_id", sort=False):
        ordered = group.sort_values("station_index", kind="stable")
        ax.plot(ordered["x_nm"], ordered["y_nm"], color="#9aa0a6", linewidth=0.8, alpha=0.30)

    xs = [point[0] for point in medoid.template_points]
    ys = [point[1] for point in medoid.template_points]
    ax.plot(xs, ys, color="#111827", linewidth=2.0, label=f"medoid {medoid.medoid_track_id}")
    for window in windows:
        segment = profile.loc[
            (profile["station_index"].astype(int) >= int(window.start_station_index))
            & (profile["station_index"].astype(int) <= int(window.end_station_index))
        ]
        if not segment.empty:
            ax.plot(
                segment["template_x_nm"],
                segment["template_y_nm"],
                color="#dc2626",
                linewidth=4.0,
                alpha=0.85,
                solid_capstyle="round",
                label=window.window_id,
            )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (NM)")
    ax.set_ylabel("y (NM)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")


def _plot_residual_energy(ax, profile: pd.DataFrame, windows: list[WindowLike]) -> None:
    ax.plot(profile["s_nm"], profile["residual_energy_nm2"], color="#2563eb", linewidth=1.7, label="median residual energy")
    _shade_windows(ax, profile, windows)
    _draw_station_guides(ax, profile)
    ax.set_ylabel("NM^2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")


def _plot_heading_dispersion(ax, profile: pd.DataFrame, windows: list[WindowLike]) -> None:
    ax.plot(profile["s_nm"], profile["heading_dispersion"], color="#047857", linewidth=1.7, label="heading dispersion")
    _shade_windows(ax, profile, windows)
    _draw_station_guides(ax, profile)
    ax.set_xlabel("station index (ticks positioned by template arc length)")
    ax.set_ylabel("dispersion")
    ax.set_ylim(bottom=0.0, top=1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")


def _draw_station_guides(ax, profile: pd.DataFrame) -> None:
    if profile.empty:
        return
    tick_rows = _station_tick_rows(profile)
    ax.set_xticks(tick_rows["s_nm"].to_numpy(dtype=float))
    ax.set_xticklabels([str(int(value)) for value in tick_rows["station_index"]], fontsize="x-small")
    for _, row in tick_rows.iterrows():
        ax.axvline(float(row["s_nm"]), color="#d1d5db", linewidth=0.5, alpha=0.45)
        ax.text(
            float(row["s_nm"]),
            0.98,
            str(int(row["station_index"])),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize="xx-small",
            color="#6b7280",
        )


def _station_tick_rows(profile: pd.DataFrame, max_ticks: int = 18) -> pd.DataFrame:
    ordered = profile.sort_values("station_index", kind="stable").reset_index(drop=True)
    if len(ordered) <= max_ticks:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, max_ticks).round().astype(int)
    positions = np.unique(positions)
    return ordered.iloc[positions]


def _shade_windows(ax, profile: pd.DataFrame, windows: list[WindowLike]) -> None:
    if profile.empty:
        return
    ordered = profile.sort_values("station_index", kind="stable")
    station_index = ordered["station_index"].astype(int)
    for window in windows:
        segment = ordered.loc[
            (station_index >= int(window.start_station_index))
            & (station_index <= int(window.end_station_index))
        ]
        if segment.empty:
            continue
        start_s = float(segment.iloc[0]["s_nm"])
        end_s = float(segment.iloc[-1]["s_nm"])
        midpoint_s = (start_s + end_s) / 2.0
        ax.axvspan(start_s, end_s, color="#dc2626", alpha=0.14)
        ax.text(
            midpoint_s,
            0.88,
            window.window_id,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize="x-small",
            color="#991b1b",
        )
