"""Interactive visualization of runway-partitioned trajectory clusters.

The viewer consumes the offline Hailmary corpus because it is the canonical
join between a flight, its runway, and its final cluster assignment.  Raw
ADS-B trajectories are clipped at their recorded terminal-entry times before
being plotted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from hailmary.data import RawADSBTrack, load_catalog_raw_adsb_tracks, load_manifest
from hailmary.scenario import TerminalEntryCorpus

ALL_RUNWAYS = "All runways"
ALL_CLUSTERS = "All clusters"


@dataclass(frozen=True, slots=True)
class ClusterTrajectory:
    """One plot-ready ADS-B trajectory with its final cluster identity."""

    runway: str
    cluster: str
    flight_id: str
    callsign: str
    lat_deg: np.ndarray
    lon_deg: np.ndarray

    def __post_init__(self) -> None:
        lat = np.array(self.lat_deg, dtype=np.float64, copy=True)
        lon = np.array(self.lon_deg, dtype=np.float64, copy=True)
        if lat.ndim != 1 or lon.ndim != 1 or len(lat) != len(lon) or len(lat) < 2:
            raise ValueError("trajectory coordinates must be equal-length 1-D arrays")
        if not self.runway or not self.cluster or not self.flight_id:
            raise ValueError("runway, cluster, and flight_id must be nonempty")
        lat.setflags(write=False)
        lon.setflags(write=False)
        object.__setattr__(self, "lat_deg", lat)
        object.__setattr__(self, "lon_deg", lon)

    @property
    def cluster_key(self) -> tuple[str, str]:
        return self.runway, self.cluster


def _cluster_sort_key(cluster: str) -> tuple[int, int | str]:
    try:
        return 0, int(cluster)
    except ValueError:
        return 1, cluster


def clip_track_at_time(
    track: RawADSBTrack, start_time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return latitude/longitude from an interpolated start time onward."""

    if start_time_s < track.time_s[0] or start_time_s >= track.time_s[-1]:
        raise ValueError(
            f"start time for {track.flight_id!r} lies outside its ADS-B track"
        )
    right = int(np.searchsorted(track.time_s, start_time_s, side="right"))
    left = right - 1
    interval = float(track.time_s[right] - track.time_s[left])
    fraction = float((start_time_s - track.time_s[left]) / interval)

    def interpolate(values: np.ndarray) -> float:
        return float(values[left] + fraction * (values[right] - values[left]))

    remaining = track.time_s > start_time_s
    lat = np.concatenate(([interpolate(track.lat_deg)], track.lat_deg[remaining]))
    lon = np.concatenate(([interpolate(track.lon_deg)], track.lon_deg[remaining]))
    if len(lat) < 2:
        raise ValueError(
            f"flight {track.flight_id!r} has no trajectory after terminal entry"
        )
    return lat, lon


def load_cluster_trajectories(
    corpus_dir: str | Path,
    manifest_path: str | Path,
    dataset_id: str | None = None,
) -> tuple[tuple[ClusterTrajectory, ...], int]:
    """Join final corpus assignments to raw ADS-B trajectories.

    Returns the plot-ready trajectories and the number of corpus arrivals that
    could not be joined or clipped.
    """

    corpus = TerminalEntryCorpus.read(Path(corpus_dir) / "traffic_corpus.json")
    requested_dataset = corpus.dataset_id if dataset_id is None else str(dataset_id)
    if requested_dataset != corpus.dataset_id:
        raise ValueError(
            f"dataset {requested_dataset!r} does not match corpus dataset {corpus.dataset_id!r}"
        )
    dataset = load_manifest(manifest_path).select(requested_dataset)
    flight_ids = {arrival.flight_id for arrival in corpus.arrivals}
    tracks = load_catalog_raw_adsb_tracks(
        dataset.require("raw_adsb"), flight_ids=flight_ids
    )
    track_by_flight = {track.flight_id: track for track in tracks}

    trajectories: list[ClusterTrajectory] = []
    missing_count = 0
    for arrival in corpus.arrivals:
        track = track_by_flight.get(arrival.flight_id)
        if track is None:
            missing_count += 1
            continue
        try:
            lat, lon = clip_track_at_time(track, arrival.terminal_entry_time_s)
        except ValueError:
            missing_count += 1
            continue
        trajectories.append(
            ClusterTrajectory(
                runway=arrival.key.runway,
                cluster=arrival.key.cluster,
                flight_id=arrival.flight_id,
                callsign=arrival.callsign or track.callsign,
                lat_deg=lat,
                lon_deg=lon,
            )
        )
    if not trajectories:
        raise ValueError("none of the clustered corpus flights could be plotted")
    trajectories.sort(
        key=lambda item: (
            item.runway,
            _cluster_sort_key(item.cluster),
            item.flight_id,
        )
    )
    return tuple(trajectories), missing_count


def runway_choices(trajectories: Iterable[ClusterTrajectory]) -> tuple[str, ...]:
    return tuple(sorted({item.runway for item in trajectories}))


def cluster_choices(
    trajectories: Iterable[ClusterTrajectory], runway: str
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {item.cluster for item in trajectories if item.runway == runway},
            key=_cluster_sort_key,
        )
    )


def select_trajectories(
    trajectories: Iterable[ClusterTrajectory],
    *,
    runway: str | None = None,
    cluster: str | None = None,
) -> tuple[ClusterTrajectory, ...]:
    """Filter trajectories for an all-runway, runway, or runway-cluster view."""

    if cluster is not None and runway is None:
        raise ValueError("a cluster can only be selected together with a runway")
    return tuple(
        item
        for item in trajectories
        if (runway is None or item.runway == runway)
        and (cluster is None or item.cluster == cluster)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hailmary-visualize-clusters",
        description="Interactively visualize clustered ADS-B trajectories by runway.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/artifacts/hailmary/corpus"),
        help="directory containing traffic_corpus.json",
    )
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest.json"))
    parser.add_argument(
        "--dataset-id",
        help="manifest dataset to use (defaults to the ID recorded in the corpus)",
    )
    return parser


def run_gui(
    trajectories: Sequence[ClusterTrajectory],
    *,
    missing_count: int = 0,
) -> None:
    """Open the Tk/Matplotlib runway-cluster trajectory viewer."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")

    import tkinter as tk
    from tkinter import ttk

    from matplotlib import colormaps
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    runway_values = runway_choices(trajectories)
    qualified_clusters = tuple(
        sorted(
            {item.cluster_key for item in trajectories},
            key=lambda key: (key[0], _cluster_sort_key(key[1])),
        )
    )
    palette = colormaps.get_cmap("turbo").resampled(max(len(qualified_clusters), 2))
    colors = {key: palette(index) for index, key in enumerate(qualified_clusters)}

    root = tk.Tk()
    root.title("Hailmary Runway-Cluster Viewer")
    root.geometry("1400x850")
    root.minsize(900, 600)

    controls = ttk.Frame(root, padding=(10, 10, 10, 4))
    controls.pack(fill=tk.X)
    ttk.Label(controls, text="Runway:").pack(side=tk.LEFT)
    selected_runway = tk.StringVar(value=ALL_RUNWAYS)
    runway_select = ttk.Combobox(
        controls,
        state="readonly",
        width=18,
        textvariable=selected_runway,
        values=(ALL_RUNWAYS, *runway_values),
    )
    runway_select.pack(side=tk.LEFT, padx=(8, 20))

    ttk.Label(controls, text="Runway-cluster:").pack(side=tk.LEFT)
    selected_cluster = tk.StringVar(value=ALL_CLUSTERS)
    cluster_select = ttk.Combobox(
        controls,
        state="disabled",
        width=22,
        textvariable=selected_cluster,
        values=(ALL_CLUSTERS,),
    )
    cluster_select.pack(side=tk.LEFT, padx=(8, 20))
    summary_text = tk.StringVar()
    ttk.Label(controls, textvariable=summary_text).pack(side=tk.LEFT)

    if missing_count:
        ttk.Label(
            root,
            text=f"Note: {missing_count} corpus trajectories could not be loaded.",
            foreground="#8a5a00",
            padding=(10, 0, 10, 3),
        ).pack(fill=tk.X)

    figure = Figure(figsize=(12.5, 7.5), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1)
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(fill=tk.X)

    def redraw(*_event: object) -> None:
        runway = None if selected_runway.get() == ALL_RUNWAYS else selected_runway.get()
        cluster = (
            None if selected_cluster.get() == ALL_CLUSTERS else selected_cluster.get()
        )
        visible = select_trajectories(trajectories, runway=runway, cluster=cluster)
        axis.clear()
        visible_keys = tuple(
            sorted(
                {item.cluster_key for item in visible},
                key=lambda key: (key[0], _cluster_sort_key(key[1])),
            )
        )
        for item in visible:
            axis.plot(
                item.lon_deg,
                item.lat_deg,
                color=colors[item.cluster_key],
                linewidth=0.9,
                alpha=0.42,
            )
        handles = [
            Line2D(
                [0],
                [0],
                color=colors[key],
                linewidth=2.2,
                label=f"{key[0]} / cluster {key[1]}",
            )
            for key in visible_keys
        ]
        if handles:
            axis.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                fontsize="small",
            )
        title_scope = ALL_RUNWAYS if runway is None else runway
        if cluster is not None:
            title_scope += f" / cluster {cluster}"
        axis.set_title(f"ADS-B terminal trajectories — {title_scope}")
        axis.set_xlabel("Longitude (deg)")
        axis.set_ylabel("Latitude (deg)")
        if visible:
            mean_latitude = float(
                np.mean(np.concatenate([item.lat_deg for item in visible]))
            )
            axis.set_aspect(1.0 / max(np.cos(np.deg2rad(mean_latitude)), 0.1))
        axis.grid(alpha=0.25)
        summary_text.set(
            f"{len(visible)} trajectories · {len(visible_keys)} runway-clusters"
        )
        canvas.draw_idle()

    def update_clusters(*_event: object) -> None:
        runway = selected_runway.get()
        selected_cluster.set(ALL_CLUSTERS)
        if runway == ALL_RUNWAYS:
            cluster_select.configure(values=(ALL_CLUSTERS,), state="disabled")
        else:
            cluster_select.configure(
                values=(ALL_CLUSTERS, *cluster_choices(trajectories, runway)),
                state="readonly",
            )
        redraw()

    runway_select.bind("<<ComboboxSelected>>", update_clusters)
    cluster_select.bind("<<ComboboxSelected>>", redraw)
    redraw()
    root.mainloop()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trajectories, missing_count = load_cluster_trajectories(
        args.corpus_dir,
        args.manifest,
        dataset_id=args.dataset_id,
    )
    run_gui(trajectories, missing_count=missing_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_CLUSTERS",
    "ALL_RUNWAYS",
    "ClusterTrajectory",
    "build_parser",
    "clip_track_at_time",
    "cluster_choices",
    "load_cluster_trajectories",
    "main",
    "run_gui",
    "runway_choices",
    "select_trajectories",
]
