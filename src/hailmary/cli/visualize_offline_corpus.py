"""Interactively inspect observed flights in an offline Hailmary corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import numpy as np
from pyproj import Geod

from hailmary.data import RawADSBTrack, load_catalog_raw_adsb_tracks, load_manifest
from hailmary.scenario import (
    ObservedArrival,
    TerminalEntryCorpus,
    materialize_arrival_variant,
)
from hailmary.templates import ClusterTemplate, TemplateStore, TrajectoryVariant


METERS_PER_FOOT = 0.3048
METERS_PER_SECOND_PER_KNOT = 0.514444


@dataclass(frozen=True, slots=True)
class CorpusSample:
    """One corpus arrival joined to its observed ADS-B trajectory."""

    arrival: ObservedArrival
    track: RawADSBTrack
    template: ClusterTemplate

    @property
    def cluster_key(self) -> tuple[str, str]:
        return self.arrival.key.runway, self.arrival.key.cluster


@dataclass(frozen=True, slots=True)
class DisplayProfile:
    """Terminal-area samples and plot-ready longitudinal profiles."""

    elapsed_minutes: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    altitude_ft: np.ndarray
    speed_elapsed_minutes: np.ndarray
    tas_kts: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hailmary-visualize-offline-corpus")
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
    parser.add_argument("--seed", type=int, help="optional random-selection seed")
    return parser


def load_samples(
    corpus_dir: str | Path,
    manifest_path: str | Path,
    dataset_id: str | None = None,
) -> tuple[tuple[CorpusSample, ...], int]:
    """Join accepted corpus arrivals to their source ADS-B tracks.

    The returned integer is the number of corpus arrivals whose source track
    or cluster template could not be found. Those arrivals are omitted from
    the GUI choices.
    """

    corpus = TerminalEntryCorpus.read(Path(corpus_dir) / "traffic_corpus.json")
    requested_dataset = corpus.dataset_id if dataset_id is None else dataset_id
    if requested_dataset != corpus.dataset_id:
        raise ValueError(
            f"dataset {requested_dataset!r} does not match corpus dataset "
            f"{corpus.dataset_id!r}"
        )
    dataset = load_manifest(manifest_path).select(requested_dataset)
    template_store = TemplateStore.read(Path(corpus_dir) / "hailmary_templates.json")
    template_by_cluster = {
        f"{template.airport_id}:{template.runway_id}:{template.cluster_id}": template
        for template in template_store.templates
    }
    flight_ids = {arrival.flight_id for arrival in corpus.arrivals}
    tracks = load_catalog_raw_adsb_tracks(
        dataset.require("raw_adsb"), flight_ids=flight_ids
    )
    track_by_flight = {track.flight_id: track for track in tracks}
    samples = tuple(
        CorpusSample(
            arrival=arrival,
            track=track_by_flight[arrival.flight_id],
            template=template_by_cluster[arrival.key.qualified_id],
        )
        for arrival in corpus.arrivals
        if arrival.flight_id in track_by_flight
        and arrival.key.qualified_id in template_by_cluster
    )
    missing_count = len(corpus.arrivals) - len(samples)
    if not samples:
        raise ValueError("none of the corpus flights could be joined to raw ADS-B tracks")
    return samples, missing_count


def make_observed_display_profile(sample: CorpusSample) -> DisplayProfile:
    """Clip a raw track at terminal entry and estimate TAS under zero wind."""

    track = sample.track
    entry_time = sample.arrival.terminal_entry_time_s
    if entry_time < track.time_s[0] or entry_time > track.time_s[-1]:
        raise ValueError(
            f"terminal entry for {track.flight_id!r} lies outside its ADS-B track"
        )

    right = int(np.searchsorted(track.time_s, entry_time, side="right"))
    if right == 0:
        right = 1
    left = min(right - 1, len(track.time_s) - 2)
    fraction = float(
        (entry_time - track.time_s[left])
        / (track.time_s[left + 1] - track.time_s[left])
    )

    def at_entry(values: np.ndarray) -> float:
        return float(values[left] + fraction * (values[left + 1] - values[left]))

    remaining = track.time_s > entry_time
    time_s = np.concatenate(([entry_time], track.time_s[remaining]))
    lat_deg = np.concatenate(([at_entry(track.lat_deg)], track.lat_deg[remaining]))
    lon_deg = np.concatenate(([at_entry(track.lon_deg)], track.lon_deg[remaining]))
    altitude_m = np.concatenate(
        ([at_entry(track.geoaltitude_m)], track.geoaltitude_m[remaining])
    )
    if len(time_s) < 2:
        raise ValueError(f"flight {track.flight_id!r} has no track after terminal entry")

    geod = Geod(ellps="WGS84")
    _, _, distance_m = geod.inv(
        lon_deg[:-1], lat_deg[:-1], lon_deg[1:], lat_deg[1:]
    )
    interval_s = np.diff(time_s)
    tas_kts = np.asarray(distance_m / interval_s, dtype=float) / METERS_PER_SECOND_PER_KNOT
    elapsed_minutes = (time_s - entry_time) / 60.0
    speed_elapsed_minutes = (elapsed_minutes[:-1] + elapsed_minutes[1:]) / 2.0
    return DisplayProfile(
        elapsed_minutes=elapsed_minutes,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        altitude_ft=altitude_m / METERS_PER_FOOT,
        speed_elapsed_minutes=speed_elapsed_minutes,
        tas_kts=tas_kts,
    )


def make_blended_display_profile(sample: CorpusSample) -> DisplayProfile:
    """Materialize the production medoid/observed blend for one arrival."""

    variant = materialize_arrival_variant(sample.template, sample.arrival)
    return display_profile_from_variant(variant)


def display_profile_from_variant(variant: TrajectoryVariant) -> DisplayProfile:
    """Convert a threshold-to-upstream variant into flight display order."""

    return DisplayProfile(
        elapsed_minutes=variant.elapsed_time_s[::-1] / 60.0,
        lat_deg=variant.lat_deg[::-1],
        lon_deg=variant.lon_deg[::-1],
        altitude_ft=variant.altitude_m[::-1] / METERS_PER_FOOT,
        speed_elapsed_minutes=variant.elapsed_time_s[::-1] / 60.0,
        tas_kts=variant.tas_mps[::-1] / METERS_PER_SECOND_PER_KNOT,
    )


def run_gui(
    samples: Sequence[CorpusSample],
    *,
    missing_count: int = 0,
    seed: int | None = None,
) -> None:
    """Create and run the Tk/Matplotlib corpus viewer."""

    import tkinter as tk
    from tkinter import messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    grouped: dict[tuple[str, str], list[CorpusSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.cluster_key, []).append(sample)
    for cluster_samples in grouped.values():
        cluster_samples.sort(key=lambda item: item.arrival.flight_id)

    def cluster_sort_key(key: tuple[str, str]) -> tuple[str, int, str]:
        runway, cluster = key
        try:
            return runway, int(cluster), cluster
        except ValueError:
            return runway, 2**31 - 1, cluster

    keys = sorted(grouped, key=cluster_sort_key)
    display_to_key = {
        f"{runway} / cluster {cluster} ({len(grouped[(runway, cluster)])} flights)": (
            runway,
            cluster,
        )
        for runway, cluster in keys
    }
    rng = random.Random(seed)

    root = tk.Tk()
    root.title("Hailmary Offline Corpus Viewer")
    root.geometry("1280x760")
    root.minsize(900, 600)

    controls = ttk.Frame(root, padding=(10, 10, 10, 4))
    controls.pack(fill=tk.X)
    ttk.Label(controls, text="Runway cluster:").pack(side=tk.LEFT)
    selected_cluster = tk.StringVar(value=next(iter(display_to_key)))
    cluster_select = ttk.Combobox(
        controls,
        state="readonly",
        width=40,
        textvariable=selected_cluster,
        values=tuple(display_to_key),
    )
    cluster_select.pack(side=tk.LEFT, padx=(8, 12))
    randomize = ttk.Button(controls, text="Randomize")
    randomize.pack(side=tk.LEFT)

    selected_mode = tk.StringVar(value="observed")
    observed_mode_button = ttk.Radiobutton(
        controls,
        text="Observed track",
        value="observed",
        variable=selected_mode,
    )
    observed_mode_button.pack(side=tk.LEFT, padx=(24, 4))
    blended_mode_button = ttk.Radiobutton(
        controls,
        text="Blend to Medoid",
        value="blended",
        variable=selected_mode,
    )
    blended_mode_button.pack(side=tk.LEFT, padx=4)

    flight_text = tk.StringVar()
    ttk.Label(root, textvariable=flight_text, padding=(10, 3)).pack(fill=tk.X)
    if missing_count:
        ttk.Label(
            root,
            text=f"Note: {missing_count} corpus flights had no matching raw ADS-B track.",
            foreground="#8a5a00",
            padding=(10, 0, 10, 3),
        ).pack(fill=tk.X)

    figure = Figure(figsize=(12, 6.5), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    lateral_axis = figure.add_subplot(layout[:, 0])
    altitude_axis = figure.add_subplot(layout[0, 1])
    tas_axis = figure.add_subplot(layout[1, 1], sharex=altitude_axis)
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(fill=tk.X)

    current_sample: CorpusSample | None = None

    def redraw_current_sample() -> None:
        if current_sample is None:
            return
        sample = current_sample
        try:
            if selected_mode.get() == "blended":
                profile = make_blended_display_profile(sample)
                mode_text = "Blend to Medoid"
            else:
                profile = make_observed_display_profile(sample)
                mode_text = "Observed track"
        except ValueError as exc:
            messagebox.showerror("Cannot display flight", str(exc), parent=root)
            return

        arrival = sample.arrival
        identity = arrival.callsign or sample.track.callsign or "unknown callsign"
        flight_text.set(
            f"Flight: {arrival.flight_id}  |  Callsign: {identity}  |  "
            f"Runway: {arrival.key.runway}  |  Cluster: {arrival.key.cluster}  |  "
            f"Mode: {mode_text}"
        )

        lateral_axis.clear()
        altitude_axis.clear()
        tas_axis.clear()
        lateral_axis.plot(profile.lon_deg, profile.lat_deg, color="#1769aa", linewidth=1.8)
        lateral_axis.scatter(
            profile.lon_deg[0], profile.lat_deg[0], color="#e07a1f", s=36, zorder=3
        )
        lateral_axis.scatter(
            profile.lon_deg[-1], profile.lat_deg[-1], color="#242424", s=24, zorder=3
        )
        lateral_title = "Lateral flight path"
        if selected_mode.get() == "blended":
            lateral_title += f" — medoid {sample.template.medoid_flight_id}"
        lateral_axis.set_title(lateral_title)
        lateral_axis.set_xlabel("Longitude (deg)")
        lateral_axis.set_ylabel("Latitude (deg)")
        mean_latitude = float(np.mean(profile.lat_deg))
        lateral_axis.set_aspect(1.0 / max(np.cos(np.deg2rad(mean_latitude)), 0.1))
        lateral_axis.grid(alpha=0.25)

        altitude_axis.plot(
            profile.elapsed_minutes, profile.altitude_ft, color="#2b8c56", linewidth=1.6
        )
        altitude_axis.set_title("Longitudinal profile")
        altitude_axis.set_ylabel("Altitude (ft)")
        altitude_axis.grid(alpha=0.25)
        altitude_axis.tick_params(labelbottom=False)

        tas_axis.plot(
            profile.speed_elapsed_minutes, profile.tas_kts, color="#a23b72", linewidth=1.4
        )
        tas_axis.set_xlabel("Time since terminal entry (min)")
        tas_axis.set_ylabel("TAS (kt)\nzero-wind model")
        tas_axis.grid(alpha=0.25)
        canvas.draw_idle()

    def show_random_sample(*_event: object) -> None:
        nonlocal current_sample
        current_sample = rng.choice(grouped[display_to_key[selected_cluster.get()]])
        redraw_current_sample()

    randomize.configure(command=show_random_sample)
    cluster_select.bind("<<ComboboxSelected>>", show_random_sample)
    observed_mode_button.configure(command=redraw_current_sample)
    blended_mode_button.configure(command=redraw_current_sample)
    show_random_sample()
    root.mainloop()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples, missing_count = load_samples(
        args.corpus_dir, args.manifest, dataset_id=args.dataset_id
    )
    run_gui(samples, missing_count=missing_count, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CorpusSample",
    "DisplayProfile",
    "build_parser",
    "display_profile_from_variant",
    "load_samples",
    "main",
    "make_blended_display_profile",
    "make_observed_display_profile",
    "run_gui",
]
