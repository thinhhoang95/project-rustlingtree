from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/project-rustlingtree-matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs

from scenario.demand_opensky.adsb_catalog_common import haversine_distance_m
from scenario.demand_opensky.adsb_catalog_io import load_runway_thresholds, normalize_callsign
from scenario.trajectory_compressor.io import load_raw_adsb
from scenario.trajectory_compressor.io import split_tracks_by_gap

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "adsb" / "raw" / "2026-04-01"
DEFAULT_RUNWAYS_PATH = PROJECT_ROOT / "data" / "kdfw_procs" / "kdfw-runways.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts" / "trajectory_plots"
DEFAULT_AIRPORT_RADIUS_M = 40_000.0
CHICAGO_TZ = ZoneInfo("America/Chicago")
UTC_TZ = timezone.utc
TRACK_COLOR = "#0b57d0"
RUNWAY_COLOR = "#8e8e8e"
THRESHOLD_COLOR = "#d9480f"
START_COLOR = "#2f9e44"
END_COLOR = "#c92a2a"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a raw ADS-B aircraft trajectory for a callsign and ICAO24 pair.",
    )
    parser.add_argument("--callsign", required=True, help="Flight callsign, for example AAL1008.")
    parser.add_argument("--icao24", required=True, help="ICAO24 hex identifier, for example a1804c.")
    return parser


def resolve_base_callsign_and_segment(callsign: str) -> tuple[str, str]:
    normalized = normalize_callsign(callsign)
    upper = normalized.upper()
    if upper.endswith("M1") or upper.endswith("M2"):
        return normalized[:-2], upper[-2:]
    return normalized, "M2"


def load_flight_track(raw_dir: Path, callsign: str, icao24: str) -> pd.DataFrame:
    tracks = load_raw_adsb(raw_dir, processes=1)
    normalized_callsign, segment_suffix = resolve_base_callsign_and_segment(callsign)
    normalized_icao24 = str(icao24).strip().lower()

    subset = tracks.loc[
        tracks["callsign"].astype(str).str.strip() == normalized_callsign
    ].copy()
    subset = subset.loc[subset["icao24"].astype(str).str.strip().str.lower() == normalized_icao24].copy()
    if subset.empty:
        raise ValueError(
            f"No raw ADS-B rows found for callsign={normalized_callsign!r} and icao24={normalized_icao24!r} in {raw_dir}"
        )

    segmented = split_tracks_by_gap(subset, split_gap_seconds=25 * 60)
    segment_name = f"{normalized_callsign}{segment_suffix}"
    selected = segmented.loc[segmented["callsign"].astype(str).str.strip() == segment_name].copy()
    if selected.empty:
        available = ", ".join(sorted(segmented["callsign"].astype(str).str.strip().unique()))
        raise ValueError(
            f"Expected an {segment_suffix} segment for {normalized_callsign}/{normalized_icao24}, but found segments: {available}"
        )

    segment_count = segmented["callsign"].astype(str).str.strip().nunique()
    if segment_count > 1:
        warnings.warn(
            (
                f"Found {segment_count} segments for {normalized_callsign}/{normalized_icao24}; "
                f"using {segment_name} only."
            ),
            stacklevel=2,
        )

    return selected.reset_index(drop=True)


def format_local_time(unix_seconds: int) -> str:
    dt = datetime.fromtimestamp(unix_seconds, tz=UTC_TZ).astimezone(CHICAGO_TZ)
    return dt.strftime("%H:%M:%S")


def nearest_threshold_summary(track: pd.DataFrame, thresholds: pd.DataFrame) -> tuple[str, float]:
    latitudes = track["lat"].to_numpy(dtype=float)
    longitudes = track["lon"].to_numpy(dtype=float)
    best_runway = ""
    best_distance_m = float("inf")

    for threshold in thresholds.to_dict("records"):
        distances = haversine_distance_m(
            latitudes,
            longitudes,
            float(threshold["threshold_lat"]),
            float(threshold["threshold_lon"]),
        )
        candidate_distance = float(np.nanmin(distances))
        if candidate_distance < best_distance_m:
            best_distance_m = candidate_distance
            best_runway = str(threshold["runway"])

    return best_runway, best_distance_m


def build_label_box(track: pd.DataFrame, thresholds: pd.DataFrame) -> str:
    valid = track.loc[track["time"].notna() & track["geoaltitude"].notna()].reset_index(drop=True)
    if valid.empty:
        raise ValueError("Track has no valid timestamps")

    first_time = int(valid.iloc[0]["time"])
    last_time = int(valid.iloc[-1]["time"])
    first_altitude = float(valid.iloc[0]["geoaltitude"])
    last_altitude = float(valid.iloc[-1]["geoaltitude"])

    nearest_runway, nearest_distance_m = nearest_threshold_summary(track, thresholds)
    return "\n".join(
        [
            f"First appear (America/Chicago): {format_local_time(first_time)} @ {first_altitude:.1f} m",
            f"Last appear (America/Chicago): {format_local_time(last_time)} @ {last_altitude:.1f} m",
            f"Nearest runway threshold: {nearest_runway} @ {nearest_distance_m:.0f} m",
        ]
    )


def _airport_extent(
    thresholds: pd.DataFrame,
    radius_m: float,
    *,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> tuple[float, float, float, float]:
    if center_lat is None:
        center_lat = float(thresholds["threshold_lat"].mean())
    if center_lon is None:
        center_lon = float(thresholds["threshold_lon"].mean())

    lat_delta = radius_m / 111_320.0
    lon_scale = max(np.cos(np.deg2rad(center_lat)), 0.2)
    lon_delta = radius_m / (111_320.0 * lon_scale)
    return center_lon - lon_delta, center_lon + lon_delta, center_lat - lat_delta, center_lat + lat_delta


def plot_trajectory(track: pd.DataFrame, thresholds: pd.DataFrame, output_path: Path, callsign: str, icao24: str) -> None:
    lats = track["lat"].to_numpy(dtype=float)
    lons = track["lon"].to_numpy(dtype=float)
    threshold_lats = thresholds["threshold_lat"].to_numpy(dtype=float)
    threshold_lons = thresholds["threshold_lon"].to_numpy(dtype=float)

    label_box = build_label_box(track, thresholds)
    min_lon, max_lon, min_lat, max_lat = _airport_extent(thresholds, DEFAULT_AIRPORT_RADIUS_M)

    fig = plt.figure(figsize=(13.5, 9.5), constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.set_facecolor("#eef3f7")
    ax.gridlines(draw_labels=True, linewidth=0.25, linestyle="--", color="#a9a9a9", alpha=0.7)

    ax.plot(
        lons,
        lats,
        color=TRACK_COLOR,
        linewidth=2.3,
        marker="o",
        markersize=3.0,
        transform=ccrs.PlateCarree(),
        label="trajectory",
    )
    ax.scatter(
        [lons[0]],
        [lats[0]],
        s=70,
        color=START_COLOR,
        edgecolors="white",
        linewidths=0.7,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label="first point",
    )
    ax.scatter(
        [lons[-1]],
        [lats[-1]],
        s=70,
        color=END_COLOR,
        edgecolors="white",
        linewidths=0.7,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label="last point",
    )

    runway_pairs = thresholds.drop_duplicates("runway_pair")
    for _, row in runway_pairs.iterrows():
        pair = thresholds.loc[thresholds["runway_pair"] == row["runway_pair"]].sort_values("runway")
        if len(pair) != 2:
            continue
        runway_pair = str(row["runway_pair"])
        pair_lons = pair["threshold_lon"].to_numpy(dtype=float)
        pair_lats = pair["threshold_lat"].to_numpy(dtype=float)
        mid_lon = float(np.mean(pair_lons))
        mid_lat = float(np.mean(pair_lats))
        ax.plot(
            pair_lons,
            pair_lats,
            color=RUNWAY_COLOR,
            linewidth=1.4,
            linestyle=":",
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        ax.text(
            mid_lon,
            mid_lat,
            runway_pair,
            transform=ccrs.PlateCarree(),
            fontsize=8.0,
            color=RUNWAY_COLOR,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": RUNWAY_COLOR,
                "alpha": 0.85,
            },
            zorder=3,
        )

    ax.scatter(
        threshold_lons,
        threshold_lats,
        s=45,
        marker="^",
        color=THRESHOLD_COLOR,
        edgecolors="white",
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
        zorder=6,
        label="runway thresholds",
    )
    for _, row in thresholds.iterrows():
        ax.text(
            float(row["threshold_lon"]) + 0.01,
            float(row["threshold_lat"]) + 0.008,
            str(row["runway"]),
            transform=ccrs.PlateCarree(),
            fontsize=8.5,
            color=THRESHOLD_COLOR,
            weight="bold",
        )

    title = f"{callsign.strip().upper()} / {str(icao24).strip().lower()} raw trajectory"
    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    ax.text(
        0.02,
        0.98,
        label_box,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11.0,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "white",
            "edgecolor": "#2f2f2f",
            "alpha": 0.94,
        },
    )
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, facecolor="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raw_dir = DEFAULT_RAW_DIR
    runways_path = DEFAULT_RUNWAYS_PATH
    normalized_callsign, segment_suffix = resolve_base_callsign_and_segment(args.callsign)
    output_path = DEFAULT_OUTPUT_DIR / f"{normalized_callsign.upper()}_{str(args.icao24).strip().lower()}_{segment_suffix.lower()}.png"

    track = load_flight_track(raw_dir, args.callsign, args.icao24)
    thresholds = load_runway_thresholds(runways_path)
    plot_trajectory(track, thresholds, output_path, args.callsign, args.icao24)

    first_time = int(track.iloc[0]["time"])
    last_time = int(track.iloc[-1]["time"])
    print(f"Saved plot to {output_path}")
    print(f"Track points: {len(track)}")
    print(f"Time span: {format_local_time(first_time)} CT to {format_local_time(last_time)} CT")
    print(f"Map extent: {DEFAULT_AIRPORT_RADIUS_M / 1000:.0f} km airport radius")


if __name__ == "__main__":
    main()
