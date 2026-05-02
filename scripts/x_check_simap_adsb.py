from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/project-rustlingtree-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

from scenario.demand_opensky.adsb_catalog_common import haversine_distance_m
from scenario.demand_opensky.adsb_catalog_io import normalize_callsign
from scenario.trajectory_compressor.io import load_raw_adsb, split_tracks_by_gap

DEFAULT_ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts" / "flights.jsonl"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "adsb" / "raw" / "2026-04-01"
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60
UTC_TZ = timezone.utc
ADSB_COLOR = "#1b6b3a"
SIMAP_COLOR = "#c2410c"
UNAVAILABLE = "N/A"


@dataclass(frozen=True)
class FlightKey:
    callsign_segment: str
    base_callsign: str
    segment_number: int
    icao24: str

    @property
    def flight_id(self) -> str:
        return f"{self.callsign_segment}{self.icao24}"


@dataclass(frozen=True)
class Trajectory:
    name: str
    time_s: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    altitude_m: np.ndarray
    speed_mps: np.ndarray

    @property
    def first_time_s(self) -> float:
        return float(self.time_s[0])

    @property
    def last_time_s(self) -> float:
        return float(self.time_s[-1])


@dataclass(frozen=True)
class Sample:
    available: bool
    time_s: float
    lat_deg: float | None = None
    lon_deg: float | None = None
    altitude_m: float | None = None
    speed_mps: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively cross-check a SIMAP trajectory against raw ADS-B for CALLSIGN_SEGMENT,ICAO24.",
    )
    parser.add_argument(
        "flight",
        help="Flight key as CALLSIGN_SEGMENT,ICAO24, for example JIA5128M2,a7cb67.",
    )
    parser.add_argument("--artifacts-path", type=Path, default=DEFAULT_ARTIFACTS_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    return parser


def parse_flight_key(value: str) -> FlightKey:
    parts = [part.strip() for part in value.split(",", maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Flight key must use CALLSIGN_SEGMENT,ICAO24 format, for example JIA5128M2,a7cb67")

    callsign_segment = normalize_callsign(parts[0]).upper()
    icao24 = parts[1].lower()
    match = re.fullmatch(r"(.+?)M([1-9][0-9]*)", callsign_segment)
    if match is None:
        raise ValueError(f"Callsign must include a segment suffix like M1 or M2: {callsign_segment!r}")
    return FlightKey(
        callsign_segment=callsign_segment,
        base_callsign=match.group(1),
        segment_number=int(match.group(2)),
        icao24=icao24,
    )


def load_simap_payload(artifacts_path: Path, key: FlightKey) -> dict[str, Any]:
    if not artifacts_path.exists():
        raise FileNotFoundError(f"SIMAP artifacts file not found: {artifacts_path}")

    matches: list[dict[str, Any]] = []
    with artifacts_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            callsign = str(payload.get("callsign", "")).strip().upper()
            icao24 = str(payload.get("icao24", "")).strip().lower()
            flight_id = str(payload.get("flight_id", "")).strip()
            if icao24 == key.icao24 and (callsign == key.callsign_segment or flight_id == key.flight_id):
                payload["_source_line_number"] = line_number
                matches.append(payload)

    if not matches:
        raise ValueError(f"No SIMAP artifact found for {key.callsign_segment},{key.icao24} in {artifacts_path}")
    if len(matches) > 1:
        raise ValueError(f"Found {len(matches)} SIMAP artifacts for {key.callsign_segment},{key.icao24}; expected one")
    return matches[0]


def trajectory_from_simap_payload(payload: dict[str, Any]) -> Trajectory:
    points = np.asarray(payload.get("points", []), dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 4:
        raise ValueError(f"SIMAP artifact {payload.get('flight_id', '<unknown>')} has no trajectory points")
    return Trajectory(
        name="SIMAP",
        time_s=points[:, 0],
        lat_deg=points[:, 1],
        lon_deg=points[:, 2],
        altitude_m=points[:, 3],
        speed_mps=derive_speed_mps(points[:, 0], points[:, 1], points[:, 2]),
    )


def load_adsb_track(raw_dir: Path, key: FlightKey, split_gap_seconds: int) -> pd.DataFrame:
    tracks = load_raw_adsb(raw_dir, processes=1)
    subset = tracks.loc[
        (tracks["callsign"].astype(str).str.strip().str.upper() == key.base_callsign)
        & (tracks["icao24"].astype(str).str.strip().str.lower() == key.icao24)
    ].copy()
    if subset.empty:
        raise ValueError(f"No raw ADS-B rows found for {key.base_callsign},{key.icao24} in {raw_dir}")

    segmented = split_tracks_by_gap(subset, split_gap_seconds=split_gap_seconds)
    selected = segmented.loc[segmented["callsign"].astype(str).str.strip().str.upper() == key.callsign_segment].copy()
    if selected.empty:
        available = ", ".join(sorted(segmented["callsign"].astype(str).str.strip().unique()))
        raise ValueError(f"No raw ADS-B segment {key.callsign_segment} found; available segments: {available}")
    return selected.reset_index(drop=True)


def trajectory_from_adsb_track(track: pd.DataFrame) -> Trajectory:
    required_columns = ["time", "lat", "lon", "geoaltitude"]
    valid = track.loc[track[required_columns].notna().all(axis=1), required_columns].copy()
    if valid.empty:
        raise ValueError("ADS-B track has no rows with complete time/lat/lon/geoaltitude values")
    valid.sort_values("time", inplace=True, kind="stable")
    valid.drop_duplicates("time", keep="last", inplace=True)
    times = valid["time"].to_numpy(dtype=float)
    lats = valid["lat"].to_numpy(dtype=float)
    lons = valid["lon"].to_numpy(dtype=float)
    altitudes = valid["geoaltitude"].to_numpy(dtype=float)
    return Trajectory(
        name="ADS-B",
        time_s=times,
        lat_deg=lats,
        lon_deg=lons,
        altitude_m=altitudes,
        speed_mps=derive_speed_mps(times, lats, lons),
    )


def derive_speed_mps(times_s: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    speeds = np.full(len(times_s), np.nan, dtype=float)
    if len(times_s) < 2:
        return speeds
    distances_m = haversine_distance_m(lat_deg[:-1], lon_deg[:-1], lat_deg[1:], lon_deg[1:])
    dts = np.diff(times_s)
    segment_speeds = np.divide(
        distances_m,
        dts,
        out=np.full(len(dts), np.nan, dtype=float),
        where=dts > 0.0,
    )
    speeds[0] = segment_speeds[0]
    if len(speeds) > 1:
        speeds[1:] = segment_speeds
    return speeds


def sample_trajectory(trajectory: Trajectory, time_s: float) -> Sample:
    if len(trajectory.time_s) == 0 or time_s < trajectory.first_time_s or time_s > trajectory.last_time_s:
        return Sample(available=False, time_s=time_s)

    return Sample(
        available=True,
        time_s=time_s,
        lat_deg=_interp_value(time_s, trajectory.time_s, trajectory.lat_deg),
        lon_deg=_interp_value(time_s, trajectory.time_s, trajectory.lon_deg),
        altitude_m=_interp_value(time_s, trajectory.time_s, trajectory.altitude_m),
        speed_mps=_interp_value(time_s, trajectory.time_s, trajectory.speed_mps),
    )


def _interp_value(time_s: float, times_s: np.ndarray, values: np.ndarray) -> float | None:
    if len(times_s) == 0 or time_s < float(times_s[0]) or time_s > float(times_s[-1]):
        return None
    index = int(np.searchsorted(times_s, time_s, side="left"))
    if index < len(times_s) and float(times_s[index]) == float(time_s):
        value = float(values[index])
        return value if np.isfinite(value) else None
    if index == 0 or index >= len(times_s):
        return None
    before = float(values[index - 1])
    after = float(values[index])
    if not np.isfinite(before) or not np.isfinite(after):
        return None
    t0 = float(times_s[index - 1])
    t1 = float(times_s[index])
    if t1 <= t0:
        return None
    fraction = (time_s - t0) / (t1 - t0)
    return before + fraction * (after - before)


def format_unix_time(time_s: float) -> str:
    return datetime.fromtimestamp(time_s, tz=UTC_TZ).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_value(value: float | None, suffix: str, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return UNAVAILABLE
    return f"{value:.{decimals}f}{suffix}"


def sample_label(source_name: str, sample: Sample) -> str:
    if not sample.available:
        return f"{source_name}: {UNAVAILABLE}"
    return (
        f"{source_name}: "
        f"lat {format_value(sample.lat_deg, '', 5)}, "
        f"lon {format_value(sample.lon_deg, '', 5)}, "
        f"alt {format_value(sample.altitude_m, ' m')}, "
        f"speed {format_value(sample.speed_mps, ' m/s')}"
    )


def _set_marker(marker, sample: Sample, x_attr: str, y_attr: str) -> None:
    x_value = getattr(sample, x_attr)
    y_value = getattr(sample, y_attr)
    if sample.available and x_value is not None and y_value is not None:
        marker.set_data([x_value], [y_value])
        marker.set_visible(True)
    else:
        marker.set_visible(False)


def plot_cross_check(adsb: Trajectory, simap: Trajectory, payload: dict[str, Any], key: FlightKey) -> None:
    start_time = min(adsb.first_time_s, simap.first_time_s)
    end_time = max(adsb.last_time_s, simap.last_time_s)
    initial_time = max(adsb.first_time_s, simap.first_time_s)

    fig = plt.figure(figsize=(15.0, 9.0))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.16], width_ratios=[1.1, 1.0])
    trajectory_ax = fig.add_subplot(grid[0:2, 0])
    altitude_ax = fig.add_subplot(grid[0, 1])
    speed_ax = fig.add_subplot(grid[1, 1])
    slider_ax = fig.add_subplot(grid[2, :])

    trajectory_ax.plot(adsb.lon_deg, adsb.lat_deg, color=ADSB_COLOR, linewidth=2.0, label="ADS-B")
    trajectory_ax.plot(simap.lon_deg, simap.lat_deg, color=SIMAP_COLOR, linewidth=2.0, label="SIMAP")
    adsb_position_dot, = trajectory_ax.plot([], [], "o", color=ADSB_COLOR, markersize=8)
    simap_position_dot, = trajectory_ax.plot([], [], "o", color=SIMAP_COLOR, markersize=8)
    trajectory_ax.set_title(f"{key.callsign_segment} / {key.icao24} trajectory")
    trajectory_ax.set_xlabel("Longitude [deg]")
    trajectory_ax.set_ylabel("Latitude [deg]")
    trajectory_ax.grid(True, alpha=0.25)
    trajectory_ax.legend(loc="best")
    trajectory_ax.axis("equal")

    altitude_ax.plot(adsb.time_s, adsb.altitude_m, color=ADSB_COLOR, linewidth=1.8, label="ADS-B")
    altitude_ax.plot(simap.time_s, simap.altitude_m, color=SIMAP_COLOR, linewidth=1.8, label="SIMAP")
    adsb_altitude_dot, = altitude_ax.plot([], [], "o", color=ADSB_COLOR, markersize=7)
    simap_altitude_dot, = altitude_ax.plot([], [], "o", color=SIMAP_COLOR, markersize=7)
    altitude_ax.set_title("Altitude")
    altitude_ax.set_ylabel("Altitude [m]")
    altitude_ax.grid(True, alpha=0.25)
    altitude_ax.legend(loc="best")

    speed_ax.plot(adsb.time_s, adsb.speed_mps, color=ADSB_COLOR, linewidth=1.8, label="ADS-B")
    speed_ax.plot(simap.time_s, simap.speed_mps, color=SIMAP_COLOR, linewidth=1.8, label="SIMAP")
    adsb_speed_dot, = speed_ax.plot([], [], "o", color=ADSB_COLOR, markersize=7)
    simap_speed_dot, = speed_ax.plot([], [], "o", color=SIMAP_COLOR, markersize=7)
    speed_ax.set_title("Ground Speed")
    speed_ax.set_xlabel("Unix time [s]")
    speed_ax.set_ylabel("Speed [m/s]")
    speed_ax.grid(True, alpha=0.25)
    speed_ax.legend(loc="best")

    label = trajectory_ax.text(
        0.02,
        0.98,
        "",
        transform=trajectory_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.0,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#333333", "alpha": 0.92},
    )

    slider = Slider(
        ax=slider_ax,
        label="Time",
        valmin=start_time,
        valmax=end_time,
        valinit=initial_time,
        valstep=1.0,
    )

    simulation = payload.get("simulation", {})

    def update(time_s: float) -> None:
        adsb_sample = sample_trajectory(adsb, float(time_s))
        simap_sample = sample_trajectory(simap, float(time_s))

        _set_marker(adsb_position_dot, adsb_sample, "lon_deg", "lat_deg")
        _set_marker(simap_position_dot, simap_sample, "lon_deg", "lat_deg")
        _set_marker(adsb_altitude_dot, adsb_sample, "time_s", "altitude_m")
        _set_marker(simap_altitude_dot, simap_sample, "time_s", "altitude_m")
        _set_marker(adsb_speed_dot, adsb_sample, "time_s", "speed_mps")
        _set_marker(simap_speed_dot, simap_sample, "time_s", "speed_mps")

        label.set_text(
            "\n".join(
                [
                    format_unix_time(float(time_s)),
                    sample_label("ADS-B", adsb_sample),
                    sample_label("SIMAP", simap_sample),
                    f"SIMAP status: {simulation.get('message', UNAVAILABLE)}",
                ]
            )
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(initial_time)
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    key = parse_flight_key(args.flight)
    payload = load_simap_payload(args.artifacts_path, key)
    simap = trajectory_from_simap_payload(payload)
    adsb = trajectory_from_adsb_track(load_adsb_track(args.raw_dir, key, args.split_gap_seconds))

    print(f"Loaded {key.flight_id}")
    print(f"ADS-B: {len(adsb.time_s)} points, {format_unix_time(adsb.first_time_s)} to {format_unix_time(adsb.last_time_s)}")
    print(f"SIMAP: {len(simap.time_s)} points, {format_unix_time(simap.first_time_s)} to {format_unix_time(simap.last_time_s)}")
    plot_cross_check(adsb, simap, payload, key)


if __name__ == "__main__":
    main()
