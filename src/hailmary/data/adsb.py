"""Minimal raw ADS-B reader and observed terminal-entry reconstruction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from hailmary.geometry.frame import LocalFrame
from hailmary.geometry.polyline import readonly_float64

RAW_COLUMNS = ("time", "icao24", "lat", "lon", "heading", "callsign", "geoaltitude")
METERS_PER_NM = 1_852.0


def normalize_callsign(value: object) -> str:
    return str(value).strip()


def build_flight_id(callsign: str, icao24: str) -> str:
    return f"{callsign if callsign else 'UNKNOWN_'}{icao24}"


@dataclass(frozen=True)
class RawADSBTrack:
    flight_id: str
    callsign: str
    icao24: str
    time_s: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    heading_deg: np.ndarray
    geoaltitude_m: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "time_s": readonly_float64(self.time_s, name="time_s", ndim=1),
            "lat_deg": readonly_float64(self.lat_deg, name="lat_deg", ndim=1),
            "lon_deg": readonly_float64(self.lon_deg, name="lon_deg", ndim=1),
            "heading_deg": readonly_float64(self.heading_deg, name="heading_deg", ndim=1),
            "geoaltitude_m": readonly_float64(self.geoaltitude_m, name="geoaltitude_m", ndim=1),
        }
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
            raise ValueError("raw ADS-B track arrays must have equal lengths of at least two")
        if np.any(np.diff(arrays["time_s"]) <= 0.0):
            raise ValueError("raw ADS-B track times must be strictly increasing")
        if np.any(np.abs(arrays["lat_deg"]) > 90.0) or np.any(np.abs(arrays["lon_deg"]) > 180.0):
            raise ValueError("raw ADS-B track contains invalid coordinates")
        if not str(self.flight_id).strip():
            raise ValueError("flight_id must be nonempty")
        object.__setattr__(self, "flight_id", str(self.flight_id).strip())
        object.__setattr__(self, "callsign", str(self.callsign).strip())
        object.__setattr__(self, "icao24", str(self.icao24).strip())
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def point_count(self) -> int:
        return len(self.time_s)


@dataclass(frozen=True)
class TerminalEntry:
    flight_id: str
    time_s: float
    lat_deg: float
    lon_deg: float
    geoaltitude_m: float
    segment_index: int
    segment_fraction: float
    radius_m: float

    def __post_init__(self) -> None:
        numeric = (self.time_s, self.lat_deg, self.lon_deg, self.geoaltitude_m, self.segment_fraction, self.radius_m)
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("terminal entry values must be finite")
        if not 0.0 <= self.segment_fraction <= 1.0:
            raise ValueError("segment_fraction must be within [0, 1]")
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")


def list_raw_csv_files(input_dir: str | Path) -> tuple[Path, ...]:
    root = Path(input_dir)
    paths = sorted(
        path
        for path in root.rglob("*.csv")
        if path.is_file() and not path.name.startswith("._") and path.stem.isdigit()
    )
    if not paths:
        raise FileNotFoundError(f"no raw ADS-B CSV files found under {root}")
    return tuple(paths)


def _parse_raw_rows(paths: Iterable[Path]) -> dict[str, list[tuple[float, float, float, float, float, str, str]]]:
    grouped: dict[str, list[tuple[float, float, float, float, float, str, str]]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, row in enumerate(csv.reader(stream), start=1):
                if len(row) < len(RAW_COLUMNS):
                    continue
                time_raw, icao_raw, lat_raw, lon_raw, heading_raw, callsign_raw, altitude_raw = row[: len(RAW_COLUMNS)]
                try:
                    time_s, lat_deg, lon_deg, altitude_m = (
                        float(value) for value in (time_raw, lat_raw, lon_raw, altitude_raw)
                    )
                except (TypeError, ValueError):
                    continue
                if not all(np.isfinite(value) for value in (time_s, lat_deg, lon_deg, altitude_m)):
                    continue
                try:
                    heading_deg = float(heading_raw)
                except (TypeError, ValueError):
                    heading_deg = float("nan")
                icao24 = str(icao_raw).strip()
                callsign = normalize_callsign(callsign_raw)
                if not icao24:
                    continue
                flight_id = build_flight_id(callsign, icao24)
                grouped.setdefault(flight_id, []).append(
                    (time_s, lat_deg, lon_deg, heading_deg, altitude_m, callsign, icao24)
                )
    return grouped


def load_raw_adsb_tracks(
    input_dir: str | Path,
    *,
    flight_ids: Iterable[str] | None = None,
    split_gap_seconds: float | None = None,
    suffix_segments: bool = False,
) -> tuple[RawADSBTrack, ...]:
    """Load raw headerless ADS-B CSVs into stable, immutable per-flight tracks."""

    wanted = None if flight_ids is None else {str(item) for item in flight_ids}
    grouped = _parse_raw_rows(list_raw_csv_files(input_dir))
    tracks: list[RawADSBTrack] = []
    if split_gap_seconds is not None and (not np.isfinite(split_gap_seconds) or split_gap_seconds <= 0.0):
        raise ValueError("split_gap_seconds must be finite and positive")
    if suffix_segments and split_gap_seconds is None:
        split_gap_seconds = 1_500.0
    for base_flight_id in sorted(grouped):
        # Stable last-observation-wins handling matches the existing compressor.
        by_time: dict[float, tuple[float, float, float, float, float, str, str]] = {}
        for row in grouped[base_flight_id]:
            by_time[row[0]] = row
        rows = [by_time[key] for key in sorted(by_time)]
        if len(rows) < 2:
            continue
        split_points = [0]
        if split_gap_seconds is not None:
            split_points.extend(
                index
                for index in range(1, len(rows))
                if rows[index][0] - rows[index - 1][0] > split_gap_seconds
            )
        split_points.append(len(rows))
        for segment_number, (start, stop) in enumerate(zip(split_points, split_points[1:], strict=False), start=1):
            segment = rows[start:stop]
            if len(segment) < 2:
                continue
            base_callsign = segment[0][5]
            icao24 = segment[0][6]
            callsign = f"{base_callsign}M{segment_number}" if suffix_segments else base_callsign
            flight_id = build_flight_id(callsign, icao24)
            if wanted is not None and flight_id not in wanted and base_flight_id not in wanted:
                continue
            heading = np.asarray([row[3] for row in segment], dtype=np.float64)
            finite_heading = np.isfinite(heading)
            if np.any(finite_heading):
                indices = np.arange(len(heading), dtype=np.float64)
                heading = np.interp(indices, indices[finite_heading], heading[finite_heading])
            else:
                heading = np.zeros(len(segment), dtype=np.float64)
            tracks.append(
                RawADSBTrack(
                    flight_id=flight_id,
                    callsign=callsign,
                    icao24=icao24,
                    time_s=np.asarray([row[0] for row in segment], dtype=np.float64),
                    lat_deg=np.asarray([row[1] for row in segment], dtype=np.float64),
                    lon_deg=np.asarray([row[2] for row in segment], dtype=np.float64),
                    heading_deg=heading,
                    geoaltitude_m=np.asarray([row[4] for row in segment], dtype=np.float64),
                )
            )
    return tuple(tracks)


def load_catalog_raw_adsb_tracks(
    input_dir: str | Path,
    *,
    flight_ids: Iterable[str] | None = None,
    split_gap_seconds: float = 1_500.0,
) -> tuple[RawADSBTrack, ...]:
    """Load tracks with the ``M1``/``M2`` segment IDs used by the catalog."""

    return load_raw_adsb_tracks(
        input_dir,
        flight_ids=flight_ids,
        split_gap_seconds=split_gap_seconds,
        suffix_segments=True,
    )


def _segment_circle_fraction(p0: np.ndarray, p1: np.ndarray, radius_m: float) -> float:
    delta = p1 - p0
    a = float(np.dot(delta, delta))
    if a <= 0.0:
        raise ValueError("terminal-boundary crossing segment has zero length")
    b = 2.0 * float(np.dot(p0, delta))
    c = float(np.dot(p0, p0) - radius_m * radius_m)
    discriminant = max(0.0, b * b - 4.0 * a * c)
    roots = sorted(((-b - np.sqrt(discriminant)) / (2.0 * a), (-b + np.sqrt(discriminant)) / (2.0 * a)))
    candidates = [float(value) for value in roots if -1.0e-12 <= value <= 1.0 + 1.0e-12]
    if not candidates:
        # Numeric fallback retains the specified along-segment interpolation.
        d0 = float(np.linalg.norm(p0))
        d1 = float(np.linalg.norm(p1))
        return float(np.clip((d0 - radius_m) / max(d0 - d1, 1.0e-12), 0.0, 1.0))
    return float(np.clip(candidates[-1], 0.0, 1.0))


def reconstruct_terminal_entry(
    track: RawADSBTrack,
    frame: LocalFrame,
    *,
    radius_nm: float = 50.0,
) -> TerminalEntry | None:
    """Interpolate the first outside-to-inside terminal-boundary crossing."""

    radius_m = float(radius_nm) * METERS_PER_NM
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_nm must be finite and positive")
    points = frame.project_points(track.lat_deg, track.lon_deg)
    distance = np.linalg.norm(points, axis=1)
    for index in range(len(points) - 1):
        if distance[index] + 1.0e-9 < radius_m or distance[index + 1] - 1.0e-9 > radius_m:
            continue
        if distance[index] <= radius_m and distance[index + 1] >= radius_m:
            continue  # outbound crossing, not an arrival release
        fraction = _segment_circle_fraction(points[index], points[index + 1], radius_m)
        local = points[index] + fraction * (points[index + 1] - points[index])
        lat, lon = frame.unproject(local[0], local[1])

        def interpolate(values: np.ndarray) -> float:
            return float(values[index] + fraction * (values[index + 1] - values[index]))

        return TerminalEntry(
            flight_id=track.flight_id,
            time_s=interpolate(track.time_s),
            lat_deg=float(lat),
            lon_deg=float(lon),
            geoaltitude_m=interpolate(track.geoaltitude_m),
            segment_index=index,
            segment_fraction=fraction,
            radius_m=radius_m,
        )
    return None


observed_release_crossing = reconstruct_terminal_entry
load_raw_tracks = load_raw_adsb_tracks
