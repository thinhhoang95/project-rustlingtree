from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from scenario.demand_opensky.adsb_catalog_common import RAW_COLUMNS
from scenario.demand_opensky.adsb_catalog_io import build_flight_id, normalize_callsign


@dataclass(frozen=True)
class FlightCompressionTask:
    flight_id: str
    callsign: str
    icao24: str
    records: tuple[tuple[int, float, float, float], ...]


@dataclass(frozen=True)
class CompressedFlightResult:
    flight_id: str
    callsign: str
    icao24: str
    first_time: int | None
    last_time: int | None
    raw_point_count: int
    compressed_point_count: int
    lateral_breakpoint_count: int
    altitude_breakpoint_count: int

    @property
    def compression_ratio(self) -> float:
        if self.raw_point_count == 0:
            return 0.0
        return self.compressed_point_count / self.raw_point_count


def list_raw_csv_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.csv")
        if path.is_file() and not path.name.startswith("._") and path.stem.isdigit()
    )


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        usecols=range(len(RAW_COLUMNS)),
    )
    frame["source_file"] = str(path)
    frame["callsign"] = frame["callsign"].map(normalize_callsign)
    frame["icao24"] = frame["icao24"].astype(str).str.strip()
    frame["flight_id"] = [
        build_flight_id(callsign, icao24)
        for callsign, icao24 in zip(frame["callsign"], frame["icao24"], strict=False)
    ]

    for column in ["time", "lat", "lon", "heading", "geoaltitude"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def load_raw_adsb(input_dir: Path, processes: int) -> pd.DataFrame:
    csv_files = list_raw_csv_files(input_dir)
    if not csv_files:
        raise FileNotFoundError(f"No raw ADS-B CSV files found under {input_dir}")

    if processes <= 1 or len(csv_files) == 1:
        frames = [load_raw_csv(path) for path in csv_files]
    else:
        from multiprocessing import Pool

        worker_count = max(1, min(processes, len(csv_files)))
        with Pool(processes=worker_count) as pool:
            frames = pool.map(load_raw_csv, csv_files)

    tracks = pd.concat(frames, ignore_index=True)
    tracks = tracks.loc[
        tracks["time"].notna()
        & tracks["lat"].notna()
        & tracks["lon"].notna()
        & tracks["geoaltitude"].notna()
    ].copy()

    tracks["time"] = tracks["time"].astype("int64")
    tracks.sort_values(["flight_id", "time"], inplace=True, kind="stable")
    tracks.drop_duplicates(["flight_id", "time"], keep="last", inplace=True)
    tracks.reset_index(drop=True, inplace=True)
    return tracks


def load_arrival_departure_flight_ids(catalog_path: Path) -> set[str]:
    catalog = pd.read_csv(catalog_path, usecols=["flight_id", "operation"])
    operations = catalog["operation"].astype(str).str.strip().str.lower()
    flight_ids = catalog.loc[operations.isin(["arrival", "departure"]), "flight_id"]
    return set(flight_ids.astype(str).str.strip())


def filter_tracks_to_flights(tracks: pd.DataFrame, flight_ids: set[str]) -> pd.DataFrame:
    filtered = tracks.loc[tracks["flight_id"].isin(flight_ids)].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def split_tracks_by_gap(tracks: pd.DataFrame, split_gap_seconds: int) -> pd.DataFrame:
    if tracks.empty:
        return tracks.copy()

    split_frames: list[pd.DataFrame] = []
    for _, flight in tracks.groupby("flight_id", sort=False):
        if flight.empty:
            continue

        times = flight["time"].to_numpy(dtype=float)
        split_points = [0]
        for index in range(1, len(flight)):
            previous_time = times[index - 1]
            current_time = times[index]
            if pd.isna(previous_time) or pd.isna(current_time):
                continue
            if current_time - previous_time > split_gap_seconds:
                split_points.append(index)
        split_points.append(len(flight))

        first = flight.iloc[0]
        base_callsign = str(first["callsign"])
        icao24 = str(first["icao24"])
        for segment_number, (start, stop) in enumerate(zip(split_points, split_points[1:], strict=False), start=1):
            segment = flight.iloc[start:stop].copy().reset_index(drop=True)
            segment_callsign = f"{base_callsign}M{segment_number}"
            segment["callsign"] = segment_callsign
            segment["flight_id"] = f"{segment_callsign}{icao24}"
            split_frames.append(segment)

    split_tracks = pd.concat(split_frames, ignore_index=True)
    split_tracks.sort_values(["flight_id", "time"], inplace=True, kind="stable")
    split_tracks.reset_index(drop=True, inplace=True)
    return split_tracks


def build_flight_tasks(tracks: pd.DataFrame) -> list[FlightCompressionTask]:
    tasks: list[FlightCompressionTask] = []
    columns = ["time", "lat", "lon", "geoaltitude"]
    for flight_id, flight in tracks.groupby("flight_id", sort=False):
        first = flight.iloc[0]
        records = tuple(
            (
                int(row.time),
                float(row.lat),
                float(row.lon),
                float(row.geoaltitude),
            )
            for row in flight[columns].itertuples(index=False)
        )
        tasks.append(
            FlightCompressionTask(
                flight_id=str(flight_id),
                callsign=str(first["callsign"]),
                icao24=str(first["icao24"]),
                records=records,
            )
        )
    return tasks


def write_jsonl(path: Path, payloads: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for payload in payloads:
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")


def write_manifest(path: Path, results: Iterable[CompressedFlightResult], flights_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "flight_id",
        "callsign",
        "icao24",
        "first_time",
        "last_time",
        "raw_point_count",
        "compressed_point_count",
        "compression_ratio",
        "lateral_breakpoint_count",
        "altitude_breakpoint_count",
        "output_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: (item.first_time is None, item.first_time or 0, item.flight_id)):
            writer.writerow(
                {
                    "flight_id": result.flight_id,
                    "callsign": result.callsign,
                    "icao24": result.icao24,
                    "first_time": result.first_time,
                    "last_time": result.last_time,
                    "raw_point_count": result.raw_point_count,
                    "compressed_point_count": result.compressed_point_count,
                    "compression_ratio": f"{result.compression_ratio:.6f}",
                    "lateral_breakpoint_count": result.lateral_breakpoint_count,
                    "altitude_breakpoint_count": result.altitude_breakpoint_count,
                    "output_path": flights_path.as_posix(),
                }
            )
