from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import cpu_count
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

DEFAULT_INPUT_DIR = Path("data/adsb/raw")
DEFAULT_OUTPUT_DIR = Path("data/adsb/compressed")
OUTPUT_FLIGHTS_FILENAME = "adsb_compressed_flights.jsonl"
DEFAULT_LATERAL_TOLERANCE_M = 100.0
DEFAULT_ALTITUDE_TOLERANCE_M = 50.0
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scenario.trajectory_compressor.compressor import compress_flights
from scenario.trajectory_compressor.io import (
    build_flight_tasks,
    filter_tracks_to_flights,
    list_raw_csv_files,
    load_arrival_departure_flight_ids,
    load_raw_adsb,
    split_tracks_by_gap,
    write_jsonl,
    write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress raw ADS-B trajectories into per-flight Douglas-Peucker breakpoint files.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--landings-departures-catalog",
        type=Path,
        default=None,
        help="Optional catalog CSV; when provided, only arrival/departure flight_ids in this file are compressed.",
    )
    parser.add_argument("--lateral-tolerance-m", type=float, default=DEFAULT_LATERAL_TOLERANCE_M)
    parser.add_argument("--altitude-tolerance-m", type=float, default=DEFAULT_ALTITUDE_TOLERANCE_M)
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument("--processes", type=int, default=max(cpu_count() - 1, 1))
    return parser


def write_metadata(
    output_dir: Path,
    input_dir: Path,
    raw_csv_count: int,
    flight_count: int,
    raw_point_count: int,
    compressed_point_count: int,
    lateral_tolerance_m: float,
    altitude_tolerance_m: float,
    landings_departures_catalog: Path | None,
    catalog_flight_count: int | None,
) -> Path:
    metadata_path = output_dir / "metadata.json"
    payload = {
        "input_dir": input_dir.as_posix(),
        "raw_csv_count": raw_csv_count,
        "flight_count": flight_count,
        "raw_point_count": raw_point_count,
        "compressed_point_count": compressed_point_count,
        "compression_ratio": compressed_point_count / raw_point_count if raw_point_count else 0.0,
        "lateral_tolerance_m": lateral_tolerance_m,
        "altitude_tolerance_m": altitude_tolerance_m,
        "landings_departures_catalog": landings_departures_catalog.as_posix() if landings_departures_catalog else None,
        "catalog_arrival_departure_flight_count": catalog_flight_count,
        "layout": {
            "manifest": "manifest.csv",
            "flights": OUTPUT_FLIGHTS_FILENAME,
            "point_columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        },
        "breakpoint_mask_bits": {
            "lateral": 1,
            "altitude": 2,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return metadata_path


def render_summary(
    console: Console,
    raw_csv_count: int,
    flight_count: int,
    raw_point_count: int,
    compressed_point_count: int,
    manifest_path: Path,
    metadata_path: Path,
    flights_path: Path,
    landings_departures_catalog: Path | None,
    catalog_flight_count: int | None,
) -> None:
    table = Table(title="ADS-B trajectory compression")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Raw CSV files", f"{raw_csv_count:,}")
    table.add_row("Flights", f"{flight_count:,}")
    table.add_row("Raw points", f"{raw_point_count:,}")
    table.add_row("Compressed points", f"{compressed_point_count:,}")
    ratio = compressed_point_count / raw_point_count if raw_point_count else 0.0
    table.add_row("Compressed/raw point ratio", f"{ratio:.2%}")
    if landings_departures_catalog is not None:
        table.add_row("Catalog arrival/departure flights", f"{catalog_flight_count or 0:,}")
        table.add_row("Catalog", landings_departures_catalog.as_posix())
    table.add_row("Manifest", manifest_path.as_posix())
    table.add_row("Flights JSONL", flights_path.as_posix())
    table.add_row("Metadata", metadata_path.as_posix())
    console.print(table)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()

    if args.lateral_tolerance_m < 0:
        parser.error("--lateral-tolerance-m must be non-negative")
    if args.altitude_tolerance_m < 0:
        parser.error("--altitude-tolerance-m must be non-negative")

    csv_files = list_raw_csv_files(args.input_dir)
    console.print(f"[bold]Found[/bold] {len(csv_files):,} raw ADS-B CSV files under {args.input_dir}")

    with console.status("[bold]Loading raw ADS-B rows...[/bold]"):
        tracks = load_raw_adsb(args.input_dir, args.processes)
        tracks = split_tracks_by_gap(tracks, args.split_gap_seconds)
        catalog_flight_ids = None
        if args.landings_departures_catalog is not None:
            catalog_flight_ids = load_arrival_departure_flight_ids(args.landings_departures_catalog)
            tracks = filter_tracks_to_flights(tracks, catalog_flight_ids)
        tasks = build_flight_tasks(tracks)

    results = []
    payloads = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        progress_task = progress.add_task("Compressing flights", total=len(tasks))
        for result, payload in compress_flights(
            tasks=tasks,
            output_dir=args.output_dir,
            lateral_tolerance_m=args.lateral_tolerance_m,
            altitude_tolerance_m=args.altitude_tolerance_m,
            processes=args.processes,
        ):
            results.append(result)
            payloads.append(payload)
            progress.advance(progress_task)

    results_and_payloads = sorted(
        zip(results, payloads, strict=True),
        key=lambda item: (item[0].first_time is None, item[0].first_time or 0, item[0].flight_id),
    )
    results = [item[0] for item in results_and_payloads]
    payloads = [item[1] for item in results_and_payloads]

    flights_path = args.output_dir / OUTPUT_FLIGHTS_FILENAME
    write_jsonl(flights_path, payloads)
    manifest_path = args.output_dir / "manifest.csv"
    write_manifest(manifest_path, results, flights_path)
    metadata_path = write_metadata(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        raw_csv_count=len(csv_files),
        flight_count=len(tasks),
        raw_point_count=sum(result.raw_point_count for result in results),
        compressed_point_count=sum(result.compressed_point_count for result in results),
        lateral_tolerance_m=args.lateral_tolerance_m,
        altitude_tolerance_m=args.altitude_tolerance_m,
        landings_departures_catalog=args.landings_departures_catalog,
        catalog_flight_count=len(catalog_flight_ids) if catalog_flight_ids is not None else None,
    )

    render_summary(
        console=console,
        raw_csv_count=len(csv_files),
        flight_count=len(tasks),
        raw_point_count=sum(result.raw_point_count for result in results),
        compressed_point_count=sum(result.compressed_point_count for result in results),
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        flights_path=flights_path,
        landings_departures_catalog=args.landings_departures_catalog,
        catalog_flight_count=len(catalog_flight_ids) if catalog_flight_ids is not None else None,
    )


if __name__ == "__main__":
    main()
