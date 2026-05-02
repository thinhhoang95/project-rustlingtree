import argparse
import datetime as dt
import re
import sys
from multiprocessing import cpu_count
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DEFAULT_INPUT_DIR = Path("data/adsb/raw/2026-04-01")
DEFAULT_RUNWAYS_PATH = Path("data/kdfw_procs/kdfw-runways.txt")
DEFAULT_FIXES_PATH = Path("data/kdfw_procs/airport_related_fixes.csv")
DEFAULT_OUTPUT_DIR = Path("data/adsb/catalogs")
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scenario.demand_opensky.adsb_authoritative_departures_and_arrivals_catalog import (  # noqa: E402
    download_authoritative_departures_and_arrivals_catalog,
    parse_datetime_arg,
    parse_timezone_arg,
)
from scenario.demand_opensky.adsb_catalog_common import (  # noqa: E402
    DEFAULT_FIX_RADIUS_M,
    DEFAULT_LOOKAROUND_SECONDS,
    DEFAULT_MIN_ALTITUDE_CHANGE_M,
    DEFAULT_MIN_PROXIMITY_DISTANCE_CHANGE_M,
    DEFAULT_RUNWAY_RADIUS_M,
)
from scenario.demand_opensky.adsb_catalog_io import (  # noqa: E402
    build_flight_id,
    load_fixes,
    load_runway_thresholds,
    load_tracks,
    resolve_date_label,
    write_outputs,
)
from scenario.demand_opensky.adsb_catalog_processing import process_tracks  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract KDFW arrivals, departures, and fix sequences from hourly ADS-B CSV chunks.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--runways", type=Path, default=DEFAULT_RUNWAYS_PATH)
    parser.add_argument("--fixes", type=Path, default=DEFAULT_FIXES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date-label", type=str, default=None)
    parser.add_argument("--from-datetime", type=parse_datetime_arg, required=True)
    parser.add_argument("--to-datetime", type=parse_datetime_arg, required=True)
    parser.add_argument("--timezone", type=parse_timezone_arg, required=True)
    parser.add_argument("--runway-radius-m", type=float, default=DEFAULT_RUNWAY_RADIUS_M)
    parser.add_argument("--fix-radius-m", type=float, default=DEFAULT_FIX_RADIUS_M)
    parser.add_argument("--lookaround-seconds", type=int, default=DEFAULT_LOOKAROUND_SECONDS)
    parser.add_argument("--min-altitude-change-m", type=float, default=DEFAULT_MIN_ALTITUDE_CHANGE_M)
    parser.add_argument(
        "--min-proximity-distance-change-m",
        type=float,
        default=DEFAULT_MIN_PROXIMITY_DISTANCE_CHANGE_M,
    )
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument("--processes", type=int, default=max(cpu_count() - 1, 1))
    return parser


def _normalize_datetime_range(
    from_datetime: dt.datetime,
    to_datetime: dt.datetime,
    timezone: ZoneInfo,
) -> tuple[dt.datetime, dt.datetime, dt.datetime, dt.datetime]:
    if from_datetime.tzinfo is None:
        local_from = from_datetime.replace(tzinfo=timezone)
    else:
        local_from = from_datetime.astimezone(timezone)

    if to_datetime.tzinfo is None:
        local_to = to_datetime.replace(tzinfo=timezone)
    else:
        local_to = to_datetime.astimezone(timezone)

    if local_from > local_to:
        raise ValueError("from_datetime must be less than or equal to to_datetime")

    return local_from, local_to, local_from.astimezone(dt.timezone.utc), local_to.astimezone(dt.timezone.utc)


def _format_dt(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S %Z%z")


def _validate_dataset_bounds(
    tracks: pd.DataFrame,
    from_utc: dt.datetime,
    to_utc: dt.datetime,
    timezone: ZoneInfo,
    console: Console,
) -> None:
    valid_times = tracks["time"].dropna()
    if valid_times.empty:
        raise ValueError("Loaded ADS-B tracks do not contain any timestamps.")

    earliest_ts = int(valid_times.min())
    latest_ts = int(valid_times.max())
    earliest_utc = dt.datetime.fromtimestamp(earliest_ts, tz=dt.timezone.utc)
    latest_utc = dt.datetime.fromtimestamp(latest_ts, tz=dt.timezone.utc)
    earliest_local = earliest_utc.astimezone(timezone)
    latest_local = latest_utc.astimezone(timezone)
    within_range = from_utc <= earliest_utc and latest_utc <= to_utc

    table = Table(title="ADS-B Dataset Timestamp Check", show_header=True, header_style="bold cyan")
    table.add_column("Item", style="bold")
    table.add_column("Local time")
    table.add_column("UTC")

    table.add_row("Requested start", _format_dt(from_utc.astimezone(timezone)), _format_dt(from_utc))
    table.add_row("Requested end", _format_dt(to_utc.astimezone(timezone)), _format_dt(to_utc))
    table.add_row("Dataset earliest", _format_dt(earliest_local), _format_dt(earliest_utc))
    table.add_row("Dataset latest", _format_dt(latest_local), _format_dt(latest_utc))
    table.add_row("Within range", "yes" if within_range else "no", "yes" if within_range else "no")
    console.print(Panel(table, border_style="cyan"))

    if not within_range:
        raise ValueError(
            "ADS-B dataset timestamps do not fall within the requested datetime range.\n"
            f"Requested UTC range: {_format_dt(from_utc)} -> {_format_dt(to_utc)}\n"
            f"Dataset UTC range: {_format_dt(earliest_utc)} -> {_format_dt(latest_utc)}"
        )


def _ensure_flight_id_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "flight_id" in frame.columns:
        result = frame.copy()
    else:
        result = frame.copy()
        result["flight_id"] = [
            build_flight_id(str(callsign).strip(), str(icao24).strip())
            for callsign, icao24 in zip(result["callsign"], result["icao24"], strict=False)
        ]

    result["callsign"] = result["callsign"].astype(str).str.strip()
    result["icao24"] = result["icao24"].astype(str).str.strip()
    result["flight_id"] = result["flight_id"].astype(str).str.strip()
    return result


def _base_callsign(value: object) -> str:
    return re.sub(r"M\d+$", "", str(value).strip())


def _catalog_key_frame(frame: pd.DataFrame, kind_column: str) -> pd.DataFrame:
    result = frame.copy()
    result["base_callsign"] = result["callsign"].map(_base_callsign)
    result["catalog_key"] = [
        f"{kind}|{callsign}|{icao24}"
        for kind, callsign, icao24 in zip(
            result[kind_column].astype(str).str.strip(),
            result["base_callsign"].astype(str).str.strip(),
            result["icao24"].astype(str).str.lower().str.strip(),
            strict=False,
        )
    ]
    return result


def _example_table(title: str, frame: pd.DataFrame, catalog_keys: list[str], kind_column: str) -> Table:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("catalog_key", overflow="fold")
    table.add_column("flight_id", overflow="fold")
    table.add_column("callsign", overflow="fold")
    table.add_column("icao24", overflow="fold")
    table.add_column(kind_column, overflow="fold")

    if not catalog_keys:
        table.add_row("-", "-", "-", "-", "-")
        return table

    indexed = frame.drop_duplicates(subset=["catalog_key"]).set_index("catalog_key", drop=False)
    for catalog_key in catalog_keys[:5]:
        row = indexed.loc[catalog_key]
        table.add_row(
            str(row["catalog_key"]),
            str(row["flight_id"]),
            str(row["callsign"]),
            str(row["icao24"]),
            str(row[kind_column]),
        )
    return table


def _report_catalog_differences(
    derived_events: pd.DataFrame,
    authoritative_catalog: pd.DataFrame,
    console: Console,
) -> None:
    derived = _catalog_key_frame(_ensure_flight_id_column(derived_events), "operation")
    authoritative = _catalog_key_frame(_ensure_flight_id_column(authoritative_catalog), "flight_type")

    derived_keys = set(derived["catalog_key"].astype(str))
    authoritative_keys = set(authoritative["catalog_key"].astype(str))

    derived_only = sorted(derived_keys - authoritative_keys)
    authoritative_only = sorted(authoritative_keys - derived_keys)

    summary = Table(title="Catalog Operation Differences", show_header=True, header_style="bold cyan")
    summary.add_column("Catalog", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_column("Examples", justify="right")
    summary.add_row("Derived only", str(len(derived_only)), str(min(5, len(derived_only))))
    summary.add_row("Authoritative only", str(len(authoritative_only)), str(min(5, len(authoritative_only))))
    summary.add_row("Shared", str(len(derived_keys & authoritative_keys)), "-")
    console.print(summary)

    console.print(
        _example_table(
            "Derived operations not present in the authoritative catalog",
            derived.loc[derived["catalog_key"].isin(derived_only)],
            derived_only,
            "operation",
        )
    )
    console.print(
        _example_table(
            "Authoritative operations not present in the derived catalog",
            authoritative.loc[authoritative["catalog_key"].isin(authoritative_only)],
            authoritative_only,
            "flight_type",
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()

    local_from, local_to, utc_from, utc_to = _normalize_datetime_range(args.from_datetime, args.to_datetime, args.timezone)
    date_label = resolve_date_label(args.input_dir, args.date_label)

    console.print(
        f"[yellow]Requested range:[/yellow] {_format_dt(local_from)} -> {_format_dt(local_to)} "
        f"({args.timezone.key})"
    )
    console.print(f"[yellow]Input directory:[/yellow] {args.input_dir}")
    console.print(f"[yellow]Output directory:[/yellow] {args.output_dir}")

    tracks = load_tracks(args.input_dir, args.processes)
    _validate_dataset_bounds(tracks, utc_from, utc_to, args.timezone, console)

    thresholds = load_runway_thresholds(args.runways)
    fixes = load_fixes(args.fixes)

    events, fix_sequences, classification_counts = process_tracks(
        tracks=tracks,
        thresholds=thresholds,
        fixes=fixes,
        runway_radius_m=args.runway_radius_m,
        fix_radius_m=args.fix_radius_m,
        lookaround_seconds=args.lookaround_seconds,
        min_altitude_change_m=args.min_altitude_change_m,
        split_gap_seconds=args.split_gap_seconds,
        date_label=date_label,
        min_proximity_distance_change_m=args.min_proximity_distance_change_m,
    )
    events_path, fixes_path = write_outputs(events, fix_sequences, args.output_dir, date_label)

    console.print(f"[green]Processed[/green] {tracks['flight_id'].nunique()} flights from {args.input_dir}")
    console.print(f"[green]Classification counts:[/green] {dict(sorted(classification_counts.items()))}")
    console.print(f"[green]Wrote[/green] {len(events)} arrival/departure rows to {events_path}")
    console.print(f"[green]Wrote[/green] {len(fix_sequences)} fix-sequence rows to {fixes_path}")

    authoritative_path = download_authoritative_departures_and_arrivals_catalog(
        from_datetime=local_from,
        to_datetime=local_to,
        timezone=args.timezone,
        output_dir=args.output_dir,
        console=console,
    )
    authoritative_catalog = pd.read_csv(authoritative_path)
    _report_catalog_differences(events, authoritative_catalog, console)


if __name__ == "__main__":
    main()
