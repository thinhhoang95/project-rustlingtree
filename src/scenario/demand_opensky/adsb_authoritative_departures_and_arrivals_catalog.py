from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "adsb" / "catalogs"
DEFAULT_AIRPORT = "KDFW"
DEFAULT_TRINO_BIN = str(PROJECT_ROOT / "trino")
DEFAULT_USERNAME = os.getenv("OPENSKY_USERNAME", "thinhhoangdinh")
DEFAULT_PASSWORD = os.getenv("OPENSKY_PASSWORD", "iQ6^yrwe7o3m")
DEFAULT_OUTPUT_BASENAME = "authoritative_departures_and_arrivals"
EXPECTED_CATALOG_COLUMNS = ["flight_type", "callsign", "icao24", "event_time"]
LEGACY_CATALOG_COLUMNS = ["flight_type", "callsign", "icao24"]


def parse_datetime_arg(value: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse wraps this path
        raise argparse.ArgumentTypeError(f"Invalid datetime value: {value!r}") from exc


def parse_timezone_arg(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except Exception as exc:  # pragma: no cover - argparse wraps this path
        raise argparse.ArgumentTypeError(f"Invalid IANA timezone: {value!r}") from exc


def get_jwt() -> str:
    response = requests.post(
        "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token",
        data={
            "client_id": "trino-client",
            "grant_type": "password",
            "username": DEFAULT_USERNAME,
            "password": DEFAULT_PASSWORD,
        },
        timeout=30,
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    print("Obtained JWT for Trino query.")
    return token


def build_query(start_day_ts: int, end_day_ts: int, airport: str) -> str:
    airport = airport.upper().strip()
    return f"""
WITH arrivals AS (
    SELECT DISTINCT
        'arrival' AS flight_type,
        TRIM(callsign) AS callsign,
        TRIM(icao24) AS icao24,
        CAST(lastseen AS bigint) AS event_time
    FROM flights_data4
    WHERE day BETWEEN {start_day_ts} AND {end_day_ts}
      AND estarrivalairport = '{airport}'
      AND callsign IS NOT NULL
      AND icao24 IS NOT NULL
      AND lastseen IS NOT NULL
      AND TRIM(callsign) <> ''
      AND TRIM(icao24) <> ''
),
departures AS (
    SELECT DISTINCT
        'departure' AS flight_type,
        TRIM(callsign) AS callsign,
        TRIM(icao24) AS icao24,
        CAST(firstseen AS bigint) AS event_time
    FROM flights_data4
    WHERE day BETWEEN {start_day_ts} AND {end_day_ts}
      AND estdepartureairport = '{airport}'
      AND callsign IS NOT NULL
      AND icao24 IS NOT NULL
      AND firstseen IS NOT NULL
      AND TRIM(callsign) <> ''
      AND TRIM(icao24) <> ''
)
SELECT flight_type, callsign, icao24, event_time
FROM arrivals
UNION ALL
SELECT flight_type, callsign, icao24, event_time
FROM departures
ORDER BY flight_type, callsign, icao24, event_time
""".strip()


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


def _day_start_ts(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp())


def build_output_path(
    output_dir: Path,
    from_datetime: dt.datetime,
    to_datetime: dt.datetime,
    timezone: ZoneInfo,
    airport: str,
) -> Path:
    local_from, local_to, _, _ = _normalize_datetime_range(from_datetime, to_datetime, timezone)
    if local_from.date() == local_to.date():
        label = local_from.strftime("%Y-%m-%d")
    else:
        label = f"{local_from.strftime('%Y-%m-%d')}_to_{local_to.strftime('%Y-%m-%d')}"
    airport = airport.lower().strip()
    return output_dir / f"{label}_{airport}_{DEFAULT_OUTPUT_BASENAME}.csv"


def run_query(query: str, output_path: Path, trino_bin: str, token: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        trino_bin,
        "--user",
        DEFAULT_USERNAME,
        "--server",
        "https://trino.opensky-network.org",
        f"--access-token={token}",
        "--catalog",
        "minio",
        "--schema",
        "osky",
        "--execute",
        query,
        "--output-format",
        "CSV",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        raise RuntimeError(
            "Trino query failed.\n"
            f"Exit code: {result.returncode}\n"
            f"Stderr:\n{result.stderr.strip()}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Query completed but {output_path} is missing or empty.")


def _read_catalog_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if set(EXPECTED_CATALOG_COLUMNS).issubset(frame.columns):
        return frame
    if set(LEGACY_CATALOG_COLUMNS).issubset(frame.columns):
        fallback = frame.copy()
        fallback["event_time"] = pd.NA
        return fallback[EXPECTED_CATALOG_COLUMNS]

    fallback = pd.read_csv(path, header=None, names=EXPECTED_CATALOG_COLUMNS)
    if not fallback.empty:
        first_row = [str(fallback.iloc[0][column]).strip() for column in EXPECTED_CATALOG_COLUMNS]
        if first_row == EXPECTED_CATALOG_COLUMNS:
            fallback = fallback.iloc[1:].reset_index(drop=True)
    return fallback


def _prune_catalog_to_requested_range(
    catalog: pd.DataFrame,
    utc_from: dt.datetime,
    utc_to: dt.datetime,
    console: Console,
) -> pd.DataFrame:
    working = catalog.copy()
    working["event_time"] = pd.to_numeric(working["event_time"], errors="coerce")
    working["event_time_utc"] = pd.to_datetime(working["event_time"], unit="s", utc=True, errors="coerce")

    before_count = len(working)
    within_range = working["event_time_utc"].between(utc_from, utc_to, inclusive="both")
    pruned = working.loc[within_range].copy()
    dropped_count = before_count - len(pruned)

    console.print(
        "[yellow]Timestamp pruning:[/yellow] "
        "using `lastseen` for arrivals and `firstseen` for departures; "
        "keeping only rows whose `event_time_utc` falls inside the exact CLI UTC range."
    )
    console.print(
        f"[yellow]  - Requested UTC window:[/yellow] {utc_from.isoformat()} -> {utc_to.isoformat()}"
    )
    console.print(
        f"[yellow]  - Pruned authoritative rows:[/yellow] kept {len(pruned)} of {before_count} "
        f"(dropped {dropped_count})"
    )

    return pruned


def download_authoritative_departures_and_arrivals_catalog(
    from_datetime: dt.datetime,
    to_datetime: dt.datetime,
    timezone: ZoneInfo,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    airport: str = DEFAULT_AIRPORT,
    trino_bin: str = DEFAULT_TRINO_BIN,
    output_path: Path | None = None,
    console: Console | None = None,
) -> Path:
    console = console or Console()
    local_from, local_to, utc_from, utc_to = _normalize_datetime_range(from_datetime, to_datetime, timezone)
    target_path = output_path or build_output_path(output_dir, local_from, local_to, timezone, airport)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    start_day = utc_from.date()
    end_day = utc_to.date()
    start_day_ts = _day_start_ts(start_day)
    end_day_ts = _day_start_ts(end_day)
    console.print(
        f"[yellow]Downloading authoritative catalog for {airport.upper().strip()} "
        f"across UTC days {start_day.isoformat()} to {end_day.isoformat()} into {target_path}"
    )

    token = get_jwt()
    query = build_query(start_day_ts, end_day_ts, airport)
    console.print(f"[yellow]  - Querying day range [{start_day_ts}, {end_day_ts}]")
    run_query(query, target_path, trino_bin, token)
    catalog = _read_catalog_csv(target_path)

    catalog = catalog.loc[catalog["callsign"].notna() & catalog["icao24"].notna()].copy()
    catalog["flight_type"] = catalog["flight_type"].astype(str).str.strip()
    catalog["callsign"] = catalog["callsign"].astype(str).str.strip()
    catalog["icao24"] = catalog["icao24"].astype(str).str.strip()
    catalog["flight_id"] = catalog["callsign"] + catalog["icao24"]
    catalog = _prune_catalog_to_requested_range(catalog, utc_from, utc_to, console)
    catalog.drop_duplicates(subset=["flight_type", "flight_id"], inplace=True)
    catalog.sort_values(["flight_type", "event_time_utc", "callsign", "icao24"], kind="stable", inplace=True)
    catalog.reset_index(drop=True, inplace=True)
    catalog.to_csv(target_path, index=False)

    console.print(f"[green]Wrote authoritative catalog with {len(catalog)} rows to {target_path}")
    return target_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download distinct KDFW arrivals and departures for a requested datetime range.",
    )
    parser.add_argument("--from-datetime", type=parse_datetime_arg, required=True)
    parser.add_argument("--to-datetime", type=parse_datetime_arg, required=True)
    parser.add_argument("--timezone", type=parse_timezone_arg, required=True)
    parser.add_argument("--airport", default=DEFAULT_AIRPORT, help="Airport ICAO code, default: KDFW")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--output", type=Path, default=None, help="Optional explicit output CSV path")
    parser.add_argument("--trino-bin", default=DEFAULT_TRINO_BIN, help="Trino CLI binary")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    console = Console()
    output_path = download_authoritative_departures_and_arrivals_catalog(
        from_datetime=args.from_datetime,
        to_datetime=args.to_datetime,
        timezone=args.timezone,
        output_dir=args.output_dir,
        airport=args.airport,
        trino_bin=args.trino_bin,
        output_path=args.output,
        console=console,
    )
    console.print(f"[green]Done:[/green] {output_path}")


if __name__ == "__main__":
    main()
