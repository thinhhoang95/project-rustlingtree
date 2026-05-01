from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "adsb" / "kdfw_2023-04-01_deps_and_arrvs.csv"
DEFAULT_AIRPORT = "KDFW"
DEFAULT_DATE = "2023-04-01"
DEFAULT_TRINO_BIN = "/mnt/d/project-rustlingtree/trino"
DEFAULT_USERNAME = os.getenv("OPENSKY_USERNAME", "thinhhoangdinh")
DEFAULT_PASSWORD = os.getenv("OPENSKY_PASSWORD", "iQ6^yrwe7o3m")


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


def build_query(day_ts: int, airport: str) -> str:
    airport = airport.upper().strip()
    return f"""
WITH arrivals AS (
    SELECT DISTINCT
        'arrival' AS flight_type,
        TRIM(callsign) AS callsign,
        TRIM(icao24) AS icao24
    FROM flights_data4
    WHERE day = {day_ts}
      AND estarrivalairport = '{airport}'
      AND callsign IS NOT NULL
      AND icao24 IS NOT NULL
      AND TRIM(callsign) <> ''
      AND TRIM(icao24) <> ''
),
departures AS (
    SELECT DISTINCT
        'departure' AS flight_type,
        TRIM(callsign) AS callsign,
        TRIM(icao24) AS icao24
    FROM flights_data4
    WHERE day = {day_ts}
      AND estdepartureairport = '{airport}'
      AND callsign IS NOT NULL
      AND icao24 IS NOT NULL
      AND TRIM(callsign) <> ''
      AND TRIM(icao24) <> ''
)
SELECT flight_type, callsign, icao24
FROM arrivals
UNION ALL
SELECT flight_type, callsign, icao24
FROM departures
ORDER BY flight_type, callsign, icao24
""".strip()


def run_query(query: str, output_path: Path, trino_bin: str) -> None:
    token = get_jwt()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download distinct KDFW arrivals and departures (callsign + icao24) for one UTC day.",
    )
    parser.add_argument("--airport", default=DEFAULT_AIRPORT, help="Airport ICAO code, default: KDFW")
    parser.add_argument("--date", default=DEFAULT_DATE, help="UTC date in YYYY-MM-DD, default: 2025-04-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--trino-bin", default=DEFAULT_TRINO_BIN, help="Trino CLI binary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    airport = args.airport.strip().upper()
    day = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
    day_ts = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp())

    print(f"Airport: {airport}")
    print(f"UTC day start: {day.isoformat()} 00:00:00+00:00")
    print(f"day_ts: {day_ts}")
    print(f"Output: {args.output}")

    query = build_query(day_ts, airport)
    run_query(query, args.output, args.trino_bin)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
