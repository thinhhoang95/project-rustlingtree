from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from typing import cast

import pandas as pd

from scenario.demand_opensky.adsb_catalog_common import RAW_COLUMNS


def list_input_csv_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix == ".csv"
        and not path.name.startswith("._")
        and path.stem.isdigit()
    )


def normalize_callsign(value: object) -> str:
    if cast(bool, pd.isna(value)):
        return ""
    return str(value).strip()


def build_flight_id(callsign: str, icao24: str) -> str:
    callsign_token = callsign if callsign else "UNKNOWN_"
    return f"{callsign_token}{icao24}"


def load_hourly_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        usecols=range(len(RAW_COLUMNS)),
    )
    frame["callsign"] = frame["callsign"].map(normalize_callsign)
    frame["icao24"] = frame["icao24"].astype(str).str.strip()
    frame["flight_id"] = [
        build_flight_id(callsign, icao24)
        for callsign, icao24 in zip(frame["callsign"], frame["icao24"], strict=False)
    ]

    numeric_columns = ["time", "lat", "lon", "heading", "geoaltitude"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def load_tracks(input_dir: Path, processes: int) -> pd.DataFrame:
    csv_files = list_input_csv_files(input_dir)
    if not csv_files:
        raise FileNotFoundError(f"No hourly CSV files found in {input_dir}")

    worker_count = max(1, min(processes, len(csv_files)))
    if worker_count == 1:
        frames = [load_hourly_csv(path) for path in csv_files]
    else:
        with Pool(processes=worker_count) as pool:
            frames = pool.map(load_hourly_csv, csv_files)

    tracks = pd.concat(frames, ignore_index=True)
    tracks.sort_values(["flight_id", "time"], inplace=True, kind="stable")
    tracks.reset_index(drop=True, inplace=True)
    return tracks


def load_runway_thresholds(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        runway_a, runway_b = str(row["runway_pair"]).split("/", maxsplit=1)
        rows.append(
            {
                "runway": runway_a,
                "runway_pair": row["runway_pair"],
                "threshold_lat": float(row["latitude_a"]),
                "threshold_lon": float(row["longitude_a"]),
            }
        )
        rows.append(
            {
                "runway": runway_b,
                "runway_pair": row["runway_pair"],
                "threshold_lat": float(row["latitude_b"]),
                "threshold_lon": float(row["longitude_b"]),
            }
        )
    return pd.DataFrame(rows).sort_values("runway").reset_index(drop=True)


def load_fixes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["fix_type"].isin(["waypoint", "vor"])].copy()
    frame = frame.loc[frame["latitude_deg"].notna() & frame["longitude_deg"].notna()].copy()
    frame["identifier"] = frame["identifier"].astype(str).str.strip()
    return frame[["identifier", "fix_type", "latitude_deg", "longitude_deg"]].reset_index(drop=True)


def resolve_date_label(input_dir: Path, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    return input_dir.name


def write_outputs(
    events: pd.DataFrame,
    fix_sequences: pd.DataFrame,
    output_dir: Path,
    date_label: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / f"{date_label}_landings_and_departures.csv"
    fixes_path = output_dir / f"{date_label}_fix_sequences.csv"

    events.to_csv(events_path, index=False)
    fix_sequences.to_csv(fixes_path, index=False)
    return events_path, fixes_path
