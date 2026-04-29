import argparse
import sys
from multiprocessing import cpu_count
from pathlib import Path

DEFAULT_INPUT_DIR = Path("data/adsb/raw/2025-04-01")
DEFAULT_RUNWAYS_PATH = Path("data/kdfw_procs/kdfw-runways.txt")
DEFAULT_FIXES_PATH = Path("data/kdfw_procs/airport_related_fixes.csv")
DEFAULT_OUTPUT_DIR = Path("data/adsb/catalogs")
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scenario.demand_opensky.adsb_catalog_common import (
    DEFAULT_FIX_RADIUS_M,
    DEFAULT_LOOKAROUND_SECONDS,
    DEFAULT_MIN_ALTITUDE_CHANGE_M,
    DEFAULT_RUNWAY_RADIUS_M,
)
from scenario.demand_opensky.adsb_catalog_io import (
    load_fixes,
    load_runway_thresholds,
    load_tracks,
    resolve_date_label,
    write_outputs,
)
from scenario.demand_opensky.adsb_catalog_processing import process_tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract KDFW arrivals, departures, and fix sequences from hourly ADS-B CSV chunks.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--runways", type=Path, default=DEFAULT_RUNWAYS_PATH)
    parser.add_argument("--fixes", type=Path, default=DEFAULT_FIXES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date-label", type=str, default=None)
    parser.add_argument("--runway-radius-m", type=float, default=DEFAULT_RUNWAY_RADIUS_M)
    parser.add_argument("--fix-radius-m", type=float, default=DEFAULT_FIX_RADIUS_M)
    parser.add_argument("--lookaround-seconds", type=int, default=DEFAULT_LOOKAROUND_SECONDS)
    parser.add_argument("--min-altitude-change-m", type=float, default=DEFAULT_MIN_ALTITUDE_CHANGE_M)
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument("--processes", type=int, default=max(cpu_count() - 1, 1))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    date_label = resolve_date_label(args.input_dir, args.date_label)
    tracks = load_tracks(args.input_dir, args.processes)
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
    )
    events_path, fixes_path = write_outputs(events, fix_sequences, args.output_dir, date_label)

    print(f"Processed {tracks['flight_id'].nunique()} flights from {args.input_dir}")
    print(f"Classification counts: {dict(sorted(classification_counts.items()))}")
    print(f"Wrote {len(events)} arrival/departure rows to {events_path}")
    print(f"Wrote {len(fix_sequences)} fix-sequence rows to {fixes_path}")


if __name__ == "__main__":
    main()
