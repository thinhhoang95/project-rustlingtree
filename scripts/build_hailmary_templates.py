#!/usr/bin/env python3
"""Build Hailmary template NPZ files and their manifest from the six-cluster artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hailmary.cli.build_templates import main as build_templates_main
from hailmary.clustering import ClusterLibrary, prepare_adsb_tracks_for_clustering
from hailmary.data import load_arrival_catalog, load_catalog_raw_adsb_tracks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data/adsb/catalogs/2026-04-01_landings_and_departures.csv"
DEFAULT_RAW_ADSB_ROOT = PROJECT_ROOT / "data/adsb/raw"
DEFAULT_CLUSTERS = PROJECT_ROOT / "artifacts/hailmary/clusters.json"
DEFAULT_MEDOID_TRACKS = PROJECT_ROOT / "artifacts/hailmary/medoid_tracks_6.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/hailmary/templates"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile six-cluster Hailmary medoids into template NPZ files and manifest.json."
    )
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--raw-adsb-root", type=Path, default=DEFAULT_RAW_ADSB_ROOT)
    parser.add_argument("--medoid-tracks", type=Path, default=DEFAULT_MEDOID_TRACKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--station-count", type=int, default=512)
    parser.add_argument("--payload-kg", type=float, default=12_000.0)
    parser.add_argument("--no-simap-validation", action="store_true")
    return parser


def _write_medoid_tracks(
    library: ClusterLibrary,
    *,
    catalog_path: Path,
    raw_adsb_root: Path,
    path: Path,
) -> None:
    arrivals = load_arrival_catalog(catalog_path, airport=library.airport)
    raw_tracks = load_catalog_raw_adsb_tracks(
        raw_adsb_root,
        flight_ids=(arrival.flight_id for arrival in arrivals),
    )
    preparation = prepare_adsb_tracks_for_clustering(
        arrivals,
        raw_tracks,
        airport=library.airport,
        runway=library.runway,
    )
    prepared_by_id = preparation.tracks_by_flight_id
    records = []
    for medoid in library.medoids:
        try:
            track = prepared_by_id[medoid.medoid_flight_id].to_medoid_track()
        except KeyError as exc:
            raise ValueError(
                f"raw ADS-B preparation did not produce medoid {medoid.medoid_flight_id!r}"
            ) from exc
        records.append(
            {
                "flight_id": track.flight_id,
                "time_s": track.time_s.tolist(),
                "lat_deg": track.lat_deg.tolist(),
                "lon_deg": track.lon_deg.tolist(),
                "altitude_m": track.altitude_m.tolist(),
                "ground_speed_mps": None
                if track.ground_speed_mps is None
                else track.ground_speed_mps.tolist(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"tracks": records}, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clusters_path = args.clusters.expanduser().resolve()
    catalog_path = args.catalog.expanduser().resolve()
    raw_adsb_root = args.raw_adsb_root.expanduser().resolve()
    medoid_tracks_path = args.medoid_tracks.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    library = ClusterLibrary.read(clusters_path)
    if len(library.medoids) != 6:
        raise ValueError(f"expected a six-cluster artifact, found {len(library.medoids)} clusters")

    _write_medoid_tracks(
        library,
        catalog_path=catalog_path,
        raw_adsb_root=raw_adsb_root,
        path=medoid_tracks_path,
    )
    template_argv = [
        "--clusters",
        str(clusters_path),
        "--medoid-tracks",
        str(medoid_tracks_path),
        "--output-dir",
        str(output_dir),
        "--station-count",
        str(args.station_count),
        "--payload-kg",
        str(args.payload_kg),
    ]
    if args.no_simap_validation:
        template_argv.append("--no-simap-validation")
    return build_templates_main(template_argv)


if __name__ == "__main__":
    raise SystemExit(main())
