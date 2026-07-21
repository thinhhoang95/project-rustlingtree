#!/usr/bin/env python3
"""Build Hailmary's tuned six-cluster KDFW/RW18R artifact from raw ADS-B."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

from hailmary.clustering import build_cluster_library_from_adsb
from hailmary.config import ClusteringConfig
from hailmary.data import load_arrival_catalog, load_catalog_raw_adsb_tracks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data/adsb/catalogs/2026-04-01_landings_and_departures.csv"
DEFAULT_RAW_ADSB_ROOT = PROJECT_ROOT / "data/adsb/raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/hailmary/clusters.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the six-cluster Hailmary KDFW/RW18R library from raw ADS-B."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--raw-adsb-root", type=Path, default=DEFAULT_RAW_ADSB_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-id", default="kdfw-2026-04-01")
    parser.add_argument("--airport", default="KDFW")
    parser.add_argument("--runway", default="18R")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog_path = args.catalog.expanduser().resolve()
    raw_adsb_root = args.raw_adsb_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    arrivals = load_arrival_catalog(catalog_path, airport=args.airport)
    raw_tracks = load_catalog_raw_adsb_tracks(
        raw_adsb_root,
        flight_ids=(arrival.flight_id for arrival in arrivals),
    )
    config = ClusteringConfig(
        min_cluster_sizes=(8,),
        min_samples=(3,),
        selection_methods=("eom",),
    )
    result = build_cluster_library_from_adsb(
        arrivals,
        raw_tracks,
        dataset_id=args.dataset_id,
        airport=args.airport,
        runway=args.runway,
        config=config,
        metadata={
            "catalog_path": str(catalog_path),
            "raw_adsb_root": str(raw_adsb_root),
            "pipeline_stage": "cluster_library",
            "tuning_note": "six-cluster preference: eom, min_cluster_size=8, min_samples=3",
        },
    )
    if len(result.library.medoids) != 6:
        raise RuntimeError(f"expected six clusters, got {len(result.library.medoids)}")

    result.library.write(output_path)
    print(
        {
            "artifact": str(output_path),
            "content_hash": result.library.artifact_content_hash,
            "cluster_count": len(result.library.medoids),
            "selected_parameters": dict(result.library.clustering.selected_parameters),
            "accepted_tracks": len(result.preparation.prepared_tracks),
            "rejection_counts": dict(
                sorted(Counter(item.reason.value for item in result.rejections).items())
            ),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
