"""Build a versioned route graph from medoid polyline geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hailmary.topology import MedoidRoute, RouteGraphConfig, build_route_graph


def _load_routes(path: Path) -> tuple[MedoidRoute, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("routes"), list):
        raise ValueError("route input must contain a routes list")
    dataset_id = str(payload.get("dataset_id", ""))
    routes: list[MedoidRoute] = []
    for record in payload["routes"]:
        if not isinstance(record, Mapping):
            raise ValueError("each route record must be an object")
        routes.append(
            MedoidRoute(
                dataset_id=str(record.get("dataset_id", dataset_id)),
                airport=str(record["airport"]),
                runway=str(record["runway"]),
                cluster_id=str(record["cluster_id"]),
                lat_deg=tuple(record["lat_deg"]),
                lon_deg=tuple(record["lon_deg"]),
                dispersion_m=float(record.get("dispersion_m", 0.0)),
                medoid_flight_id=str(record.get("medoid_flight_id", "")),
                source_hash=str(record.get("source_hash", "")),
            )
        )
    return tuple(routes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hailmary-build-route-graph")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lateral-floor-nm", type=float, default=0.5)
    parser.add_argument("--tangent-tolerance-deg", type=float, default=15.0)
    parser.add_argument("--minimum-common-length-nm", type=float, default=5.0)
    parser.add_argument("--resample-step-nm", type=float, default=0.25)
    parser.add_argument("--hysteresis-gap-nm", type=float, default=0.75)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    routes = _load_routes(args.input)
    config = RouteGraphConfig(
        lateral_floor_nm=args.lateral_floor_nm,
        tangent_tolerance_deg=args.tangent_tolerance_deg,
        minimum_common_length_nm=args.minimum_common_length_nm,
        resample_step_nm=args.resample_step_nm,
        hysteresis_gap_nm=args.hysteresis_gap_nm,
    )
    artifact = build_route_graph(
        routes,
        config=config,
        provenance={
            "input": args.input.resolve().as_posix(),
            "source_hashes": sorted({item.source_hash for item in routes if item.source_hash}),
        },
    )
    artifact.write(args.output)
    print(
        json.dumps(
            {
                "artifact_content_hash": artifact.artifact_content_hash,
                "cluster_count": len({item.qualified_cluster_id for item in routes}),
                "segment_count": len(artifact.segments),
                "node_count": len(artifact.nodes),
                "output": args.output.resolve().as_posix(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
