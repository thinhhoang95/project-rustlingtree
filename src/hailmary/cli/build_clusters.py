"""Build a canonical cluster library from resampled local-coordinate tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from hailmary.clustering import build_cluster_library, canonical_json_dumps
from hailmary.geometry import LocalFrame


def _load_tracks(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "track_ids" not in payload or "tracks_m" not in payload:
                raise ValueError("NPZ input must contain track_ids and tracks_m")
            ids = np.asarray(payload["track_ids"])
            tracks = np.asarray(payload["tracks_m"], dtype=np.float64)
        if ids.ndim != 1 or tracks.ndim != 3 or tracks.shape[0] != len(ids):
            raise ValueError("NPZ tracks must have shapes (track,) and (track, station, 2)")
        return {str(flight_id): tracks[index] for index, flight_id in enumerate(ids.tolist())}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "track_ids" in payload and "tracks_m" in payload:
            ids = [str(item) for item in payload["track_ids"]]
            tracks = np.asarray(payload["tracks_m"], dtype=np.float64)
            if tracks.ndim != 3 or tracks.shape[0] != len(ids):
                raise ValueError("JSON tracks_m must have shape (track, station, 2)")
            return {flight_id: tracks[index] for index, flight_id in enumerate(ids)}
        if isinstance(payload, dict) and "tracks" in payload:
            payload = payload["tracks"]
        if not isinstance(payload, dict):
            raise ValueError("JSON input must map flight IDs to resampled point arrays")
        return {str(key): np.asarray(value, dtype=np.float64) for key, value in payload.items()}
    raise ValueError("track input must be .npz or .json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hailmary-build-clusters",
        description=(
            "Build a deterministic runway-partitioned cluster artifact from "
            "equally resampled local XY tracks."
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="NPZ/JSON resampled-track input")
    parser.add_argument("--output", type=Path, required=True, help="cluster-library JSON output")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--airport", required=True)
    parser.add_argument("--runway", required=True)
    parser.add_argument("--origin-lat", type=float, required=True, help="runway-threshold latitude")
    parser.add_argument("--origin-lon", type=float, required=True, help="runway-threshold longitude")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tracks = _load_tracks(args.input)
    frame = LocalFrame(args.origin_lat, args.origin_lon)
    artifact = build_cluster_library(
        tracks,
        dataset_id=args.dataset_id,
        airport=args.airport,
        runway=args.runway,
        projection=frame.to_dict(),
    )
    artifact.write(args.output)
    print(
        canonical_json_dumps(
            {
                "artifact_content_hash": artifact.artifact_content_hash,
                "assignment_count": len(artifact.assignments),
                "cluster_count": len(artifact.medoids),
                "output": args.output.resolve().as_posix(),
                "used_fallback": artifact.clustering.used_fallback,
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - console script is the normal boundary
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
