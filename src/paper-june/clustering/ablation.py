"""CLI entrypoint for the paper-June clustering ablation."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CLUSTERING_ROOT = Path(__file__).resolve().parent
if str(CLUSTERING_ROOT) not in sys.path:
    sys.path.insert(0, str(CLUSTERING_ROOT))

from clustering_ablation_density import run_density_ablation
from clustering_ablation_kmeans import *  # noqa: F403 - preserve existing test/import surface.
from clustering_ablation_kmeans import build_parser as _build_kmeans_parser
from clustering_ablation_kmeans import run_ablation as _run_kmeans_ablation


def run_ablation(args) -> dict[str, Any]:
    summary = _run_kmeans_ablation(args)
    density_summary = run_density_ablation(args, summary)
    summary["density_and_community"] = density_summary
    output_dir = Path(summary["config"]["output_dir"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def build_parser():
    return _build_kmeans_parser()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_ablation(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
