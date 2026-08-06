"""Build a deterministic traffic-batch audit from a prepared ADS-B corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hailmary.scenario import (
    DemandWindowConfig,
    TerminalEntryCorpus,
    TrafficScaleConfig,
    TrafficScenarioBuilder,
    iter_demand_windows,
)
from hailmary.templates import TemplateStore
from hailmary.topology import RouteGraphArtifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hailmary-build-traffic-batch")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--route-graph", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--stop", type=float, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-partition", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = TerminalEntryCorpus.read(args.corpus)
    dataset_id = corpus.dataset_id
    arrivals = corpus.arrivals
    rejections = dict(corpus.rejection_counts)
    store = TemplateStore.read(args.templates)
    templates = {
        f"{template.airport_id}:{template.runway_id}:{template.cluster_id}": template
        for template in store.templates
    }
    route_graph = None if args.route_graph is None else RouteGraphArtifact.read(args.route_graph)
    builder = TrafficScenarioBuilder(
        arrivals,
        templates_by_cluster=templates,
        route_graph=route_graph,
        rejection_counts=rejections,
    )
    windows = iter_demand_windows(
        args.start,
        args.stop,
        config=DemandWindowConfig(),
    )
    batch = builder.build_batch(
        windows,
        scale_config=TrafficScaleConfig(
            global_scale=args.scale,
            replicate=args.replicate,
            master_seed=args.seed,
        ),
        source_partition=args.source_partition,
    )
    output = {
        "schema_version": batch.schema_version,
        "dataset_id": dataset_id,
        "batch_content_hash": batch.batch_content_hash,
        "global_scale": batch.scale_config.global_scale,
        "audit": dict(batch.audit),
        "scenarios": [
            {
                "scenario_id": item.definition.scenario_id,
                "definition_hash": item.definition.definition_hash,
                "window": {"start_s": item.window.start_s, "end_s": item.window.end_s},
                "observed_cluster_counts": {
                    count.key.qualified_id: count.count for count in item.observed_cluster_counts
                },
                "target_cluster_counts": {
                    count.key.qualified_id: count.count for count in item.target_cluster_counts
                },
                "observed_runway_counts": {
                    f"{key[0]}:{key[1]}": value
                    for key, value in item.observed_runway_counts.items()
                },
                "target_runway_counts": {
                    f"{key[0]}:{key[1]}": value
                    for key, value in item.target_runway_counts.items()
                },
            }
            for item in batch.scenarios
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "batch_content_hash": batch.batch_content_hash,
                "scenario_count": len(batch.scenarios),
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
