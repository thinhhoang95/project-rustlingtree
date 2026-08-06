"""Verify the three-command Phase 0/1 pipeline on the repository ADS-B day."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hailmary.actions import ActionCatalog, ActionLever
from hailmary.features import build_current_segment_anchors
from hailmary.scenario import (
    DemandWindow,
    TerminalEntryCorpus,
    TrafficScaleConfig,
    TrafficScenarioBuilder,
)
from hailmary.simulator import Simulator
from hailmary.templates import TemplateStore
from hailmary.topology import RouteGraphArtifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    return parser


def main() -> None:
    root = build_parser().parse_args().artifact_dir
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    batch_audit = json.loads(
        (root / "traffic_batch_scale1.json").read_text(encoding="utf-8")
    )
    corpus = TerminalEntryCorpus.read(root / "traffic_corpus.json")
    store = TemplateStore.read(root / "hailmary_templates.json")
    graph = RouteGraphArtifact.read(root / "route_graph.json")
    templates = {
        f"{item.airport_id}:{item.runway_id}:{item.cluster_id}": item
        for item in store.templates
    }
    builder = TrafficScenarioBuilder(
        corpus.arrivals,
        templates_by_cluster=templates,
        route_graph=graph,
        rejection_counts=dict(corpus.rejection_counts),
    )
    scenario = builder.build_scenario(
        DemandWindow(1_775_038_800.0, 1_775_042_400.0),
        scale_config=TrafficScaleConfig(global_scale=1.0),
    )
    simulator = Simulator(scenario.definition)
    first_trainable: dict[str, object] | None = None
    while first_trainable is None:
        event_batch = simulator.advance_next()
        if event_batch is None:
            raise RuntimeError("real 10:20 window has no trainable segment epoch")
        if event_batch.decision_epoch is None:
            continue
        for anchor in build_current_segment_anchors(simulator).leader_follower:
            candidates = ActionCatalog().enumerate_for_batch(
                simulator,
                event_batch,
                anchor_id=anchor.anchor_id,
                bound_flight_id=anchor.follower_id,
                resource_id=anchor.resource_id,
                segment_id=anchor.segment_id,
            )
            if any(item.lever is not ActionLever.NO_OP for item in candidates):
                first_trainable = {
                    "leader": anchor.leader_id,
                    "follower": anchor.follower_id,
                    "candidate_actions": [
                        f"{item.lever.value}:{item.band}" for item in candidates
                    ],
                }
                break

    exact_scale_one = all(
        item["observed_cluster_counts"] == item["target_cluster_counts"]
        and item["observed_runway_counts"] == item["target_runway_counts"]
        for item in batch_audit["scenarios"]
    )
    represented_runways = sorted(
        {
            key.split(":", 1)[1]
            for item in batch_audit["scenarios"]
            for key, count in item["target_runway_counts"].items()
            if count
        }
    )
    result = {
        "catalog_arrivals": audit["catalog_arrival_count"],
        "reconstructable_arrivals": len(corpus.arrivals),
        "runways": represented_runways,
        "template_count": len(store.templates),
        "route_segment_count": len(graph.segments),
        "route_graph_hash": graph.artifact_content_hash,
        "traffic_corpus_hash": corpus.corpus_content_hash,
        "traffic_batch_hash": batch_audit["batch_content_hash"],
        "scale_1_window_count": len(batch_audit["scenarios"]),
        "scale_1_counts_exact": exact_scale_one,
        "real_1020_window_flight_count": len(scenario.definition.flights),
        "real_1020_window_runways": sorted(
            {item.runway for item in scenario.definition.flights}
        ),
        "first_trainable_pair": first_trainable,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
