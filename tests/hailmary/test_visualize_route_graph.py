from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np
from fastapi.testclient import TestClient

from hailmary.cli.visualize_route_graph import (
    build_parser,
    build_route_graph_view,
    create_app,
)
from hailmary.geometry import LocalFrame
from hailmary.scenario import ArrivalClusterKey, TerminalEntryCorpus
from hailmary.topology import MedoidRoute, build_route_graph

from .test_phase01_traffic import _arrival


def _route(cluster: str, points: tuple[tuple[float, float], ...]) -> MedoidRoute:
    frame = LocalFrame(0.0, 0.0)
    lat, lon = frame.unproject(
        np.asarray([item[0] for item in points]),
        np.asarray([item[1] for item in points]),
    )
    return MedoidRoute(
        dataset_id="phase01-test",
        airport="KATL",
        runway="RW18R",
        cluster_id=cluster,
        lat_deg=tuple(lat),
        lon_deg=tuple(lon),
    )


def _view():
    graph = build_route_graph(
        (
            _route("C1", ((-30_000.0, 10_000.0), (-16_000.0, 0.0), (0.0, 0.0))),
            _route("C2", ((30_000.0, 10_000.0), (-16_000.0, 0.0), (0.0, 0.0))),
        )
    )
    c1 = ArrivalClusterKey("KATL", "RW18R", "C1")
    c2 = ArrivalClusterKey("KATL", "RW18R", "C2")
    corpus = TerminalEntryCorpus(
        dataset_id="phase01-test",
        arrivals=(
            _arrival(1, c1, 100.0),
            _arrival(2, c1, 200.0),
            _arrival(3, c2, 300.0),
        ),
        rejection_counts=(),
        airport="KATL",
    )
    return build_route_graph_view(
        graph,
        corpus=corpus,
        route_graph_source="/corpus/route_graph.json",
        corpus_source="/corpus/traffic_corpus.json",
    )


def test_cli_help_and_console_script_registration() -> None:
    help_text = build_parser().format_help()
    scripts = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["scripts"]

    assert "hailmary-visualize-route-graph" in help_text
    assert "--corpus-dir" in help_text
    assert (
        scripts["hailmary-visualize-route-graph"]
        == "hailmary.cli.visualize_route_graph:main"
    )


def test_view_uses_canonical_shared_membership_and_runtime_resource_ids() -> None:
    payload = _view().to_dict()
    shared = [item for item in payload["segments"] if item["shared"]]

    assert shared
    assert shared[0]["cluster_count"] == 2
    assert shared[0]["runway_ids"] == ["RW18R"]
    assert shared[0]["observed_arrival_count"] == 3
    assert shared[0]["entry_resource_id"].endswith(":entry")
    assert shared[0]["exit_resource_id"].endswith(":exit")
    assert payload["summary"]["shared_segment_count"] == len(shared)
    assert payload["coverage"]["corpus_clusters_missing_from_graph"] == []
    assert "Physical occupants first" in payload["fidelity"]["runtime_queue_order"]

    for route in payload["routes"]:
        assert [item["ordinal"] for item in route["traversals"]] == list(
            range(len(route["traversals"]))
        )


def test_local_web_app_serves_map_and_canonical_payload() -> None:
    client = TestClient(create_app(_view()))

    page = client.get("/")
    response = client.get("/api/route-graph")

    assert page.status_code == 200
    assert "Common traffic segment" in page.text
    assert "Runtime interpretation" in page.text
    assert response.status_code == 200
    assert response.json()["schema_version"] == "hailmary.route_graph.viewer.v3"
