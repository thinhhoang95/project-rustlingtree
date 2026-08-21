"""Browser-based verifier for a pre-computed Hailmary route graph."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
import webbrowser

from hailmary.config import M_PER_NM
from hailmary.scenario import TerminalEntryCorpus
from hailmary.topology import ClusterSegmentTraversal, RouteGraphArtifact

from .route_graph_web import ROUTE_GRAPH_HTML


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hailmary-visualize-route-graph",
        description=(
            "Open a local verifier GUI for the pre-computed Hailmary route graph, "
            "highlighting shared traffic segments and their runtime queue gates."
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/artifacts/hailmary/corpus"),
        help="offline corpus directory (default: data/artifacts/hailmary/corpus)",
    )
    parser.add_argument(
        "--route-graph",
        type=Path,
        help="compiled route_graph.json (defaults to CORPUS_DIR/route_graph.json)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help=(
            "traffic_corpus.json used for observed traffic counts (defaults to "
            "CORPUS_DIR/traffic_corpus.json when that file exists)"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="serve the GUI without opening the system browser",
    )
    return parser


@dataclass(frozen=True, slots=True)
class RouteGraphView:
    """Self-contained, browser-ready audit of one immutable route graph."""

    payload: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def build_route_graph_view(
    artifact: RouteGraphArtifact,
    *,
    corpus: TerminalEntryCorpus | None = None,
    route_graph_source: str | None = None,
    corpus_source: str | None = None,
) -> RouteGraphView:
    """Convert canonical graph records into a GUI payload without re-inferring topology."""

    if corpus is not None and corpus.dataset_id != artifact.dataset_id:
        raise ValueError("route graph and traffic corpus must use the same dataset")

    traffic_by_cluster: Counter[str] = Counter()
    if corpus is not None:
        traffic_by_cluster.update(item.key.qualified_id for item in corpus.arrivals)

    traversals_by_cluster: dict[str, list[ClusterSegmentTraversal]] = defaultdict(list)
    traversals_by_segment: dict[str, list[ClusterSegmentTraversal]] = defaultdict(list)
    for traversal in artifact.traversals:
        traversals_by_cluster[traversal.qualified_cluster_id].append(traversal)
        traversals_by_segment[traversal.segment_id].append(traversal)

    graph_clusters = set(traversals_by_cluster)
    corpus_clusters = set(traffic_by_cluster)
    nodes = [asdict(item) for item in artifact.nodes]
    segments: list[dict[str, object]] = []
    for segment in artifact.segments:
        cluster_ids = list(segment.cluster_ids)
        segment_traversals = sorted(
            traversals_by_segment[segment.segment_id],
            key=lambda item: (item.qualified_cluster_id, item.ordinal),
        )
        segments.append(
            {
                "segment_id": segment.segment_id,
                "airport": segment.airport,
                "runway": segment.runway,
                "entry_node_id": segment.entry_node_id,
                "exit_node_id": segment.exit_node_id,
                "entry_resource_id": segment.entry_resource_id,
                "exit_resource_id": segment.exit_resource_id,
                "cluster_ids": cluster_ids,
                "cluster_count": len(cluster_ids),
                "shared": len(cluster_ids) > 1,
                "observed_arrival_count": sum(
                    traffic_by_cluster[item] for item in cluster_ids
                ),
                "lat_deg": list(segment.lat_deg),
                "lon_deg": list(segment.lon_deg),
                "length_m": segment.length_m,
                "length_nm": segment.length_m / M_PER_NM,
                "corridor_width_m": segment.corridor_width_m,
                "traversals": [
                    {
                        "qualified_cluster_id": item.qualified_cluster_id,
                        "ordinal": item.ordinal,
                        "entry_s_m": item.entry_s_m,
                        "exit_s_m": item.exit_s_m,
                    }
                    for item in segment_traversals
                ],
            }
        )

    routes: list[dict[str, object]] = []
    for cluster_id, records in sorted(traversals_by_cluster.items()):
        ordered = sorted(records, key=lambda item: item.ordinal)
        first_segment = artifact.segment(ordered[0].segment_id)
        routes.append(
            {
                "qualified_cluster_id": cluster_id,
                "airport": first_segment.airport,
                "runway": first_segment.runway,
                "observed_arrival_count": traffic_by_cluster[cluster_id],
                "traversals": [
                    {
                        "ordinal": item.ordinal,
                        "segment_id": item.segment_id,
                        "entry_s_m": item.entry_s_m,
                        "exit_s_m": item.exit_s_m,
                        "shared": len(artifact.segment(item.segment_id).cluster_ids)
                        > 1,
                    }
                    for item in ordered
                ],
            }
        )

    partition_keys = sorted(
        {(segment.airport, segment.runway) for segment in artifact.segments}
    )
    partitions = []
    for airport, runway in partition_keys:
        partition_segments = [
            item
            for item in segments
            if item["airport"] == airport and item["runway"] == runway
        ]
        partition_clusters = {
            cluster_id
            for item in partition_segments
            for cluster_id in item["cluster_ids"]
        }
        partitions.append(
            {
                "key": f"{airport}:{runway}",
                "airport": airport,
                "runway": runway,
                "segment_count": len(partition_segments),
                "shared_segment_count": sum(
                    bool(item["shared"]) for item in partition_segments
                ),
                "cluster_count": len(partition_clusters),
                "observed_arrival_count": sum(
                    traffic_by_cluster[item] for item in partition_clusters
                ),
            }
        )

    shared_segments = [item for item in segments if item["shared"]]
    merge_nodes = [item for item in nodes if item["kind"] == "merge"]
    payload: dict[str, object] = {
        "schema_version": "hailmary.route_graph.viewer.v1",
        "dataset_id": artifact.dataset_id,
        "artifact_content_hash": artifact.artifact_content_hash,
        "route_graph_source": route_graph_source,
        "corpus_source": corpus_source,
        "config": asdict(artifact.config),
        "summary": {
            "partition_count": len(partitions),
            "cluster_count": len(graph_clusters),
            "segment_count": len(segments),
            "shared_segment_count": len(shared_segments),
            "merge_node_count": len(merge_nodes),
            "observed_arrival_count": 0 if corpus is None else len(corpus.arrivals),
        },
        "coverage": {
            "corpus_loaded": corpus is not None,
            "corpus_clusters_missing_from_graph": sorted(
                corpus_clusters - graph_clusters
            ),
            "graph_clusters_without_observed_traffic": (
                [] if corpus is None else sorted(graph_clusters - corpus_clusters)
            ),
        },
        "fidelity": {
            "topology_source": (
                "Canonical RouteGraphArtifact records; the viewer does not infer "
                "sharing from rendered geometry."
            ),
            "shared_segment_rule": "A segment is shared exactly when cluster_count > 1.",
            "geometry_direction": (
                "Every segment polyline runs from upstream entry gate to downstream "
                "exit gate, matching traversal order toward the runway."
            ),
            "runtime_queue_order": (
                "Physical occupants first by progress; committed future entrants next "
                "by entry ETA with flight-ID tie-break. Exit ETA evaluates spacing and "
                "never reorders the established queue."
            ),
            "queue_resource": (
                "Each static segment becomes :entry and :exit resources; live flow "
                "anchors are bound to the segment exit resource."
            ),
        },
        "partitions": partitions,
        "nodes": nodes,
        "segments": segments,
        "routes": routes,
    }
    return RouteGraphView(payload)


def _load_view(args: argparse.Namespace) -> RouteGraphView:
    graph_path = args.route_graph or args.corpus_dir / "route_graph.json"
    if not graph_path.is_file():
        raise ValueError(
            f"compiled route graph not found: {graph_path}; run "
            "hailmary-build-route-graph --input "
            f"{args.corpus_dir / 'route_graph_input.json'} --output {graph_path}"
        )
    corpus_path = args.corpus or args.corpus_dir / "traffic_corpus.json"
    if args.corpus is not None and not corpus_path.is_file():
        raise ValueError(f"traffic corpus not found: {corpus_path}")
    corpus = TerminalEntryCorpus.read(corpus_path) if corpus_path.is_file() else None
    return build_route_graph_view(
        RouteGraphArtifact.read(graph_path),
        corpus=corpus,
        route_graph_source=graph_path.resolve().as_posix(),
        corpus_source=None if corpus is None else corpus_path.resolve().as_posix(),
    )


def create_app(view: RouteGraphView):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(
        title="Hailmary route-graph verifier",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return ROUTE_GRAPH_HTML

    @app.get("/api/route-graph", response_class=JSONResponse)
    def route_graph_payload() -> dict[str, object]:
        return view.to_dict()

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def serve(
    view: RouteGraphView,
    *,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    import uvicorn

    if not 1 <= int(port) <= 65_535:
        raise ValueError("port must lie in [1, 65535]")
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in url_host and not url_host.startswith("["):
        url_host = f"[{url_host}]"
    url = f"http://{url_host}:{port}"
    summary = view.payload["summary"]
    assert isinstance(summary, Mapping)
    print(
        f"Hailmary route-graph verifier: {url} "
        f"({summary['segment_count']} segments, "
        f"{summary['shared_segment_count']} shared)"
    )
    if open_browser:
        timer = threading.Timer(0.7, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    uvicorn.run(create_app(view), host=host, port=int(port), log_level="warning")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        view = _load_view(args)
        serve(
            view,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RouteGraphView",
    "build_parser",
    "build_route_graph_view",
    "create_app",
    "main",
    "serve",
]
