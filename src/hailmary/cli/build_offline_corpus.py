"""Build complete runway-scoped clusters, templates, and route input."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import sys
from typing import Sequence

import numpy as np

from hailmary.clustering import build_all_runway_cluster_libraries_from_adsb
from hailmary.config import M_PER_NM
from hailmary.data import load_arrival_catalog, load_catalog_raw_adsb_tracks, load_manifest
from hailmary.scenario import ArrivalClusterKey, ObservedArrival, TerminalEntryCorpus
from hailmary.templates import ClusterTemplate, TemplateCompiler, TemplateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hailmary-build-offline-corpus")
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest.json"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--airport", default="KDFW")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-medoid-dispersion-nm", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not np.isfinite(args.maximum_medoid_dispersion_nm)
        or args.maximum_medoid_dispersion_nm <= 0.0
    ):
        raise ValueError("maximum medoid dispersion must be finite and positive")
    dataset = load_manifest(args.manifest).select(args.dataset_id)
    catalog = load_arrival_catalog(
        dataset.require("landings_and_departures"), airport=args.airport
    )
    tracks = load_catalog_raw_adsb_tracks(dataset.require("raw_adsb"))
    result = build_all_runway_cluster_libraries_from_adsb(
        catalog,
        tracks,
        dataset_id=dataset.dataset_id,
        airport=args.airport,
        metadata={"data_manifest": dataset.manifest_path.as_posix()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compiler = TemplateCompiler(require_observed_profile_fit=False)
    store = TemplateStore()
    route_records: list[dict[str, object]] = []
    template_rejections: dict[str, str] = {}
    template_by_cluster: dict[str, ClusterTemplate] = {}
    traffic_arrivals: list[ObservedArrival] = []
    rejection_counts = Counter(dict(result.rejection_counts))
    uncertain_clusters: set[str] = set()
    hdbscan_outlier_rejections_by_runway: dict[str, int] = {}
    cluster_diagnostics_by_runway: dict[str, list[dict[str, object]]] = {}
    selected_clustering_by_runway: dict[str, dict[str, object]] = {}
    for runway_result in result.runway_results:
        library = runway_result.library
        cluster_diagnostics_by_runway[library.runway] = [
            item.to_dict() for item in library.clustering.cluster_diagnostics
        ]
        selected_clustering_by_runway[library.runway] = {
            "algorithm": library.clustering.algorithm,
            "parameters": library.clustering.parameter_dict,
            "cluster_count": len(library.medoids),
        }
        library.write(args.output_dir / f"clusters_{library.runway}.json")
        medoid_tracks = runway_result.template_medoid_tracks
        for medoid in library.medoids:
            qualified_cluster = (
                f"{library.airport}:{library.runway}:{medoid.cluster_id}"
            )
            dispersion_nm = medoid.mean_distance_m / M_PER_NM
            if dispersion_nm > args.maximum_medoid_dispersion_nm:
                uncertain_clusters.add(qualified_cluster)
                print(
                    "uncertain medoid excluded: "
                    f"{qualified_cluster} dispersion_nm={dispersion_nm:.3f} "
                    f"limit_nm={args.maximum_medoid_dispersion_nm:.3f}",
                    file=sys.stderr,
                )
                continue
            source = medoid_tracks[medoid.medoid_flight_id]
            try:
                template = compiler.compile(
                    source,
                    cluster_id=str(medoid.cluster_id),
                    member_count=medoid.member_count,
                    dataset_id=library.dataset_id,
                    airport_id=library.airport,
                    runway_id=library.runway,
                    dispersion_m=medoid.mean_distance_m,
                    threshold_resource_id=f"{library.airport}:{library.runway}:threshold",
                )
            except ValueError as exc:
                template_rejections[qualified_cluster] = str(exc)
                continue
            store.add_template(template)
            template_by_cluster[qualified_cluster] = template
            variant = template.baseline_variant
            route_records.append(
                {
                    "airport": library.airport,
                    "runway": library.runway,
                    "cluster_id": qualified_cluster.split(":", 2)[2],
                    "lat_deg": variant.lat_deg.tolist()[::-1],
                    "lon_deg": variant.lon_deg.tolist()[::-1],
                    "dispersion_m": template.dispersion_m,
                    "medoid_flight_id": template.medoid_flight_id,
                    "source_hash": library.artifact_content_hash,
                }
            )
        assignment_by_flight = {
            item.flight_id: item for item in library.corpus_assignments
        }
        for prepared in runway_result.preparation.prepared_tracks:
            assignment = assignment_by_flight.get(prepared.flight_id)
            if assignment is None:
                # Nearest-medoid assignment remains in the portable cluster
                # artifact as explicit provenance, but an HDBSCAN -1 label is
                # a rejection for the observed traffic corpus.  Keeping the
                # fallback identity out of the corpus also keeps it out of all
                # corpus-based visualizations and scenario generation.
                hdbscan_outlier_rejections_by_runway[library.runway] = (
                    hdbscan_outlier_rejections_by_runway.get(library.runway, 0) + 1
                )
                rejection_counts[f"{library.runway}:hdbscan_outlier"] += 1
                continue
            qualified_cluster = (
                f"{library.airport}:{library.runway}:{assignment.cluster_id}"
            )
            template = template_by_cluster.get(qualified_cluster)
            if template is None:
                if qualified_cluster not in uncertain_clusters:
                    rejection_counts[
                        f"{library.runway}:template_compilation_failed"
                    ] += 1
                continue
            raw = prepared.raw_track
            index = prepared.terminal_entry.segment_index
            dt = float(raw.time_s[index + 1] - raw.time_s[index])
            segment = runway_result.preparation.frame.project_points(
                raw.lat_deg[index : index + 2],
                raw.lon_deg[index : index + 2],
            )
            ground_speed_mps = float(np.linalg.norm(segment[1] - segment[0]) / dt)
            if not np.isfinite(ground_speed_mps) or ground_speed_mps <= 0.0:
                rejection_counts[f"{library.runway}:invalid_terminal_entry_speed"] += 1
                continue
            traffic_arrivals.append(
                ObservedArrival(
                    dataset_id=dataset.dataset_id,
                    key=ArrivalClusterKey(
                        library.airport,
                        library.runway,
                        str(assignment.cluster_id),
                    ),
                    flight_id=prepared.flight_id,
                    callsign=prepared.catalog_arrival.callsign,
                    icao24=prepared.catalog_arrival.icao24,
                    terminal_entry_time_s=prepared.terminal_entry.time_s,
                    terminal_entry_ground_speed_mps=ground_speed_mps,
                    terminal_entry_altitude_m=prepared.terminal_entry.geoaltitude_m,
                    baseline_variant_id=template.baseline_variant.variant_id,
                    source_day=datetime.fromtimestamp(
                        prepared.terminal_entry.time_s, tz=UTC
                    ).date().isoformat(),
                )
            )
        hdbscan_outlier_rejections_by_runway.setdefault(library.runway, 0)
    for runway, diagnostics in cluster_diagnostics_by_runway.items():
        accepted_diagnostics = [
            item
            for item in diagnostics
            if f"{str(args.airport).strip().upper()}:{runway}:{item['cluster_id']}"
            not in uncertain_clusters
        ]
        cluster_diagnostics_by_runway[runway] = accepted_diagnostics
        selected_clustering_by_runway[runway]["cluster_count"] = len(
            accepted_diagnostics
        )
    store.write(args.output_dir / "hailmary_templates.json")
    route_input = {
        "dataset_id": dataset.dataset_id,
        "routes": route_records,
    }
    (args.output_dir / "route_graph_input.json").write_text(
        json.dumps(route_input, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    corpus = TerminalEntryCorpus(
        dataset_id=dataset.dataset_id,
        arrivals=tuple(traffic_arrivals),
        rejection_counts=tuple(sorted(rejection_counts.items())),
        airport=str(args.airport).strip().upper(),
    )
    corpus.write(args.output_dir / "traffic_corpus.json")
    audit = {
        "dataset_id": dataset.dataset_id,
        "airport": args.airport,
        "catalog_arrival_count": len(catalog),
        "accepted_terminal_entry_count": result.accepted_track_count,
        "runway_count": len(result.runway_results),
        "runways": [item.library.runway for item in result.runway_results],
        "cluster_count": len(route_records),
        "template_count": len(store.templates),
        "traffic_arrival_count": len(traffic_arrivals),
        "traffic_corpus_hash": corpus.corpus_content_hash,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "template_rejections": template_rejections,
        "hdbscan_outlier_rejections_by_runway": dict(
            sorted(hdbscan_outlier_rejections_by_runway.items())
        ),
        "selected_clustering_by_runway": dict(
            sorted(selected_clustering_by_runway.items())
        ),
        "cluster_diagnostics_by_runway": dict(
            sorted(cluster_diagnostics_by_runway.items())
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True, separators=(",", ":")))
    for runway in sorted(selected_clustering_by_runway):
        selection = selected_clustering_by_runway[runway]
        print(
            f"{runway}: HDBSCAN outliers rejected from corpus="
            f"{hdbscan_outlier_rejections_by_runway[runway]}; "
            f"selected={selection['algorithm']} {selection['parameters']}; "
            f"clusters={selection['cluster_count']}",
            file=sys.stderr,
        )
        for diagnostic in cluster_diagnostics_by_runway[runway]:
            probability = diagnostic["membership_probability"]
            distance = diagnostic["centroid_distance_m"]
            bearing = diagnostic["entry_bearing_deg"]
            assert isinstance(probability, dict)
            assert isinstance(distance, dict)
            assert isinstance(bearing, dict)
            print(
                f"  cluster={diagnostic['cluster_id']} "
                f"members={diagnostic['member_count']} "
                f"membership_median={float(probability['median']):.3f} "
                f"centroid_p90_km={float(distance['p90']) / 1000.0:.2f} "
                f"entry_bearing_mean_deg={float(bearing['circular_mean']):.1f} "
                f"entry_bearing_p90_span_deg={float(bearing['p90_span']):.1f}",
                file=sys.stderr,
            )
    visualize_command = shlex.join(
        [
            "./.venv/bin/python",
            "src/hailmary/cli/visualize_offline_corpus.py",
            "--corpus-dir",
            str(args.output_dir),
            "--manifest",
            str(args.manifest),
            "--dataset-id",
            dataset.dataset_id,
        ]
    )
    print(
        f"Visualize the built corpus with:\n  {visualize_command}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
