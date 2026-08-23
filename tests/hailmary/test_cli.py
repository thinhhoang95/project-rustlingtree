from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from hailmary.cli import (
    build_clusters,
    build_offline_corpus,
    build_route_graph,
    build_templates,
    build_traffic_batch,
    simulate,
    visualize_event_queue,
    visualize_offline_corpus,
    visualize_route_graph,
)
from hailmary.clustering import ClusterLibrary
from hailmary.topology import RouteGraphArtifact

from .test_adapters import _variant


@pytest.mark.parametrize(
    "entrypoint",
    [
        build_clusters.main,
        build_templates.main,
        build_offline_corpus.main,
        build_route_graph.main,
        build_traffic_batch.main,
        simulate.main,
        visualize_event_queue.main,
        visualize_offline_corpus.main,
        visualize_route_graph.main,
    ],
)
def test_cli_help_is_headless_and_successful(entrypoint, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        entrypoint(["--help"])

    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_pyproject_registers_hailmary_console_scripts() -> None:
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]

    assert scripts["hailmary-build-clusters"] == "hailmary.cli.build_clusters:main"
    assert scripts["hailmary-build-templates"] == "hailmary.cli.build_templates:main"
    assert scripts["hailmary-build-offline-corpus"] == "hailmary.cli.build_offline_corpus:main"
    assert (
        scripts["hailmary-visualize-clusters"]
        == "hailmary.clustering.visualization:main"
    )
    assert scripts["hailmary-build-route-graph"] == "hailmary.cli.build_route_graph:main"
    assert scripts["hailmary-build-traffic-batch"] == "hailmary.cli.build_traffic_batch:main"
    assert scripts["hailmary-simulate"] == "hailmary.cli.simulate:main"
    assert (
        scripts["hailmary-visualize-event-queue"]
        == "hailmary.cli.visualize_event_queue:main"
    )
    assert (
        scripts["hailmary-visualize-route-graph"]
        == "hailmary.cli.visualize_route_graph:main"
    )


def test_cluster_cli_builds_canonical_artifact_from_npz(tmp_path: Path, capsys) -> None:
    station = np.linspace(0.0, 10_000.0, 8)
    tracks = []
    ids = []
    for prefix, base in (("A", 0.0), ("B", 5_000.0)):
        for index, offset in enumerate((-50.0, 0.0, 50.0)):
            ids.append(f"{prefix}{index}")
            tracks.append(np.column_stack((np.full_like(station, base + offset), station)))
    source = tmp_path / "tracks.npz"
    np.savez(source, track_ids=np.asarray(ids), tracks_m=np.asarray(tracks))
    output = tmp_path / "clusters.json"

    result = build_clusters.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--dataset-id",
            "synthetic",
            "--airport",
            "KDFW",
            "--runway",
            "35C",
            "--origin-lat",
            "32.9",
            "--origin-lon",
            "-97.0",
        ]
    )

    artifact = ClusterLibrary.read(output)
    summary = json.loads(capsys.readouterr().out)
    assert result == 0
    assert len(artifact.assignments) == 6
    assert summary["artifact_content_hash"] == artifact.artifact_content_hash


def test_route_graph_cli_builds_hashed_artifact(tmp_path: Path, capsys) -> None:
    help_text = build_route_graph.build_parser().format_help()
    assert "--pair-match-tolerance-nm" in help_text
    assert "--component-diameter-limit-nm" in help_text
    assert "--gate-alignment-tolerance-nm" in help_text
    assert "--lateral-floor-nm" not in help_text

    source = tmp_path / "routes.json"
    source.write_text(
        json.dumps(
            {
                "dataset_id": "cli-route-test",
                "routes": [
                    {
                        "airport": "KATL",
                        "runway": "18R",
                        "cluster_id": cluster,
                        "lat_deg": [0.0, 0.0, 0.0],
                        "lon_deg": [0.0, 0.1, 0.2],
                        "dispersion_m": 0.0,
                    }
                    for cluster in ("C1", "C2")
                ]
                + [
                    {
                        "airport": "KATL",
                        "runway": "18R",
                        "cluster_id": "UNCERTAIN",
                        "lat_deg": [0.0, 0.0, 0.0],
                        "lon_deg": [0.0, 0.1, 0.2],
                        "dispersion_m": 6.0 * 1_852.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "route_graph.json"

    result = build_route_graph.main(["--input", str(source), "--output", str(output)])

    artifact = RouteGraphArtifact.read(output)
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert result == 0
    assert summary["artifact_content_hash"] == artifact.artifact_content_hash
    assert summary["cluster_count"] == 2
    assert artifact.traversals_for("KATL:RW18R:C1")
    assert "KATL:RW18R:UNCERTAIN" in captured.err
    assert not artifact.traversals_for("KATL:RW18R:UNCERTAIN")


def test_variant_npz_round_trip_is_content_exact_and_byte_deterministic(tmp_path: Path) -> None:
    variant = _variant()
    first = build_templates.write_variant_npz(variant, tmp_path / "a.npz")
    second = build_templates.write_variant_npz(variant, tmp_path / "b.npz")

    restored = build_templates.read_variant_npz(first)

    assert restored.variant_id == variant.variant_id
    np.testing.assert_array_equal(restored.s_m, variant.s_m)
    assert not restored.s_m.flags.writeable
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()


def test_simulate_cli_runs_json_npz_scenario_and_writes_trace(tmp_path: Path, capsys) -> None:
    variant = _variant()
    build_templates.write_variant_npz(variant, tmp_path / "variant.npz")
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "scenario_id": "CLI",
                "seed": 11,
                "variants": [{"npz": "variant.npz"}],
                "resources": [{"resource_id": "RWY"}],
                "flights": [
                    {
                        "flight_id": "F1",
                        "release_time_s": 100.0,
                        "baseline_variant_id": variant.variant_id,
                        "callsign": "CALL1",
                        "icao24": "abc",
                        "runway": "RW35C",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "trace.json"

    result = simulate.main(["--scenario", str(scenario_path), "--output", str(output)])

    trace = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)
    assert result == 0
    assert trace["scenario_id"] == "CLI"
    assert trace["final_state"]["flights"][0]["lifecycle"] == "completed"
    assert trace["arrival_schedule"][0]["flight_id"] == "F1"
    assert summary["batch_count"] == len(trace["batches"])


def test_requested_notebooks_are_valid() -> None:
    for name in ("01_clusters.ipynb", "02_templates.ipynb"):
        payload = json.loads((Path("notebooks/hailmary") / name).read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert any("hailmary" in "".join(cell["source"]).lower() for cell in payload["cells"])
