from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vlm_ppe.agents.graph import run_graph
from vlm_ppe.agents.vlm_client import ClusterReviewClient
from vlm_ppe.config import load_config
from vlm_ppe.schemas import ClusterReview, EvidenceImage, KMetric


class FakeReviewClient(ClusterReviewClient):
    def __init__(self, *, retry_once: bool = False) -> None:
        self.retry_once = retry_once
        self.calls = 0

    def review_clusters(
        self,
        *,
        evidence_images: list[EvidenceImage],
        metrics: list[KMetric],
        available_k: list[int],
        attempt: int,
        max_retries: int,
        prompt: str | None = None,
    ) -> ClusterReview:
        self.calls += 1
        assert prompt
        assert evidence_images
        assert metrics
        if self.retry_once and self.calls == 1:
            return ClusterReview(
                chosen_k=available_k[-1],
                confidence=0.45,
                rationale=["Expand K once for clearer separation."],
                retry_requested=True,
                requested_k_max=max(available_k) + 1,
                suggested_action="retry",
            )
        return ClusterReview(
            chosen_k=2 if 2 in available_k else available_k[-1],
            confidence=0.91,
            rationale=["Two stable geometry groups are visible."],
            rejected_alternatives=["K=1 merges separate offsets."],
            clusters_to_recheck=[],
            retry_requested=False,
            suggested_action="accept",
        )


def _write_fixture(tmp_path: Path) -> Path:
    catalog_path = tmp_path / "catalog.csv"
    compressed_path = tmp_path / "compressed.jsonl"
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.yaml"

    catalog = pd.DataFrame(
        {
            "date": ["2026-04-01"] * 4,
            "flight_id": ["T1", "T2", "T3", "T4"],
            "callsign": ["T1", "T2", "T3", "T4"],
            "icao24": ["a", "b", "c", "d"],
            "operation": ["arrival"] * 4,
            "runway": ["18R"] * 4,
            "event_time": [100, 100, 100, 100],
            "event_lat": [32.0, 32.0, 32.0, 32.0],
            "event_lon": [-97.0, -97.0, -97.0, -97.0],
            "threshold_lat": [32.0, 32.0, 32.0, 32.0],
            "threshold_lon": [-97.0, -97.0, -97.0, -97.0],
        }
    )
    catalog.to_csv(catalog_path, index=False)

    tracks = {
        "T1": [(0, 32.00, -97.000, 1000), (60, 32.02, -97.000, 900), (120, 32.04, -97.000, 800)],
        "T2": [(0, 32.00, -97.002, 1000), (60, 32.02, -97.002, 900), (120, 32.04, -97.002, 800)],
        "T3": [(0, 32.00, -97.050, 1000), (60, 32.02, -97.050, 900), (120, 32.04, -97.050, 800)],
        "T4": [(0, 32.00, -97.052, 1000), (60, 32.02, -97.052, 900), (120, 32.04, -97.052, 800)],
    }
    with compressed_path.open("w", encoding="utf-8") as stream:
        for flight_id, points in tracks.items():
            payload = {
                "flight_id": flight_id,
                "callsign": flight_id,
                "icao24": flight_id.lower(),
                "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
                "points": [[time, lat, lon, altitude, 3] for time, lat, lon, altitude in points],
            }
            stream.write(json.dumps(payload) + "\n")

    manifest_path.write_text(
        json.dumps(
            {
                "2026-04-01": {
                    "landings_and_departures": catalog_path.as_posix(),
                    "adsb_compressed_trajectories": compressed_path.as_posix(),
                    "default": True,
                }
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                'dataset_id: "2026-04-01"',
                'operation: "arrival"',
                "runway: null",
                f'manifest_path: "{manifest_path.as_posix()}"',
                f'output_root: "{(tmp_path / "out").as_posix()}"',
                "n_resample: 8",
                "k_min: 1",
                "k_max: 2",
                "kmeans_n_init: 3",
                "kmeans_random_state: 5",
                "max_retries: 1",
                "max_k_expansion: 3",
                'vlm_model: "gemini-3.5-flash"',
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_graph_runs_through_medoid_with_fake_vlm(tmp_path: Path) -> None:
    config = load_config(_write_fixture(tmp_path))
    client = FakeReviewClient()

    result = run_graph(config, run_id="test-run", vlm_client=client)

    assert result["status"] == "complete"
    assert result["chosen_k"] == 2
    assert client.calls == 1
    assert Path(result["state_path"]).exists()
    assert Path(result["medoids_path"]).exists()
    assert Path(result["medoid_report_path"]).exists()
    assert Path(result["run_dir"], "audit.log").exists()
    assert Path(result["run_dir"], "vlm_interactions.jsonl").exists()
    assert Path(result["run_dir"], "vlm_reviews", "attempt_00_request.json").exists()
    assert Path(result["run_dir"], "vlm_reviews", "attempt_00_prompt.txt").exists()
    assert len(result["vlm_reviews"]) == 1


def test_graph_honors_single_retry_requested_by_vlm(tmp_path: Path) -> None:
    config = load_config(_write_fixture(tmp_path))
    client = FakeReviewClient(retry_once=True)

    result = run_graph(config, run_id="retry-run", vlm_client=client)

    assert result["status"] == "complete"
    assert result["retry_count"] == 1
    assert result["k_max_current"] == 3
    assert client.calls == 2
    assert len(result["vlm_reviews"]) == 2


def test_offline_chosen_k_bypasses_missing_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = load_config(_write_fixture(tmp_path))

    result = run_graph(config, chosen_k=2, run_id="offline-run")

    assert result["status"] == "complete"
    assert result["vlm_reviews"][-1]["confidence"] == 1.0


def test_missing_api_key_without_override_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = load_config(_write_fixture(tmp_path))

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        run_graph(config, run_id="missing-key-run")
