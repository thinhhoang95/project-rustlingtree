from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hailmary.topology import MedoidRoute, build_route_graph


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPOSITORY_ROOT / "tests/fixtures/hailmary/phase01_rw18r_20260401_1020.json"
OVERLAY = REPOSITORY_ROOT / "tests/fixtures/hailmary/phase01_rw18r_20260401_1020.svg"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_pairing_fixture_is_versioned_and_source_locked() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["fixture_version"] == "hailmary.phase01.real_pairing_fixture.v1"
    assert payload["geometry_thresholds"] == {
        "lateral_floor_nm": 0.5,
        "minimum_common_length_nm": 5.0,
        "tangent_tolerance_deg": 15.0,
    }
    for prefix in ("clusters", "medoids", "catalog"):
        source_path = REPOSITORY_ROOT / payload["source"][f"{prefix}_path"]
        assert _sha256(source_path) == payload["source"][f"{prefix}_sha256"]
    assert OVERLAY.read_text(encoding="utf-8").startswith("<svg")


def test_real_fixture_retains_exact_window_arrivals_and_medoid_coordinates() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    arrivals = payload["arrivals"]
    medoids = payload["medoids"]
    start = 1775038800.0
    end = 1775042400.0

    assert [item["flight_id"] for item in arrivals] == payload["extraction"]["flight_ids"]
    assert len(arrivals) == 8
    assert all(start <= item["terminal_entry_time_s"] < end for item in arrivals)
    assert {item["cluster_id"] for item in medoids} == {"0", "1", "2", "3", "5"}
    assert all(len(item["lat_deg"]) == len(item["lon_deg"]) >= 2 for item in medoids)
    assert {item["medoid_flight_id"] for item in medoids} == {
        "AAL1845M1a353c8",
        "AAL3125M2ac7a5e",
        "ENY3828M2a2e1c7",
        "ENY4144M2a23c5d",
        "JIA5597M1a657ad",
    }


def test_reviewed_segment_pairs_exclude_threshold_only_adjacency() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    expected = payload["expected"]
    segments = {
        item["segment_id"]: tuple(item["cluster_ids"])
        for item in expected["segments"]
    }
    accepted = {
        (item["leader_id"], item["follower_id"], item["segment_id"])
        for item in expected["accepted_pairs"]
    }
    rejected = {
        (item["leader_id"], item["follower_id"])
        for item in expected["rejected_threshold_only_pairs"]
    }
    runway_order = [
        item["flight_id"]
        for item in sorted(payload["arrivals"], key=lambda item: item["runway_event_time_s"])
    ]
    threshold_adjacency = set(zip(runway_order, runway_order[1:], strict=False))

    assert segments["reviewed_rw18r_common_123"] == ("1", "2", "3")
    assert segments["reviewed_rw18r_parallel_5"] == ("5",)
    assert segments["reviewed_rw18r_west_0"] == ("0",)
    assert accepted == {
        ("AAL2149M1ad385f", "AAL2563M1aa1be0", "reviewed_rw18r_common_123"),
        ("AAL2563M1aa1be0", "NKS1673M1a9cc41", "reviewed_rw18r_common_123"),
        ("NKS1673M1a9cc41", "ENY4020M1a34c68", "reviewed_rw18r_common_123"),
        ("JIA5113M1a67b75", "JIA5463M1a71aa5", "reviewed_rw18r_parallel_5"),
        ("JIA5066M1a7cfe6", "AAL1286M1ac0dd9", "reviewed_rw18r_west_0"),
    }
    assert rejected == {
        ("ENY4020M1a34c68", "JIA5113M1a67b75"),
        ("JIA5463M1a71aa5", "JIA5066M1a7cfe6"),
    }
    assert rejected.issubset(threshold_adjacency)
    assert all((leader, follower) not in rejected for leader, follower, _ in accepted)


def test_production_topology_recovers_reviewed_123_corridor_not_cluster_5() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    routes = tuple(
        MedoidRoute(
            dataset_id="adsb_2026-04-01",
            airport="KDFW",
            runway="RW18R",
            cluster_id=item["cluster_id"],
            lat_deg=tuple(item["lat_deg"]),
            lon_deg=tuple(item["lon_deg"]),
            medoid_flight_id=item["medoid_flight_id"],
        )
        for item in payload["medoids"]
    )

    graph = build_route_graph(routes)
    expected_clusters = (
        "KDFW:RW18R:1",
        "KDFW:RW18R:2",
        "KDFW:RW18R:3",
    )
    shared = [
        segment for segment in graph.segments
        if segment.cluster_ids == expected_clusters
    ]

    assert len(shared) == 1
    assert 13.0 <= shared[0].length_m / 1_852.0 <= 15.0
    assert "KDFW:RW18R:5" not in shared[0].cluster_ids
