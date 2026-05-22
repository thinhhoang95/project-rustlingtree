from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from mcp_tools.advisors import FeasibilityAdvisory, SpeedControlAdvisory, VectoringAdvisory
from mcp_tools.scenario_manager.api import create_app
from mcp_tools.scenario_manager.manager import ScenarioManager
from mcp_tools.scenario_manager.models import ScenarioResourceConfig


def write_fixture_resources(tmp_path: Path) -> ScenarioResourceConfig:
    events_path = tmp_path / "landings_and_departures.csv"
    events_path.write_text(
        "date,flight_id,callsign,icao24,operation,runway,event_time,event_time_utc\n"
        "2025-04-01,DEP2,CALLDEP2,dep002,departure,17R,200,1970-01-01T00:03:20Z\n"
        "2025-04-01,ARR1,CALLARR1,arr001,arrival,35C,500,1970-01-01T00:08:20Z\n"
        "2025-04-01,DEP1,CALLDEP1,dep001,departure,18L,100,1970-01-01T00:01:40Z\n"
        "2025-04-01,ARR2,CALLARR2,arr002,arrival,36L,600,1970-01-01T00:10:00Z\n",
        encoding="utf-8",
    )

    fix_sequences_path = tmp_path / "fix_sequences.csv"
    fix_sequences_path.write_text(
        "date,flight_id,callsign,icao24,first_time,last_time,fix_sequence,fix_count\n"
        "2025-04-01,ARR1,CALLARR1,arr001,300,500,FIXA>FIXB,2\n"
        "2025-04-01,ARR2,CALLARR2,arr002,250,600,FIXC,1\n",
        encoding="utf-8",
    )

    simap_arrival_trajectories_path = tmp_path / "simap_arrival_flights.jsonl"
    simap_arrival_payload = {
        "flight_id": "ARR1",
        "callsign": "CALLARR1",
        "icao24": "arr001",
        "runway": "RW35C",
        "route_type": "base-route",
        "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        "breakpoint_mask_bits": {"lateral": 1, "altitude": 2},
        "points": [[310, 32.0, -97.0, 1000.0, 3], [500, 32.1, -97.1, 200.0, 3]],
        "cas_profile": {
            "columns": ["time", "cas_kts"],
            "units": {"cas_kts": "kt"},
            "source": "simap_fms_bichannel",
            "points": [[310, 190.0], [312, 188.5], [314, 187.0]],
        },
        "lateral_breakpoint_times": [310, 500],
        "altitude_breakpoint_times": [310, 500],
        "first_time": 310,
        "last_time": 500,
        "raw_point_count": 2,
        "compressed_point_count": 2,
        "lateral_tolerance_m": 100.0,
        "altitude_tolerance_m": 50.0,
        "wait_atc_point": {
            "source": "fix",
            "identifier": "FIXB",
            "lat": 32.1,
            "lon": -97.1,
            "lateral_path_token": "FIXB",
            "route_index": 1,
            "distance_nm": 37.0,
            "ring_inner_nm": 35.0,
            "ring_outer_nm": 40.0,
        },
        "baseline_final_fix": {
            "identifier": "FINAL35C",
            "lat": 32.9,
            "lon": -97.0,
        },
        "base_route": {
            "type": "base-route",
            "fix_sequence": "FIXA>FIXB>FINAL35C>RW35C",
            "fix_count": 4,
            "lateral_path": ["FIXA", "FIXB", "FINAL35C", "RW35C"],
        },
        "simulation": {
            "success": False,
            "message": "infeasible: not enough along-track distance to complete FMS profile before threshold",
            "final_threshold_error_m": 1_852.0,
        },
    }
    simap_arrival_trajectories_path.write_text(json.dumps(simap_arrival_payload) + "\n", encoding="utf-8")

    adsb_compressed_trajectories_path = tmp_path / "adsb_compressed_flights.jsonl"
    adsb_arrival_payload = dict(simap_arrival_payload)
    adsb_arrival_payload["points"] = [[300, 0.0, 0.0, 0.0, 3]]
    adsb_departure_payload = {
        "flight_id": "DEP1",
        "callsign": "CALLDEP1",
        "icao24": "dep001",
        "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        "breakpoint_mask_bits": {"lateral": 1, "altitude": 2},
        "points": [[90, 33.0, -98.0, 300.0, 3], [140, 33.2, -98.2, 1500.0, 3]],
        "lateral_breakpoint_times": [90, 140],
        "altitude_breakpoint_times": [90, 140],
        "first_time": 90,
        "last_time": 140,
        "raw_point_count": 5,
        "compressed_point_count": 2,
        "lateral_tolerance_m": 100.0,
        "altitude_tolerance_m": 50.0,
    }
    adsb_compressed_trajectories_path.write_text(
        json.dumps(adsb_arrival_payload) + "\n" + json.dumps(adsb_departure_payload) + "\n",
        encoding="utf-8",
    )

    simap_arrival_artifact_manifest_path = tmp_path / "manifest.json"
    simap_arrival_artifact_manifest_path.write_text(
        json.dumps({"generated_count": 1, "skipped_arrival_count": 1, "skipped_departure_count": 2}) + "\n",
        encoding="utf-8",
    )
    adsb_compressed_metadata_path = tmp_path / "metadata.json"
    adsb_compressed_metadata_path.write_text(json.dumps({"flight_count": 2}) + "\n", encoding="utf-8")
    fixes_path = tmp_path / "fixes.csv"
    fixes_path.write_text(
        "identifier,latitude_deg,longitude_deg,fix_type,elevation_ft\n"
        "FIXA,32.0,-97.0,fix,\n"
        "FIXB,32.1,-97.1,fix,\n"
        "FINAL35C,32.9,-97.0,fix,\n"
        "RW35C,33.0,-97.0,runway,620\n",
        encoding="utf-8",
    )

    return ScenarioResourceConfig(
        events_path=events_path,
        fix_sequences_path=fix_sequences_path,
        simap_arrival_trajectories_path=simap_arrival_trajectories_path,
        adsb_compressed_trajectories_path=adsb_compressed_trajectories_path,
        fixes_path=fixes_path,
        simap_arrival_artifact_manifest_path=simap_arrival_artifact_manifest_path,
        adsb_compressed_metadata_path=adsb_compressed_metadata_path,
    )


def test_resource_config_loads_default_entry_from_data_manifest(tmp_path: Path) -> None:
    config = write_fixture_resources(tmp_path)
    assert config.simap_arrival_artifact_manifest_path is not None
    assert config.adsb_compressed_metadata_path is not None
    data_manifest_path = tmp_path / "data_manifest.json"
    data_manifest_path.write_text(
        json.dumps(
            {
                "2025-04-01": {
                    "landings_and_departures": config.events_path.as_posix(),
                    "fix_sequences": config.fix_sequences_path.as_posix(),
                    "simap_arrival_trajectories": config.simap_arrival_trajectories_path.as_posix(),
                    "simap_arrival_artifact_manifest": config.simap_arrival_artifact_manifest_path.as_posix(),
                    "adsb_compressed_trajectories": config.adsb_compressed_trajectories_path.as_posix(),
                    "adsb_compressed_metadata": config.adsb_compressed_metadata_path.as_posix(),
                    "fixes": config.fixes_path.as_posix(),
                    "default": True,
                },
                "2025-04-02": {
                    "landings_and_departures": "unused_events.csv",
                    "fix_sequences": "unused_fix_sequences.csv",
                    "simap_arrival_trajectories": "unused_simap_arrival_flights.jsonl",
                    "adsb_compressed_trajectories": "unused_adsb_compressed_flights.jsonl",
                    "default": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = ScenarioResourceConfig.from_manifest(data_manifest_path)

    assert loaded.data_manifest_path == data_manifest_path
    assert loaded.data_date == "2025-04-01"
    assert loaded.events_path == config.events_path
    assert loaded.fix_sequences_path == config.fix_sequences_path
    assert loaded.simap_arrival_trajectories_path == config.simap_arrival_trajectories_path
    assert loaded.simap_arrival_artifact_manifest_path == config.simap_arrival_artifact_manifest_path
    assert loaded.adsb_compressed_trajectories_path == config.adsb_compressed_trajectories_path
    assert loaded.adsb_compressed_metadata_path == config.adsb_compressed_metadata_path
    assert loaded.fixes_path == config.fixes_path


def test_departure_schedule_returns_adsb_trajectory_and_skips_missing_flights(tmp_path: Path) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))

    schedule = manager.departure_schedule()

    assert [item["flight_id"] for item in schedule] == ["DEP1"]
    departure = schedule[0]
    assert departure["callsign"] == "CALLDEP1"
    assert departure["icao24"] == "dep001"
    assert departure["departure_time"] == 100
    assert departure["departure_time_utc"] == "1970-01-01T00:01:40Z"
    assert departure["runway"] == "18L"
    assert departure["columns"] == ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"]
    assert departure["points"] == [[90, 33.0, -98.0, 300.0, 3], [140, 33.2, -98.2, 1500.0, 3]]
    assert departure["first_time"] == 90
    assert departure["last_time"] == 140


def test_departure_schedule_uses_diff_mutated_payload(tmp_path: Path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))

    def apply_diff(payload: dict[str, object]) -> dict[str, object]:
        if payload.get("flight_id") != "DEP1":
            return payload
        diffed = dict(payload)
        diffed["departure_time"] = 115
        diffed["runway"] = "RW18R"
        return diffed

    monkeypatch.setattr(manager, "_apply_diff", apply_diff)

    schedule = manager.departure_schedule()

    assert len(schedule) == 1
    departure = schedule[0]
    assert departure["flight_id"] == "DEP1"
    assert departure["departure_time"] == 115
    assert departure["departure_time_utc"] == "1970-01-01T00:01:55Z"
    assert departure["runway"] == "RW18R"


def test_arrival_schedule_uses_artifact_start_time_and_preserves_compressed_points(tmp_path: Path) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))

    schedule = manager.arrival_schedule()

    assert len(schedule) == 1
    arrival = schedule[0]
    assert arrival["flight_id"] == "ARR1"
    assert arrival["time_at_first_fix"] == 310
    assert arrival["time_at_first_fix_utc"] == "1970-01-01T00:05:10Z"
    assert arrival["time_at_last_event"] == 500
    assert arrival["time_at_last_event_utc"] == "1970-01-01T00:08:20Z"
    assert arrival["runway"] == "RW35C"
    assert arrival["route_type"] == "base-route"
    assert arrival["fix_sequence"] == "FIXA>FIXB>FINAL35C>RW35C"
    assert arrival["fix_count"] == 4
    assert "original_fix_sequence" not in arrival
    assert "original_fix_count" not in arrival
    assert arrival["final_fix"]["identifier"] == "FINAL35C"
    assert arrival["baseline_final_fix"]["identifier"] == "FINAL35C"
    assert arrival["base_route"]["lateral_path"] == ["FIXA", "FIXB", "FINAL35C", "RW35C"]
    assert arrival["columns"] == ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"]
    assert arrival["points"] == [[310, 32.0, -97.0, 1000.0, 3], [500, 32.1, -97.1, 200.0, 3]]
    assert arrival["cas_profile"] == {
        "columns": ["time", "cas_kts"],
        "units": {"cas_kts": "kt"},
        "source": "simap_fms_bichannel",
        "points": [[310, 190.0], [312, 188.5], [314, 187.0]],
    }
    expected_wait_point = {
        "source": "fix",
        "identifier": "FIXB",
        "lat": 32.1,
        "lon": -97.1,
        "lateral_path_token": "FIXB",
        "route_index": 1,
        "distance_nm": 37.0,
        "ring_inner_nm": 35.0,
        "ring_outer_nm": 40.0,
    }
    assert arrival["atc_wait_point"] == expected_wait_point
    assert arrival["wait_atc_point"] == expected_wait_point


def test_health_reports_missing_arrival_trajectories(tmp_path: Path) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))

    health = manager.health()

    assert health["events_count"] == 4
    assert health["arrivals_count"] == 2
    assert health["departures_count"] == 2
    assert health["compressed_flights_count"] == 2
    assert health["artifact_flights_count"] == 1
    assert health["simap_arrival_flights_count"] == 1
    assert health["adsb_compressed_flights_count"] == 2
    assert health["arrivals_missing_fix_sequences_count"] == 0
    assert health["arrivals_missing_trajectories_count"] == 1
    assert health["arrivals_missing_artifacts_count"] == 1
    assert health["arrivals_missing_simap_trajectories_count"] == 1
    assert health["departures_missing_trajectories_count"] == 1
    assert health["departures_missing_adsb_trajectories_count"] == 1
    assert health["skipped_departures_count"] == 2


def test_fastapi_app_exposes_scenario_routes(tmp_path: Path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    monkeypatch.setattr("mcp_tools.scenario_manager.api.ScenarioManager", lambda _config: manager)

    class FakeFeasibilityAdvisor:
        def __init__(self, _manager: ScenarioManager) -> None:
            pass

        def evaluate(self, flight_id: str | None = None) -> list[FeasibilityAdvisory]:
            assert flight_id == "ARR1"
            return [
                FeasibilityAdvisory(
                    flight_number="CALLARR1",
                    icao24="arr001",
                    flight_id="ARR1",
                    runway="RW35C",
                    miles_to_gain_nmi=2.0,
                    miles_to_gain_m=3_704.0,
                    minutes_to_gain=4.0,
                    baseline_success=False,
                    baseline_message="infeasible",
                    what_if_success=True,
                    what_if_message="ok",
                    baseline_time_s=100.0,
                    what_if_time_s=340.0,
                    search_converged=True,
                )
            ]

    class FakeVectoringAdvisor:
        def __init__(self, _manager: ScenarioManager) -> None:
            pass

        def advise(self, *, flight_id: str, extra_distance_nmi: float) -> VectoringAdvisory:
            assert flight_id == "ARR1"
            assert extra_distance_nmi == 2.0
            return VectoringAdvisory(
                flight_number="CALLARR1",
                icao24="arr001",
                flight_id="ARR1",
                runway="RW35C",
                miles_to_gain_nmi=2.0,
                minutes_to_gain=3.0,
                baseline_success=False,
                baseline_message="infeasible",
                what_if_success=False,
                what_if_message="still infeasible",
                baseline_time_s=100.0,
                what_if_time_s=280.0,
                requested_extension_nmi=2.0,
                requested_extension_m=3_704.0,
                required_feasibility_miles_nmi=4.0,
                required_feasibility_m=7_408.0,
                feasibility_search_converged=True,
            )

    class FakeSpeedControlAdvisor:
        def __init__(self, _manager: ScenarioManager) -> None:
            pass

        def advise(self, *, flight_id: str, s_m: float, cas_kts: float) -> SpeedControlAdvisory:
            assert flight_id == "ARR1"
            assert s_m == 50_000.0
            assert cas_kts == 180.0
            return SpeedControlAdvisory(
                flight_number="CALLARR1",
                icao24="arr001",
                flight_id="ARR1",
                runway="RW35C",
                miles_to_gain_nmi=0.0,
                minutes_to_gain=2.0,
                baseline_success=True,
                baseline_message="ok",
                what_if_success=True,
                what_if_message="ok",
                baseline_time_s=100.0,
                what_if_time_s=220.0,
                s_m=50_000.0,
                cas_kts=180.0,
                equivalent_vectoring_miles_nmi=6.0,
                equivalent_vectoring_m=11_112.0,
            )

    monkeypatch.setattr("mcp_tools.scenario_manager.api.FeasibilityAdvisor", FakeFeasibilityAdvisor)
    monkeypatch.setattr("mcp_tools.scenario_manager.api.VectoringAdvisor", FakeVectoringAdvisor)
    monkeypatch.setattr("mcp_tools.scenario_manager.api.SpeedControlAdvisor", FakeSpeedControlAdvisor)
    app = create_app()

    route_paths = {str(getattr(route, "path", "")) for route in app.routes}

    assert {
        "/health",
        "/departures",
        "/arrivals",
        "/tools/evals/feasibility",
        "/tools/evals/conflicts",
        "/tools/evals/runway-overlaps",
        "/tools/advisors/feasibility",
        "/tools/advisors/vectoring",
        "/tools/advisors/speed-control",
        "/diff",
        "/tools/path-stretch/simulate",
        "/diff/path-stretch/{flight_id}",
        "/tools/speed-intervention/simulate",
        "/diff/speed-intervention/{flight_id}",
    } <= route_paths
    assert manager.health()["arrivals_missing_trajectories_count"] == 1
    assert [item["flight_id"] for item in manager.departure_schedule()] == ["DEP1"]
    assert manager.arrival_schedule()[0]["time_at_first_fix"] == 310
    assert manager.intervention_diff() == []

    with TestClient(app) as client:
        departures = client.get("/departures")
        arrivals = client.get("/arrivals")
        feasibility = client.get("/tools/evals/feasibility")
        conflicts = client.get("/tools/evals/conflicts")
        runway_overlaps = client.get("/tools/evals/runway-overlaps")
        advisory_feasibility = client.get("/tools/advisors/feasibility", params={"flight_id": "ARR1"})
        advisory_vectoring = client.get(
            "/tools/advisors/vectoring",
            params={"flight_id": "ARR1", "extra_distance_nmi": 2.0},
        )
        advisory_speed = client.get(
            "/tools/advisors/speed-control",
            params={"flight_id": "ARR1", "s_m": 50_000.0, "cas_kts": 180.0},
        )

    assert departures.status_code == 200
    assert arrivals.status_code == 200
    assert feasibility.status_code == 200
    assert conflicts.status_code == 200
    assert runway_overlaps.status_code == 200
    assert advisory_feasibility.status_code == 200
    assert advisory_vectoring.status_code == 200
    assert advisory_speed.status_code == 200
    assert departures.json()[0]["points"] == [[90, 33.0, -98.0, 300.0, 3], [140, 33.2, -98.2, 1500.0, 3]]
    assert arrivals.json()[0]["points"] == [[310, 32.0, -97.0, 1000.0, 3], [500, 32.1, -97.1, 200.0, 3]]
    assert arrivals.json()[0]["cas_profile"]["columns"] == ["time", "cas_kts"]
    assert arrivals.json()[0]["cas_profile"]["points"] == [[310, 190.0], [312, 188.5], [314, 187.0]]
    assert arrivals.json()[0]["fix_sequence"] == "FIXA>FIXB>FINAL35C>RW35C"
    assert arrivals.json()[0]["final_fix"]["identifier"] == "FINAL35C"
    assert arrivals.json()[0]["baseline_final_fix"]["identifier"] == "FINAL35C"
    assert arrivals.json()[0]["atc_wait_point"]["identifier"] == "FIXB"
    assert arrivals.json()[0]["wait_atc_point"]["identifier"] == "FIXB"
    assert feasibility.json() == [
        {
            "flight_number": "CALLARR1",
            "icao24": "arr001",
            "flight_id": "ARR1",
            "runway": "RW35C",
            "missing_distance_nmi": 1.0,
            "missing_distance_m": 1_852.0,
            "simulation_message": (
                "infeasible: not enough along-track distance to complete FMS profile before threshold"
            ),
        }
    ]
    assert conflicts.json() == []
    assert runway_overlaps.json() == []
    assert advisory_feasibility.json()[0]["miles_to_gain_nmi"] == 2.0
    assert advisory_vectoring.json()["required_feasibility_miles_nmi"] == 4.0
    assert advisory_speed.json()["equivalent_vectoring_miles_nmi"] == 6.0
