from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

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

    compressed_flights_path = tmp_path / "flights.jsonl"
    payload = {
        "flight_id": "ARR1",
        "callsign": "CALLARR1",
        "icao24": "arr001",
        "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        "breakpoint_mask_bits": {"lateral": 1, "altitude": 2},
        "points": [[300, 32.0, -97.0, 1000.0, 3], [500, 32.1, -97.1, 200.0, 3]],
        "lateral_breakpoint_times": [300, 500],
        "altitude_breakpoint_times": [300, 500],
        "first_time": 300,
        "last_time": 500,
        "raw_point_count": 2,
        "compressed_point_count": 2,
        "lateral_tolerance_m": 100.0,
        "altitude_tolerance_m": 50.0,
    }
    compressed_flights_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    return ScenarioResourceConfig(
        events_path=events_path,
        fix_sequences_path=fix_sequences_path,
        compressed_flights_path=compressed_flights_path,
    )


def test_departure_schedule_returns_time_and_runway_sorted_by_departure_time(tmp_path: Path) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))

    schedule = manager.departure_schedule()

    assert [item["flight_id"] for item in schedule] == ["DEP1", "DEP2"]
    assert schedule[0] == {
        "flight_id": "DEP1",
        "callsign": "CALLDEP1",
        "icao24": "dep001",
        "departure_time": 100,
        "departure_time_utc": "1970-01-01T00:01:40Z",
        "runway": "18L",
    }


def test_arrival_schedule_uses_first_fix_time_and_preserves_compressed_points(tmp_path: Path) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))

    schedule = manager.arrival_schedule()

    assert len(schedule) == 1
    arrival = schedule[0]
    assert arrival["flight_id"] == "ARR1"
    assert arrival["arrival_time"] == 300
    assert arrival["arrival_time_utc"] == "1970-01-01T00:05:00Z"
    assert arrival["runway"] == "35C"
    assert arrival["original_fix_sequence"] == "FIXA>FIXB"
    assert arrival["original_fix_count"] == 2
    assert arrival["columns"] == ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"]
    assert arrival["points"] == [[300, 32.0, -97.0, 1000.0, 3], [500, 32.1, -97.1, 200.0, 3]]


def test_health_reports_missing_arrival_trajectories(tmp_path: Path) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))

    health = manager.health()

    assert health["events_count"] == 4
    assert health["arrivals_count"] == 2
    assert health["departures_count"] == 2
    assert health["arrivals_missing_fix_sequences_count"] == 0
    assert health["arrivals_missing_trajectories_count"] == 1
    assert health["departures_missing_trajectories_count"] == 2


def test_fastapi_routes_reuse_loaded_manager(tmp_path: Path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    monkeypatch.setattr("mcp_tools.scenario_manager.api.ScenarioManager", lambda: manager)
    app = create_app()

    with TestClient(app) as client:
        health = client.get("/health")
        departures = client.get("/departures")
        arrivals = client.get("/arrivals")

    assert health.status_code == 200
    assert health.json()["arrivals_missing_trajectories_count"] == 1
    assert departures.status_code == 200
    assert [item["flight_id"] for item in departures.json()] == ["DEP1", "DEP2"]
    assert arrivals.status_code == 200
    assert arrivals.json()[0]["arrival_time"] == 300
