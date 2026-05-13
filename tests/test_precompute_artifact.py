from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mcp_tools.scenario_manager import precompute_artifact
from simap.path_geometry import ReferencePath
from simap.nlp_colloc.tactical.models import PathWaypoint


def _write_catalogs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    events_path = tmp_path / "events.csv"
    events_path.write_text(
        "date,flight_id,callsign,icao24,operation,runway,event_time,event_time_utc\n"
        "2025-04-01,ARR1,CALLARR1,abc001,arrival,35C,120,1970-01-01T00:02:00Z\n"
        "2025-04-01,ARR2,CALLARR2,abc002,arrival,36L,180,1970-01-01T00:03:00Z\n"
        "2025-04-01,DEP1,CALLDEP1,def001,departure,17R,240,1970-01-01T00:04:00Z\n",
        encoding="utf-8",
    )
    fix_sequences_path = tmp_path / "fix_sequences.csv"
    fix_sequences_path.write_text(
        "date,flight_id,callsign,icao24,first_time,last_time,fix_sequence,fix_count\n"
        "2025-04-01,ARR1,CALLARR1,abc001,100,160,FIXA>FIXB,2\n"
        "2025-04-01,ARR2,CALLARR2,abc002,100,160,FIXC>FIXD,2\n",
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    fixes_csv = tmp_path / "fixes.csv"
    fixes_csv.write_text(
        "identifier,latitude_deg,longitude_deg\n"
        "FIXA,32.0,-97.0\n"
        "FIXB,32.1,-97.1\n"
        "FIXC,33.0,-98.0\n"
        "FIXD,33.1,-98.1\n"
        "DAYZZ,32.78,-97.9\n"
        "RW17C,33.1,-97.9\n"
        "RW35C,32.9,-97.9\n"
        "RW36L,33.9,-98.9\n",
        encoding="utf-8",
    )
    return events_path, fix_sequences_path, raw_dir, fixes_csv


def test_precompute_writes_arrival_artifacts_and_manifest(tmp_path: Path, monkeypatch) -> None:
    events_path, fix_sequences_path, raw_dir, fixes_csv = _write_catalogs(tmp_path)
    output_dir = tmp_path / "artifacts"
    raw_tracks = {
        "ARR1": pd.DataFrame(
            {
                "time": [90, 100, 110],
                "lat": [31.5, 32.0, 32.01],
                "lon": [-96.5, -97.0, -97.01],
                "heading": [179.0, 180.0, 181.0],
                "geoaltitude": [1_200.0, 1_000.0, 900.0],
            }
        ),
        "ARR2": pd.DataFrame(
            {
                "time": [90],
                "lat": [32.0],
                "lon": [-97.0],
                "heading": [180.0],
                "geoaltitude": [1_000.0],
            }
        ),
    }
    fake_result = SimpleNamespace(
        t_s=np.asarray([0.0, 10.0, 20.0]),
        lat_deg=np.asarray([32.0, 32.01, 32.02]),
        lon_deg=np.asarray([-97.0, -97.01, -97.02]),
        h_m=np.asarray([1_000.0, 800.0, 600.0]),
        v_cas_mps=np.asarray([100.0, 105.0, 110.0]),
        success=True,
        message="ok",
        max_abs_cross_track_m=0.0,
        max_abs_track_error_rad=0.0,
        final_threshold_error_m=0.0,
    )

    monkeypatch.setattr(precompute_artifact, "_flight_raw_tracks", lambda *args, **kwargs: raw_tracks)
    monkeypatch.setattr(precompute_artifact, "_build_request", lambda *args, **kwargs: (object(), None))
    monkeypatch.setattr(precompute_artifact, "plan_fms_bichannel", lambda _request, **_kwargs: fake_result)
    monkeypatch.setattr(
        precompute_artifact,
        "detect_wait_atc_point",
        lambda *args, **kwargs: {
            "source": "fix",
            "identifier": "FIXB",
            "lat": 32.1,
            "lon": -97.1,
            "lateral_path_token": "FIXB",
            "route_index": 1,
            "distance_nm": 45.0,
            "arrival_cluster": "SE",
        },
    )

    manifest = precompute_artifact.precompute_artifacts(
        events_path=events_path,
        fix_sequences_path=fix_sequences_path,
        raw_adsb_dir=raw_dir,
        fixes_csv=fixes_csv,
        output_dir=output_dir,
        processes=1,
    )

    flights_path = output_dir / "simap_arrival_flights.jsonl"
    stored_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    payloads = [json.loads(line) for line in flights_path.read_text(encoding="utf-8").splitlines()]

    assert manifest["generated_count"] == 1
    assert stored_manifest["generated_count"] == 1
    assert stored_manifest["skipped_arrival_count"] == 1
    assert stored_manifest["skipped_departure_count"] == 1
    assert payloads[0]["flight_id"] == "ARR1"
    assert payloads[0]["columns"] == ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"]
    assert payloads[0]["breakpoint_mask_bits"] == {"lateral": 1, "altitude": 2}
    assert payloads[0]["points"][0][0] == 100
    assert payloads[0]["cas_profile"]["columns"] == ["time", "cas_kts"]
    assert payloads[0]["cas_profile"]["units"] == {"cas_kts": "kt"}
    assert payloads[0]["cas_profile"]["source"] == "simap_fms_bichannel"
    assert [point[0] for point in payloads[0]["cas_profile"]["points"]] == [100, 110, 120]
    assert payloads[0]["route_type"] == "base-route"
    assert payloads[0]["runway"] == "RW35C"
    assert payloads[0]["fix_sequence"] == "FIXA>FIXB>DAYZZ>RW35C"
    assert payloads[0]["fix_count"] == 4
    assert payloads[0]["wait_atc_point"]["identifier"] == "FIXB"
    assert payloads[0]["baseline_final_fix"]["identifier"] == "DAYZZ"
    assert payloads[0]["base_route"]["type"] == "base-route"
    assert payloads[0]["base_route"]["fix_sequence"] == "FIXA>FIXB>DAYZZ>RW35C"
    assert payloads[0]["base_route"]["fix_count"] == 4
    assert payloads[0]["base_route"]["lateral_path"] == ["FIXA", "FIXB", "DAYZZ", "RW35C"]
    assert payloads[0]["base_route"]["final_fix"]["identifier"] == "DAYZZ"
    assert payloads[0]["simulation"]["lateral_guidance"]["lookahead_m"] == 1500.0
    assert stored_manifest["lateral_guidance"]["integration_step_s"] == 0.5
    assert stored_manifest["layout"]["cas_profile_columns"] == ["time", "cas_kts"]
    assert stored_manifest["layout"]["cas_profile_units"] == {"cas_kts": "kt"}


def test_payload_uses_bichannel_map_coordinates_when_reference_path_is_available() -> None:
    reference_path = ReferencePath.from_geographic(
        lat_deg=np.asarray([32.0, 32.1], dtype=float),
        lon_deg=np.asarray([-97.0, -97.0], dtype=float),
    )
    base_route = precompute_artifact.BaseRoute(
        lateral_path=["FIXA", "RW35C"],
        upstream_identifier="FIXA",
        runway_identifier="RW35C",
        final_fix=precompute_artifact.FinalFixSelection(
            waypoint=PathWaypoint("FIXA", 32.0, -97.0),
            distance_nm=7.0,
            along_track_nm=7.0,
            cross_track_nm=0.0,
            runway_true_heading_deg=350.0,
        ),
        atc_point={"identifier": "FIXA"},
        target_final_fix_distance_nm=7.0,
        final_fix_cross_track_tolerance_nm=0.15,
    )
    result = SimpleNamespace(
        t_s=np.asarray([0.0, 10.0]),
        s_m=np.asarray([reference_path.total_length_m, 0.0]),
        lat_deg=np.asarray([33.0, 33.1]),
        lon_deg=np.asarray([-98.0, -98.1]),
        h_m=np.asarray([1000.0, 900.0]),
        v_cas_mps=np.asarray([100.0, 90.0]),
        success=True,
        message="ok",
        max_abs_cross_track_m=42.0,
        max_abs_track_error_rad=0.1,
        final_threshold_error_m=12.0,
    )

    _artifact, payload = precompute_artifact._payload_from_result(
        row=pd.Series({"flight_id": "ARR1", "callsign": "CALLARR1", "icao24": "abc001", "runway": "35C"}),
        seed=precompute_artifact.SeedState(
            time_s=100,
            lat_deg=33.0,
            lon_deg=-98.0,
            geoaltitude_m=1000.0,
            heading_deg=180.0,
            ground_speed_mps=100.0,
        ),
        wait_atc_point=None,
        base_route=base_route,
        result=result,
        reference_path=reference_path,
        guidance=precompute_artifact._default_lateral_guidance(),
        lateral_tolerance_m=1.0,
        altitude_tolerance_m=1.0,
    )

    assert payload["points"][0][1:3] == [33.0, -98.0]
    assert payload["points"][-1][1:3] == [33.1, -98.1]


def test_payload_adds_full_resolution_cas_profile_in_knots() -> None:
    base_route = precompute_artifact.BaseRoute(
        lateral_path=["FIXA", "RW35C"],
        upstream_identifier="FIXA",
        runway_identifier="RW35C",
        final_fix=precompute_artifact.FinalFixSelection(
            waypoint=PathWaypoint("FIXA", 32.0, -97.0),
            distance_nm=7.0,
            along_track_nm=7.0,
            cross_track_nm=0.0,
            runway_true_heading_deg=350.0,
        ),
        atc_point={"identifier": "FIXA"},
        target_final_fix_distance_nm=7.0,
        final_fix_cross_track_tolerance_nm=0.15,
    )
    result = SimpleNamespace(
        t_s=np.asarray([0.0, 2.0, 4.0]),
        lat_deg=np.asarray([32.0, 32.01, 32.02]),
        lon_deg=np.asarray([-97.0, -97.01, -97.02]),
        h_m=np.asarray([1000.0, 950.0, 900.0]),
        v_cas_mps=np.asarray([100.0, 105.0, 110.0]),
        success=True,
        message="ok",
        max_abs_cross_track_m=0.0,
        max_abs_track_error_rad=0.0,
        final_threshold_error_m=0.0,
    )

    _artifact, payload = precompute_artifact._payload_from_result(
        row=pd.Series({"flight_id": "ARR1", "callsign": "CALLARR1", "icao24": "abc001", "runway": "35C"}),
        seed=precompute_artifact.SeedState(
            time_s=100,
            lat_deg=32.0,
            lon_deg=-97.0,
            geoaltitude_m=1000.0,
            heading_deg=180.0,
            ground_speed_mps=100.0,
        ),
        wait_atc_point=None,
        base_route=base_route,
        result=result,
        guidance=precompute_artifact._default_lateral_guidance(),
        lateral_tolerance_m=1.0,
        altitude_tolerance_m=1.0,
    )

    assert payload["cas_profile"]["columns"] == ["time", "cas_kts"]
    assert payload["cas_profile"]["units"] == {"cas_kts": "kt"}
    assert payload["cas_profile"]["points"] == [
        [100, precompute_artifact.mps_to_kts(100.0)],
        [102, precompute_artifact.mps_to_kts(105.0)],
        [104, precompute_artifact.mps_to_kts(110.0)],
    ]


def test_seed_for_flight_uses_adsb_point_closest_to_first_fix() -> None:
    flight = pd.DataFrame(
        {
            "time": [100, 110, 120],
            "lat": [32.0, 32.05, 32.2],
            "lon": [-97.0, -97.05, -97.2],
            "heading": [180.0, 181.0, 182.0],
            "geoaltitude": [1_000.0, 900.0, 800.0],
        }
    )

    seed = precompute_artifact._seed_for_flight_at_fix(
        flight,
        fix_lat_deg=32.049,
        fix_lon_deg=-97.049,
    )

    assert seed is not None
    assert seed.time_s == 110
    assert seed.geoaltitude_m == 900.0
    assert seed.heading_deg == 181.0
    assert seed.ground_speed_mps > 1.0


def test_final_fix_selection_uses_extended_runway_alignment_for_parallel_runways() -> None:
    catalog = {
        "RW17C": PathWaypoint("RW17C", 33.10, -97.00, source="runway"),
        "RW35C": PathWaypoint("RW35C", 32.90, -97.00, source="runway"),
        "RW17L": PathWaypoint("RW17L", 33.10, -97.02, source="runway"),
        "RW35R": PathWaypoint("RW35R", 32.90, -97.02, source="runway"),
        "CENTER_FINAL": PathWaypoint("CENTER_FINAL", 32.7834, -97.00),
        "RIGHT_FINAL": PathWaypoint("RIGHT_FINAL", 32.7834, -97.02),
        "MISALIGNED": PathWaypoint("MISALIGNED", 32.7834, -97.04),
    }

    center = precompute_artifact._select_final_fix(
        runway_identifier="RW35C",
        fix_catalog=catalog,
        target_distance_nm=7.0,
        cross_track_tolerance_nm=0.15,
    )
    right = precompute_artifact._select_final_fix(
        runway_identifier="RW35R",
        fix_catalog=catalog,
        target_distance_nm=7.0,
        cross_track_tolerance_nm=0.15,
    )

    assert center.waypoint.identifier == "CENTER_FINAL"
    assert right.waypoint.identifier == "RIGHT_FINAL"
