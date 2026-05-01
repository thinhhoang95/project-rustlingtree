from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mcp_tools.scenario_manager import precompute_artifact


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
    fixes_csv.write_text("identifier,latitude_deg,longitude_deg\n", encoding="utf-8")
    return events_path, fix_sequences_path, raw_dir, fixes_csv


def test_precompute_writes_arrival_artifacts_and_manifest(tmp_path: Path, monkeypatch) -> None:
    events_path, fix_sequences_path, raw_dir, fixes_csv = _write_catalogs(tmp_path)
    output_dir = tmp_path / "artifacts"
    raw_tracks = {
        "ARR1": pd.DataFrame(
            {
                "time": [100, 110],
                "lat": [32.0, 32.01],
                "lon": [-97.0, -97.01],
                "heading": [180.0, 181.0],
                "geoaltitude": [1_000.0, 900.0],
            }
        ),
        "ARR2": pd.DataFrame(
            {
                "time": [90, 110],
                "lat": [32.0, 32.01],
                "lon": [-97.0, -97.01],
                "heading": [180.0, 181.0],
                "geoaltitude": [1_000.0, 900.0],
            }
        ),
    }
    fake_result = SimpleNamespace(
        t_s=np.asarray([0.0, 10.0, 20.0]),
        lat_deg=np.asarray([32.0, 32.01, 32.02]),
        lon_deg=np.asarray([-97.0, -97.01, -97.02]),
        h_m=np.asarray([1_000.0, 800.0, 600.0]),
        success=True,
        message="ok",
        max_abs_cross_track_m=0.0,
        max_abs_track_error_rad=0.0,
        final_threshold_error_m=0.0,
    )

    monkeypatch.setattr(precompute_artifact, "_flight_raw_tracks", lambda *args, **kwargs: raw_tracks)
    monkeypatch.setattr(precompute_artifact, "_build_request", lambda *args, **kwargs: (object(), None))
    monkeypatch.setattr(precompute_artifact, "plan_fms_bichannel", lambda _request: fake_result)

    manifest = precompute_artifact.precompute_artifacts(
        events_path=events_path,
        fix_sequences_path=fix_sequences_path,
        raw_adsb_dir=raw_dir,
        fixes_csv=fixes_csv,
        output_dir=output_dir,
        processes=1,
    )

    flights_path = output_dir / "flights.jsonl"
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
