from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hailmary.data import (
    RawADSBTrack,
    load_catalog_raw_adsb_tracks,
    load_arrival_catalog,
    load_manifest,
    load_raw_adsb_tracks,
    reconstruct_terminal_entry,
)
from hailmary.geometry import LocalFrame


def test_manifest_resolves_relative_paths_and_selects_declared_default(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "day-b": {"landings_and_departures": "b/events.csv", "default": False},
                "day-a": {
                    "landings_and_departures": "a/events.csv",
                    "raw_adsb": "a/raw",
                    "default": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)
    selected = manifest.select()

    assert manifest.dataset_ids == ("day-a", "day-b")
    assert selected.dataset_id == "day-a"
    assert selected.landings_and_departures == (tmp_path / "a/events.csv").resolve()
    assert selected.raw_adsb == (tmp_path / "a/raw").resolve()
    assert manifest.select("day-b").dataset_id == "day-b"


def test_manifest_rejects_ambiguous_defaults(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"a": {"default": True}, "b": {"default": True}}), encoding="utf-8")

    with pytest.raises(ValueError, match="more than one default"):
        load_manifest(path)


def test_arrival_catalog_filters_partition_and_normalizes_runway(tmp_path: Path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "flight_id,callsign,icao24,operation,runway,event_time,threshold_lat,threshold_lon,airport\n"
        "D1,DEP1,d1,departure,35C,10,32.9,-97.0,KDFW\n"
        "A2,ARR2,a2,arrival,36L,30,32.8,-97.1,KDFW\n"
        "A1,ARR1,a1,arrival,35C,20,32.9,-97.0,KDFW\n",
        encoding="utf-8",
    )

    arrivals = load_arrival_catalog(path, runway="RW35C", airport="kdfw")

    assert [item.flight_id for item in arrivals] == ["A1"]
    assert arrivals[0].runway == "RW35C"
    assert arrivals[0].threshold_lat_deg == pytest.approx(32.9)


def test_single_airport_catalog_without_airport_column_uses_requested_partition(tmp_path: Path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "flight_id,operation,runway,event_time,threshold_lat,threshold_lon\n"
        "A1,arrival,35C,20,32.9,-97.0\n",
        encoding="utf-8",
    )

    arrivals = load_arrival_catalog(path, airport="KDFW")

    assert arrivals[0].airport == "KDFW"


def test_raw_adsb_loader_is_stable_last_observation_wins_and_arrays_are_immutable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "00.csv").write_text(
        "0,abc,32.0,-97.0,180,CALL,1000\n"
        "10,abc,32.1,-97.1,181,CALL,900\n"
        "10,abc,32.2,-97.2,182,CALL,800\n"
        "0,def,33.0,-98.0,170,OTHER,1200\n"
        "10,def,33.1,-98.1,171,OTHER,1100\n",
        encoding="utf-8",
    )
    (raw / "._01.csv").write_text("not,a,real,file\n", encoding="utf-8")

    tracks = load_raw_adsb_tracks(raw, flight_ids={"CALLabc"})

    assert len(tracks) == 1
    track = tracks[0]
    assert track.flight_id == "CALLabc"
    np.testing.assert_allclose(track.lat_deg, [32.0, 32.2])
    assert track.lat_deg.dtype == np.float64
    assert track.lat_deg.flags.c_contiguous
    assert not track.lat_deg.flags.writeable
    with pytest.raises(ValueError):
        track.lat_deg[0] = 0.0


def test_catalog_raw_loader_splits_gaps_and_tolerates_missing_heading(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "00.csv").write_text(
        "0,abc,32.0,-97.0,,CALL,1000\n"
        "10,abc,32.1,-97.1,,CALL,900\n"
        "2000,abc,33.0,-98.0,170,CALL,1200\n"
        "2010,abc,33.1,-98.1,171,CALL,1100\n",
        encoding="utf-8",
    )

    tracks = load_catalog_raw_adsb_tracks(raw)

    assert [item.flight_id for item in tracks] == ["CALLM1abc", "CALLM2abc"]
    np.testing.assert_allclose(tracks[0].heading_deg, [0.0, 0.0])


def test_terminal_entry_interpolates_the_observed_50_nm_crossing() -> None:
    frame = LocalFrame(32.9, -97.0)
    lat, lon = frame.unproject(np.zeros(2), np.asarray([60.0, 40.0]) * 1_852.0)
    track = RawADSBTrack(
        flight_id="A1",
        callsign="A1",
        icao24="abc",
        time_s=[100.0, 200.0],
        lat_deg=lat,
        lon_deg=lon,
        heading_deg=[180.0, 180.0],
        geoaltitude_m=[3_000.0, 2_000.0],
    )

    entry = reconstruct_terminal_entry(track, frame, radius_nm=50.0)

    assert entry is not None
    assert entry.time_s == pytest.approx(150.0, abs=1.0e-5)
    assert entry.geoaltitude_m == pytest.approx(2_500.0, abs=1.0e-5)
    assert entry.segment_fraction == pytest.approx(0.5, abs=1.0e-7)
    east_m, north_m = frame.project(entry.lat_deg, entry.lon_deg)
    assert np.hypot(float(east_m), float(north_m)) == pytest.approx(50.0 * 1_852.0, abs=1.0e-4)


def test_terminal_entry_does_not_substitute_a_track_without_a_crossing() -> None:
    frame = LocalFrame(32.9, -97.0)
    lat, lon = frame.unproject(np.zeros(2), np.asarray([40.0, 20.0]) * 1_852.0)
    track = RawADSBTrack(
        flight_id="A1",
        callsign="A1",
        icao24="abc",
        time_s=[100.0, 200.0],
        lat_deg=lat,
        lon_deg=lon,
        heading_deg=[180.0, 180.0],
        geoaltitude_m=[2_000.0, 1_000.0],
    )

    assert reconstruct_terminal_entry(track, frame) is None
