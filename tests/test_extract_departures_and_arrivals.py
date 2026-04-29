from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scenario.demand_opensky.adsb_catalog_io import (
    build_flight_id,
    list_input_csv_files,
    load_runway_thresholds,
    normalize_callsign,
)
from scenario.demand_opensky.adsb_catalog_processing import (
    classify_flight_track,
    extract_fix_sequence,
    process_tracks,
    split_flight_track,
)


THRESHOLDS = pd.DataFrame(
    [
        {
            "runway": "17C",
            "runway_pair": "17C/35C",
            "threshold_lat": 32.915706694444445,
            "threshold_lon": -97.02597491666667,
        }
    ]
)


def make_flight(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["callsign"] = frame["callsign"].map(normalize_callsign)
    frame["icao24"] = frame["icao24"].astype(str)
    frame["flight_id"] = [
        build_flight_id(callsign, icao24)
        for callsign, icao24 in zip(frame["callsign"], frame["icao24"], strict=False)
    ]
    return frame


class ExtractDeparturesAndArrivalsTests(unittest.TestCase):
    def test_list_input_csv_files_skips_sidecars_and_non_numeric_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "1743465600.csv").write_text("", encoding="utf-8")
            (directory / "._1743465600.csv").write_text("", encoding="utf-8")
            (directory / "notes.csv").write_text("", encoding="utf-8")
            (directory / "1743469200.txt").write_text("", encoding="utf-8")

            result = list_input_csv_files(directory)

            self.assertEqual([path.name for path in result], ["1743465600.csv"])

    def test_load_runway_thresholds_expands_both_runway_ends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runways.csv"
            path.write_text(
                "runway_pair,length_ft,latitude_a,longitude_a,latitude_b,longitude_b\n"
                "13L/31R,9000,32.91,-97.02,32.89,-97.00\n",
                encoding="utf-8",
            )

            thresholds = load_runway_thresholds(path)

            self.assertEqual(list(thresholds["runway"]), ["13L", "31R"])
            self.assertAlmostEqual(float(thresholds.iloc[0]["threshold_lat"]), 32.91)
            self.assertAlmostEqual(float(thresholds.iloc[1]["threshold_lon"]), -97.00)

    def test_classify_flight_track_detects_arrival(self) -> None:
        flight = make_flight(
            [
                {"time": 0, "icao24": "abc123", "lat": 33.10, "lon": -97.20, "heading": 180.0, "callsign": "AAL100  ", "geoaltitude": 900.0},
                {"time": 60, "icao24": "abc123", "lat": 33.00, "lon": -97.15, "heading": 180.0, "callsign": "AAL100  ", "geoaltitude": 800.0},
                {"time": 120, "icao24": "abc123", "lat": 32.98, "lon": -97.10, "heading": 180.0, "callsign": "AAL100  ", "geoaltitude": 700.0},
                {"time": 180, "icao24": "abc123", "lat": 32.94, "lon": -97.05, "heading": 180.0, "callsign": "AAL100  ", "geoaltitude": 600.0},
                {"time": 240, "icao24": "abc123", "lat": 32.9158, "lon": -97.0260, "heading": 180.0, "callsign": "AAL100  ", "geoaltitude": 180.0},
                {"time": 300, "icao24": "abc123", "lat": 32.9157, "lon": -97.0260, "heading": 180.0, "callsign": "AAL100  ", "geoaltitude": 120.0},
            ]
        )

        classification, event = classify_flight_track(
            flight=flight,
            thresholds=THRESHOLDS,
            runway_radius_m=1_000.0,
            lookaround_seconds=300,
            min_altitude_change_m=75.0,
            date_label="2025-04-01",
        )

        self.assertEqual(classification, "arrival")
        assert event is not None
        self.assertEqual(event["runway"], "17C")
        self.assertEqual(event["event_time"], 240)
        self.assertLess(float(event["altitude_delta_m"]), 0.0)

    def test_classify_flight_track_detects_departure(self) -> None:
        flight = make_flight(
            [
                {"time": 0, "icao24": "def456", "lat": 32.9157, "lon": -97.0260, "heading": 0.0, "callsign": "DAL200", "geoaltitude": 110.0},
                {"time": 60, "icao24": "def456", "lat": 32.9157, "lon": -97.0261, "heading": 0.0, "callsign": "DAL200", "geoaltitude": 190.0},
                {"time": 120, "icao24": "def456", "lat": 32.94, "lon": -97.03, "heading": 0.0, "callsign": "DAL200", "geoaltitude": 410.0},
                {"time": 180, "icao24": "def456", "lat": 32.98, "lon": -97.04, "heading": 0.0, "callsign": "DAL200", "geoaltitude": 650.0},
                {"time": 240, "icao24": "def456", "lat": 33.02, "lon": -97.08, "heading": 0.0, "callsign": "DAL200", "geoaltitude": 820.0},
                {"time": 300, "icao24": "def456", "lat": 33.08, "lon": -97.12, "heading": 0.0, "callsign": "DAL200", "geoaltitude": 980.0},
            ]
        )

        classification, event = classify_flight_track(
            flight=flight,
            thresholds=THRESHOLDS,
            runway_radius_m=1_000.0,
            lookaround_seconds=300,
            min_altitude_change_m=75.0,
            date_label="2025-04-01",
        )

        self.assertEqual(classification, "departure")
        assert event is not None
        self.assertEqual(event["runway"], "17C")
        self.assertEqual(event["event_time"], 60)
        self.assertGreater(float(event["altitude_delta_m"]), 0.0)

    def test_classify_flight_track_returns_overflight_when_not_ground_relevant(self) -> None:
        flight = make_flight(
            [
                {"time": 0, "icao24": "ghi789", "lat": 33.10, "lon": -97.20, "heading": 90.0, "callsign": "UAL300", "geoaltitude": 2_000.0},
                {"time": 60, "icao24": "ghi789", "lat": 33.00, "lon": -97.10, "heading": 90.0, "callsign": "UAL300", "geoaltitude": 2_100.0},
                {"time": 120, "icao24": "ghi789", "lat": 32.95, "lon": -97.00, "heading": 90.0, "callsign": "UAL300", "geoaltitude": 2_300.0},
            ]
        )

        classification, event = classify_flight_track(
            flight=flight,
            thresholds=THRESHOLDS,
            runway_radius_m=1_000.0,
            lookaround_seconds=300,
            min_altitude_change_m=75.0,
            date_label="2025-04-01",
        )

        self.assertEqual(classification, "overflight")
        self.assertIsNone(event)

    def test_extract_fix_sequence_preserves_order_and_collapses_consecutive_duplicates(self) -> None:
        fixes = pd.DataFrame(
            [
                {"identifier": "SPERA", "fix_type": "waypoint", "latitude_deg": 32.7827, "longitude_deg": -96.9950},
                {"identifier": "JGIRL", "fix_type": "waypoint", "latitude_deg": 32.7651, "longitude_deg": -96.9393},
                {"identifier": "CORTS", "fix_type": "waypoint", "latitude_deg": 32.7688, "longitude_deg": -96.8316},
            ]
        )
        flight = make_flight(
            [
                {"time": 0, "icao24": "jkl012", "lat": 32.7827, "lon": -96.9950, "heading": 0.0, "callsign": "AAL400", "geoaltitude": 1000.0},
                {"time": 60, "icao24": "jkl012", "lat": 32.7828, "lon": -96.9951, "heading": 0.0, "callsign": "AAL400", "geoaltitude": 1200.0},
                {"time": 120, "icao24": "jkl012", "lat": 32.7651, "lon": -96.9393, "heading": 0.0, "callsign": "AAL400", "geoaltitude": 1500.0},
                {"time": 180, "icao24": "jkl012", "lat": 32.7652, "lon": -96.9394, "heading": 0.0, "callsign": "AAL400", "geoaltitude": 1800.0},
                {"time": 240, "icao24": "jkl012", "lat": 32.7688, "lon": -96.8316, "heading": 0.0, "callsign": "AAL400", "geoaltitude": 2200.0},
            ]
        )

        result = extract_fix_sequence(
            flight=flight,
            fixes=fixes,
            fix_radius_m=2_000.0,
            date_label="2025-04-01",
        )

        self.assertEqual(result["flight_id"], "AAL400jkl012")
        self.assertEqual(result["fix_sequence"], "SPERA>JGIRL>CORTS")
        self.assertEqual(result["fix_count"], 3)

    def test_split_flight_track_splits_large_time_gaps_and_suffixes_callsigns(self) -> None:
        flight = make_flight(
            [
                {"time": 0, "icao24": "abc123", "lat": 32.9157, "lon": -97.0260, "heading": 0.0, "callsign": "AAL1061", "geoaltitude": 110.0},
                {"time": 60, "icao24": "abc123", "lat": 32.94, "lon": -97.03, "heading": 0.0, "callsign": "AAL1061", "geoaltitude": 410.0},
                {"time": 2000, "icao24": "abc123", "lat": 33.10, "lon": -97.02, "heading": 180.0, "callsign": "AAL1061", "geoaltitude": 1200.0},
                {"time": 2060, "icao24": "abc123", "lat": 32.9157, "lon": -97.0260, "heading": 180.0, "callsign": "AAL1061", "geoaltitude": 150.0},
            ]
        )

        segments = split_flight_track(flight, split_gap_seconds=1500)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["callsign"].tolist(), ["AAL1061M1", "AAL1061M1"])
        self.assertEqual(segments[1]["callsign"].tolist(), ["AAL1061M2", "AAL1061M2"])
        self.assertEqual(segments[0]["flight_id"].iloc[0], "AAL1061M1abc123")
        self.assertEqual(segments[1]["flight_id"].iloc[0], "AAL1061M2abc123")

    def test_process_tracks_classifies_split_segments_independently(self) -> None:
        tracks = make_flight(
            [
                {"time": 0, "icao24": "abc123", "lat": 32.9157, "lon": -97.0260, "heading": 0.0, "callsign": "AAL1061", "geoaltitude": 110.0},
                {"time": 60, "icao24": "abc123", "lat": 32.94, "lon": -97.03, "heading": 0.0, "callsign": "AAL1061", "geoaltitude": 410.0},
                {"time": 120, "icao24": "abc123", "lat": 33.02, "lon": -97.08, "heading": 0.0, "callsign": "AAL1061", "geoaltitude": 820.0},
                {"time": 2000, "icao24": "abc123", "lat": 33.10, "lon": -97.20, "heading": 180.0, "callsign": "AAL1061", "geoaltitude": 900.0},
                {"time": 2060, "icao24": "abc123", "lat": 32.98, "lon": -97.10, "heading": 180.0, "callsign": "AAL1061", "geoaltitude": 700.0},
                {"time": 2120, "icao24": "abc123", "lat": 32.9157, "lon": -97.0260, "heading": 180.0, "callsign": "AAL1061", "geoaltitude": 120.0},
            ]
        )
        fixes = pd.DataFrame(
            [
                {"identifier": "DEPFX", "fix_type": "waypoint", "latitude_deg": 33.02, "longitude_deg": -97.08},
                {"identifier": "ARRFX", "fix_type": "waypoint", "latitude_deg": 32.98, "longitude_deg": -97.10},
            ]
        )

        events, fix_sequences, classification_counts = process_tracks(
            tracks=tracks,
            thresholds=THRESHOLDS,
            fixes=fixes,
            runway_radius_m=1_000.0,
            fix_radius_m=2_000.0,
            lookaround_seconds=300,
            min_altitude_change_m=75.0,
            split_gap_seconds=1500,
            date_label="2025-04-01",
        )

        self.assertEqual(classification_counts["departure"], 1)
        self.assertEqual(classification_counts["arrival"], 1)
        self.assertEqual(events["flight_id"].tolist(), ["AAL1061M1abc123", "AAL1061M2abc123"])
        self.assertEqual(events["operation"].tolist(), ["departure", "arrival"])
        self.assertEqual(fix_sequences["flight_id"].tolist(), ["AAL1061M1abc123", "AAL1061M2abc123"])


if __name__ == "__main__":
    unittest.main()
