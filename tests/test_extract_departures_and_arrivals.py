from __future__ import annotations

import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from rich.console import Console

from scenario.demand_opensky.adsb_authoritative_departures_and_arrivals_catalog import build_output_path
from scenario.demand_opensky.adsb_authoritative_departures_and_arrivals_catalog import _read_catalog_csv
from scenario.demand_opensky.adsb_authoritative_departures_and_arrivals_catalog import download_authoritative_departures_and_arrivals_catalog
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
from scenario.demand_opensky.extract_departures_and_arrivals import _report_catalog_differences


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
        self.assertEqual(event["event_time"], 0)
        self.assertGreater(float(event["altitude_delta_m"]), 0.0)
        self.assertGreater(float(event["distance_delta_m"]), 0.0)

    def test_classify_flight_track_uses_wide_low_altitude_proximity(self) -> None:
        flight = make_flight(
            [
                {"time": 0, "icao24": "abc124", "lat": 33.20, "lon": -97.20, "heading": 180.0, "callsign": "AAL101", "geoaltitude": 3_600.0},
                {"time": 60, "icao24": "abc124", "lat": 32.955, "lon": -97.0260, "heading": 180.0, "callsign": "AAL101", "geoaltitude": 2_600.0},
                {"time": 120, "icao24": "abc124", "lat": 32.930, "lon": -97.0260, "heading": 180.0, "callsign": "AAL101", "geoaltitude": 1_500.0},
                {"time": 180, "icao24": "abc124", "lat": 32.916, "lon": -97.0260, "heading": 180.0, "callsign": "AAL101", "geoaltitude": float("nan")},
            ]
        )

        classification, event = classify_flight_track(
            flight=flight,
            thresholds=THRESHOLDS,
            runway_radius_m=5_000.0,
            lookaround_seconds=300,
            min_altitude_change_m=75.0,
            date_label="2025-04-01",
        )

        self.assertEqual(classification, "arrival")
        assert event is not None
        self.assertEqual(event["event_time"], 60)
        self.assertLess(float(event["distance_delta_m"]), 0.0)

    def test_classify_flight_track_returns_overflight_when_not_ground_relevant(self) -> None:
        flight = make_flight(
            [
                {"time": 0, "icao24": "ghi789", "lat": 33.10, "lon": -97.20, "heading": 90.0, "callsign": "UAL300", "geoaltitude": 7_000.0},
                {"time": 60, "icao24": "ghi789", "lat": 32.94, "lon": -97.0260, "heading": 90.0, "callsign": "UAL300", "geoaltitude": 6_500.0},
                {"time": 120, "icao24": "ghi789", "lat": 32.92, "lon": -97.0260, "heading": 90.0, "callsign": "UAL300", "geoaltitude": 6_000.0},
            ]
        )

        classification, event = classify_flight_track(
            flight=flight,
            thresholds=THRESHOLDS,
            runway_radius_m=5_000.0,
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

    def test_build_output_path_uses_local_range_and_airport(self) -> None:
        from_datetime = pd.Timestamp("2025-04-01T00:00:00")
        to_datetime = pd.Timestamp("2025-04-01T23:59:59")

        output_path = build_output_path(
            Path("data/adsb/catalogs"),
            from_datetime.to_pydatetime(),
            to_datetime.to_pydatetime(),
            ZoneInfo("America/Chicago"),
            "KDFW",
        )

        self.assertEqual(output_path.name, "2025-04-01_kdfw_authoritative_departures_and_arrivals.csv")

    def test_report_catalog_differences_shows_both_sides(self) -> None:
        derived = pd.DataFrame(
            [
                {"flight_id": "DER1abc001", "callsign": "DER1", "icao24": "abc001", "operation": "arrival"},
                {"flight_id": "DER2abc002", "callsign": "DER2", "icao24": "abc002", "operation": "departure"},
            ]
        )
        authoritative = pd.DataFrame(
            [
                {"flight_type": "arrival", "callsign": "DER1", "icao24": "abc001", "flight_id": "DER1abc001"},
                {"flight_type": "departure", "callsign": "AUTH1", "icao24": "def001", "flight_id": "AUTH1def001"},
            ]
        )

        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=100)

        _report_catalog_differences(derived, authoritative, console)
        output = buffer.getvalue()

        self.assertIn("Derived only", output)
        self.assertIn("Authoritative only", output)
        self.assertIn("DER2abc002", output)
        self.assertIn("AUTH1def001", output)

    def test_read_catalog_csv_handles_headerless_trino_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.csv"
            path.write_text(
                "arrival,AAL100,abc123,100\n"
                "departure,DAL200,def456,200\n",
                encoding="utf-8",
            )

            frame = _read_catalog_csv(path)

            self.assertEqual(frame.columns.tolist(), ["flight_type", "callsign", "icao24", "event_time"])
            self.assertEqual(frame["callsign"].tolist(), ["AAL100", "DAL200"])

    def test_download_authoritative_catalog_uses_single_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "catalogs"
            output_dir.mkdir()
            output_path = output_dir / "authoritative.csv"
            inside_ts = int(pd.Timestamp("2025-04-01T05:00:59Z").timestamp())
            outside_ts = int(pd.Timestamp("2025-03-31T23:00:00Z").timestamp())
            departure_ts = int(pd.Timestamp("2025-04-01T12:00:00Z").timestamp())

            def fake_run_query(query: str, path: Path, trino_bin: str, token: str) -> None:
                self.assertIn("day BETWEEN", query)
                self.assertIn("estarrivalairport", query)
                self.assertIn("estdepartureairport", query)
                path.write_text(
                    "flight_type,callsign,icao24,event_time\n"
                    f"arrival,AAL100,abc123,{inside_ts}\n"
                    f"arrival,OUTSIDE1,abc999,{outside_ts}\n"
                    f"departure,DAL200,def456,{departure_ts}\n",
                    encoding="utf-8",
                )

            with patch(
                "scenario.demand_opensky.adsb_authoritative_departures_and_arrivals_catalog.get_jwt",
                return_value="token",
            ), patch(
                "scenario.demand_opensky.adsb_authoritative_departures_and_arrivals_catalog.run_query",
                side_effect=fake_run_query,
            ) as run_query_mock:
                result_path = download_authoritative_departures_and_arrivals_catalog(
                    from_datetime=pd.Timestamp("2025-04-01T00:00:00", tz="America/Chicago").to_pydatetime(),
                    to_datetime=pd.Timestamp("2025-04-01T23:59:59", tz="America/Chicago").to_pydatetime(),
                    timezone=ZoneInfo("America/Chicago"),
                    output_dir=output_dir,
                    output_path=output_path,
                    console=Console(file=io.StringIO(), force_terminal=True, width=100),
                )

            self.assertEqual(result_path, output_path)
            self.assertEqual(run_query_mock.call_count, 1)
            written = pd.read_csv(output_path)
            self.assertEqual(written["flight_id"].tolist(), ["AAL100abc123", "DAL200def456"])
            self.assertNotIn("OUTSIDE1abc999", written["flight_id"].tolist())
            self.assertIn("event_time_utc", written.columns)


if __name__ == "__main__":
    unittest.main()
