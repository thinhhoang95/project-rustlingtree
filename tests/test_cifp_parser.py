from __future__ import annotations

import warnings
import unittest
import tempfile
from pathlib import Path

import pandas as pd

from scenario.cifp_parser.airport_related_extractor import (
    FIX_COLUMNS,
    TRANSITION_COLUMNS,
    build_airport_procedure_dataframes,
)


FIXTURE_LINES = [
    "SUSAP KDFWK4CBELFR K40    W     N33533935W098402606                       E0038     NAR           BELFR                    696172403",
    "SUSAP KDFWK4CSPERA K40    W     N32465775W096594205                       E0028     NAR           SPERA                    697902403",
    "SUSAEAENRT   GEEKY K40    W     N32164632W099530109                       E0044     NAR           GEEKY                    359612403",
    "SUSAD        ABI   K4011370VTHW N32285279W099514843    N32285279W099514843E0100018092     NARABILENE                       248641810",
    "SUSAD        TTT   K4011310VDHW N32520898W097022581    N32520898W097022581E0060005362     NARMAVERICK                      258392010",
    "SUSAP KDFWK4GRW17C   0134001760 N32545654W097013351         +0144100562000058150IIFLQ3                                     719682401",
    "SUSAP KDFWK4EBOOVE74GEEKY 010GEEKYK4EA0E       IF                                             18000                        709792407",
    "SUSAP KDFWK4DCYOTE43ABI   010TTT  K4D 0V       IF                                             18000                        700632401",
    "SUSAP KDFWK4EJOVEM64BELFR 010BELFRK4PC0E       IF                                             18000                        711462407",
    "SUSAP KDFWK4DAKUNA94RW17C 020SPERAK4PC0E       CF PGO K4      2231164815720049D                                            698392012",
    "SUSAP KDFWK4DJACKY1TRW17B 010         0        VA                     1763        + 01006     18000                        703702401",
]

REPORT_DETAIL_LINES = [
    "SUSAD        TTT   K4011310VDHW N32520898W097022581    N32520898W097022581E0060005362     NARMAVERICK                      258392010",
    "SUSAP KDFWK4DCYOTE42ALL   010TTT  K4D 0VE      IF                                             18000       TTT   K4D        700622401",
    "SUSAP KDFWK4DJACKY1TRW17B 010         0        VA                     1763        + 01006     18000                        703702401",
]


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.txt"
    path.write_text("\n".join(FIXTURE_LINES) + "\n", encoding="utf-8")
    return path


def _write_report_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "report_fixture.txt"
    path.write_text("\n".join(REPORT_DETAIL_LINES) + "\n", encoding="utf-8")
    return path


def _row_by(df, identifier: str):
    rows = df.loc[df["identifier"] == identifier]
    if len(rows) != 1:
        raise AssertionError(rows)
    return rows.iloc[0]


class CifpParserTests(unittest.TestCase):
    def test_build_airport_procedure_dataframes_resolves_transitions_and_fixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cifp_path = _write_fixture(Path(tmpdir))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                transitions_df, fixes_df, summary = build_airport_procedure_dataframes(cifp_path, "KDFW")

                self.assertEqual(list(transitions_df.columns), TRANSITION_COLUMNS)
                self.assertEqual(list(fixes_df.columns), FIX_COLUMNS)
                self.assertEqual(summary.filter_hit_count, 8)
                self.assertEqual(summary.processed_row_count, 5)
                self.assertEqual(summary.fixes_added_count, 6)
                self.assertEqual(summary.unresolved_fix_reference_count, 1)
                self.assertEqual(summary.unresolved_fix_identifier_count, 1)
                self.assertEqual(summary.filter_parse_failure_count, 0)
                self.assertEqual(summary.deduplicated_fix_row_count, 0)
                self.assertEqual(
                    list(transitions_df["sid_star_approach_identifier"]),
                    ["BOOVE7", "CYOTE4", "JOVEM6", "AKUNA9", "JACKY1"],
                )

                boove7 = transitions_df.iloc[0]
                self.assertEqual(boove7["transition_identifier"], "GEEKY")
                self.assertEqual(boove7["fix_identifier"], "GEEKY")
                self.assertEqual(boove7["section_code"], "PE")

                cyote4 = transitions_df.iloc[1]
                self.assertEqual(cyote4["transition_identifier"], "ABI")
                self.assertEqual(cyote4["fix_identifier"], "TTT")
                self.assertEqual(cyote4["section_code"], "PD")

                jovem6 = transitions_df.iloc[2]
                self.assertEqual(jovem6["transition_identifier"], "BELFR")
                self.assertEqual(jovem6["fix_identifier"], "BELFR")

                akun9 = transitions_df.iloc[3]
                self.assertEqual(akun9["transition_identifier"], "RW17C")
                self.assertEqual(akun9["fix_identifier"], "SPERA")

                jacky1 = transitions_df.iloc[4]
                self.assertEqual(jacky1["transition_identifier"], "RW17B")
                self.assertTrue(pd.isna(jacky1["fix_identifier"]) or jacky1["fix_identifier"] == "")

                self.assertEqual(
                    set(fixes_df["identifier"]),
                    {"ABI", "BELFR", "GEEKY", "RW17C", "SPERA", "TTT"},
                )
                self.assertNotIn("RW17B", set(fixes_df["identifier"]))

                belfr = _row_by(fixes_df, "BELFR")
                self.assertEqual(belfr["fix_type"], "waypoint")
                self.assertEqual(belfr["source_section_code"], "PC")
                self.assertEqual(belfr["airport_icao"], "KDFW")
                self.assertEqual(belfr["latitude_raw"], "N33533935")
                self.assertEqual(belfr["longitude_raw"], "W098402606")
                self.assertAlmostEqual(belfr["latitude_deg"], 33.8942638889, places=9)
                self.assertAlmostEqual(belfr["longitude_deg"], -98.6739055556, places=9)

                geeky = _row_by(fixes_df, "GEEKY")
                self.assertEqual(geeky["fix_type"], "waypoint")
                self.assertEqual(geeky["source_section_code"], "EA")
                self.assertAlmostEqual(geeky["latitude_deg"], 32.2795333333, places=9)
                self.assertAlmostEqual(geeky["longitude_deg"], -99.8836361111, places=9)

                abi = _row_by(fixes_df, "ABI")
                self.assertEqual(abi["fix_type"], "vor")
                self.assertEqual(abi["source_section_code"], "D")
                self.assertEqual(abi["latitude_raw"], "N32285279")
                self.assertEqual(abi["longitude_raw"], "W099514843")

                ttt = _row_by(fixes_df, "TTT")
                self.assertEqual(ttt["fix_type"], "vor")
                self.assertEqual(ttt["source_section_code"], "D")
                self.assertAlmostEqual(ttt["latitude_deg"], 32.8691611111, places=9)
                self.assertAlmostEqual(ttt["longitude_deg"], -97.0405027778, places=9)

                rw17c = _row_by(fixes_df, "RW17C")
                self.assertEqual(rw17c["fix_type"], "runway")
                self.assertEqual(rw17c["source_section_code"], "PG")
                self.assertEqual(rw17c["airport_icao"], "KDFW")
                self.assertEqual(rw17c["elevation_raw"], "00562")
                self.assertEqual(rw17c["elevation_ft"], 562)
                self.assertAlmostEqual(rw17c["latitude_deg"], 32.9157055556, places=9)
                self.assertAlmostEqual(rw17c["longitude_deg"], -97.025975, places=9)

                spera = _row_by(fixes_df, "SPERA")
                self.assertEqual(spera["fix_type"], "waypoint")
                self.assertEqual(spera["source_section_code"], "PC")
                self.assertEqual(spera["airport_icao"], "KDFW")

                self.assertTrue(any("RW17B" in str(w.message) for w in caught))

    def test_report_includes_lookup_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cifp_path = _write_report_fixture(tmp_path)
            report_path = tmp_path / "report.txt"

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                build_airport_procedure_dataframes(
                    cifp_path,
                    "KDFW",
                    report_path=report_path,
                )

            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("- ALL (named): 1", report_text)
            self.assertIn("Looked up: airport waypoint lookup for KDFW/ALL", report_text)
            self.assertIn("Why it failed: no matching fix record was found for identifier ALL", report_text)
            self.assertIn("- RW17B (runway): 1", report_text)
            self.assertIn("Looked up: runway lookup for KDFW/RW17B", report_text)
            self.assertIn("Why it failed: no runway record matched KDFW/RW17B", report_text)


if __name__ == "__main__":
    unittest.main()
