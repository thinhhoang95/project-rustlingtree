from __future__ import annotations

import unittest
from pathlib import Path

from simap.units import ft_to_m
from simap.units import kts_to_mps
from simap import planned_cas_bounds_mps

from tactical import (
    AltitudeConstraint,
    TacticalCommand,
    TacticalCondition,
    build_tactical_plan_request,
    load_fix_catalog,
    load_procedure_altitude_constraints,
    parse_altitude_ft,
    resolve_lateral_path,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data/kdfw_procs"


class TacticalNavdataTests(unittest.TestCase):
    def test_parse_altitude_accepts_flight_levels_and_feet(self) -> None:
        self.assertEqual(parse_altitude_ft("FL260"), 26_000.0)
        self.assertEqual(parse_altitude_ft("08000"), 8_000.0)
        self.assertIsNone(parse_altitude_ft(""))

    def test_resolve_lateral_path_mixes_fixes_and_coordinates(self) -> None:
        catalog = load_fix_catalog(DATA_DIR / "airport_related_fixes.csv")
        path = resolve_lateral_path("JUSST 33.95,-95.75 RW17C", catalog)

        self.assertEqual(path.identifiers[0], "JUSST")
        self.assertEqual(path.identifiers[1], "COORD02")
        self.assertEqual(path.identifiers[2], "RW17C")

    def test_load_procedure_constraints_for_seevr4_common_route(self) -> None:
        constraints = load_procedure_altitude_constraints(
            DATA_DIR / "airport_related.csv",
            procedure_identifier="SEEVR4",
            transition_identifier="ALL",
        )

        by_fix = {constraint.fix_identifier: constraint for constraint in constraints}
        self.assertEqual(by_fix["BRDJE"], AltitudeConstraint("BRDJE", lower_ft=11_000.0, upper_ft=12_000.0))
        self.assertEqual(by_fix["NUSSS"], AltitudeConstraint("NUSSS", lower_ft=8_000.0, upper_ft=10_000.0))


class TacticalBuilderTests(unittest.TestCase):
    def test_build_request_from_kdfw_tactical_command(self) -> None:
        command = TacticalCommand(
            lateral_path="JUSST SWTCH THEMM RW17C",
            upstream=TacticalCondition("JUSST", cas_kts=290.0, altitude_ft=26_000.0),
            runway_altitude_ft=620.0,
        )
        bundle = build_tactical_plan_request(
            command,
            fixes_csv=DATA_DIR / "airport_related_fixes.csv",
        )

        self.assertEqual(bundle.path.identifiers, ("JUSST", "SWTCH", "THEMM", "RW17C"))
        self.assertGreater(bundle.request.reference_path.total_length_m, 0.0)
        self.assertGreaterEqual(bundle.request.constraints.s_m[-1], bundle.request.reference_path.total_length_m)
        _, upper_m = bundle.request.constraints.h_bounds(0.75 * bundle.request.reference_path.total_length_m)
        self.assertGreaterEqual(upper_m, ft_to_m(26_000.0))
        self.assertEqual(bundle.request.upstream.cas_window_mps, (kts_to_mps(290.0), kts_to_mps(290.0)))

    def test_tactical_speed_envelope_is_compatible_with_mode_limits(self) -> None:
        command = TacticalCommand(
            lateral_path="JUSST SWTCH THEMM TUSLE SEEVR BRDJE NUSSS YAHBT ZINGG RW17C",
            upstream=TacticalCondition("JUSST", cas_kts=290.0, altitude_ft=26_000.0),
            runway_altitude_ft=620.0,
        )
        bundle = build_tactical_plan_request(
            command,
            fixes_csv=DATA_DIR / "airport_related_fixes.csv",
        )

        for s_m in bundle.request.constraints.s_m:
            route_lower, route_upper = bundle.request.constraints.cas_bounds(float(s_m))
            mode_lower, mode_upper = planned_cas_bounds_mps(bundle.request.cfg, float(s_m))
            self.assertLessEqual(max(route_lower, mode_lower), min(route_upper, mode_upper))


if __name__ == "__main__":
    unittest.main()
