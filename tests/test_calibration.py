from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from openap import aero

from simap.calibration import build_default_aircraft_config
from simap.openap_adapter import extract_aircraft_data, load_openap
from tests.helpers import a320_fixture


class CalibrationTests(unittest.TestCase):
    def test_default_mode_coefficients_are_positive(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        self.assertGreater(cfg.clean.cd0, 0.0)
        self.assertGreater(cfg.clean.k, 0.0)
        self.assertGreater(cfg.approach.cd0, 0.0)
        self.assertGreater(cfg.approach.k, 0.0)
        self.assertGreater(cfg.final.cd0, 0.0)
        self.assertGreater(cfg.final.k, 0.0)

    def test_effective_backend_tracks_openap_nonclean_reference_samples(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        openap = fixture["openap"]

        approach_drag = perf.drag_newtons(
            mode=cfg.approach,
            mass_kg=cfg.mass_kg,
            wing_area_m2=cfg.wing_area_m2,
            v_tas_mps=150.0 * aero.kts,
            h_m=3_000.0 * aero.ft,
            vs_mps=-700.0 * aero.fpm,
        )
        approach_ref = float(
            openap.drag.nonclean(
                mass=cfg.mass_kg,
                tas=150.0,
                alt=3_000.0,
                flap_angle=15.0,
                vs=-700.0,
                landing_gear=False,
            )
        )
        final_drag = perf.drag_newtons(
            mode=cfg.final,
            mass_kg=cfg.mass_kg,
            wing_area_m2=cfg.wing_area_m2,
            v_tas_mps=135.0 * aero.kts,
            h_m=1_500.0 * aero.ft,
            vs_mps=-650.0 * aero.fpm,
        )
        final_ref = float(
            openap.drag.nonclean(
                mass=cfg.mass_kg,
                tas=135.0,
                alt=1_500.0,
                flap_angle=30.0,
                vs=-650.0,
                landing_gear=True,
            )
        )

        self.assertLess(abs(approach_drag - approach_ref) / approach_ref, 0.15)
        self.assertLess(abs(final_drag - final_ref) / final_ref, 0.15)

    def test_extract_aircraft_data_uses_selected_engine(self) -> None:
        openap = load_openap("A320", engine_name="V2527-A5")
        aircraft_data = extract_aircraft_data(openap)

        self.assertEqual(aircraft_data.engine_name, "V2527-A5")

    def test_build_default_aircraft_config_keeps_selected_engine_metadata(self) -> None:
        cfg, _ = build_default_aircraft_config("A320", mass_kg=65_000.0, engine_name="V2527-A5")

        self.assertEqual(cfg.engine_name, "V2527-A5")


if __name__ == "__main__":
    unittest.main()
