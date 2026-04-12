from __future__ import annotations

import os
import unittest

import numpy as np
from openap import aero

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.longitudinal_dynamics import LongitudinalState, longitudinal_rhs
from simap.longitudinal_profiles import (
    FeasibilityConfig,
    ScalarProfile,
    build_feasible_cas_schedule,
    build_simple_glidepath,
)
from simap.weather import ConstantWeather
from tests.helpers import a320_fixture


class ScalarProfileTests(unittest.TestCase):
    def test_value_and_slope_interpolate_between_nodes(self) -> None:
        profile = ScalarProfile(
            s_m=np.asarray([0.0, 10.0, 20.0], dtype=float),
            y=np.asarray([0.0, 10.0, 30.0], dtype=float),
        )
        self.assertAlmostEqual(profile.value(5.0), 5.0)
        self.assertAlmostEqual(profile.value(15.0), 20.0)
        self.assertAlmostEqual(profile.slope(5.0), 1.0)
        self.assertAlmostEqual(profile.slope(15.0), 2.0)

    def test_simple_glidepath_hits_threshold_and_caps_at_intercept_altitude(self) -> None:
        profile = build_simple_glidepath(
            threshold_elevation_m=450.0,
            intercept_distance_m=55_000.0,
            intercept_altitude_m=3_500.0,
            glide_deg=3.0,
            n=20,
        )
        self.assertAlmostEqual(profile.value(0.0), 450.0)
        self.assertLessEqual(profile.value(55_000.0), 3_500.0)
        self.assertGreater(profile.slope(20_000.0), 0.0)

    def test_feasible_schedule_respects_mode_cas_limits(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        altitude_profile = build_simple_glidepath(
            threshold_elevation_m=450.0,
            intercept_distance_m=60_000.0,
            intercept_altitude_m=3_500.0,
        )
        raw_schedule = ScalarProfile(
            s_m=np.asarray([0.0, 20_000.0, 60_000.0], dtype=float),
            y=np.asarray([10.0, 200.0, 220.0], dtype=float),
        )

        feasible = build_feasible_cas_schedule(
            raw_speed_schedule_cas=raw_schedule,
            altitude_profile=altitude_profile,
            cfg=cfg,
            perf=perf,
            feasibility=FeasibilityConfig(distance_step_m=250.0),
        )

        self.assertAlmostEqual(feasible.value(0.0), cfg.final.cas_min_mps)
        self.assertLessEqual(feasible.value(cfg.final_gate_m + 1_000.0), cfg.approach.cas_max_mps)
        self.assertLessEqual(feasible.value(50_000.0), cfg.clean.cas_max_mps)

    def test_longitudinal_rhs_clamps_schedule_to_mode_limits(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        altitude_profile = ScalarProfile(
            s_m=np.asarray([0.0, 20_000.0], dtype=float),
            y=np.asarray([450.0, 450.0], dtype=float),
        )
        raw_schedule = ScalarProfile(
            s_m=np.asarray([0.0, 20_000.0], dtype=float),
            y=np.asarray([10.0, 10.0], dtype=float),
        )
        h_m = 450.0
        v_tas_mps = float(aero.cas2tas(cfg.final.cas_min_mps, h_m, dT=0.0))
        state = LongitudinalState(t_s=0.0, s_m=1_000.0, h_m=h_m, v_tas_mps=v_tas_mps)

        rates = longitudinal_rhs(
            state=state,
            cfg=cfg,
            perf=perf,
            altitude_profile=altitude_profile,
            speed_schedule_cas=raw_schedule,
            weather=ConstantWeather(),
        )

        self.assertGreaterEqual(float(rates[2]), -1e-9)


if __name__ == "__main__":
    unittest.main()
