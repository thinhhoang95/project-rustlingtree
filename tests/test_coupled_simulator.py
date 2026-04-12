from __future__ import annotations

import os
import unittest

import numpy as np
from openap import aero

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap import (
    ApproachSimulator,
    ConstantWeather,
    Scenario,
    State,
    build_simple_glidepath,
    build_speed_schedule_from_wrap,
)
from tests.helpers import a320_fixture


class CoupledSimulatorTests(unittest.TestCase):
    def test_turning_path_produces_banked_flight_while_progress_stays_monotone(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        openap = fixture["openap"]
        reference_path = fixture["reference_path"]
        intercept_distance_m = fixture["intercept_distance_m"]
        intercept_altitude_m = fixture["intercept_altitude_m"]

        scenario = Scenario(
            altitude_profile=build_simple_glidepath(
                threshold_elevation_m=450.0,
                intercept_distance_m=intercept_distance_m,
                intercept_altitude_m=intercept_altitude_m,
                glide_deg=3.0,
            ),
            raw_speed_schedule_cas=build_speed_schedule_from_wrap(openap.wrap),
            reference_path=reference_path,
            weather=ConstantWeather(),
        )
        simulator = ApproachSimulator(cfg=cfg, perf=perf, scenario=scenario)
        v0_cas_mps = simulator.feasible_speed_schedule_cas.value(intercept_distance_m)
        initial = State.on_reference_path(
            t_s=0.0,
            s_m=intercept_distance_m,
            h_m=intercept_altitude_m,
            v_tas_mps=float(aero.cas2tas(v0_cas_mps, intercept_altitude_m, dT=0.0)),
            reference_path=reference_path,
        )
        trajectory = simulator.run(initial, dt_s=1.0, t_max_s=3_000.0)

        self.assertGreater(len(trajectory), 100)
        self.assertTrue((trajectory.s_m[1:] <= trajectory.s_m[:-1]).all())
        self.assertGreater(np.max(np.abs(trajectory.bank_rad)), np.deg2rad(1.0))
        self.assertLess(np.max(np.abs(trajectory.bank_rad)), cfg.clean.phi_comfort_max_rad + 1e-6)

    def test_crosswind_tracking_crabs_heading_into_the_wind(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        openap = fixture["openap"]
        intercept_altitude_m = fixture["intercept_altitude_m"]
        reference_path = fixture["reference_path"]
        intercept_distance_m = reference_path.total_length_m

        straight_path = type(reference_path).from_geographic(
            lat_deg=np.asarray([48.7600, 48.3538], dtype=float),
            lon_deg=np.asarray([11.7861, 11.7861], dtype=float),
        )
        scenario = Scenario(
            altitude_profile=build_simple_glidepath(
                threshold_elevation_m=450.0,
                intercept_distance_m=intercept_distance_m,
                intercept_altitude_m=intercept_altitude_m,
                glide_deg=3.0,
            ),
            raw_speed_schedule_cas=build_speed_schedule_from_wrap(openap.wrap),
            reference_path=straight_path,
            weather=ConstantWeather(wind_east_mps=20.0),
        )
        simulator = ApproachSimulator(cfg=cfg, perf=perf, scenario=scenario)
        v0_cas_mps = simulator.feasible_speed_schedule_cas.value(intercept_distance_m)
        initial = State.on_reference_path(
            t_s=0.0,
            s_m=intercept_distance_m,
            h_m=intercept_altitude_m,
            v_tas_mps=float(aero.cas2tas(v0_cas_mps, intercept_altitude_m, dT=0.0)),
            reference_path=straight_path,
        )
        trajectory = simulator.run(initial, dt_s=1.0, t_max_s=1_500.0)

        final_heading_rad = float(trajectory.heading_rad[-1])
        final_track_rad = straight_path.track_angle_rad(float(trajectory.s_m[-1]))
        self.assertGreater(abs(final_heading_rad - final_track_rad), np.deg2rad(2.0))
        self.assertLess(abs(trajectory.cross_track_m[-1]), 200.0)


if __name__ == "__main__":
    unittest.main()
