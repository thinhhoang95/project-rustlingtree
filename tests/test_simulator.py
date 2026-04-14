from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from openap import aero

from simap import (
    ConstantWeather,
    FeasibilityConfig,
    LongitudinalApproachSimulator,
    LongitudinalScenario,
    LongitudinalState,
    mode_for_s,
)
from tests.helpers import a320_fixture


class SimulatorTests(unittest.TestCase):
    def test_mode_gate_selection_is_deterministic(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        self.assertEqual(mode_for_s(cfg, cfg.approach_gate_m + 1.0).name, "clean")
        self.assertEqual(mode_for_s(cfg, cfg.approach_gate_m - 1.0).name, "approach")
        self.assertEqual(mode_for_s(cfg, cfg.final_gate_m - 1.0).name, "final")

    def test_tailwind_feasible_schedule_slows_earlier_than_headwind(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        scenario = fixture["longitudinal_scenario"]
        tailwind_sim = LongitudinalApproachSimulator(
            cfg=cfg,
            perf=perf,
            scenario=LongitudinalScenario(
                altitude_profile=scenario.altitude_profile,
                raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
                weather=scenario.weather,
                feasibility=FeasibilityConfig(planning_tailwind_mps=10.0, distance_step_m=250.0),
                reference_track_rad=scenario.reference_track_rad,
            ),
        )
        headwind_sim = LongitudinalApproachSimulator(
            cfg=cfg,
            perf=perf,
            scenario=LongitudinalScenario(
                altitude_profile=scenario.altitude_profile,
                raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
                weather=scenario.weather,
                feasibility=FeasibilityConfig(planning_tailwind_mps=-10.0, distance_step_m=250.0),
                reference_track_rad=scenario.reference_track_rad,
            ),
        )

        s_probe_m = 12_000.0
        self.assertLess(
            tailwind_sim.feasible_speed_schedule_cas.value(s_probe_m),
            headwind_sim.feasible_speed_schedule_cas.value(s_probe_m),
        )

    def test_runtime_weather_changes_groundspeed_immediately(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        scenario = fixture["longitudinal_scenario"]
        initial = LongitudinalState(
            t_s=0.0,
            s_m=fixture["intercept_distance_m"],
            h_m=fixture["intercept_altitude_m"],
            v_tas_mps=float(
                aero.cas2tas(
                    scenario.raw_speed_schedule_cas.value(fixture["intercept_distance_m"]),
                    fixture["intercept_altitude_m"],
                    dT=0.0,
                )
            ),
        )
        headwind_traj = LongitudinalApproachSimulator(
            cfg=cfg,
            perf=perf,
            scenario=LongitudinalScenario(
                altitude_profile=scenario.altitude_profile,
                raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
                weather=ConstantWeather(wind_east_mps=-10.0),
                feasibility=scenario.feasibility,
                reference_track_rad=scenario.reference_track_rad,
            ),
        ).run(initial, dt_s=1.0, t_max_s=1.0)
        tailwind_traj = LongitudinalApproachSimulator(
            cfg=cfg,
            perf=perf,
            scenario=LongitudinalScenario(
                altitude_profile=scenario.altitude_profile,
                raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
                weather=ConstantWeather(wind_east_mps=10.0),
                feasibility=scenario.feasibility,
                reference_track_rad=scenario.reference_track_rad,
            ),
        ).run(initial, dt_s=1.0, t_max_s=1.0)
        self.assertAlmostEqual(tailwind_traj.gs_mps[0] - headwind_traj.gs_mps[0], 20.0, places=6)

    def test_a320_smoke_run_produces_monotone_distance_and_dataframe_schema(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        scenario = fixture["longitudinal_scenario"]
        simulator = LongitudinalApproachSimulator(cfg=cfg, perf=perf, scenario=scenario)

        v0_cas_mps = simulator.feasible_speed_schedule_cas.value(fixture["intercept_distance_m"])
        initial = LongitudinalState(
            t_s=0.0,
            s_m=fixture["intercept_distance_m"],
            h_m=fixture["intercept_altitude_m"],
            v_tas_mps=float(aero.cas2tas(v0_cas_mps, fixture["intercept_altitude_m"], dT=0.0)),
        )
        trajectory = simulator.run(initial, dt_s=1.0, t_max_s=3_000.0)
        df = trajectory.to_pandas()

        self.assertGreater(len(trajectory), 100)
        self.assertTrue((trajectory.s_m[1:] <= trajectory.s_m[:-1]).all())
        self.assertTrue((trajectory.h_m >= 0.0).all())
        self.assertTrue((trajectory.v_tas_mps > 0.0).all())
        self.assertEqual(len(trajectory.v_ref_tas_mps), len(trajectory))
        self.assertEqual(len(trajectory.vdot_cmd_mps2), len(trajectory))
        self.assertEqual(len(trajectory.vdot_mps2), len(trajectory))
        self.assertIn("final", set(trajectory.mode))
        self.assertEqual(
            list(df.columns),
            [
                "t_s",
                "s_m",
                "h_m",
                "v_tas_mps",
                "v_cas_mps",
                "gs_mps",
                "h_ref_m",
                "v_ref_cas_mps",
                "v_ref_tas_mps",
                "vdot_cmd_mps2",
                "vdot_mps2",
                "mode",
            ],
        )


if __name__ == "__main__":
    unittest.main()
