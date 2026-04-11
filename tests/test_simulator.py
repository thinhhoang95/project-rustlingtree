from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from openap import aero

from simap import ApproachSimulator, ConstantWeather, FeasibilityConfig, Scenario, State, mode_for_x
from tests.helpers import a320_fixture


class SimulatorTests(unittest.TestCase):
    def test_mode_gate_selection_is_deterministic(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        self.assertEqual(mode_for_x(cfg, cfg.approach_gate_m + 1.0).name, "clean")
        self.assertEqual(mode_for_x(cfg, cfg.approach_gate_m - 1.0).name, "approach")
        self.assertEqual(mode_for_x(cfg, cfg.final_gate_m - 1.0).name, "final")

    def test_tailwind_feasible_schedule_slows_earlier_than_headwind(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        scenario = fixture["scenario"]
        tailwind_sim = ApproachSimulator(
            cfg=cfg,
            perf=perf,
            scenario=Scenario(
                altitude_profile=scenario.altitude_profile,
                raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
                centerline=scenario.centerline,
                weather=scenario.weather,
                feasibility=FeasibilityConfig(planning_tailwind_mps=10.0, distance_step_m=250.0),
            ),
        )
        headwind_sim = ApproachSimulator(
            cfg=cfg,
            perf=perf,
            scenario=Scenario(
                altitude_profile=scenario.altitude_profile,
                raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
                centerline=scenario.centerline,
                weather=scenario.weather,
                feasibility=FeasibilityConfig(planning_tailwind_mps=-10.0, distance_step_m=250.0),
            ),
        )

        x_probe_m = 30_000.0
        self.assertLess(
            tailwind_sim.feasible_speed_schedule_cas.value(x_probe_m),
            headwind_sim.feasible_speed_schedule_cas.value(x_probe_m),
        )

    def test_runtime_weather_changes_groundspeed_immediately(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        scenario = fixture["scenario"]
        initial = State(
            t_s=0.0,
            x_m=fixture["intercept_distance_m"],
            h_m=fixture["intercept_altitude_m"],
            v_tas_mps=float(
                aero.cas2tas(
                    scenario.raw_speed_schedule_cas.value(fixture["intercept_distance_m"]),
                    fixture["intercept_altitude_m"],
                    dT=0.0,
                )
            ),
        )
        headwind_traj = ApproachSimulator(
            cfg=cfg,
            perf=perf,
            scenario=Scenario(
                altitude_profile=scenario.altitude_profile,
                raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
                centerline=scenario.centerline,
                weather=ConstantWeather(alongtrack_wind_mps=-10.0),
                feasibility=scenario.feasibility,
            ),
        ).run(initial, dt_s=1.0, t_max_s=1.0)
        tailwind_traj = ApproachSimulator(
            cfg=cfg,
            perf=perf,
            scenario=Scenario(
                altitude_profile=scenario.altitude_profile,
                raw_speed_schedule_cas=scenario.raw_speed_schedule_cas,
                centerline=scenario.centerline,
                weather=ConstantWeather(alongtrack_wind_mps=10.0),
                feasibility=scenario.feasibility,
            ),
        ).run(initial, dt_s=1.0, t_max_s=1.0)
        self.assertAlmostEqual(tailwind_traj.gs_mps[0] - headwind_traj.gs_mps[0], 20.0, places=6)

    def test_a320_smoke_run_produces_monotone_distance_and_dataframe_schema(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        perf = fixture["perf"]
        scenario = fixture["scenario"]
        simulator = ApproachSimulator(cfg=cfg, perf=perf, scenario=scenario)

        v0_cas_mps = simulator.feasible_speed_schedule_cas.value(fixture["intercept_distance_m"])
        initial = State(
            t_s=0.0,
            x_m=fixture["intercept_distance_m"],
            h_m=fixture["intercept_altitude_m"],
            v_tas_mps=float(aero.cas2tas(v0_cas_mps, fixture["intercept_altitude_m"], dT=0.0)),
        )
        trajectory = simulator.run(initial, dt_s=1.0, t_max_s=3_000.0)
        df = trajectory.to_pandas()

        self.assertGreater(len(trajectory), 100)
        self.assertTrue((trajectory.x_m[1:] <= trajectory.x_m[:-1]).all())
        self.assertTrue((trajectory.h_m >= 0.0).all())
        self.assertTrue((trajectory.v_tas_mps > 0.0).all())
        self.assertIn("final", set(trajectory.mode))
        self.assertEqual(
            list(df.columns),
            [
                "t_s",
                "x_m",
                "h_m",
                "v_tas_mps",
                "v_cas_mps",
                "gs_mps",
                "h_ref_m",
                "v_ref_cas_mps",
                "mode",
                "lat_deg",
                "lon_deg",
            ],
        )


if __name__ == "__main__":
    unittest.main()
