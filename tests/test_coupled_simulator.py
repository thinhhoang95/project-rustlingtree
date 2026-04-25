from __future__ import annotations

import os
import unittest
from dataclasses import replace

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.coupled_descent_planner import LateralBoundary, OptimizerConfig, plan_coupled_descent
from simap.path_geometry import ReferencePath
from simap.simulator import SimulationRequest, State, simulate_plan
from tests.test_simulator import build_test_request


def _straight_reference_path() -> ReferencePath:
    return ReferencePath.from_geographic(
        lat_deg=np.asarray([0.0, 0.0], dtype=float),
        lon_deg=np.asarray([0.70, 0.0], dtype=float),
    )


class CoupledSimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan_request = build_test_request()
        cls.plan = plan_coupled_descent(cls.plan_request)
        cls.reference_path = _straight_reference_path()

    def test_simulator_replays_plan_on_straight_path(self) -> None:
        result = simulate_plan(
            SimulationRequest(
                cfg=self.plan_request.cfg,
                plan=self.plan,
                reference_path=self.reference_path,
                weather=self.plan_request.weather,
                dt_s=1.0,
                threshold_tolerance_m=0.0,
            )
        )

        self.assertTrue(result.success, msg=result.message)
        self.assertLessEqual(abs(result.s_m[-1]), 1e-9)
        self.assertLess(result.final_threshold_error_m, 1.0)
        self.assertLess(result.max_abs_cross_track_m, 1e-6)
        self.assertTrue(np.all(np.diff(result.s_m) <= 1e-9))
        self.assertTrue(np.all(np.diff(result.t_s) > 0.0))
        self.assertTrue(np.all(np.isfinite(result.h_m)))
        self.assertTrue(np.all(np.isfinite(result.v_tas_mps)))
        self.assertTrue(np.all(np.isfinite(result.cross_track_m)))
        self.assertGreater(result.min_alongtrack_speed_mps, 0.0)

    def test_simulator_recovers_from_cross_track_offset(self) -> None:
        start_sample = self.plan.s_m[-1]
        initial_state = State.on_reference_path(
            t_s=0.0,
            s_m=float(start_sample),
            h_m=float(self.plan.h_m[-1]),
            v_tas_mps=float(self.plan.v_tas_mps[-1]),
            reference_path=self.reference_path,
            cross_track_m=150.0,
        )
        result = simulate_plan(
            SimulationRequest(
                cfg=self.plan_request.cfg,
                plan=self.plan,
                reference_path=self.reference_path,
                weather=self.plan_request.weather,
                dt_s=1.0,
                threshold_tolerance_m=0.0,
                initial_state=initial_state,
            )
        )

        self.assertTrue(result.success, msg=result.message)
        self.assertGreater(abs(result.cross_track_m[0]), 100.0)
        self.assertLess(abs(result.cross_track_m[-1]), 5.0)
        self.assertLess(result.final_threshold_error_m, 5.0)

    def test_joint_planner_recovers_from_upstream_cross_track_offset(self) -> None:
        request = replace(
            self.plan_request,
            upstream_lateral=LateralBoundary(cross_track_m=150.0),
            optimizer=OptimizerConfig(num_nodes=9, maxiter=120, verbose=0),
        )
        plan = plan_coupled_descent(request)

        self.assertAlmostEqual(plan.cross_track_m[-1], 150.0, delta=1e-2)
        self.assertAlmostEqual(plan.cross_track_m[0], 0.0, delta=1e-2)
        self.assertGreater(np.max(np.abs(plan.phi_rad)), 1e-5)
        self.assertTrue(np.all(np.abs(plan.phi_rad) <= plan.phi_max_rad + 1e-6))
        self.assertGreater(np.min(plan.alongtrack_speed_mps), 0.0)

    def test_simulator_rejects_reference_path_shorter_than_tod(self) -> None:
        short_path = ReferencePath.from_geographic(
            lat_deg=np.asarray([0.0, 0.0], dtype=float),
            lon_deg=np.asarray([0.20, 0.0], dtype=float),
        )

        with self.assertRaisesRegex(ValueError, "top of descent"):
            SimulationRequest(
                cfg=self.plan_request.cfg,
                plan=self.plan,
                reference_path=short_path,
            )


if __name__ == "__main__":
    unittest.main()
