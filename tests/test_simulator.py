from __future__ import annotations

import os
import unittest
from dataclasses import replace

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.config import AircraftConfig, ModeConfig, planned_cas_bounds_mps
from simap.coupled_descent_planner import (
    CoupledDescentPlanRequest,
    OptimizerConfig,
    ThresholdBoundary,
    UpstreamBoundary,
    _PlannerScale,
    _TrajectoryEvaluationCache,
    _SolverProfilingState,
    _inequality_constraints,
    _initial_guess,
    _initial_tod_guess,
    _unpack,
    plan_coupled_descent,
)
from simap.longitudinal_profiles import ConstraintEnvelope, ScalarProfile
from simap.path_geometry import ReferencePath
from simap.weather import ConstantWeather


class SmoothBackend:
    def drag_newtons(
        self,
        *,
        mode: ModeConfig,
        mass_kg: float,
        wing_area_m2: float,
        v_tas_mps: float,
        h_m: float,
        gamma_rad: float = 0.0,
        bank_rad: float = 0.0,
        delta_isa_K: float = 0.0,
    ) -> float:
        del mode, mass_kg, wing_area_m2, bank_rad, delta_isa_K
        return float(3_500.0 + 0.45 * v_tas_mps**2 + 0.02 * h_m + 250.0 * abs(gamma_rad))

    def idle_thrust_newtons(
        self,
        *,
        v_tas_mps: float,
        h_m: float,
        delta_isa_K: float = 0.0,
    ) -> float:
        del v_tas_mps, delta_isa_K
        return float(max(1_200.0, 2_500.0 - 0.08 * h_m))

    def thrust_bounds_newtons(
        self,
        *,
        mode: ModeConfig,
        v_tas_mps: float,
        h_m: float,
        delta_isa_K: float = 0.0,
    ) -> tuple[float, float]:
        idle = self.idle_thrust_newtons(v_tas_mps=v_tas_mps, h_m=h_m, delta_isa_K=delta_isa_K)
        phase_scale = {"clean": 1.0, "approach": 0.92, "final": 0.85}[mode.name]
        return idle, float(25_000.0 * phase_scale)


def build_test_request() -> CoupledDescentPlanRequest:
    mode = ModeConfig(
        name="clean",
        tau_v_s=20.0,
        vs_min_mps=-12.0,
        vs_max_mps=3.0,
        cd0=0.02,
        k=0.05,
        phi_comfort_max_rad=np.deg2rad(25.0),
        phi_procedure_max_rad=np.deg2rad(25.0),
        tau_phi_s=2.0,
        p_max_rps=np.deg2rad(5.0),
        vs_1g_ref_cas_mps=60.0,
        cas_min_mps=65.0,
        cas_max_mps=120.0,
    )
    cfg = AircraftConfig(
        typecode="TEST",
        engine_name="TEST-ENG",
        mass_kg=62_000.0,
        reference_mass_kg=62_000.0,
        wing_area_m2=122.0,
        vmo_kts=250.0,
        mmo=0.78,
        clean=mode,
        approach=ModeConfig(**{**mode.__dict__, "name": "approach", "cas_min_mps": 68.0, "cas_max_mps": 110.0}),
        final=ModeConfig(**{**mode.__dict__, "name": "final", "cas_min_mps": 70.0, "cas_max_mps": 95.0}),
    )
    threshold = ThresholdBoundary(h_m=450.0, cas_mps=70.0, gamma_rad=-np.deg2rad(3.0))
    upstream = UpstreamBoundary(h_m=3_000.0, cas_window_mps=(79.0, 105.0))
    max_s_m = 60_000.0
    envelope = ConstraintEnvelope.from_profiles(
        altitude_lower=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([threshold.h_m, upstream.h_m - 30.0], dtype=float),
        ),
        altitude_upper=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([threshold.h_m + 40.0, upstream.h_m + 80.0], dtype=float),
        ),
        cas_lower=ScalarProfile(
            s_m=np.asarray([0.0, 20_000.0, max_s_m], dtype=float),
            y=np.asarray([70.0, 78.0, 88.0], dtype=float),
        ),
        cas_upper=ScalarProfile(
            s_m=np.asarray([0.0, 20_000.0, max_s_m], dtype=float),
            y=np.asarray([74.0, 92.0, 108.0], dtype=float),
        ),
        gamma_lower=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([-np.deg2rad(4.5), -np.deg2rad(0.5)], dtype=float),
        ),
        gamma_upper=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([-np.deg2rad(2.0), np.deg2rad(0.25)], dtype=float),
        ),
        cl_max=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([1.6, 1.6], dtype=float),
        ),
    )
    reference_path = ReferencePath.from_geographic(
        lat_deg=np.asarray([0.0, 0.0], dtype=float),
        lon_deg=np.asarray([0.70, 0.0], dtype=float),
    )
    return CoupledDescentPlanRequest(
        cfg=cfg,
        perf=SmoothBackend(),
        threshold=threshold,
        upstream=upstream,
        constraints=envelope,
        reference_path=reference_path,
        weather=ConstantWeather(),
        optimizer=OptimizerConfig(num_nodes=9, maxiter=80, verbose=0),
        initial_tod_guess_m=48_000.0,
    )


class LongitudinalPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.request = build_test_request()
        cls.plan = plan_coupled_descent(cls.request)

    def test_mode_gate_selection_is_deterministic(self) -> None:
        cfg = self.request.cfg
        self.assertEqual(planned_cas_bounds_mps(cfg, cfg.approach_gate_m + 1.0)[1], cfg.clean.cas_max_mps)
        self.assertEqual(planned_cas_bounds_mps(cfg, cfg.final_gate_m - 1.0)[1], cfg.final.cas_max_mps)

    def test_solver_returns_finite_endpoint_consistent_plan(self) -> None:
        plan = self.plan
        request = self.request

        self.assertAlmostEqual(plan.h_m[0], request.threshold.h_m, delta=1e-2)
        self.assertAlmostEqual(plan.gamma_rad[0], request.threshold.gamma_rad, delta=1e-4)
        self.assertAlmostEqual(plan.cross_track_m[0], request.threshold_lateral.cross_track_m, delta=1e-2)
        self.assertAlmostEqual(plan.heading_error_rad[0], request.threshold_lateral.heading_error_rad, delta=1e-4)
        self.assertAlmostEqual(plan.phi_rad[0], request.threshold_lateral.bank_rad, delta=1e-4)
        self.assertAlmostEqual(plan.h_m[-1], request.upstream.h_m, delta=1e-2)
        self.assertAlmostEqual(plan.gamma_rad[-1], request.upstream.gamma_rad, delta=1e-3)
        self.assertAlmostEqual(plan.cross_track_m[-1], request.upstream_lateral.cross_track_m, delta=1e-2)
        self.assertAlmostEqual(plan.heading_error_rad[-1], request.upstream_lateral.heading_error_rad, delta=1e-4)
        self.assertAlmostEqual(plan.phi_rad[-1], request.upstream_lateral.bank_rad, delta=1e-4)
        self.assertEqual(len(plan.s_m), self.request.optimizer.num_nodes)
        self.assertTrue(np.all(np.isfinite(plan.h_m)))
        self.assertTrue(np.all(np.isfinite(plan.v_tas_mps)))
        self.assertTrue(np.all(np.isfinite(plan.v_cas_mps)))
        self.assertTrue(np.all(np.isfinite(plan.gamma_rad)))
        self.assertTrue(np.all(np.isfinite(plan.thrust_n)))
        self.assertTrue(np.all(np.isfinite(plan.cross_track_m)))
        self.assertTrue(np.all(np.isfinite(plan.heading_error_rad)))
        self.assertTrue(np.all(np.isfinite(plan.phi_rad)))
        self.assertTrue(np.all(np.isfinite(plan.roll_rate_rps)))
        self.assertTrue(np.all(plan.v_tas_mps > 0.0))
        self.assertTrue(np.all(plan.v_cas_mps > 0.0))
        self.assertGreater(np.min(plan.alongtrack_speed_mps), 0.0)
        self.assertLess(plan.constraint_slack, 5.0)

    def test_exact_upstream_cas_boundary_is_enforced(self) -> None:
        request = replace(
            self.request,
            upstream=replace(self.request.upstream, cas_window_mps=(95.0, 95.0)),
            optimizer=replace(self.request.optimizer, maxiter=100),
        )
        plan = plan_coupled_descent(request)

        self.assertAlmostEqual(plan.v_cas_mps[-1], 95.0, delta=1e-4)

    def test_solver_is_deterministic_for_fixed_boundaries(self) -> None:
        plan_a = self.plan
        plan_b = plan_coupled_descent(self.request)

        self.assertAlmostEqual(plan_a.tod_m, plan_b.tod_m, delta=1e-6)
        self.assertTrue(np.allclose(plan_a.h_m, plan_b.h_m))
        self.assertTrue(np.allclose(plan_a.v_tas_mps, plan_b.v_tas_mps))
        self.assertTrue(np.allclose(plan_a.cross_track_m, plan_b.cross_track_m))

    def test_replay_and_collocation_diagnostics_are_reported(self) -> None:
        plan = self.plan

        self.assertTrue(np.isfinite(plan.collocation_residual_max))
        self.assertTrue(np.isfinite(plan.replay_h_error_m))
        self.assertTrue(np.isfinite(plan.replay_v_error_mps))
        self.assertTrue(np.isfinite(plan.replay_t_error_s))
        self.assertTrue(np.isfinite(plan.replay_residual_max))
        self.assertGreaterEqual(plan.collocation_residual_max, 0.0)
        self.assertGreaterEqual(plan.replay_residual_max, 0.0)

    def test_solver_profile_is_reported(self) -> None:
        profile = self.plan.solve_profile

        self.assertGreater(profile.total_wall_time_s, 0.0)
        self.assertGreaterEqual(profile.postprocess_wall_time_s, 0.0)
        self.assertGreater(profile.objective_calls, 0)
        self.assertGreater(profile.equality_calls, 0)
        self.assertGreater(profile.inequality_calls, 0)
        self.assertGreater(profile.objective_time_s, 0.0)
        self.assertGreater(profile.equality_time_s, 0.0)
        self.assertGreater(profile.inequality_time_s, 0.0)
        self.assertGreater(profile.trajectory_evaluations, 0)

    def test_idle_thrust_margin_initial_guess_starts_inside_band(self) -> None:
        request = replace(
            self.request,
            optimizer=replace(self.request.optimizer, idle_thrust_margin_fraction=0.03),
        )
        scale = _PlannerScale.from_request(request)
        threshold_v_tas = 70.0
        max_tod_m = float(min(request.constraints.s_m[-1], request.reference_path.total_length_m))
        z0 = _initial_guess(
            request=request,
            threshold_v_tas=threshold_v_tas,
            initial_tod_guess_m=_initial_tod_guess(
                request,
                threshold_v_tas=threshold_v_tas,
                max_tod_m=max_tod_m,
            ),
            scale=scale,
        )
        _, _, _, _, _, _, _, thrust_n, _, tod_m, _ = _unpack(z0, request.optimizer.num_nodes)
        evaluation = _TrajectoryEvaluationCache(
            request=request,
            profiling=_SolverProfilingState(),
        ).evaluate(z0)

        thrust_lower, thrust_upper = evaluation.thrust_bounds_backend
        idle_upper = evaluation.idle_thrust_n + 0.03 * np.maximum(thrust_upper - evaluation.idle_thrust_n, 0.0)
        self.assertAlmostEqual(tod_m, evaluation.tod_m)
        self.assertTrue(np.all(thrust_n >= evaluation.idle_thrust_n - 1e-9))
        self.assertTrue(np.all(thrust_n <= idle_upper + 1e-9))
        self.assertTrue(np.all(thrust_n >= thrust_lower - 1e-9))

    def test_idle_thrust_margin_is_not_relaxed_by_generic_slack(self) -> None:
        request = replace(
            self.request,
            optimizer=replace(self.request.optimizer, idle_thrust_margin_fraction=0.03),
        )
        scale = _PlannerScale.from_request(request)
        threshold_v_tas = 70.0
        max_tod_m = float(min(request.constraints.s_m[-1], request.reference_path.total_length_m))
        z0 = _initial_guess(
            request=request,
            threshold_v_tas=threshold_v_tas,
            initial_tod_guess_m=_initial_tod_guess(
                request,
                threshold_v_tas=threshold_v_tas,
                max_tod_m=max_tod_m,
            ),
            scale=scale,
        )
        z_bad = np.array(z0, dtype=float, copy=True)
        num_nodes = request.optimizer.num_nodes
        thrust_start = 7 * num_nodes
        slack_col = 9 * num_nodes + 1
        cache = _TrajectoryEvaluationCache(request=request, profiling=_SolverProfilingState())
        evaluation = cache.evaluate(z0)
        z_bad[thrust_start] = evaluation.idle_thrust_n[0] - 10.0
        z_bad[thrust_start + 1] = evaluation.idle_thrust_n[1] + 0.03 * (
            evaluation.thrust_bounds_backend[1][1] - evaluation.idle_thrust_n[1]
        ) + 10.0
        z_bad[slack_col] = 5.0

        residual = _inequality_constraints(
            z_bad,
            request=request,
            scale=scale,
            evaluation_cache=_TrajectoryEvaluationCache(request=request, profiling=_SolverProfilingState()),
        )
        idle_lower_start = 11 * num_nodes
        idle_upper_start = idle_lower_start + num_nodes

        self.assertLess(residual[idle_lower_start], 0.0)
        self.assertLess(residual[idle_upper_start + 1], 0.0)


if __name__ == "__main__":
    unittest.main()
