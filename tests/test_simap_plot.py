from __future__ import annotations

import os
import unittest
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
matplotlib.use("Agg", force=True)

from simap.longitudinal_profiles import ConstraintEnvelope
from simap.simap_plot import (
    plot_altitude_response,
    plot_cas_response,
    plot_constraint_envelope,
    plot_gamma_response,
    plot_longitudinal_plan,
    plot_tas_response,
    plot_thrust_response,
)


@dataclass(frozen=True)
class DummyPlan:
    s_m: np.ndarray
    h_m: np.ndarray
    v_tas_mps: np.ndarray
    v_cas_mps: np.ndarray
    t_s: np.ndarray
    gamma_rad: np.ndarray
    thrust_n: np.ndarray


class SimapPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = DummyPlan(
            s_m=np.asarray([0.0, 10_000.0, 20_000.0, 30_000.0], dtype=float),
            h_m=np.asarray([450.0, 900.0, 1_700.0, 3_000.0], dtype=float),
            v_tas_mps=np.asarray([72.0, 78.0, 84.0, 92.0], dtype=float),
            v_cas_mps=np.asarray([70.0, 74.0, 80.0, 92.0], dtype=float),
            t_s=np.asarray([0.0, 130.0, 270.0, 430.0], dtype=float),
            gamma_rad=np.asarray([-0.05, -0.045, -0.03, 0.0], dtype=float),
            thrust_n=np.asarray([2_500.0, 4_000.0, 5_500.0, 7_000.0], dtype=float),
        )
        self.envelope = ConstraintEnvelope(
            s_m=self.plan.s_m,
            h_lower_m=np.asarray([430.0, 850.0, 1_650.0, 2_950.0], dtype=float),
            h_upper_m=np.asarray([500.0, 980.0, 1_850.0, 3_100.0], dtype=float),
            cas_lower_mps=np.asarray([68.0, 72.0, 78.0, 90.0], dtype=float),
            cas_upper_mps=np.asarray([74.0, 80.0, 86.0, 96.0], dtype=float),
            gamma_lower_rad=np.asarray([-0.06, -0.05, -0.04, -0.01], dtype=float),
            gamma_upper_rad=np.asarray([-0.03, -0.025, -0.015, 0.01], dtype=float),
            thrust_lower_n=np.asarray([2_000.0, 3_000.0, 4_500.0, 6_000.0], dtype=float),
            thrust_upper_n=np.asarray([4_000.0, 5_500.0, 7_000.0, 8_500.0], dtype=float),
        )

    def test_plan_overview_draws_four_panels(self) -> None:
        fig, axes = plot_longitudinal_plan(self.plan, envelope=self.envelope, show=False)
        self.assertEqual(axes.shape, (2, 2))
        self.assertEqual(axes[0, 0].get_title(), "Altitude")
        self.assertEqual(axes[0, 1].get_title(), "CAS")
        self.assertEqual(axes[1, 0].get_title(), "Flight-Path Angle")
        self.assertEqual(axes[1, 1].get_title(), "Thrust")
        plt.close(fig)

    def test_constraint_envelope_plot_draws_two_panels(self) -> None:
        fig, axes = plot_constraint_envelope(self.envelope, show=False)
        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_title(), "Altitude Envelope")
        self.assertEqual(axes[1].get_title(), "CAS Envelope")
        plt.close(fig)

    def test_individual_plot_helpers_accept_plan_and_envelope(self) -> None:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        plot_altitude_response(self.plan, envelope=self.envelope, ax=axes[0, 0])
        plot_tas_response(self.plan, ax=axes[0, 1])
        plot_cas_response(self.plan, envelope=self.envelope, ax=axes[1, 0])
        plot_gamma_response(self.plan, envelope=self.envelope, ax=axes[1, 1])
        plot_thrust_response(self.plan, envelope=self.envelope, ax=axes[2, 0])

        self.assertEqual(axes[0, 0].get_ylabel(), "h [m]")
        self.assertEqual(axes[0, 1].get_ylabel(), "v_tas [m/s]")
        self.assertEqual(axes[1, 0].get_ylabel(), "v_cas [m/s]")
        self.assertEqual(axes[1, 1].get_ylabel(), "gamma [deg]")
        self.assertEqual(axes[2, 0].get_ylabel(), "T [N]")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
