from __future__ import annotations

import os
import unittest
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
matplotlib.use("Agg", force=True)

from simap.simap_plot import (
    plot_all_state_responses,
    plot_phi_response,
    plot_psi_response,
    plot_state_overview,
)


@dataclass(frozen=True)
class DummyTrajectory:
    t_s: np.ndarray
    s_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    h_m: np.ndarray
    h_ref_m: np.ndarray
    heading_rad: np.ndarray
    bank_rad: np.ndarray


@dataclass(frozen=True)
class DummyLegacyAngleTrajectory:
    t_s: np.ndarray
    psi_rad: np.ndarray
    phi_rad: np.ndarray


class SimapPlotTests(unittest.TestCase):
    def test_plot_state_overview_draws_six_panels(self) -> None:
        n = 40
        t_s = np.linspace(0.0, 39.0, n)
        trajectory = DummyTrajectory(
            t_s=t_s,
            s_m=np.linspace(12_000.0, 0.0, n),
            lat_deg=np.linspace(48.8, 48.35, n),
            lon_deg=np.linspace(11.0, 11.8, n),
            h_m=np.linspace(3_500.0, 450.0, n),
            h_ref_m=np.linspace(3_450.0, 450.0, n),
            heading_rad=np.linspace(np.deg2rad(170.0), np.deg2rad(180.0), n),
            bank_rad=np.deg2rad(np.sin(np.linspace(0.0, 2.0 * np.pi, n)) * 10.0),
        )

        fig, axes = plot_state_overview(trajectory, show=False)
        self.assertEqual(axes.shape, (3, 2))
        self.assertEqual(fig.axes[0].get_title(), "Along-track Distance")
        self.assertEqual(fig.axes[5].get_title(), "Bank Angle (phi)")

        plt.close(fig)

    def test_angle_helpers_accept_legacy_psi_phi_fields(self) -> None:
        n = 20
        t_s = np.linspace(0.0, 19.0, n)
        trajectory = DummyLegacyAngleTrajectory(
            t_s=t_s,
            psi_rad=np.linspace(-0.2, 0.4, n),
            phi_rad=np.linspace(-0.1, 0.1, n),
        )

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        plot_psi_response(trajectory, ax=axes[0], in_degrees=False)
        plot_phi_response(trajectory, ax=axes[1], in_degrees=False)

        self.assertEqual(axes[0].get_ylabel(), "psi [rad]")
        self.assertEqual(axes[1].get_ylabel(), "phi [rad]")

        plt.close(fig)

    def test_plot_all_state_responses_alias(self) -> None:
        n = 10
        t_s = np.linspace(0.0, 9.0, n)
        trajectory = DummyTrajectory(
            t_s=t_s,
            s_m=np.linspace(3_000.0, 0.0, n),
            lat_deg=np.linspace(48.5, 48.35, n),
            lon_deg=np.linspace(11.2, 11.8, n),
            h_m=np.linspace(1_500.0, 450.0, n),
            h_ref_m=np.linspace(1_500.0, 450.0, n),
            heading_rad=np.linspace(0.0, 0.1, n),
            bank_rad=np.linspace(-0.05, 0.05, n),
        )

        fig, axes = plot_all_state_responses(trajectory, show=False)
        self.assertEqual(axes.shape, (3, 2))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
