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
    plot_initial_profiles,
    plot_phi_response,
    plot_psi_response,
    plot_state_overview,
    plot_trajectory_map_scrubber,
)
from simap.longitudinal_profiles import ScalarProfile
from simap.path_geometry import ReferencePath


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
    v_tas_mps: np.ndarray
    v_ref_tas_mps: np.ndarray
    vdot_cmd_mps2: np.ndarray
    vdot_mps2: np.ndarray
    v_cas_mps: np.ndarray | None = None
    v_ref_cas_mps: np.ndarray | None = None


@dataclass(frozen=True)
class DummyLegacyAngleTrajectory:
    t_s: np.ndarray
    psi_rad: np.ndarray
    phi_rad: np.ndarray


class SimapPlotTests(unittest.TestCase):
    def test_plot_initial_profiles_overlays_feasible_cas(self) -> None:
        reference_path = ReferencePath.from_geographic(
            lat_deg=np.asarray([48.82, 48.58, 48.34], dtype=float),
            lon_deg=np.asarray([11.02, 11.40, 11.78], dtype=float),
        )
        s_nodes = np.asarray([0.0, 4_000.0, 8_000.0], dtype=float)
        altitude_profile = ScalarProfile(s_m=s_nodes, y=np.asarray([450.0, 700.0, 1_000.0], dtype=float))
        raw_speed_schedule_cas = ScalarProfile(s_m=s_nodes, y=np.asarray([62.0, 70.0, 76.0], dtype=float))
        feasible_speed_schedule_cas = ScalarProfile(s_m=s_nodes, y=np.asarray([60.0, 66.0, 71.0], dtype=float))

        fig, axes = plot_initial_profiles(
            reference_path=reference_path,
            altitude_profile=altitude_profile,
            raw_speed_schedule_cas=raw_speed_schedule_cas,
            feasible_speed_schedule_cas=feasible_speed_schedule_cas,
            show=False,
            add_features=False,
        )

        speed_ax = axes[2]
        labels_to_lines = {line.get_label(): line for line in speed_ax.get_lines()}
        self.assertIn("Raw CAS", labels_to_lines)
        self.assertIn("Feasible CAS", labels_to_lines)
        self.assertEqual(labels_to_lines["Feasible CAS"].get_linestyle(), "--")
        plt.close(fig)

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
            v_tas_mps=np.linspace(78.0, 64.0, n),
            v_ref_tas_mps=np.linspace(76.0, 62.0, n),
            vdot_cmd_mps2=np.linspace(-0.5, 0.1, n),
            vdot_mps2=np.linspace(-0.4, 0.1, n),
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
            v_tas_mps=np.linspace(75.0, 60.0, n),
            v_ref_tas_mps=np.linspace(74.0, 58.0, n),
            vdot_cmd_mps2=np.linspace(-0.2, 0.1, n),
            vdot_mps2=np.linspace(-0.15, 0.08, n),
            v_cas_mps=np.linspace(73.0, 58.0, n),
            v_ref_cas_mps=np.linspace(72.0, 56.0, n),
        )
        reference_path = ReferencePath.from_geographic(
            lat_deg=np.asarray([48.52, 48.42, 48.34], dtype=float),
            lon_deg=np.asarray([11.20, 11.48, 11.80], dtype=float),
        )

        fig, axes = plot_all_state_responses(trajectory, reference_path=reference_path, show=False)
        self.assertEqual(axes.shape, (4, 2))
        self.assertEqual(axes[0, 1].get_title(), "TAS")
        self.assertEqual(axes[1, 0].get_title(), "CAS")
        self.assertEqual(axes[1, 1].get_title(), "Altitude")
        self.assertEqual(axes[3, 0].name, "cartopy.geoaxes")
        self.assertIs(axes[3, 0], axes[3, 1])
        plt.close(fig)

    def test_plot_trajectory_map_scrubber_moves_marker(self) -> None:
        n = 12
        t_s = np.linspace(0.0, 110.0, n)
        trajectory = DummyTrajectory(
            t_s=t_s,
            s_m=np.linspace(12_000.0, 0.0, n),
            lat_deg=np.linspace(48.8, 48.35, n),
            lon_deg=np.linspace(11.0, 11.8, n),
            h_m=np.linspace(3_500.0, 450.0, n),
            h_ref_m=np.linspace(3_450.0, 450.0, n),
            heading_rad=np.linspace(0.0, 0.1, n),
            bank_rad=np.linspace(-0.05, 0.05, n),
            v_tas_mps=np.linspace(78.0, 62.0, n),
            v_ref_tas_mps=np.linspace(77.0, 60.0, n),
            vdot_cmd_mps2=np.linspace(-0.3, 0.1, n),
            vdot_mps2=np.linspace(-0.25, 0.08, n),
        )

        fig, ax, slider, aircraft = plot_trajectory_map_scrubber(
            trajectory,
            show=False,
            add_features=False,
        )

        self.assertEqual(ax.name, "cartopy.geoaxes")
        initial_offsets = aircraft.get_offsets().copy()
        initial_u = float(aircraft.U[0])
        initial_v = float(aircraft.V[0])
        slider.set_val(float(t_s[-1]))
        updated_offsets = aircraft.get_offsets().copy()

        self.assertNotEqual(tuple(initial_offsets[0]), tuple(updated_offsets[0]))
        self.assertAlmostEqual(float(updated_offsets[0][0]), float(trajectory.lon_deg[-1]), places=6)
        self.assertAlmostEqual(float(updated_offsets[0][1]), float(trajectory.lat_deg[-1]), places=6)
        self.assertNotEqual(initial_u, float(aircraft.U[0]))
        self.assertNotEqual(initial_v, float(aircraft.V[0]))

        plt.close(fig)

    def test_plot_trajectory_map_scrubber_draws_reference_path(self) -> None:
        n = 12
        t_s = np.linspace(0.0, 110.0, n)
        trajectory = DummyTrajectory(
            t_s=t_s,
            s_m=np.linspace(12_000.0, 0.0, n),
            lat_deg=np.linspace(48.8, 48.35, n),
            lon_deg=np.linspace(11.0, 11.8, n),
            h_m=np.linspace(3_500.0, 450.0, n),
            h_ref_m=np.linspace(3_450.0, 450.0, n),
            heading_rad=np.linspace(0.0, 0.1, n),
            bank_rad=np.linspace(-0.05, 0.05, n),
            v_tas_mps=np.linspace(78.0, 62.0, n),
            v_ref_tas_mps=np.linspace(77.0, 60.0, n),
            vdot_cmd_mps2=np.linspace(-0.3, 0.1, n),
            vdot_mps2=np.linspace(-0.25, 0.08, n),
        )
        reference_path = ReferencePath.from_geographic(
            lat_deg=np.asarray([48.82, 48.58, 48.34], dtype=float),
            lon_deg=np.asarray([11.02, 11.40, 11.78], dtype=float),
        )

        fig, ax, slider, aircraft = plot_trajectory_map_scrubber(
            trajectory,
            reference_path=reference_path,
            show=False,
            add_features=False,
        )

        status_texts = [text for text in ax.texts if text.get_text().startswith("Reference s:")]
        self.assertEqual(len(status_texts), 1)
        initial_status = status_texts[0].get_text()
        self.assertIn("TAS:", initial_status)
        self.assertIn("Altitude:", initial_status)
        self.assertIn("Cross-track error:", initial_status)
        self.assertIn("Track-error:", initial_status)
        self.assertIn("Heading:", initial_status)
        self.assertIn("Bank angle:", initial_status)

        reference_points = [line for line in ax.lines if line.get_marker() == "o"]
        self.assertEqual(len(reference_points), 1)
        initial_ref_lat, initial_ref_lon = reference_path.latlon(float(trajectory.s_m[0]))
        self.assertAlmostEqual(float(reference_points[0].get_xdata()[0]), initial_ref_lon, places=6)
        self.assertAlmostEqual(float(reference_points[0].get_ydata()[0]), initial_ref_lat, places=6)

        dashed_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        self.assertEqual(len(dashed_lines), 1)
        self.assertAlmostEqual(float(dashed_lines[0].get_xdata()[0]), float(reference_path.lon_deg[0]), places=6)
        self.assertAlmostEqual(float(dashed_lines[0].get_ydata()[0]), float(reference_path.lat_deg[0]), places=6)

        slider.set_val(float(t_s[-1]))
        offsets = aircraft.get_offsets()
        self.assertAlmostEqual(float(offsets[0][0]), float(trajectory.lon_deg[-1]), places=6)
        self.assertAlmostEqual(float(offsets[0][1]), float(trajectory.lat_deg[-1]), places=6)
        final_ref_lat, final_ref_lon = reference_path.latlon(float(trajectory.s_m[-1]))
        self.assertAlmostEqual(float(reference_points[0].get_xdata()[0]), final_ref_lon, places=6)
        self.assertAlmostEqual(float(reference_points[0].get_ydata()[0]), final_ref_lat, places=6)
        updated_status = status_texts[0].get_text()
        self.assertNotEqual(initial_status, updated_status)

        plt.close(fig)

    def test_plot_trajectory_map_scrubber_can_toggle_reference_turning_points(self) -> None:
        n = 12
        t_s = np.linspace(0.0, 110.0, n)
        trajectory = DummyTrajectory(
            t_s=t_s,
            s_m=np.linspace(12_000.0, 0.0, n),
            lat_deg=np.linspace(48.8, 48.35, n),
            lon_deg=np.linspace(11.0, 11.8, n),
            h_m=np.linspace(3_500.0, 450.0, n),
            h_ref_m=np.linspace(3_450.0, 450.0, n),
            heading_rad=np.linspace(0.0, 0.1, n),
            bank_rad=np.linspace(-0.05, 0.05, n),
            v_tas_mps=np.linspace(78.0, 62.0, n),
            v_ref_tas_mps=np.linspace(77.0, 60.0, n),
            vdot_cmd_mps2=np.linspace(-0.3, 0.1, n),
            vdot_mps2=np.linspace(-0.25, 0.08, n),
        )
        reference_path = ReferencePath.from_geographic(
            lat_deg=np.asarray([48.82, 48.58, 48.34], dtype=float),
            lon_deg=np.asarray([11.02, 11.40, 11.78], dtype=float),
        )

        fig, ax, _, _ = plot_trajectory_map_scrubber(
            trajectory,
            reference_path=reference_path,
            show_reference_turning_points=False,
            show=False,
            add_features=False,
        )
        waypoint_collections = [
            collection
            for collection in ax.collections
            if len(collection.get_offsets()) == len(reference_path.waypoint_lat_deg)
        ]
        self.assertEqual(len(waypoint_collections), 0)
        plt.close(fig)

        fig, ax, _, _ = plot_trajectory_map_scrubber(
            trajectory,
            reference_path=reference_path,
            show_reference_turning_points=True,
            show=False,
            add_features=False,
        )
        waypoint_collections = [
            collection
            for collection in ax.collections
            if len(collection.get_offsets()) == len(reference_path.waypoint_lat_deg)
        ]
        self.assertEqual(len(waypoint_collections), 1)
        offsets = waypoint_collections[0].get_offsets()
        self.assertEqual(len(offsets), len(reference_path.waypoint_lat_deg))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
