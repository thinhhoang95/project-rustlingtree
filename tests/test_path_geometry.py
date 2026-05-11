from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.path_geometry import EARTH_RADIUS_M, ReferencePath


class ReferencePathTests(unittest.TestCase):
    def test_remaining_arc_length_maps_to_end_of_path(self) -> None:
        path = ReferencePath.from_geographic(
            lat_deg=np.asarray([48.1000, 48.0500, 48.0000], dtype=float),
            lon_deg=np.asarray([11.0000, 11.1200, 11.2500], dtype=float),
        )

        self.assertAlmostEqual(path.s_m[0], path.total_length_m)
        self.assertAlmostEqual(path.s_m[-1], 0.0)
        end_lat_deg, end_lon_deg = path.latlon(0.0)
        self.assertAlmostEqual(end_lat_deg, 48.0000, places=3)
        self.assertAlmostEqual(end_lon_deg, 11.2500, places=3)

    def test_smoothed_path_has_finite_curvature_and_continuous_heading(self) -> None:
        path = ReferencePath.from_geographic(
            lat_deg=np.asarray([48.2000, 48.1200, 48.0000], dtype=float),
            lon_deg=np.asarray([11.0000, 11.0000, 11.1800], dtype=float),
        )

        self.assertTrue(np.all(np.isfinite(path.curvature_inv_m)))
        self.assertLess(np.max(np.abs(np.diff(path.track_rad))), 0.2)

    def test_project_s_m_returns_closest_along_path_station(self) -> None:
        path = ReferencePath.from_geographic(
            lat_deg=np.asarray([0.0, 0.0, 0.0], dtype=float),
            lon_deg=np.asarray([0.0, 0.01, 0.02], dtype=float),
        )

        east_m, north_m = path.position_ne(0.5 * path.total_length_m)
        projected_s_m = path.project_s_m(east_m, north_m + 250.0)

        self.assertAlmostEqual(projected_s_m, 0.5 * path.total_length_m, delta=1.0)

    def test_flyby_path_anticipates_waypoint_turn(self) -> None:
        path = ReferencePath.from_geographic(
            lat_deg=np.asarray([0.0, 0.0, 0.05], dtype=float),
            lon_deg=np.asarray([-0.05, 0.0, 0.0], dtype=float),
        )

        fix_east_m = EARTH_RADIUS_M * np.cos(np.deg2rad(0.05)) * np.deg2rad(0.0)
        fix_north_m = EARTH_RADIUS_M * np.deg2rad(0.0 - 0.05)
        distance_to_fix_m = np.hypot(path.east_m - fix_east_m, path.north_m - fix_north_m)
        fix_s_m = path.project_s_m(fix_east_m, fix_north_m)

        self.assertGreater(float(np.min(distance_to_fix_m)), 250.0)
        self.assertGreater(np.rad2deg(path.track_angle_rad(fix_s_m + 1_000.0)), 5.0)
        self.assertLess(np.rad2deg(path.track_angle_rad(fix_s_m + 1_000.0)), 85.0)


if __name__ == "__main__":
    unittest.main()
