from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.path_geometry import ReferencePath


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


if __name__ == "__main__":
    unittest.main()
