from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.longitudinal_profiles import ScalarProfile, build_simple_glidepath


class ScalarProfileTests(unittest.TestCase):
    def test_value_and_slope_interpolate_between_nodes(self) -> None:
        profile = ScalarProfile(
            s_m=np.asarray([0.0, 10.0, 20.0], dtype=float),
            y=np.asarray([0.0, 10.0, 30.0], dtype=float),
        )
        self.assertAlmostEqual(profile.value(5.0), 5.0)
        self.assertAlmostEqual(profile.value(15.0), 20.0)
        self.assertAlmostEqual(profile.slope(5.0), 1.0)
        self.assertAlmostEqual(profile.slope(15.0), 2.0)

    def test_simple_glidepath_hits_threshold_and_caps_at_intercept_altitude(self) -> None:
        profile = build_simple_glidepath(
            threshold_elevation_m=450.0,
            intercept_distance_m=55_000.0,
            intercept_altitude_m=3_500.0,
            glide_deg=3.0,
            n=20,
        )
        self.assertAlmostEqual(profile.value(0.0), 450.0)
        self.assertLessEqual(profile.value(55_000.0), 3_500.0)
        self.assertGreater(profile.slope(20_000.0), 0.0)


if __name__ == "__main__":
    unittest.main()
