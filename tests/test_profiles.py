from __future__ import annotations

import os
import unittest
from math import isclose

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.longitudinal_profiles import ConstraintEnvelope, ScalarProfile


class ScalarProfileTests(unittest.TestCase):
    def test_value_and_slope_interpolate_between_nodes(self) -> None:
        profile = ScalarProfile(
            s_m=np.asarray([0.0, 10.0, 20.0], dtype=float),
            y=np.asarray([0.0, 10.0, 30.0], dtype=float),
        )
        self.assertTrue(isclose(profile.value(5.0), 5.0))
        self.assertTrue(isclose(profile.value(15.0), 20.0))
        self.assertTrue(isclose(profile.slope(5.0), 1.0))
        self.assertTrue(isclose(profile.slope(15.0), 2.0))

    def test_constraint_envelope_is_piecewise_constant_between_nodes(self) -> None:
        envelope = ConstraintEnvelope(
            s_m=np.asarray([0.0, 20_000.0, 40_000.0], dtype=float),
            h_lower_m=np.asarray([450.0, 1_500.0, 3_000.0], dtype=float),
            h_upper_m=np.asarray([500.0, 1_700.0, 3_250.0], dtype=float),
            cas_lower_mps=np.asarray([68.0, 72.0, 78.0], dtype=float),
            cas_upper_mps=np.asarray([72.0, 82.0, 92.0], dtype=float),
            gamma_lower_rad=np.asarray([-0.08, -0.07, -0.04], dtype=float),
            gamma_upper_rad=np.asarray([-0.04, -0.03, 0.0], dtype=float),
        )

        h_lower, h_upper = envelope.h_bounds(10_000.0)
        cas_lower, cas_upper = envelope.cas_bounds(10_000.0)
        gamma_lower, gamma_upper = envelope.gamma_bounds(10_000.0)
        assert gamma_lower is not None
        assert gamma_upper is not None

        self.assertTrue(isclose(h_lower, 450.0))
        self.assertTrue(isclose(h_upper, 500.0))
        self.assertTrue(isclose(cas_lower, 68.0))
        self.assertTrue(isclose(cas_upper, 72.0))
        self.assertTrue(isclose(gamma_lower, -0.08))
        self.assertTrue(isclose(gamma_upper, -0.04))

        h_lower_after, h_upper_after = envelope.h_bounds(20_001.0)
        self.assertTrue(isclose(h_lower_after, 1_500.0))
        self.assertTrue(isclose(h_upper_after, 1_700.0))

    def test_constraint_envelope_from_profiles_unifies_node_grid(self) -> None:
        altitude_lower = ScalarProfile(
            s_m=np.asarray([0.0, 15_000.0, 40_000.0], dtype=float),
            y=np.asarray([450.0, 1_400.0, 3_000.0], dtype=float),
        )
        altitude_upper = ScalarProfile(
            s_m=np.asarray([0.0, 10_000.0, 40_000.0], dtype=float),
            y=np.asarray([500.0, 1_300.0, 3_150.0], dtype=float),
        )
        cas_lower = ScalarProfile(
            s_m=np.asarray([0.0, 8_000.0, 40_000.0], dtype=float),
            y=np.asarray([68.0, 70.0, 82.0], dtype=float),
        )
        cas_upper = ScalarProfile(
            s_m=np.asarray([0.0, 20_000.0, 40_000.0], dtype=float),
            y=np.asarray([72.0, 84.0, 92.0], dtype=float),
        )

        envelope = ConstraintEnvelope.from_profiles(
            altitude_lower=altitude_lower,
            altitude_upper=altitude_upper,
            cas_lower=cas_lower,
            cas_upper=cas_upper,
        )

        self.assertGreater(len(envelope.s_m), 3)
        self.assertEqual(float(envelope.s_m[0]), 0.0)
        self.assertEqual(float(envelope.s_m[-1]), 40_000.0)
        self.assertLessEqual(envelope.h_lower_m[0], envelope.h_upper_m[0])
        self.assertLessEqual(envelope.cas_lower_mps[-1], envelope.cas_upper_mps[-1])
        self.assertTrue(isclose(envelope.h_lower_m[1], 956.6666666666666))
        self.assertTrue(isclose(envelope.h_upper_m[1], 1140.0))
        self.assertTrue(isclose(envelope.cas_upper_mps[1], 76.8))

        h_lower, h_upper = envelope.h_bounds(9_000.0)
        self.assertTrue(isclose(h_lower, float(envelope.h_lower_m[1])))
        self.assertTrue(isclose(h_upper, float(envelope.h_upper_m[1])))


if __name__ == "__main__":
    unittest.main()
