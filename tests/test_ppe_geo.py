from __future__ import annotations

import numpy as np

from vlm_ppe.geo.projection import LocalProjection
from vlm_ppe.geo.resample import arc_length_resample_points


def test_local_projection_round_trips_near_origin() -> None:
    projection = LocalProjection.from_origin(32.9, -97.0)
    lat = np.asarray([32.9, 32.91])
    lon = np.asarray([-97.0, -96.99])

    x_nm, y_nm = projection.project_nm(lat, lon)
    round_lat, round_lon = projection.unproject_nm(x_nm, y_nm)

    np.testing.assert_allclose(round_lat, lat, atol=1e-9)
    np.testing.assert_allclose(round_lon, lon, atol=1e-9)


def test_arc_length_resample_preserves_endpoints_and_spacing() -> None:
    points = np.asarray([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])

    resampled, s_nm = arc_length_resample_points(points, 8)

    np.testing.assert_allclose(resampled[0], points[0])
    np.testing.assert_allclose(resampled[-1], points[-1])
    np.testing.assert_allclose(s_nm, np.linspace(0.0, 7.0, 8))
