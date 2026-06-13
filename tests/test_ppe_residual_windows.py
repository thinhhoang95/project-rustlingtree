from __future__ import annotations

import numpy as np
import pandas as pd

from vlm_ppe.clustering.residual_windows import (
    compute_cluster_residual_profiles,
    compute_heading_dispersion,
    compute_residual_energy,
)
from vlm_ppe.schemas import ClusterMedoid


def test_residual_energy_uses_median_squared_distance() -> None:
    template = np.column_stack([np.arange(3, dtype=float), np.zeros(3)])
    tracks = np.stack(
        [
            template,
            template + np.asarray([[0.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
            template + np.asarray([[0.0, 0.0], [0.0, 4.0], [0.0, 0.0]]),
        ],
        axis=0,
    )

    energy = compute_residual_energy(tracks, template)

    np.testing.assert_allclose(energy, [0.0, 4.0, 0.0])


def test_heading_dispersion_is_circular() -> None:
    east = np.column_stack([np.arange(4, dtype=float), np.zeros(4)])
    north = np.column_stack([np.zeros(4), np.arange(4, dtype=float)])

    dispersion = compute_heading_dispersion(np.stack([east, north], axis=0))

    np.testing.assert_allclose(dispersion, np.full(4, 1.0 - np.sqrt(0.5)), atol=1e-12)


def test_compute_cluster_residual_profiles_preserves_station_metrics() -> None:
    station_x = np.linspace(0.0, 10.0, 11)
    template = np.column_stack([station_x, np.zeros_like(station_x)])
    excursion = template.copy()
    excursion[3:7, 1] = 3.0
    rows: list[dict] = []
    for flight_id, points in {"T0": template, "T1": excursion, "T2": excursion}.items():
        for station_index, (x_nm, y_nm) in enumerate(points):
            rows.append(
                {
                    "flight_id": flight_id,
                    "station_index": station_index,
                    "x_nm": float(x_nm),
                    "y_nm": float(y_nm),
                }
            )
    resampled = pd.DataFrame(rows)
    labels = pd.DataFrame({"flight_id": ["T0", "T1", "T2"], "cluster_id": [0, 0, 0]})
    medoid = ClusterMedoid(
        cluster_id=0,
        medoid_track_id="T0",
        n_tracks=3,
        mean_distance_nm=0.0,
        max_distance_nm=0.0,
        template_points=[(float(x), float(y)) for x, y in template],
    )

    profiles = compute_cluster_residual_profiles(resampled, labels, [medoid])

    assert len(profiles) == 11
    assert set(profiles.columns) >= {
        "cluster_id",
        "station_index",
        "s_fraction",
        "s_nm",
        "template_x_nm",
        "template_y_nm",
        "residual_energy_nm2",
        "heading_dispersion",
    }
    assert float(profiles["residual_energy_nm2"].max()) == 9.0
    assert profiles.loc[profiles["residual_energy_nm2"] == 9.0, "station_index"].astype(int).tolist() == [3, 4, 5, 6]
