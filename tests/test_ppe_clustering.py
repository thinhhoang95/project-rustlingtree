from __future__ import annotations

import numpy as np
import pandas as pd

from vlm_ppe.clustering.community_detection import run_candidate_community_detection
from vlm_ppe.clustering.features import build_shape_features
from vlm_ppe.clustering.medoid import compute_cluster_medoids


def _resampled_frame() -> pd.DataFrame:
    rows: list[dict] = []
    tracks = {
        "A": np.column_stack([np.linspace(0.0, 4.0, 5), np.zeros(5)]),
        "B": np.column_stack([np.linspace(0.0, 4.0, 5), np.ones(5)]),
        "C": np.column_stack([np.linspace(0.0, 4.0, 5), np.full(5, 3.0)]),
    }
    for flight_id, points in tracks.items():
        for station_index, (x_nm, y_nm) in enumerate(points):
            rows.append(
                {
                    "flight_id": flight_id,
                    "station_index": station_index,
                    "x_nm": float(x_nm),
                    "y_nm": float(y_nm),
                }
            )
    return pd.DataFrame(rows)


def test_shape_features_standardize_flattened_resampled_points() -> None:
    features = build_shape_features(_resampled_frame())

    assert features.track_ids == ["A", "B", "C"]
    assert features.raw.shape == (3, 10)
    np.testing.assert_allclose(features.standardized.mean(axis=0), np.zeros(10), atol=1e-12)


def test_community_detection_candidate_runs_include_metrics() -> None:
    features = build_shape_features(_resampled_frame())

    runs = run_candidate_community_detection(
        features,
        threshold_min_nm=0.0,
        threshold_max_nm=3.0,
        threshold_steps=2,
    )

    assert [run.threshold_nm for run in runs] == [0.0, 3.0]
    assert [run.metric.community_count for run in runs] == [3, 1]
    assert runs[0].metric.singleton_count == 3
    assert runs[1].metric.edge_density == 1.0


def test_medoid_selects_track_with_lowest_pairwise_distance_sum() -> None:
    resampled = _resampled_frame()
    labels = pd.DataFrame({"flight_id": ["A", "B", "C"], "cluster_id": [0, 0, 0]})

    medoids = compute_cluster_medoids(resampled, labels)

    assert len(medoids) == 1
    assert medoids[0].medoid_track_id == "B"
    assert medoids[0].n_tracks == 3
