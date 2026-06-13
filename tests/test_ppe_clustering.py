from __future__ import annotations

import numpy as np
import pandas as pd

from vlm_ppe.clustering.features import build_shape_features
from vlm_ppe.clustering.kmeans_runner import run_candidate_kmeans
from vlm_ppe.clustering.medoid import compute_cluster_medoids
from vlm_ppe.clustering.polygon_capture import assign_tracks_to_subcluster_polygons, convex_hull
from vlm_ppe.schemas import SubclusterPolygon


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


def test_kmeans_candidate_runs_include_metrics() -> None:
    features = build_shape_features(_resampled_frame())

    runs = run_candidate_kmeans(features, k_min=1, k_max=2, n_init=5, random_state=3)

    assert [run.k for run in runs] == [1, 2]
    assert runs[0].metric.silhouette is None
    assert runs[1].metric.silhouette is not None


def test_medoid_selects_track_with_lowest_pairwise_distance_sum() -> None:
    resampled = _resampled_frame()
    labels = pd.DataFrame({"flight_id": ["A", "B", "C"], "cluster_id": [0, 0, 0]})

    medoids = compute_cluster_medoids(resampled, labels)

    assert len(medoids) == 1
    assert medoids[0].medoid_track_id == "B"
    assert medoids[0].n_tracks == 3


def test_convex_hull_normalizes_unordered_polygon_points() -> None:
    hull = convex_hull([(1.0, 1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)])

    assert hull == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_polygon_capture_assigns_tracks_crossing_convex_gates_once() -> None:
    resampled = _resampled_frame()
    subclusters = [
        SubclusterPolygon(
            subcluster_id=1,
            label="Low gate",
            polygon=[(1.5, -0.2), (2.5, -0.2), (2.5, 0.2), (1.5, 0.2)],
        ),
        SubclusterPolygon(
            subcluster_id=2,
            label="Middle gate",
            polygon=[(1.5, 0.8), (2.5, 0.8), (2.5, 1.2), (1.5, 1.2)],
        ),
    ]

    result = assign_tracks_to_subcluster_polygons(resampled, ["A", "B", "C"], subclusters)

    assert result.captures[0].track_ids == ["A"]
    assert result.captures[1].track_ids == ["B"]
    assert result.uncaptured_track_ids == ["C"]
    assert result.overlapping_track_ids == {}
