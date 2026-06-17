from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ppe_evaluation.artifacts import MedoidTrajectory


def _load_ablation_module():
    path = Path(__file__).resolve().parents[1] / "src" / "paper-june" / "clustering" / "ablation.py"
    spec = importlib.util.spec_from_file_location("paper_june_clustering_ablation", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_density_module():
    path = Path(__file__).resolve().parents[1] / "src" / "paper-june" / "clustering" / "clustering_ablation_density.py"
    spec = importlib.util.spec_from_file_location("paper_june_clustering_ablation_density", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_density_algorithms_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "paper-june"
        / "clustering"
        / "clustering_ablation_density_algorithms.py"
    )
    spec = importlib.util.spec_from_file_location("paper_june_clustering_ablation_density_algorithms", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_k_by_indices_uses_expected_direction_per_metric() -> None:
    ablation = _load_ablation_module()
    metrics = pd.DataFrame(
        {
            "k": [1, 2, 3, 4],
            "inertia": [100.0, 60.0, 35.0, 30.0],
            "silhouette": [np.nan, 0.2, 0.5, 0.4],
            "calinski_harabasz": [np.nan, 90.0, 70.0, 80.0],
            "davies_bouldin": [np.nan, 1.1, 0.9, 0.7],
        }
    )

    selected = ablation.select_k_by_indices(metrics)

    assert selected["silhouette"] == 3
    assert selected["calinski_harabasz"] == 2
    assert selected["davies_bouldin"] == 4
    assert selected["inertia_elbow"] in {2, 3}


def test_match_medoids_scores_duplicate_ground_truth_ids_as_instances() -> None:
    ablation = _load_ablation_module()
    line_a = np.column_stack([np.linspace(0.0, 1.0, 4), np.zeros(4)])
    line_b = np.column_stack([np.linspace(0.0, 1.0, 4), np.ones(4)])
    ground_truth = [
        ablation.GroundTruthMedoid("GT005#1", "GT005", MedoidTrajectory("GT005#1", line_a)),
        ablation.GroundTruthMedoid("GT005#2", "GT005", MedoidTrajectory("GT005#2", line_b)),
    ]
    predicted = [
        MedoidTrajectory("K02_C00", line_a, cluster_id=0, medoid_track_id="A"),
        MedoidTrajectory("K02_C01", line_b, cluster_id=1, medoid_track_id="B"),
    ]

    summary, assignments, pairwise = ablation.match_medoids(
        ground_truth,
        predicted,
        threshold_nm=0.01,
        k=2,
        mode="all",
    )

    assert summary["tp"] == 2
    assert summary["fp"] == 0
    assert summary["tn"] == 0
    assert summary["fn"] == 0
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert set(assignments["gt_instance_id"]) == {"GT005#1", "GT005#2"}
    assert len(pairwise) == 4


def test_density_selection_ranks_pruned_f1_then_tie_breakers() -> None:
    density = _load_density_module()
    scores = pd.DataFrame(
        [
            {
                "algorithm": "dbscan",
                "run_id": "low_precision",
                "mode": "all",
                "f1": 0.7,
                "recall": 0.7,
                "precision": 0.7,
                "fp": 3,
                "pred_clusters": 8,
                "gt_subclusters": 7,
                "mean_nearest_gt_distance_nm": 2.0,
            },
            {
                "algorithm": "dbscan",
                "run_id": "low_precision",
                "mode": "pruned",
                "f1": 0.8,
                "recall": 0.7,
                "precision": 0.9,
                "fp": 2,
                "pred_clusters": 7,
                "gt_subclusters": 7,
                "mean_nearest_gt_distance_nm": 2.0,
            },
            {
                "algorithm": "dbscan",
                "run_id": "best",
                "mode": "all",
                "f1": 0.75,
                "recall": 0.75,
                "precision": 0.75,
                "fp": 2,
                "pred_clusters": 8,
                "gt_subclusters": 7,
                "mean_nearest_gt_distance_nm": 3.0,
            },
            {
                "algorithm": "dbscan",
                "run_id": "best",
                "mode": "pruned",
                "f1": 0.8,
                "recall": 0.8,
                "precision": 0.85,
                "fp": 1,
                "pred_clusters": 7,
                "gt_subclusters": 7,
                "mean_nearest_gt_distance_nm": 3.0,
            },
            {
                "algorithm": "dbscan",
                "run_id": "third",
                "mode": "pruned",
                "f1": 0.6,
                "recall": 0.9,
                "precision": 0.5,
                "fp": 0,
                "pred_clusters": 6,
                "gt_subclusters": 7,
                "mean_nearest_gt_distance_nm": 1.0,
            },
        ]
    )

    selected = density.select_top_density_runs(scores)

    assert selected["run_id"].tolist() == ["best", "best", "low_precision", "low_precision"]
    assert selected["mode"].tolist() == ["all", "pruned", "all", "pruned"]
    assert selected["method"].tolist() == [
        "dbscan_1st (all)",
        "dbscan_1st (pruned)",
        "dbscan_2nd (all)",
        "dbscan_2nd (pruned)",
    ]


def test_density_medoids_ignore_noise_label() -> None:
    density = _load_density_module()
    from vlm_ppe.clustering.features import build_shape_features

    rows = []
    for flight_id, y in [("noise", 10.0), ("a", 0.0), ("b", 0.2), ("c", 3.0)]:
        for station_index, x in enumerate([0.0, 1.0, 2.0]):
            rows.append({"flight_id": flight_id, "station_index": station_index, "x_nm": x, "y_nm": y})
    resampled = pd.DataFrame(rows)
    features = build_shape_features(resampled)
    labels = np.asarray([-1, 0, 0, 1], dtype=int)

    medoids = density._compute_density_medoids(features, resampled, labels)

    assert sorted(item.cluster_id for item in medoids) == [0, 1]
    assert {item.medoid_track_id for item in medoids}.isdisjoint({"noise"})


def test_leiden_sweep_uses_binary_knn_graph_with_seeded_labels() -> None:
    algorithms = _load_density_algorithms_module()
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.1, 0.0],
            [4.0, 4.0],
            [4.1, 4.0],
            [4.0, 4.1],
        ],
        dtype=float,
    )

    runs = algorithms.run_leiden_sweep(
        features,
        random_state=7,
        neighbor_values=(2,),
        resolution_values=(0.5,),
    )

    assert len(runs) == 1
    assert runs[0].algorithm == "leiden"
    assert runs[0].params["graph"] == "knn_binary"
    assert runs[0].params["n_neighbors"] == 2
    assert len(runs[0].labels) == len(features)
    assert int(runs[0].metrics["noise_tracks"]) == 0
