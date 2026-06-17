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
