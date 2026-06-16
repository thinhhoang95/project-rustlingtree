from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ppe_evaluation.artifacts import MedoidTrajectory
from ppe_evaluation.frechet import discrete_frechet_distance


DEFAULT_FRECHET_THRESHOLD_NM = 0.75


@dataclass(frozen=True)
class MedoidMatch:
    gt_medoid_id: str
    pred_cluster_id: int
    pred_medoid_id: str
    distance_nm: float


@dataclass(frozen=True)
class MedoidMatchResult:
    matches: list[MedoidMatch]
    unmatched_gt_ids: list[str]
    unmatched_pred_cluster_ids: list[int]
    pairwise_distances: pd.DataFrame


def match_medoids(
    ground_truth: list[MedoidTrajectory],
    predicted: list[MedoidTrajectory],
    *,
    threshold_nm: float = DEFAULT_FRECHET_THRESHOLD_NM,
) -> MedoidMatchResult:
    if not ground_truth and not predicted:
        return MedoidMatchResult([], [], [], pd.DataFrame(columns=["gt_medoid_id", "pred_cluster_id", "distance_nm"]))

    distances = np.empty((len(ground_truth), len(predicted)), dtype=float)
    rows: list[dict[str, object]] = []
    for gt_index, gt_medoid in enumerate(ground_truth):
        for pred_index, pred_medoid in enumerate(predicted):
            distance = discrete_frechet_distance(gt_medoid.points, pred_medoid.points)
            distances[gt_index, pred_index] = distance
            rows.append(
                {
                    "gt_medoid_id": gt_medoid.medoid_id,
                    "pred_cluster_id": int(pred_medoid.cluster_id if pred_medoid.cluster_id is not None else pred_index),
                    "pred_medoid_id": pred_medoid.medoid_id,
                    "distance_nm": distance,
                }
            )

    matches: list[MedoidMatch] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    if len(ground_truth) and len(predicted):
        gt_indices, pred_indices = linear_sum_assignment(distances)
        for gt_index, pred_index in zip(gt_indices.tolist(), pred_indices.tolist(), strict=True):
            distance = float(distances[gt_index, pred_index])
            if distance < threshold_nm:
                gt_medoid = ground_truth[gt_index]
                pred_medoid = predicted[pred_index]
                matches.append(
                    MedoidMatch(
                        gt_medoid_id=gt_medoid.medoid_id,
                        pred_cluster_id=int(pred_medoid.cluster_id if pred_medoid.cluster_id is not None else pred_index),
                        pred_medoid_id=pred_medoid.medoid_id,
                        distance_nm=distance,
                    )
                )
                matched_gt.add(gt_index)
                matched_pred.add(pred_index)

    unmatched_gt_ids = [item.medoid_id for index, item in enumerate(ground_truth) if index not in matched_gt]
    unmatched_pred_cluster_ids = [
        int(item.cluster_id if item.cluster_id is not None else index)
        for index, item in enumerate(predicted)
        if index not in matched_pred
    ]
    return MedoidMatchResult(
        matches=matches,
        unmatched_gt_ids=unmatched_gt_ids,
        unmatched_pred_cluster_ids=unmatched_pred_cluster_ids,
        pairwise_distances=pd.DataFrame(rows),
    )
