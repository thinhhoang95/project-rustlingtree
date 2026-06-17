from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ppe_evaluation.artifacts import MedoidTrajectory
from ppe_evaluation.frechet import discrete_frechet_distance


DEFAULT_FRECHET_THRESHOLD_NM = 3.0


def detection_confusion_summary(
    *,
    n_gt: int,
    n_pred: int,
    n_tp: int,
    n_tn: int = 0,
) -> dict[str, Any]:
    """Summarize object-detection style counts.

    Medoid and window evaluation compare finite sets of positives, so there is
    no well-defined true-negative universe. Keep TN explicit and default it to
    zero so downstream reports do not silently invent a different denominator.
    """
    tp = max(0, int(n_tp))
    tn = max(0, int(n_tn))
    fp = max(0, int(n_pred) - tp)
    fn = max(0, int(n_gt) - tp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "accuracy": _safe_div(tp + tn, tp + fp + tn + fn),
    }


@dataclass(frozen=True)
class MedoidMatch:
    gt_medoid_id: str
    pred_cluster_id: int
    pred_medoid_id: str
    distance_nm: float


@dataclass(frozen=True)
class MedoidAssignment:
    gt_medoid_id: str
    pred_cluster_id: int
    pred_medoid_id: str
    distance_nm: float
    matched: bool


@dataclass(frozen=True)
class MedoidMatchResult:
    matches: list[MedoidMatch]
    assignments: list[MedoidAssignment]
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
        return MedoidMatchResult(
            matches=[],
            assignments=[],
            unmatched_gt_ids=[],
            unmatched_pred_cluster_ids=[],
            pairwise_distances=pd.DataFrame(columns=["gt_medoid_id", "pred_cluster_id", "pred_medoid_id", "distance_nm"]),
        )

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
    assignments: list[MedoidAssignment] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    if len(ground_truth) and len(predicted):
        gt_indices, pred_indices = linear_sum_assignment(distances)
        for gt_index, pred_index in zip(gt_indices.tolist(), pred_indices.tolist(), strict=True):
            distance = float(distances[gt_index, pred_index])
            gt_medoid = ground_truth[gt_index]
            pred_medoid = predicted[pred_index]
            pred_cluster_id = int(pred_medoid.cluster_id if pred_medoid.cluster_id is not None else pred_index)
            matched = distance < threshold_nm
            assignments.append(
                MedoidAssignment(
                    gt_medoid_id=gt_medoid.medoid_id,
                    pred_cluster_id=pred_cluster_id,
                    pred_medoid_id=pred_medoid.medoid_id,
                    distance_nm=distance,
                    matched=matched,
                )
            )
            if matched:
                matches.append(
                    MedoidMatch(
                        gt_medoid_id=gt_medoid.medoid_id,
                        pred_cluster_id=pred_cluster_id,
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
        assignments=assignments,
        unmatched_gt_ids=unmatched_gt_ids,
        unmatched_pred_cluster_ids=unmatched_pred_cluster_ids,
        pairwise_distances=pd.DataFrame(rows),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)
