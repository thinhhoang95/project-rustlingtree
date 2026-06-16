from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppe_evaluation.artifacts import (
    GroundTruth,
    PredictionArtifacts,
    default_ground_truth_dir,
    load_ground_truth,
    load_run_artifacts,
)
from ppe_evaluation.matching import DEFAULT_FRECHET_THRESHOLD_NM, MedoidMatch, match_medoids


DEFAULT_WINDOW_IOU_THRESHOLD = 0.1
DEFAULT_WINDOW_IOU_THRESHOLDS = (0.1, 0.25, 0.5, 0.75, 0.9)
MAP_IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class EvaluationReport:
    run_id: str
    dataset_id: str
    medoid_summary: dict[str, Any]
    window_summary: dict[str, Any]
    window_classification_summary: dict[str, Any]
    medoid_matches: pd.DataFrame
    window_matches: pd.DataFrame
    window_classification_matches: pd.DataFrame
    pairwise_medoid_distances: pd.DataFrame
    truth: GroundTruth | None = None
    predictions: PredictionArtifacts | None = None
    output_dir: Path | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "medoid_summary": self.medoid_summary,
            "window_summary": self.window_summary,
            "window_classification_summary": self.window_classification_summary,
            "output_dir": self.output_dir.as_posix() if self.output_dir else None,
        }


def evaluate_run(
    run_dir: str | Path,
    ground_truth_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    frechet_threshold_nm: float = DEFAULT_FRECHET_THRESHOLD_NM,
    window_iou_threshold: float = DEFAULT_WINDOW_IOU_THRESHOLD,
    window_iou_thresholds: Sequence[float] = DEFAULT_WINDOW_IOU_THRESHOLDS,
    class_aware: bool = True,
) -> EvaluationReport:
    predictions = load_run_artifacts(run_dir)
    truth_path = Path(ground_truth_dir) if ground_truth_dir is not None else default_ground_truth_dir(predictions.run_dir)
    truth = load_ground_truth(truth_path)
    medoid_result = match_medoids(truth.medoids, predictions.medoids, threshold_nm=frechet_threshold_nm)
    medoid_summary = _medoid_summary(
        n_gt=len(truth.medoids),
        n_pred=len(predictions.medoids),
        n_tp=len(medoid_result.matches),
        threshold_nm=frechet_threshold_nm,
    )
    medoid_matches = _medoid_match_frame(medoid_result.matches)
    window_summary, window_matches = evaluate_windows(
        truth,
        predictions,
        medoid_result.matches,
        iou_threshold=window_iou_threshold,
        iou_thresholds=window_iou_thresholds,
        class_aware=class_aware,
    )
    window_classification_summary, window_classification_matches = evaluate_window_classification(
        truth,
        predictions,
        medoid_result.matches,
        iou_threshold=window_iou_threshold,
        iou_thresholds=window_iou_thresholds,
    )
    report = EvaluationReport(
        run_id=predictions.run_id,
        dataset_id=predictions.dataset_id,
        medoid_summary=medoid_summary,
        window_summary=window_summary,
        window_classification_summary=window_classification_summary,
        medoid_matches=medoid_matches,
        window_matches=window_matches,
        window_classification_matches=window_classification_matches,
        pairwise_medoid_distances=medoid_result.pairwise_distances,
        truth=truth,
        predictions=predictions,
        output_dir=Path(output_dir) if output_dir is not None else predictions.run_dir / "evaluation",
    )
    if report.output_dir is not None:
        write_report(report, report.output_dir)
    return report


def evaluate_windows(
    truth: GroundTruth,
    predictions: PredictionArtifacts,
    medoid_matches: list[MedoidMatch],
    *,
    iou_threshold: float = DEFAULT_WINDOW_IOU_THRESHOLD,
    iou_thresholds: Sequence[float] = DEFAULT_WINDOW_IOU_THRESHOLDS,
    class_aware: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pred_to_gt = {int(match.pred_cluster_id): match.gt_medoid_id for match in medoid_matches}
    gt_objects = _window_objects(truth.windows, id_column="gt_medoid_id")
    pred_objects = _prediction_window_objects(predictions.windows, pred_to_gt)
    labels = (
        sorted({item["class_name"] for item in gt_objects} | {item["class_name"] for item in pred_objects})
        if class_aware
        else ["all"]
    )

    thresholds = _normalized_thresholds(iou_thresholds, primary=iou_threshold)
    threshold_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    primary_summary: dict[str, Any] | None = None
    for threshold in thresholds:
        threshold_summary, threshold_rows = _evaluate_windows_at_threshold(
            labels,
            gt_objects,
            pred_objects,
            iou_threshold=threshold,
            class_aware=class_aware,
        )
        threshold_summaries.append(threshold_summary)
        all_rows.extend(threshold_rows)
        if threshold == iou_threshold:
            primary_summary = threshold_summary

    if primary_summary is None:
        primary_summary = threshold_summaries[0] if threshold_summaries else _empty_window_summary(iou_threshold, class_aware)
    map_at_0_5 = next(
        (item["map"] for item in threshold_summaries if item["iou_threshold"] == MAP_IOU_THRESHOLD),
        primary_summary["map"],
    )
    summary = {
        **primary_summary,
        "map_at_0_5": map_at_0_5,
        "primary_iou_threshold": iou_threshold,
        "iou_thresholds": thresholds,
        "by_iou_threshold": threshold_summaries,
    }
    return summary, pd.DataFrame(all_rows)


def evaluate_window_classification(
    truth: GroundTruth,
    predictions: PredictionArtifacts,
    medoid_matches: list[MedoidMatch],
    *,
    iou_threshold: float = DEFAULT_WINDOW_IOU_THRESHOLD,
    iou_thresholds: Sequence[float] = DEFAULT_WINDOW_IOU_THRESHOLDS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pred_to_gt = {int(match.pred_cluster_id): match.gt_medoid_id for match in medoid_matches}
    gt_objects = _window_objects(truth.windows, id_column="gt_medoid_id")
    pred_objects = _prediction_window_objects(predictions.windows, pred_to_gt)
    thresholds = _normalized_thresholds(iou_thresholds, primary=iou_threshold)
    threshold_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    primary_summary: dict[str, Any] | None = None
    for threshold in thresholds:
        _, localization_rows = _evaluate_windows_at_threshold(
            ["all"],
            gt_objects,
            pred_objects,
            iou_threshold=threshold,
            class_aware=False,
        )
        threshold_summary, threshold_rows = _classification_at_threshold(
            localization_rows,
            gt_objects,
            pred_objects,
            iou_threshold=threshold,
        )
        threshold_summaries.append(threshold_summary)
        all_rows.extend(threshold_rows)
        if threshold == iou_threshold:
            primary_summary = threshold_summary

    if primary_summary is None:
        primary_summary = threshold_summaries[0] if threshold_summaries else _empty_classification_summary(iou_threshold)
    summary = {
        **primary_summary,
        "primary_iou_threshold": iou_threshold,
        "iou_thresholds": thresholds,
        "by_iou_threshold": threshold_summaries,
    }
    return summary, pd.DataFrame(all_rows)


def _classification_at_threshold(
    localization_rows: list[dict[str, Any]],
    gt_objects: list[dict[str, Any]],
    pred_objects: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_class: dict[str, dict[str, Any]] = {
        class_name: {"class_name": class_name, "tp": 0, "fp": 0, "fn": 0}
        for class_name in sorted({item["class_name"] for item in gt_objects} | {item["class_name"] for item in pred_objects})
    }
    localized_matches = 0
    correct = 0
    incorrect = 0
    unmatched_predictions = 0
    missed_ground_truth = 0
    for row in localization_rows:
        reason = str(row["reason"])
        if reason == "matched":
            localized_matches += 1
            gt_class = str(row["gt_class_name"])
            pred_class = str(row["pred_class_name"])
            classification_correct = gt_class == pred_class
            if classification_correct:
                correct += 1
                by_class[gt_class]["tp"] += 1
                classification_reason = "correct"
            else:
                incorrect += 1
                by_class[pred_class]["fp"] += 1
                by_class[gt_class]["fn"] += 1
                classification_reason = "wrong_class"
            rows.append(
                {
                    **row,
                    "classification_correct": classification_correct,
                    "classification_reason": classification_reason,
                    "iou_threshold": iou_threshold,
                }
            )
        elif reason == "unmatched_prediction":
            unmatched_predictions += 1
            pred_class = str(row["pred_class_name"])
            if pred_class in by_class:
                by_class[pred_class]["fp"] += 1
            rows.append(
                {
                    **row,
                    "classification_correct": False,
                    "classification_reason": "unlocalized_prediction",
                    "iou_threshold": iou_threshold,
                }
            )
        elif reason == "missed_ground_truth":
            missed_ground_truth += 1
            gt_class = str(row["gt_class_name"])
            if gt_class in by_class:
                by_class[gt_class]["fn"] += 1
            rows.append(
                {
                    **row,
                    "classification_correct": False,
                    "classification_reason": "missed_ground_truth",
                    "iou_threshold": iou_threshold,
                }
            )

    tp = correct
    fp = incorrect + unmatched_predictions
    fn = incorrect + missed_ground_truth
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    classes = [_classification_class_summary(item) for item in by_class.values()]
    summary = {
        "iou_threshold": iou_threshold,
        "gt_windows": len(gt_objects),
        "pred_windows": len(pred_objects),
        "localized_matches": localized_matches,
        "correct": correct,
        "incorrect": incorrect,
        "unmatched_predictions": unmatched_predictions,
        "missed_ground_truth": missed_ground_truth,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "accuracy": _safe_div(correct, localized_matches),
        "overall_accuracy": _safe_div(correct, len(gt_objects)),
        "by_class": classes,
    }
    return summary, rows


def _classification_class_summary(counts: dict[str, Any]) -> dict[str, Any]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "class_name": str(counts["class_name"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _evaluate_windows_at_threshold(
    labels: list[str],
    gt_objects: list[dict[str, Any]],
    pred_objects: list[dict[str, Any]],
    *,
    iou_threshold: float,
    class_aware: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    aps: list[float] = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for label in labels:
        class_gt = [item for item in gt_objects if item["class_name"] == label] if class_aware else gt_objects
        class_pred = [item for item in pred_objects if item["class_name"] == label] if class_aware else pred_objects
        ap, class_rows, tp, fp, fn = _evaluate_class_windows(
            label,
            class_gt,
            class_pred,
            iou_threshold,
            class_aware=class_aware,
        )
        aps.append(ap)
        for row in class_rows:
            row["iou_threshold"] = iou_threshold
        rows.extend(class_rows)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = _safe_div(tp_total, tp_total + fp_total)
    recall = _safe_div(tp_total, tp_total + fn_total)
    summary = {
        "iou_threshold": iou_threshold,
        "class_aware": class_aware,
        "gt_windows": len(gt_objects),
        "pred_windows": len(pred_objects),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "map": float(sum(aps) / len(aps)) if aps else 0.0,
    }
    return summary, rows


def _normalized_thresholds(thresholds: Sequence[float], *, primary: float) -> list[float]:
    normalized = {float(threshold) for threshold in thresholds}
    normalized.add(float(primary))
    return sorted(normalized)


def _empty_window_summary(iou_threshold: float, class_aware: bool) -> dict[str, Any]:
    return {
        "iou_threshold": float(iou_threshold),
        "class_aware": class_aware,
        "gt_windows": 0,
        "pred_windows": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "map": 0.0,
    }


def _empty_classification_summary(iou_threshold: float) -> dict[str, Any]:
    return {
        "iou_threshold": float(iou_threshold),
        "gt_windows": 0,
        "pred_windows": 0,
        "localized_matches": 0,
        "correct": 0,
        "incorrect": 0,
        "unmatched_predictions": 0,
        "missed_ground_truth": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
        "overall_accuracy": 0.0,
        "by_class": [],
    }


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    a0, a1 = sorted((float(a_start), float(a_end)))
    b0, b1 = sorted((float(b_start), float(b_end)))
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0.0:
        return 1.0 if a0 == b0 else 0.0
    return float(intersection / union)


def write_report(report: EvaluationReport, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")
    report.medoid_matches.to_csv(root / "medoid_matches.csv", index=False)
    report.window_matches.to_csv(root / "window_matches.csv", index=False)
    report.window_classification_matches.to_csv(root / "window_classification_matches.csv", index=False)
    report.pairwise_medoid_distances.to_csv(root / "medoid_pairwise_distances.csv", index=False)
    (root / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    (root / "evaluation_log.md").write_text(_evaluation_log_markdown(report), encoding="utf-8")
    _write_window_overlay_plot(report, root / "window_overlay.png")


def _write_window_overlay_plot(report: EvaluationReport, path: Path) -> None:
    if report.truth is None or report.predictions is None:
        return
    rows = report.window_classification_matches
    if rows.empty:
        return
    primary_iou = float(report.window_classification_summary["primary_iou_threshold"])
    primary_rows = rows.loc[rows["iou_threshold"].astype(float) == primary_iou].copy()
    if primary_rows.empty:
        return

    gt_windows = _records_by_key(report.truth.windows, "window_id")
    pred_windows = _prediction_windows_by_key(report.predictions.windows)
    gt_medoids = _ground_truth_medoids_by_key(report.truth)
    pred_medoids = {
        int(medoid.cluster_id): medoid
        for medoid in report.predictions.medoids
        if medoid.cluster_id is not None
    }

    primary_rows = primary_rows.sort_values(
        ["classification_reason", "gt_medoid_id", "pred_cluster_id", "gt_window_id", "pred_window_id"],
        na_position="last",
    )
    n_panels = len(primary_rows)
    n_cols = min(3, max(1, math.ceil(math.sqrt(n_panels))))
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 6.5 * n_rows), squeeze=False)
    fig.suptitle(
        f"PPE Window Overlay: {report.run_id} (GT blue, model red, IoU threshold {primary_iou:.2f})",
        fontsize=16,
    )

    for axis, row in zip(axes.ravel(), primary_rows.to_dict("records"), strict=False):
        _plot_window_row(axis, row, gt_windows, pred_windows, gt_medoids, pred_medoids, primary_iou)
    for axis in axes.ravel()[n_panels:]:
        axis.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_window_row(
    axis: Any,
    row: dict[str, Any],
    gt_windows: dict[str, dict[str, Any]],
    pred_windows: dict[tuple[int, str], dict[str, Any]],
    gt_medoids: dict[tuple[str, int | None], Any],
    pred_medoids: dict[int, Any],
    primary_iou: float,
) -> None:
    gt_window_id = str(row["gt_window_id"]) if _has_value(row.get("gt_window_id")) else ""
    pred_window_id = str(row["pred_window_id"]) if _has_value(row.get("pred_window_id")) else ""
    pred_cluster_id = int(float(row["pred_cluster_id"])) if _has_value(row.get("pred_cluster_id")) else None
    gt_window = gt_windows.get(gt_window_id)
    pred_window = pred_windows.get((pred_cluster_id, pred_window_id)) if pred_cluster_id is not None else None
    gt_medoid = _medoid_for_ground_truth_window(gt_window, gt_medoids)
    pred_medoid = pred_medoids.get(pred_cluster_id) if pred_cluster_id is not None else None

    if gt_medoid is not None:
        _plot_medoid(axis, gt_medoid.points, color="#4b5563", linestyle="-", label="GT medoid")
    if pred_medoid is not None:
        _plot_medoid(axis, pred_medoid.points, color="#9ca3af", linestyle="--", label="Model medoid")
    if gt_medoid is not None and gt_window is not None:
        _plot_window_segment(axis, gt_medoid.points, gt_window, color="#2563eb", linewidth=5.0, label="Ground truth window")
    if pred_medoid is not None and pred_window is not None:
        _plot_window_segment(axis, pred_medoid.points, pred_window, color="#dc2626", linewidth=4.0, label="Model window")

    iou = float(row["iou"]) if _has_value(row.get("iou")) else 0.0
    result = _window_result(row)
    axis.set_title(f"{result} | IoU {_percent(iou)}", fontsize=12)
    axis.set_xlabel("x (NM)")
    axis.set_ylabel("y (NM)")
    axis.grid(True, alpha=0.25)
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(loc="best", fontsize=8)
    axis.text(
        0.01,
        0.01,
        (
            f"GT: {_window_side_label(row, prefix='gt')}\n"
            f"Model: {_window_side_label(row, prefix='pred')}\n"
            f"{_window_reason(row, primary_iou)}"
        ),
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )


def _plot_medoid(axis: Any, points: np.ndarray, *, color: str, linestyle: str, label: str) -> None:
    if len(points) == 0:
        return
    axis.plot(points[:, 0], points[:, 1], color=color, linestyle=linestyle, linewidth=1.4, alpha=0.85, label=label)
    axis.scatter(points[0, 0], points[0, 1], color=color, s=18, marker="o", zorder=3)
    axis.scatter(points[-1, 0], points[-1, 1], color=color, s=24, marker="x", zorder=3)


def _plot_window_segment(axis: Any, points: np.ndarray, window: dict[str, Any], *, color: str, linewidth: float, label: str) -> None:
    if len(points) == 0:
        return
    start = max(0, min(len(points) - 1, int(window.get("start_station_index", 0))))
    end = max(0, min(len(points) - 1, int(window.get("end_station_index", start))))
    start, end = sorted((start, end))
    segment = points[start : end + 1]
    if len(segment) == 0:
        return
    axis.plot(segment[:, 0], segment[:, 1], color=color, linewidth=linewidth, alpha=0.9, solid_capstyle="round", label=label)


def _records_by_key(frame: pd.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {}
    return {str(row[key]): row for row in frame.to_dict("records") if key in row}


def _prediction_windows_by_key(frame: pd.DataFrame) -> dict[tuple[int, str], dict[str, Any]]:
    if frame.empty:
        return {}
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        rows[(int(row["cluster_id"]), str(row["window_id"]))] = row
    return rows


def _ground_truth_medoids_by_key(truth: GroundTruth) -> dict[tuple[str, int | None], Any]:
    medoids: dict[tuple[str, int | None], Any] = {}
    for medoid in truth.medoids:
        medoids[(medoid.medoid_id, medoid.cluster_id)] = medoid
        medoids.setdefault((medoid.medoid_id, None), medoid)
    return medoids


def _medoid_for_ground_truth_window(window: dict[str, Any] | None, medoids: dict[tuple[str, int | None], Any]) -> Any | None:
    if window is None:
        return None
    gt_medoid_id = str(window.get("gt_medoid_id", ""))
    source_cluster_id = _source_cluster_id_from_window_id(str(window.get("window_id", "")))
    return medoids.get((gt_medoid_id, source_cluster_id)) or medoids.get((gt_medoid_id, None))


def _source_cluster_id_from_window_id(window_id: str) -> int | None:
    if not window_id.startswith("C"):
        return None
    digits = []
    for char in window_id[1:]:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else None


def _medoid_summary(*, n_gt: int, n_pred: int, n_tp: int, threshold_nm: float) -> dict[str, Any]:
    fp = n_pred - n_tp
    fn = n_gt - n_tp
    precision = _safe_div(n_tp, n_tp + fp)
    recall = _safe_div(n_tp, n_tp + fn)
    return {
        "frechet_threshold_nm": threshold_nm,
        "gt_medoids": n_gt,
        "pred_medoids": n_pred,
        "tp": n_tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "accuracy": _safe_div(n_tp, n_tp + fp + fn),
    }


def _medoid_match_frame(matches: list[MedoidMatch]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gt_medoid_id": match.gt_medoid_id,
                "pred_cluster_id": match.pred_cluster_id,
                "pred_medoid_id": match.pred_medoid_id,
                "frechet_distance_nm": match.distance_nm,
            }
            for match in matches
        ],
        columns=["gt_medoid_id", "pred_cluster_id", "pred_medoid_id", "frechet_distance_nm"],
    )


def _window_objects(frame: pd.DataFrame, *, id_column: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    objects: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        medoid_id = str(row[id_column])
        objects.append(
            {
                "medoid_id": medoid_id,
                "window_id": str(row["window_id"]),
                "class_name": str(row["class_name"]),
                "start": _interval_start(row),
                "end": _interval_end(row),
                "start_station_index": int(row["start_station_index"]) if "start_station_index" in row and pd.notna(row["start_station_index"]) else None,
                "end_station_index": int(row["end_station_index"]) if "end_station_index" in row and pd.notna(row["end_station_index"]) else None,
                "confidence": float(row.get("confidence", 1.0) or 1.0),
            }
        )
    return objects


def _prediction_window_objects(frame: pd.DataFrame, pred_to_gt: dict[int, str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    objects: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        cluster_id = int(row["cluster_id"])
        medoid_id = pred_to_gt.get(cluster_id)
        objects.append(
            {
                "cluster_id": cluster_id,
                "medoid_id": medoid_id,
                "window_id": str(row["window_id"]),
                "class_name": str(row["class_name"]),
                "start": _interval_start(row),
                "end": _interval_end(row),
                "start_station_index": int(row["start_station_index"]) if "start_station_index" in row and pd.notna(row["start_station_index"]) else None,
                "end_station_index": int(row["end_station_index"]) if "end_station_index" in row and pd.notna(row["end_station_index"]) else None,
                "confidence": float(row.get("confidence", 1.0) or 1.0),
            }
        )
    return objects


def _evaluate_class_windows(
    class_name: str,
    gt_objects: list[dict[str, Any]],
    pred_objects: list[dict[str, Any]],
    iou_threshold: float,
    *,
    class_aware: bool = True,
) -> tuple[float, list[dict[str, Any]], int, int, int]:
    unmatched_gt = set(range(len(gt_objects)))
    rows: list[dict[str, Any]] = []
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for pred in sorted(pred_objects, key=lambda item: item["confidence"], reverse=True):
        best_index: int | None = None
        best_iou = 0.0
        for gt_index in sorted(unmatched_gt):
            gt = gt_objects[gt_index]
            if pred["medoid_id"] != gt["medoid_id"]:
                continue
            iou = interval_iou(pred["start"], pred["end"], gt["start"], gt["end"])
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        matched = best_index is not None and best_iou >= iou_threshold
        if matched:
            unmatched_gt.remove(best_index)
            gt = gt_objects[best_index]
            tp_flags.append(1)
            fp_flags.append(0)
            rows.append(
                {
                    "class_name": class_name,
                    "gt_medoid_id": gt["medoid_id"],
                    "pred_cluster_id": pred.get("cluster_id"),
                    "gt_window_id": gt["window_id"],
                    "pred_window_id": pred["window_id"],
                    "gt_class_name": gt["class_name"],
                    "pred_class_name": pred["class_name"],
                    "gt_start_station_index": gt["start_station_index"],
                    "gt_end_station_index": gt["end_station_index"],
                    "pred_start_station_index": pred["start_station_index"],
                    "pred_end_station_index": pred["end_station_index"],
                    "gt_start_s_fraction": gt["start"],
                    "gt_end_s_fraction": gt["end"],
                    "pred_start_s_fraction": pred["start"],
                    "pred_end_s_fraction": pred["end"],
                    "iou": best_iou,
                    "matched": True,
                    "reason": "matched",
                    "class_aware": class_aware,
                }
            )
        else:
            gt = gt_objects[best_index] if best_index is not None else None
            tp_flags.append(0)
            fp_flags.append(1)
            rows.append(
                {
                    "class_name": class_name,
                    "gt_medoid_id": gt["medoid_id"] if gt is not None else pred["medoid_id"],
                    "pred_cluster_id": pred.get("cluster_id"),
                    "gt_window_id": gt["window_id"] if gt is not None else "",
                    "pred_window_id": pred["window_id"],
                    "gt_class_name": gt["class_name"] if gt is not None else "",
                    "pred_class_name": pred["class_name"],
                    "gt_start_station_index": gt["start_station_index"] if gt is not None else None,
                    "gt_end_station_index": gt["end_station_index"] if gt is not None else None,
                    "pred_start_station_index": pred["start_station_index"],
                    "pred_end_station_index": pred["end_station_index"],
                    "gt_start_s_fraction": gt["start"] if gt is not None else None,
                    "gt_end_s_fraction": gt["end"] if gt is not None else None,
                    "pred_start_s_fraction": pred["start"],
                    "pred_end_s_fraction": pred["end"],
                    "iou": best_iou,
                    "matched": False,
                    "reason": "unmatched_prediction",
                    "class_aware": class_aware,
                }
            )
    for gt_index in sorted(unmatched_gt):
        gt = gt_objects[gt_index]
        rows.append(
            {
                "class_name": class_name,
                "gt_medoid_id": gt["medoid_id"],
                "pred_cluster_id": None,
                "gt_window_id": gt["window_id"],
                "pred_window_id": "",
                "gt_class_name": gt["class_name"],
                "pred_class_name": "",
                "gt_start_station_index": gt["start_station_index"],
                "gt_end_station_index": gt["end_station_index"],
                "pred_start_station_index": None,
                "pred_end_station_index": None,
                "gt_start_s_fraction": gt["start"],
                "gt_end_s_fraction": gt["end"],
                "pred_start_s_fraction": None,
                "pred_end_s_fraction": None,
                "iou": 0.0,
                "matched": False,
                "reason": "missed_ground_truth",
                "class_aware": class_aware,
            }
        )

    tp = sum(tp_flags)
    fp = sum(fp_flags)
    fn = len(unmatched_gt)
    ap = _average_precision(tp_flags, fp_flags, len(gt_objects)) if gt_objects else (0.0 if pred_objects else 0.0)
    return ap, rows, tp, fp, fn


def _average_precision(tp_flags: list[int], fp_flags: list[int], n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    tp_cum = 0
    fp_cum = 0
    recalls = [0.0]
    precisions = [1.0]
    for tp, fp in zip(tp_flags, fp_flags, strict=True):
        tp_cum += tp
        fp_cum += fp
        recalls.append(tp_cum / n_gt)
        precisions.append(_safe_div(tp_cum, tp_cum + fp_cum))
    recalls.append(1.0)
    precisions.append(0.0)
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    ap = 0.0
    for index in range(1, len(recalls)):
        ap += (recalls[index] - recalls[index - 1]) * precisions[index]
    return float(ap)


def _interval_start(row: dict[str, Any]) -> float:
    if "start_s_fraction" in row and pd.notna(row["start_s_fraction"]):
        return float(row["start_s_fraction"])
    return float(row["start_station_index"])


def _interval_end(row: dict[str, Any]) -> float:
    if "end_s_fraction" in row and pd.notna(row["end_s_fraction"]):
        return float(row["end_s_fraction"])
    return float(row["end_station_index"])


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def _summary_markdown(report: EvaluationReport) -> str:
    medoid = report.medoid_summary
    window = report.window_summary
    classification = report.window_classification_summary
    threshold_lines = [
        "| IoU | TP | FP | FN | Precision | Recall | F1 | mAP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for suite in window.get("by_iou_threshold", []):
        threshold_lines.append(
            "| "
            f"{suite['iou_threshold']:.2f} | "
            f"{suite['tp']} | "
            f"{suite['fp']} | "
            f"{suite['fn']} | "
            f"{suite['precision']:.3f} | "
            f"{suite['recall']:.3f} | "
            f"{suite['f1']:.3f} | "
            f"{suite['map']:.3f} |"
        )
    return "\n".join(
        [
            f"# PPE Evaluation: {report.run_id}",
            "",
            "## Medoids",
            f"- TP/FP/FN: {medoid['tp']}/{medoid['fp']}/{medoid['fn']}",
            f"- Precision: {medoid['precision']:.3f}",
            f"- Recall: {medoid['recall']:.3f}",
            f"- F1: {medoid['f1']:.3f}",
            f"- Accuracy: {medoid['accuracy']:.3f}",
            "",
            "## Windows",
            f"- TP/FP/FN: {window['tp']}/{window['fp']}/{window['fn']}",
            f"- Precision: {window['precision']:.3f}",
            f"- Recall: {window['recall']:.3f}",
            f"- F1: {window['f1']:.3f}",
            f"- mAP@0.5: {window['map_at_0_5']:.3f}",
            "",
            "## Window Classification",
            f"- TP/FP/FN: {classification['tp']}/{classification['fp']}/{classification['fn']}",
            f"- Precision: {classification['precision']:.3f}",
            f"- Recall: {classification['recall']:.3f}",
            f"- F1: {classification['f1']:.3f}",
            f"- Accuracy on localized matches: {classification['accuracy']:.3f}",
            f"- Overall accuracy: {classification['overall_accuracy']:.3f}",
            "",
            "### Window Metrics by IoU",
            *threshold_lines,
            "",
        ]
    )


def _evaluation_log_markdown(report: EvaluationReport) -> str:
    primary_iou = float(report.window_classification_summary["primary_iou_threshold"])
    return "\n".join(
        [
            f"# PPE Evaluation Log: {report.run_id}",
            "",
            f"- Dataset: {report.dataset_id}",
            f"- Medoid Frechet threshold: {report.medoid_summary['frechet_threshold_nm']:.2f} NM",
            f"- Primary window IoU threshold: {primary_iou:.2f}",
            "",
            "## Clusters",
            *_cluster_log_lines(report),
            "",
            "## Windows",
            *_window_log_lines(report, primary_iou),
            "",
        ]
    )


def _cluster_log_lines(report: EvaluationReport) -> list[str]:
    lines = [
        "| Ground Truth | Model | Result | Distance | Why |",
        "| --- | --- | --- | ---: | --- |",
    ]
    matches = report.medoid_matches.copy()
    pairwise = report.pairwise_medoid_distances.copy()
    matched_gt = set(matches["gt_medoid_id"].astype(str).tolist()) if not matches.empty else set()
    matched_pred = set(matches["pred_cluster_id"].astype(int).tolist()) if not matches.empty else set()
    for row in matches.sort_values(["gt_medoid_id", "pred_cluster_id"]).to_dict("records"):
        distance = float(row["frechet_distance_nm"])
        lines.append(
            "| "
            f"{row['gt_medoid_id']} | "
            f"C{int(row['pred_cluster_id'])} | "
            "Correct | "
            f"{distance:.3f} NM | "
            f"Frechet distance is below {report.medoid_summary['frechet_threshold_nm']:.2f} NM |"
        )
    if not pairwise.empty:
        for gt_id in sorted(set(pairwise["gt_medoid_id"].astype(str)) - matched_gt):
            best = pairwise.loc[pairwise["gt_medoid_id"].astype(str) == gt_id].sort_values("distance_nm").iloc[0]
            lines.append(
                "| "
                f"{gt_id} | "
                f"C{int(best['pred_cluster_id'])} | "
                "Incorrect | "
                f"{float(best['distance_nm']):.3f} NM | "
                f"Best available model cluster is above {report.medoid_summary['frechet_threshold_nm']:.2f} NM |"
            )
        for cluster_id in sorted(set(pairwise["pred_cluster_id"].astype(int)) - matched_pred):
            best = pairwise.loc[pairwise["pred_cluster_id"].astype(int) == cluster_id].sort_values("distance_nm").iloc[0]
            lines.append(
                "| "
                "None | "
                f"C{cluster_id} | "
                "Incorrect | "
                f"{float(best['distance_nm']):.3f} NM | "
                "Extra model cluster with no accepted ground-truth match |"
            )
    return lines


def _window_log_lines(report: EvaluationReport, primary_iou: float) -> list[str]:
    lines = [
        "| Ground Truth | Model | Overlap | Result | Why |",
        "| --- | --- | ---: | --- | --- |",
    ]
    rows = report.window_classification_matches
    if rows.empty:
        lines.append("| None | None | 0.0% | No windows | No ground-truth or model windows were available |")
        return lines
    primary = rows.loc[rows["iou_threshold"].astype(float) == primary_iou].copy()
    if primary.empty:
        lines.append("| None | None | 0.0% | No rows | No rows were produced at the primary IoU threshold |")
        return lines
    primary = primary.sort_values(["reason", "gt_medoid_id", "pred_cluster_id", "gt_window_id", "pred_window_id"], na_position="last")
    for row in primary.to_dict("records"):
        gt_label = _window_side_label(row, prefix="gt")
        pred_label = _window_side_label(row, prefix="pred")
        iou = float(row["iou"]) if pd.notna(row["iou"]) else 0.0
        result = _window_result(row)
        why = _window_reason(row, primary_iou)
        lines.append(f"| {gt_label} | {pred_label} | {_percent(iou)} | {result} | {why} |")
    return lines


def _window_side_label(row: dict[str, Any], *, prefix: str) -> str:
    window_id = row.get(f"{prefix}_window_id")
    class_name = row.get(f"{prefix}_class_name")
    start = row.get(f"{prefix}_start_station_index")
    end = row.get(f"{prefix}_end_station_index")
    if not _has_value(window_id):
        return "None"
    station_range = _station_range(start, end)
    label = str(window_id)
    if station_range:
        label += f": {station_range}"
    if _has_value(class_name):
        label += f" ({class_name})"
    return label


def _window_result(row: dict[str, Any]) -> str:
    reason = str(row.get("classification_reason") or row.get("reason"))
    if reason == "correct":
        return "Correct"
    if reason == "wrong_class":
        return "Incorrect"
    if reason == "unlocalized_prediction":
        return "False positive"
    if reason == "missed_ground_truth":
        return "False negative"
    return "Incorrect"


def _window_reason(row: dict[str, Any], primary_iou: float) -> str:
    reason = str(row.get("classification_reason") or row.get("reason"))
    if reason == "correct":
        return f"IoU >= {primary_iou:.2f} and class matches"
    if reason == "wrong_class":
        return f"IoU >= {primary_iou:.2f}, but class is {row.get('pred_class_name')} instead of {row.get('gt_class_name')}"
    if reason == "unlocalized_prediction":
        return f"Best same-medoid overlap is below {primary_iou:.2f}, so the model window is rejected"
    if reason == "missed_ground_truth":
        return "No model window was accepted for this ground-truth window"
    return str(row.get("reason") or "unmatched")


def _station_range(start: Any, end: Any) -> str:
    if not _has_value(start) or not _has_value(end):
        return ""
    return f"{int(float(start))}-{int(float(end))}"


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _has_value(value: Any) -> bool:
    return value is not None and pd.notna(value) and str(value) != ""
