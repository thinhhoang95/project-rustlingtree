"""Evaluate density and community clustering sweeps for the paper-June ablation."""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

CLUSTERING_ROOT = Path(__file__).resolve().parent
if str(CLUSTERING_ROOT) not in sys.path:
    sys.path.insert(0, str(CLUSTERING_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering_ablation_density_algorithms import DensityClusteringRun, run_density_sweeps
from clustering_ablation_kmeans import (
    DEFAULT_FRECHET_THRESHOLD_NM,
    PROJECT_ROOT,
    GroundTruthMedoid,
    load_json,
    make_ground_truth_instances,
    resolve_feature_inputs,
    resolve_path,
    resolve_run_dir,
)
from ppe_evaluation.artifacts import MedoidTrajectory
from ppe_evaluation.frechet import discrete_frechet_distance
from ppe_evaluation.matching import detection_confusion_summary
from ppe_evaluation.matching import match_medoids as match_evaluation_medoids
from vlm_ppe.clustering.features import FeatureSet
from vlm_ppe.clustering.medoid import compute_cluster_medoids
from vlm_ppe.schemas import ClusterMedoid


ORDINALS = {1: "1st", 2: "2nd"}


def run_density_ablation(args, kmeans_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    ground_truth_dir = resolve_path(args.ground_truth)
    if ground_truth_dir is None or not ground_truth_dir.exists():
        raise FileNotFoundError(f"ground truth directory does not exist: {args.ground_truth}")
    run_dir = resolve_run_dir(ground_truth_dir, args.run_dir)
    gt_instances, gt_metadata = make_ground_truth_instances(ground_truth_dir)

    output_dir = args.output_dir or (ground_truth_dir.parent / "paper-june" / "clustering-ablation")
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features, resampled, _, _ = resolve_feature_inputs(
        run_dir,
        features_path=args.features,
        resampled_path=args.resampled_tracks,
    )
    random_state = _resolve_random_state(args, run_dir)
    frechet_threshold = float(
        args.frechet_threshold_nm
        if args.frechet_threshold_nm is not None
        else DEFAULT_FRECHET_THRESHOLD_NM
    )

    runs = run_density_sweeps(features.standardized, random_state=random_state)
    metrics = pd.DataFrame([run.metrics for run in runs])
    metrics_path = output_dir / "density_sweep_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    scored = _score_runs(
        runs,
        features,
        resampled,
        gt_instances,
        frechet_threshold=frechet_threshold,
        noise_min_tracks=int(args.noise_min_tracks),
    )
    scores = scored["scores"]
    scores_path = output_dir / "density_match_scores.csv"
    scores.to_csv(scores_path, index=False)

    _write_frame(scored["assignments"], output_dir / "density_medoid_assignments.csv")
    _write_frame(scored["pairwise"], output_dir / "density_pairwise_medoid_distances.csv")
    _write_frame(scored["nearest"], output_dir / "density_nearest_ground_truth_by_cluster.csv")

    selected = select_top_density_runs(scores)
    selected = _write_selected_artifacts(
        selected,
        runs,
        scored["medoids_by_run_id"],
        features,
        resampled,
        gt_instances,
        output_dir,
    )
    selected["source_summary"] = scores_path.relative_to(PROJECT_ROOT).as_posix()
    selected_path = output_dir / "selected_density_scores.csv"
    selected.to_csv(selected_path, index=False)

    report_path = append_density_report(
        output_dir=output_dir,
        metrics_path=metrics_path,
        scores_path=scores_path,
        selected_path=selected_path,
        selected=selected,
    )

    summary = {
        "random_state": random_state,
        "sweep_metrics_path": metrics_path.as_posix(),
        "match_scores_path": scores_path.as_posix(),
        "selected_scores_path": selected_path.as_posix(),
        "medoid_assignments_path": (output_dir / "density_medoid_assignments.csv").as_posix(),
        "pairwise_medoid_distances_path": (output_dir / "density_pairwise_medoid_distances.csv").as_posix(),
        "nearest_ground_truth_by_cluster_path": (output_dir / "density_nearest_ground_truth_by_cluster.csv").as_posix(),
        "findings_path": report_path.as_posix(),
        "selected": _json_safe(selected.to_dict("records")),
    }
    if kmeans_summary is not None:
        kmeans_summary["density_and_community"] = summary
    return summary


def select_top_density_runs(scores: pd.DataFrame, *, per_algorithm: int = 2) -> pd.DataFrame:
    pruned = scores.loc[scores["mode"] == "pruned"].copy()
    if pruned.empty:
        return pruned
    pruned["cluster_count_delta"] = (
        pruned["pred_clusters"].astype(float) - pruned["gt_subclusters"].astype(float)
    ).abs()
    rows: list[pd.Series] = []
    for algorithm, group in pruned.groupby("algorithm", sort=True):
        ranked = group.sort_values(
            [
                "f1",
                "recall",
                "precision",
                "fp",
                "cluster_count_delta",
                "mean_nearest_gt_distance_nm",
                "run_id",
            ],
            ascending=[False, False, False, True, True, True, True],
            kind="stable",
        )
        for rank, (_, row) in enumerate(ranked.head(per_algorithm).iterrows(), start=1):
            base_method = f"{algorithm}_{ORDINALS.get(rank, str(rank))}"
            selected_modes = scores.loc[
                (scores["algorithm"] == algorithm)
                & (scores["run_id"] == row["run_id"])
                & (scores["mode"].isin(["all", "pruned"]))
            ].copy()
            mode_order = {"all": 0, "pruned": 1}
            selected_modes["_mode_order"] = selected_modes["mode"].map(mode_order).fillna(99)
            selected_modes = selected_modes.sort_values("_mode_order", kind="stable").drop(columns=["_mode_order"])
            for _, mode_row in selected_modes.iterrows():
                payload = mode_row.copy()
                payload["rank"] = int(rank)
                payload["base_method"] = base_method
                payload["method"] = f"{base_method} ({payload['mode']})"
                rows.append(payload)
    selected = pd.DataFrame(rows)
    if "cluster_count_delta" in selected:
        selected = selected.drop(columns=["cluster_count_delta"])
    return selected.reset_index(drop=True)


def _score_runs(
    runs: list[DensityClusteringRun],
    features: FeatureSet,
    resampled: pd.DataFrame,
    ground_truth: list[GroundTruthMedoid],
    *,
    frechet_threshold: float,
    noise_min_tracks: int,
) -> dict[str, Any]:
    medoid_cache: dict[tuple[int, ...], list[ClusterMedoid]] = {}
    medoids_by_run_id: dict[str, list[ClusterMedoid]] = {}
    score_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    nearest_frames: list[pd.DataFrame] = []

    for run in runs:
        labels_key = tuple(int(value) for value in run.labels.tolist())
        if labels_key not in medoid_cache:
            medoid_cache[labels_key] = _compute_density_medoids(features, resampled, run.labels)
        medoids = medoid_cache[labels_key]
        medoids_by_run_id[run.run_id] = medoids

        all_predicted = [_cluster_medoid_to_evaluation_medoid(run.run_id, medoid) for medoid in medoids]
        pruned_medoids = [medoid for medoid in medoids if int(medoid.n_tracks) >= int(noise_min_tracks)]
        pruned_predicted = [
            _cluster_medoid_to_evaluation_medoid(run.run_id, medoid) for medoid in pruned_medoids
        ]

        for mode, predicted in [("all", all_predicted), ("pruned", pruned_predicted)]:
            summary, assignments, pairwise = _match_density_medoids(
                run,
                ground_truth,
                predicted,
                threshold_nm=frechet_threshold,
                mode=mode,
            )
            score_rows.append(summary)
            if not assignments.empty:
                assignment_frames.append(assignments)
            if not pairwise.empty:
                pairwise_frames.append(pairwise)
        nearest = _nearest_ground_truth_by_cluster(run.run_id, run.algorithm, medoids, ground_truth)
        if not nearest.empty:
            nearest_frames.append(nearest)

    return {
        "scores": pd.DataFrame(score_rows),
        "assignments": pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame(),
        "pairwise": pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame(),
        "nearest": pd.concat(nearest_frames, ignore_index=True) if nearest_frames else pd.DataFrame(),
        "medoids_by_run_id": medoids_by_run_id,
    }


def _compute_density_medoids(
    features: FeatureSet,
    resampled: pd.DataFrame,
    labels: np.ndarray,
) -> list[ClusterMedoid]:
    clustered = labels >= 0
    if not bool(clustered.any()):
        return []
    label_frame = pd.DataFrame(
        {
            "flight_id": np.asarray(features.track_ids, dtype=object)[clustered],
            "cluster_id": labels[clustered].astype(int),
        }
    )
    return compute_cluster_medoids(resampled, label_frame)


def _match_density_medoids(
    run: DensityClusteringRun,
    ground_truth: list[GroundTruthMedoid],
    predicted: list[MedoidTrajectory],
    *,
    threshold_nm: float,
    mode: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    result = match_evaluation_medoids(
        [item.medoid for item in ground_truth],
        predicted,
        threshold_nm=threshold_nm,
    )
    gt_by_instance = {item.instance_id: item for item in ground_truth}
    pred_by_medoid_id = {item.medoid_id: item for item in predicted}

    assignment_rows: list[dict[str, Any]] = []
    for assignment in result.assignments:
        gt_item = gt_by_instance[assignment.gt_medoid_id]
        pred_medoid = pred_by_medoid_id[assignment.pred_medoid_id]
        assignment_rows.append(
            {
                **_run_identity(run),
                "mode": mode,
                "gt_instance_id": gt_item.instance_id,
                "gt_medoid_id": gt_item.original_id,
                "pred_medoid_id": assignment.pred_medoid_id,
                "pred_cluster_id": assignment.pred_cluster_id,
                "pred_medoid_track_id": pred_medoid.medoid_track_id,
                "distance_nm": assignment.distance_nm,
                "threshold_nm": float(threshold_nm),
                "matched": assignment.matched,
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    for row in result.pairwise_distances.to_dict("records"):
        gt_item = gt_by_instance[str(row["gt_medoid_id"])]
        pred_medoid = pred_by_medoid_id[str(row["pred_medoid_id"])]
        pairwise_rows.append(
            {
                **_run_identity(run),
                "mode": mode,
                "gt_instance_id": gt_item.instance_id,
                "gt_medoid_id": gt_item.original_id,
                "pred_medoid_id": row["pred_medoid_id"],
                "pred_cluster_id": int(row["pred_cluster_id"]),
                "pred_medoid_track_id": pred_medoid.medoid_track_id,
                "distance_nm": float(row["distance_nm"]),
            }
        )

    summary = _density_match_summary(
        run,
        mode,
        n_gt=len(ground_truth),
        n_pred=len(predicted),
        tp=len(result.matches),
        threshold_nm=threshold_nm,
    )
    pairwise_frame = pd.DataFrame(pairwise_rows)
    if not pairwise_frame.empty:
        nearest = pairwise_frame.sort_values("distance_nm", kind="stable").groupby(
            ["run_id", "mode", "gt_instance_id"],
            as_index=False,
            sort=False,
        ).first()
        summary["mean_nearest_gt_distance_nm"] = float(nearest["distance_nm"].mean())
        summary["max_nearest_gt_distance_nm"] = float(nearest["distance_nm"].max())
    return summary, pd.DataFrame(assignment_rows), pairwise_frame


def _density_match_summary(
    run: DensityClusteringRun,
    mode: str,
    *,
    n_gt: int,
    n_pred: int,
    tp: int,
    threshold_nm: float,
) -> dict[str, Any]:
    metrics = detection_confusion_summary(n_gt=n_gt, n_pred=n_pred, n_tp=tp)
    return {
        **_run_identity(run),
        "mode": mode,
        "k": math.nan,
        "threshold_nm": float(threshold_nm),
        "gt_subclusters": int(n_gt),
        "pred_clusters": int(n_pred),
        **metrics,
        "same_count_as_ground_truth": bool(n_pred == n_gt),
        "mean_nearest_gt_distance_nm": math.nan,
        "max_nearest_gt_distance_nm": math.nan,
        "noise_tracks": int(run.metrics["noise_tracks"]),
        "noise_fraction": float(run.metrics["noise_fraction"]),
        "n_clustered_tracks": int(run.metrics["n_clustered_tracks"]),
    }


def _run_identity(run: DensityClusteringRun) -> dict[str, Any]:
    return {
        "algorithm": run.algorithm,
        "run_id": run.run_id,
        "params": _params_json(run.params),
    }


def _cluster_medoid_to_evaluation_medoid(run_id: str, medoid: ClusterMedoid) -> MedoidTrajectory:
    return MedoidTrajectory(
        medoid_id=f"{run_id}_C{int(medoid.cluster_id):02d}",
        points=np.asarray(medoid.template_points, dtype=float),
        cluster_id=int(medoid.cluster_id),
        medoid_track_id=medoid.medoid_track_id,
    )


def _nearest_ground_truth_by_cluster(
    run_id: str,
    algorithm: str,
    medoids: list[ClusterMedoid],
    ground_truth: list[GroundTruthMedoid],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for medoid in medoids:
        pred = _cluster_medoid_to_evaluation_medoid(run_id, medoid)
        best_gt = None
        best_distance = math.inf
        for gt_item in ground_truth:
            distance = discrete_frechet_distance(gt_item.medoid.points, pred.points)
            if distance < best_distance:
                best_distance = float(distance)
                best_gt = gt_item
        rows.append(
            {
                "algorithm": algorithm,
                "run_id": run_id,
                "cluster_id": int(medoid.cluster_id),
                "medoid_track_id": medoid.medoid_track_id,
                "n_tracks": int(medoid.n_tracks),
                "gt_instance_id": best_gt.instance_id if best_gt is not None else "",
                "gt_medoid_id": best_gt.original_id if best_gt is not None else "",
                "distance_nm": float(best_distance),
            }
        )
    return pd.DataFrame(rows)


def _write_selected_artifacts(
    selected: pd.DataFrame,
    runs: list[DensityClusteringRun],
    medoids_by_run_id: dict[str, list[ClusterMedoid]],
    features: FeatureSet,
    resampled: pd.DataFrame,
    ground_truth: list[GroundTruthMedoid],
    output_dir: Path,
) -> pd.DataFrame:
    if selected.empty:
        return selected
    run_by_id = {run.run_id: run for run in runs}
    labels_dir = output_dir / "labels" / "density"
    medoids_dir = output_dir / "medoids" / "density"
    contact_dir = output_dir / "figures" / "contact_sheets"
    labels_dir.mkdir(parents=True, exist_ok=True)
    medoids_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    selected = selected.copy()
    written: dict[str, dict[str, str]] = {}
    for index, row in selected.iterrows():
        base_method = str(row.get("base_method") or str(row["method"]).split(" (", 1)[0])
        run = run_by_id[str(row["run_id"])]
        medoids = medoids_by_run_id[str(run.run_id)]

        if base_method not in written:
            label_path = labels_dir / f"{base_method}_labels.csv"
            pd.DataFrame({"flight_id": features.track_ids, "cluster_id": run.labels.astype(int)}).to_csv(
                label_path,
                index=False,
            )

            medoid_path = medoids_dir / f"{base_method}_medoids.parquet"
            _write_medoids_parquet(medoids, medoid_path, run.run_id, run.algorithm)

            contact_path = render_density_contact_sheet(
                method=base_method,
                run=run,
                medoids=medoids,
                features=features,
                resampled=resampled,
                ground_truth=ground_truth,
                output_dir=contact_dir,
            )
            written[base_method] = {
                "label_path": label_path.relative_to(PROJECT_ROOT).as_posix(),
                "medoid_path": medoid_path.relative_to(PROJECT_ROOT).as_posix(),
                "contact_sheet_path": contact_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        for column, value in written[base_method].items():
            selected.loc[index, column] = value
    return selected


def _write_medoids_parquet(
    medoids: list[ClusterMedoid],
    path: Path,
    run_id: str,
    algorithm: str,
) -> None:
    rows: list[dict[str, Any]] = []
    for medoid in medoids:
        for station_index, (x_nm, y_nm) in enumerate(medoid.template_points):
            rows.append(
                {
                    "algorithm": algorithm,
                    "run_id": run_id,
                    "cluster_id": int(medoid.cluster_id),
                    "medoid_track_id": medoid.medoid_track_id,
                    "n_tracks": int(medoid.n_tracks),
                    "mean_distance_nm": float(medoid.mean_distance_nm),
                    "max_distance_nm": float(medoid.max_distance_nm),
                    "station_index": int(station_index),
                    "x_nm": float(x_nm),
                    "y_nm": float(y_nm),
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def render_density_contact_sheet(
    *,
    method: str,
    run: DensityClusteringRun,
    medoids: list[ClusterMedoid],
    features: FeatureSet,
    resampled: pd.DataFrame,
    ground_truth: list[GroundTruthMedoid],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{method}_contact_sheet.png"
    if not medoids:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"{method}: no non-noise clusters")
        ax.axis("off")
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return path

    labels = pd.DataFrame({"flight_id": features.track_ids, "cluster_id": run.labels.astype(int)})
    nearest = _nearest_ground_truth_by_cluster(run.run_id, run.algorithm, medoids, ground_truth)
    clusters = sorted(int(item.cluster_id) for item in medoids)
    cols = min(3, max(1, len(clusters)))
    rows = int(math.ceil(len(clusters) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.5 * rows), squeeze=False)
    grouped = {str(flight_id): group for flight_id, group in resampled.groupby("flight_id", sort=False)}
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for panel_index, cluster_id in enumerate(clusters):
        ax = axes[panel_index // cols][panel_index % cols]
        track_ids = labels.loc[labels["cluster_id"].astype(int) == cluster_id, "flight_id"].astype(str).tolist()
        color = colors[cluster_id % len(colors)] if colors else "#1f77b4"
        for track_id in track_ids:
            group = grouped.get(track_id)
            if group is None:
                continue
            ordered = group.sort_values("station_index", kind="stable")
            ax.plot(ordered["x_nm"], ordered["y_nm"], color=color, linewidth=0.65, alpha=0.18)
        medoid = next(item for item in medoids if int(item.cluster_id) == cluster_id)
        points = np.asarray(medoid.template_points, dtype=float)
        ax.plot(points[:, 0], points[:, 1], color="black", linewidth=2.2, label="medoid")
        for gt_item in ground_truth:
            gt_points = np.asarray(gt_item.medoid.points, dtype=float)
            ax.plot(gt_points[:, 0], gt_points[:, 1], color="#777777", linestyle="--", linewidth=0.55, alpha=0.22)
        nearest_row = nearest.loc[nearest["cluster_id"] == cluster_id].iloc[0]
        ax.set_title(
            f"C{cluster_id} n={len(track_ids)} | medoid {medoid.medoid_track_id}\n"
            f"nearest {nearest_row['gt_instance_id']} d={nearest_row['distance_nm']:.2f} NM",
            fontsize=9,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("x (NM)")
        ax.set_ylabel("y (NM)")
    for empty_index in range(len(clusters), rows * cols):
        axes[empty_index // cols][empty_index % cols].axis("off")
    fig.suptitle(f"{method}: {run.algorithm} selected parameter set")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def append_density_report(
    *,
    output_dir: Path,
    metrics_path: Path,
    scores_path: Path,
    selected_path: Path,
    selected: pd.DataFrame,
) -> Path:
    report_path = output_dir / "findings.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Clustering Ablation Findings\n"
    lines = [
        existing.rstrip(),
        "",
        "## Density and Community Sweep",
        "",
        "| Method | Algorithm | Pred clusters | Noise tracks | TP | FP | TN | FN | Precision | Recall | F1 | Params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| {row['method']} | {row['algorithm']} | {int(row['pred_clusters'])} | "
            f"{int(row['noise_tracks'])} | {int(row['tp'])} | {int(row['fp'])} | "
            f"{int(row.get('tn', 0))} | {int(row['fn'])} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | `{row['params']}` |"
        )
    lines.extend(
        [
            "",
            "Additional density/community output files:",
            "",
            f"- `{metrics_path.relative_to(PROJECT_ROOT).as_posix()}`: intrinsic metrics for every swept parameter set.",
            f"- `{scores_path.relative_to(PROJECT_ROOT).as_posix()}`: medoid scores for every swept parameter set.",
            f"- `{selected_path.relative_to(PROJECT_ROOT).as_posix()}`: top two selected rows per algorithm.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _resolve_random_state(args, run_dir: Path) -> int:
    if args.random_state is not None:
        return int(args.random_state)
    state = load_json(run_dir / "state.json")
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    return int(config.get("kmeans_random_state", 17))


def _params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
