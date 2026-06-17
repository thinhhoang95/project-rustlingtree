from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from ppe_evaluation.artifacts import MedoidTrajectory, load_ground_truth
from ppe_evaluation.frechet import discrete_frechet_distance
from ppe_evaluation.matching import (
    DEFAULT_FRECHET_THRESHOLD_NM,
    detection_confusion_summary,
    match_medoids as match_evaluation_medoids,
)
from vlm_ppe.clustering.features import FeatureSet, build_shape_features, load_features
from vlm_ppe.clustering.medoid import compute_cluster_medoids
from vlm_ppe.io.parquet_store import read_parquet
from vlm_ppe.schemas import ClusterMedoid


DEFAULT_GROUND_TRUTH_DIR = Path("data/artifacts/ppe/2026-04-01/ground_truth")
DEFAULT_NOISE_MIN_TRACKS = 4


@dataclass(frozen=True)
class KMeansResult:
    k: int
    labels: np.ndarray
    metrics: dict[str, Any]
    medoids: list[ClusterMedoid]


@dataclass(frozen=True)
class GroundTruthMedoid:
    instance_id: str
    original_id: str
    medoid: MedoidTrajectory


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value)
    if path.exists():
        return path
    if not path.is_absolute():
        rooted = PROJECT_ROOT / path
        if rooted.exists():
            return rooted
        if base_dir is not None:
            based = base_dir / path
            if based.exists():
                return based
    return path


def resolve_run_dir(ground_truth_dir: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        resolved = resolve_path(run_dir)
        if resolved is not None and resolved.exists():
            return resolved
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    manifest = load_json(ground_truth_dir / "manifest.json")
    source_run_dir = resolve_path(manifest.get("source_run_dir"))
    if source_run_dir is not None and source_run_dir.exists():
        return source_run_dir

    seed_run_id = str(manifest.get("seed_run_id") or "").strip()
    if seed_run_id:
        candidate = ground_truth_dir.parent / "runs" / seed_run_id
        if candidate.exists():
            return candidate

    raise ValueError(
        "could not infer source run directory from ground truth manifest; pass --run-dir explicitly"
    )


def resolve_feature_inputs(
    run_dir: Path,
    *,
    features_path: Path | None,
    resampled_path: Path | None,
) -> tuple[FeatureSet, pd.DataFrame, Path | None, Path]:
    state = load_json(run_dir / "state.json")
    resolved_features = resolve_path(features_path or state.get("features_path"), base_dir=run_dir)
    resolved_resampled = resolve_path(resampled_path or state.get("resampled_tracks_path"), base_dir=run_dir)

    if resolved_resampled is None or not resolved_resampled.exists():
        default_resampled = run_dir / "processed" / "resampled_tracks.parquet"
        if default_resampled.exists():
            resolved_resampled = default_resampled
    if resolved_resampled is None or not resolved_resampled.exists():
        raise FileNotFoundError("resampled tracks are required; pass --resampled-tracks")

    resampled = read_parquet(resolved_resampled)
    if resolved_features is not None and resolved_features.exists():
        features = load_features(resolved_features)
    else:
        features = build_shape_features(resampled)
        resolved_features = None

    resampled_track_ids = set(resampled["flight_id"].astype(str).unique())
    missing = [track_id for track_id in features.track_ids if track_id not in resampled_track_ids]
    if missing:
        raise ValueError(f"feature matrix references tracks absent from resampled data: {missing[:5]}")
    return features, resampled, resolved_features, resolved_resampled


def make_ground_truth_instances(ground_truth_dir: Path) -> tuple[list[GroundTruthMedoid], dict[str, Any]]:
    truth = load_ground_truth(ground_truth_dir)
    totals = Counter(medoid.medoid_id for medoid in truth.medoids)
    seen: defaultdict[str, int] = defaultdict(int)
    instances: list[GroundTruthMedoid] = []
    duplicate_ids = sorted([medoid_id for medoid_id, count in totals.items() if count > 1])
    for medoid in truth.medoids:
        seen[medoid.medoid_id] += 1
        instance_id = medoid.medoid_id
        if totals[medoid.medoid_id] > 1:
            instance_id = f"{medoid.medoid_id}#{seen[medoid.medoid_id]}"
        instances.append(
            GroundTruthMedoid(
                instance_id=instance_id,
                original_id=medoid.medoid_id,
                medoid=MedoidTrajectory(
                    medoid_id=instance_id,
                    points=medoid.points,
                    cluster_id=medoid.cluster_id,
                    medoid_track_id=medoid.medoid_track_id,
                    source_run_id=medoid.source_run_id,
                ),
            )
        )
    metadata = {
        "dataset_id": truth.dataset_id,
        "n_ground_truth_subclusters": len(instances),
        "n_ground_truth_ids": len(totals),
        "duplicate_ground_truth_ids": duplicate_ids,
        "manifest": truth.manifest,
    }
    return instances, metadata


def run_kmeans_sweep(
    features: FeatureSet,
    resampled: pd.DataFrame,
    *,
    k_min: int,
    k_max: int,
    n_init: int,
    random_state: int,
) -> list[KMeansResult]:
    n_tracks = len(features.track_ids)
    if n_tracks < 2:
        raise ValueError("at least two tracks are required for a clustering ablation")

    effective_k_min = max(1, int(k_min))
    effective_k_max = min(int(k_max), n_tracks - 1)
    if effective_k_max < effective_k_min:
        raise ValueError(f"invalid K range: {effective_k_min}..{effective_k_max}")

    results: list[KMeansResult] = []
    for k in range(effective_k_min, effective_k_max + 1):
        model = KMeans(n_clusters=k, n_init=int(n_init), random_state=int(random_state))
        labels = model.fit_predict(features.standardized).astype(int)
        counts = np.bincount(labels, minlength=k)
        metrics = {
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette": math.nan,
            "calinski_harabasz": math.nan,
            "davies_bouldin": math.nan,
            "cluster_count_min": int(counts.min()),
            "cluster_count_max": int(counts.max()),
            "cluster_count_mean": float(counts.mean()),
        }
        unique_labels = np.unique(labels)
        if 1 < len(unique_labels) < n_tracks:
            metrics["silhouette"] = _metric_or_nan(silhouette_score, features.standardized, labels)
            metrics["calinski_harabasz"] = _metric_or_nan(calinski_harabasz_score, features.standardized, labels)
            metrics["davies_bouldin"] = _metric_or_nan(davies_bouldin_score, features.standardized, labels)

        label_frame = pd.DataFrame({"flight_id": features.track_ids, "cluster_id": labels})
        medoids = compute_cluster_medoids(resampled, label_frame)
        results.append(KMeansResult(k=k, labels=labels, metrics=metrics, medoids=medoids))
    return results


def _metric_or_nan(func, features: np.ndarray, labels: np.ndarray) -> float:
    try:
        value = float(func(features, labels))
    except ValueError:
        return math.nan
    return value


def cluster_medoid_to_evaluation_medoid(k: int, medoid: ClusterMedoid) -> MedoidTrajectory:
    return MedoidTrajectory(
        medoid_id=f"K{k:02d}_C{int(medoid.cluster_id):02d}",
        points=np.asarray(medoid.template_points, dtype=float),
        cluster_id=int(medoid.cluster_id),
        medoid_track_id=medoid.medoid_track_id,
    )


def match_medoids(
    ground_truth: list[GroundTruthMedoid],
    predicted: list[MedoidTrajectory],
    *,
    threshold_nm: float,
    k: int,
    mode: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    assignment_rows: list[dict[str, Any]] = []
    n_gt = len(ground_truth)
    n_pred = len(predicted)

    result = match_evaluation_medoids(
        [item.medoid for item in ground_truth],
        predicted,
        threshold_nm=threshold_nm,
    )
    gt_by_instance = {item.instance_id: item for item in ground_truth}
    pred_by_medoid_id = {item.medoid_id: item for item in predicted}

    for assignment in result.assignments:
        gt_item = gt_by_instance[assignment.gt_medoid_id]
        pred_medoid = pred_by_medoid_id[assignment.pred_medoid_id]
        assignment_rows.append(
            {
                "k": int(k),
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
                "k": int(k),
                "mode": mode,
                "gt_instance_id": gt_item.instance_id,
                "gt_medoid_id": gt_item.original_id,
                "pred_medoid_id": row["pred_medoid_id"],
                "pred_cluster_id": int(row["pred_cluster_id"]),
                "pred_medoid_track_id": pred_medoid.medoid_track_id,
                "distance_nm": float(row["distance_nm"]),
            }
        )

    summary = _match_summary(k, mode, n_gt=n_gt, n_pred=n_pred, tp=len(result.matches), threshold_nm=threshold_nm)
    pairwise_frame = pd.DataFrame(pairwise_rows)
    if not pairwise_frame.empty:
        nearest = pairwise_frame.sort_values("distance_nm", kind="stable").groupby(
            ["k", "mode", "gt_instance_id"],
            as_index=False,
            sort=False,
        ).first()
        summary["mean_nearest_gt_distance_nm"] = float(nearest["distance_nm"].mean())
        summary["max_nearest_gt_distance_nm"] = float(nearest["distance_nm"].max())
    return summary, pd.DataFrame(assignment_rows), pairwise_frame


def _match_summary(
    k: int,
    mode: str,
    *,
    n_gt: int,
    n_pred: int,
    tp: int,
    threshold_nm: float,
) -> dict[str, Any]:
    metrics = detection_confusion_summary(n_gt=n_gt, n_pred=n_pred, n_tp=tp)
    return {
        "k": int(k),
        "mode": mode,
        "threshold_nm": float(threshold_nm),
        "gt_subclusters": int(n_gt),
        "pred_clusters": int(n_pred),
        **metrics,
        "same_count_as_ground_truth": bool(n_pred == n_gt),
        "mean_nearest_gt_distance_nm": math.nan,
        "max_nearest_gt_distance_nm": math.nan,
    }


def select_k_by_indices(metrics: pd.DataFrame) -> dict[str, int]:
    selectors: dict[str, int] = {}
    if not metrics.empty:
        selectors["inertia_elbow"] = int(_select_inertia_elbow(metrics))
    for column, direction in [
        ("silhouette", "max"),
        ("calinski_harabasz", "max"),
        ("davies_bouldin", "min"),
    ]:
        valid = metrics.loc[metrics[column].notna()]
        if valid.empty:
            continue
        if direction == "max":
            row = valid.sort_values([column, "k"], ascending=[False, True], kind="stable").iloc[0]
        else:
            row = valid.sort_values([column, "k"], ascending=[True, True], kind="stable").iloc[0]
        selectors[column] = int(row["k"])
    return selectors


def _select_inertia_elbow(metrics: pd.DataFrame) -> int:
    valid = metrics.loc[metrics["inertia"].notna(), ["k", "inertia"]].sort_values("k", kind="stable")
    if len(valid) <= 2:
        return int(valid.iloc[-1]["k"])
    x = valid["k"].to_numpy(dtype=float)
    y = valid["inertia"].to_numpy(dtype=float)
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    if x_range <= 0.0 or y_range <= 0.0:
        return int(valid.iloc[0]["k"])
    points = np.column_stack(((x - x.min()) / x_range, (y - y.min()) / y_range))
    start = points[0]
    end = points[-1]
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm <= 0.0:
        return int(valid.iloc[0]["k"])
    deltas = points - start
    distances = np.abs((line[0] * deltas[:, 1] - line[1] * deltas[:, 0]) / line_norm)
    return int(valid.iloc[int(np.argmax(distances))]["k"])


def write_result_artifacts(
    results: list[KMeansResult],
    features: FeatureSet,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[int, Path], dict[int, Path]]:
    labels_dir = output_dir / "labels"
    medoids_dir = output_dir / "medoids"
    labels_dir.mkdir(parents=True, exist_ok=True)
    medoids_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([result.metrics for result in results])
    metrics_path = output_dir / "validity_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    label_paths: dict[int, Path] = {}
    medoid_paths: dict[int, Path] = {}
    for result in results:
        labels = pd.DataFrame({"flight_id": features.track_ids, "cluster_id": result.labels})
        label_path = labels_dir / f"k_{result.k:02d}_labels.csv"
        labels.to_csv(label_path, index=False)
        label_paths[result.k] = label_path

        rows: list[dict[str, Any]] = []
        for medoid in result.medoids:
            for station_index, (x_nm, y_nm) in enumerate(medoid.template_points):
                rows.append(
                    {
                        "k": int(result.k),
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
        medoid_path = medoids_dir / f"k_{result.k:02d}_medoids.parquet"
        pd.DataFrame(rows).to_parquet(medoid_path, index=False)
        medoid_paths[result.k] = medoid_path
    return metrics, label_paths, medoid_paths


def render_metrics_chart(metrics: pd.DataFrame, selectors: dict[str, int], gt_count: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validity_metrics.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    specs = [
        ("inertia", "Inertia", axes[0][0]),
        ("silhouette", "Silhouette", axes[0][1]),
        ("calinski_harabasz", "Calinski-Harabasz", axes[1][0]),
        ("davies_bouldin", "Davies-Bouldin", axes[1][1]),
    ]
    for column, title, ax in specs:
        ax.plot(metrics["k"], metrics[column], marker="o", linewidth=1.5)
        ax.axvline(gt_count, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8, label="GT count")
        for selector_name, selected_k in selectors.items():
            if selector_name == "inertia_elbow" and column != "inertia":
                continue
            if selector_name == column:
                ax.axvline(selected_k, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.8, label=selector_name)
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize="small")
    fig.suptitle("Classical K selection indices")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_contact_sheet(
    *,
    k: int,
    result: KMeansResult,
    features: FeatureSet,
    resampled: pd.DataFrame,
    ground_truth: list[GroundTruthMedoid],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame({"flight_id": features.track_ids, "cluster_id": result.labels})
    nearest = nearest_ground_truth_by_cluster(k, result.medoids, ground_truth)

    clusters = sorted(int(item.cluster_id) for item in result.medoids)
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
        medoid = next(item for item in result.medoids if int(item.cluster_id) == cluster_id)
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
    fig.suptitle(f"KMeans K={k}: identified clusters with medoid paths")
    fig.tight_layout()
    path = output_dir / f"k_{k:02d}_contact_sheet.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def nearest_ground_truth_by_cluster(
    k: int,
    medoids: list[ClusterMedoid],
    ground_truth: list[GroundTruthMedoid],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for medoid in medoids:
        pred = cluster_medoid_to_evaluation_medoid(k, medoid)
        best_gt = None
        best_distance = math.inf
        for gt_item in ground_truth:
            distance = discrete_frechet_distance(gt_item.medoid.points, pred.points)
            if distance < best_distance:
                best_distance = float(distance)
                best_gt = gt_item
        rows.append(
            {
                "k": int(k),
                "cluster_id": int(medoid.cluster_id),
                "medoid_track_id": medoid.medoid_track_id,
                "n_tracks": int(medoid.n_tracks),
                "gt_instance_id": best_gt.instance_id if best_gt is not None else "",
                "gt_medoid_id": best_gt.original_id if best_gt is not None else "",
                "distance_nm": float(best_distance),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    *,
    output_dir: Path,
    config: dict[str, Any],
    gt_metadata: dict[str, Any],
    selectors: dict[str, int],
    metrics: pd.DataFrame,
    scores: pd.DataFrame,
    selected_contact_sheets: dict[int, Path],
) -> Path:
    gt_count = int(gt_metadata["n_ground_truth_subclusters"])
    duplicate_ids = gt_metadata["duplicate_ground_truth_ids"]
    selected_rows = _selected_score_table(selectors, scores)
    best_all = _best_score(scores, mode="all")
    best_pruned = _best_score(scores, mode="pruned")
    exact_count_selectors = [name for name, k in selectors.items() if int(k) == gt_count]

    lines = [
        "# Clustering Ablation Findings",
        "",
        "## Inputs",
        "",
        f"- Dataset: `{gt_metadata['dataset_id']}`",
        f"- Ground truth: `{config['ground_truth_dir']}`",
        f"- Source run: `{config['run_dir']}`",
        f"- Resampled tracks: `{config['resampled_tracks_path']}`",
        f"- Feature matrix: `{config.get('features_path') or 'rebuilt from resampled tracks'}`",
        f"- K sweep: `{config['k_min']}..{config['k_max']}`",
        f"- KMeans: `n_init={config['n_init']}`, `random_state={config['random_state']}`",
        f"- Ground-truth subcluster medoids: `{gt_count}`",
        f"- Frechet match threshold: `{config['frechet_threshold_nm']:.3f} NM`",
        f"- Pruned/noisy cluster rule: discard clusters with fewer than `{config['noise_min_tracks']}` tracks",
    ]
    if duplicate_ids:
        lines.append(
            "- Ground-truth note: duplicate medoid IDs were present "
            f"({', '.join(f'`{item}`' for item in duplicate_ids)}); they were scored as separate instances."
        )

    lines.extend(
        [
            "",
            "## Validity-Index K Choices",
            "",
            "| Criterion | Selected K | Matches GT count? |",
            "| --- | ---: | --- |",
        ]
    )
    for name, selected_k in selectors.items():
        lines.append(f"| {name} | {selected_k} | {'yes' if selected_k == gt_count else 'no'} |")

    lines.extend(
        [
            "",
            "## Selected-K Medoid Matching",
            "",
            "| Criterion | Mode | K | Pred clusters | TP | FP | TN | FN | Precision | Recall | F1 | Mean nearest GT dist (NM) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selected_rows:
        lines.append(
            f"| {row['criterion']} | {row['mode']} | {int(row['k'])} | {int(row['pred_clusters'])} | "
            f"{int(row['tp'])} | {int(row['fp'])} | {int(row.get('tn', 0))} | {int(row['fn'])} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | "
            f"{row['mean_nearest_gt_distance_nm']:.3f} |"
        )

    lines.extend(["", "## Full-Sweep Best Case", ""])
    for label, row in [("All clusters", best_all), ("Pruned clusters", best_pruned)]:
        if row is None:
            continue
        lines.append(
            f"- {label}: best F1 `{row['f1']:.3f}` at K=`{int(row['k'])}` "
            f"with `{int(row['pred_clusters'])}` predicted clusters "
            f"(TP `{int(row['tp'])}`, FP `{int(row['fp'])}`, TN `{int(row.get('tn', 0))}`, FN `{int(row['fn'])}`)."
        )

    lines.extend(["", "## Interpretation", ""])
    if exact_count_selectors:
        lines.append(
            "- At least one validity index selected the ground-truth subcluster count "
            f"(`{gt_count}`): {', '.join(exact_count_selectors)}."
        )
    else:
        lines.append(
            "- None of the tested validity indices selected the ground-truth subcluster count "
            f"of `{gt_count}`."
        )
    oracle_count_scores = scores.loc[(scores["k"].astype(int) == gt_count) & (scores["mode"] == "all")]
    if not oracle_count_scores.empty:
        row = oracle_count_scores.iloc[0]
        lines.append(
            f"- Even forcing K to the ground-truth count gives F1 `{row['f1']:.3f}` "
            f"and recall `{row['recall']:.3f}` at the configured Frechet threshold."
        )
    if best_pruned is not None:
        lines.append(
            f"- Allowing noisy-cluster discard with the `{config['noise_min_tracks']}`-track rule "
            f"does not change the underlying K selected by the validity indices; it only changes "
            "the effective predicted-cluster count before medoid matching."
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `validity_metrics.csv`: KMeans validity metrics for the sweep.",
            "- `match_scores.csv`: medoid precision/recall/F1 for all and pruned clusters.",
            "- `medoid_assignments.csv`: Hungarian medoid assignments at each K.",
            "- `pairwise_medoid_distances.csv`: all GT/predicted medoid Frechet distances.",
            "- `figures/validity_metrics.png`: validity-index curves.",
            "- `figures/contact_sheets/*.png`: trajectory cluster contact sheets for selected K values.",
        ]
    )
    if selected_contact_sheets:
        lines.extend(["", "Rendered contact sheets:"])
        for k, path in sorted(selected_contact_sheets.items()):
            lines.append(f"- K={k}: `{path.as_posix()}`")

    report_path = output_dir / "findings.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _selected_score_table(selectors: dict[str, int], scores: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for criterion, selected_k in selectors.items():
        for mode in ["all", "pruned"]:
            match = scores.loc[(scores["k"].astype(int) == int(selected_k)) & (scores["mode"] == mode)]
            if match.empty:
                continue
            payload = match.iloc[0].to_dict()
            payload["criterion"] = criterion
            rows.append(payload)
    return rows


def _best_score(scores: pd.DataFrame, *, mode: str) -> dict[str, Any] | None:
    subset = scores.loc[scores["mode"] == mode]
    if subset.empty:
        return None
    row = subset.sort_values(["f1", "recall", "precision", "k"], ascending=[False, False, False, True], kind="stable").iloc[0]
    return row.to_dict()


def infer_default_k_bounds(run_dir: Path, gt_count: int) -> tuple[int, int, int, int]:
    state = load_json(run_dir / "state.json")
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    k_min = int(config.get("k_min", 1))
    configured_k_max = int(config.get("k_max", max(8, gt_count)))
    max_k_expansion = int(config.get("max_k_expansion", configured_k_max))
    k_max = max(configured_k_max, max_k_expansion, gt_count)
    n_init = int(config.get("kmeans_n_init", 50))
    random_state = int(config.get("kmeans_random_state", 17))
    return k_min, k_max, n_init, random_state


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    ground_truth_dir = resolve_path(args.ground_truth)
    if ground_truth_dir is None or not ground_truth_dir.exists():
        raise FileNotFoundError(f"ground truth directory does not exist: {args.ground_truth}")
    run_dir = resolve_run_dir(ground_truth_dir, args.run_dir)
    gt_instances, gt_metadata = make_ground_truth_instances(ground_truth_dir)
    gt_count = int(gt_metadata["n_ground_truth_subclusters"])

    default_k_min, default_k_max, default_n_init, default_random_state = infer_default_k_bounds(run_dir, gt_count)
    k_min = int(args.k_min if args.k_min is not None else default_k_min)
    k_max = int(args.k_max if args.k_max is not None else default_k_max)
    n_init = int(args.n_init if args.n_init is not None else default_n_init)
    random_state = int(args.random_state if args.random_state is not None else default_random_state)
    frechet_threshold = float(
        args.frechet_threshold_nm
        if args.frechet_threshold_nm is not None
        else DEFAULT_FRECHET_THRESHOLD_NM
    )

    output_dir = args.output_dir or (ground_truth_dir.parent / "paper-june" / "clustering-ablation")
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features, resampled, features_path, resampled_path = resolve_feature_inputs(
        run_dir,
        features_path=args.features,
        resampled_path=args.resampled_tracks,
    )
    results = run_kmeans_sweep(
        features,
        resampled,
        k_min=k_min,
        k_max=k_max,
        n_init=n_init,
        random_state=random_state,
    )
    metrics, label_paths, medoid_paths = write_result_artifacts(results, features, output_dir)
    selectors = select_k_by_indices(metrics)

    score_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    nearest_frames: list[pd.DataFrame] = []
    for result in results:
        all_predicted = [cluster_medoid_to_evaluation_medoid(result.k, medoid) for medoid in result.medoids]
        pruned_medoids = [
            medoid for medoid in result.medoids if int(medoid.n_tracks) >= int(args.noise_min_tracks)
        ]
        pruned_predicted = [cluster_medoid_to_evaluation_medoid(result.k, medoid) for medoid in pruned_medoids]
        for mode, predicted in [("all", all_predicted), ("pruned", pruned_predicted)]:
            summary, assignments, pairwise = match_medoids(
                gt_instances,
                predicted,
                threshold_nm=frechet_threshold,
                k=result.k,
                mode=mode,
            )
            score_rows.append(summary)
            if not assignments.empty:
                assignment_frames.append(assignments)
            if not pairwise.empty:
                pairwise_frames.append(pairwise)
        nearest_frames.append(nearest_ground_truth_by_cluster(result.k, result.medoids, gt_instances))

    scores = pd.DataFrame(score_rows)
    scores.to_csv(output_dir / "match_scores.csv", index=False)
    assignments_frame = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    assignments_frame.to_csv(output_dir / "medoid_assignments.csv", index=False)
    pairwise_frame = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    pairwise_frame.to_csv(output_dir / "pairwise_medoid_distances.csv", index=False)
    nearest_frame = pd.concat(nearest_frames, ignore_index=True) if nearest_frames else pd.DataFrame()
    nearest_frame.to_csv(output_dir / "nearest_ground_truth_by_cluster.csv", index=False)

    figures_dir = output_dir / "figures"
    metrics_chart = render_metrics_chart(metrics, selectors, gt_count, figures_dir)
    best_all = _best_score(scores, mode="all")
    best_pruned = _best_score(scores, mode="pruned")
    contact_k_values = set(selectors.values()) | {gt_count}
    if best_all is not None:
        contact_k_values.add(int(best_all["k"]))
    if best_pruned is not None:
        contact_k_values.add(int(best_pruned["k"]))
    result_by_k = {result.k: result for result in results}
    contact_sheets: dict[int, Path] = {}
    for k in sorted(contact_k_values):
        if k in result_by_k:
            contact_sheets[k] = render_contact_sheet(
                k=k,
                result=result_by_k[k],
                features=features,
                resampled=resampled,
                ground_truth=gt_instances,
                output_dir=figures_dir / "contact_sheets",
            )

    config = {
        "ground_truth_dir": ground_truth_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "features_path": features_path.as_posix() if features_path else None,
        "resampled_tracks_path": resampled_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "k_min": k_min,
        "k_max": k_max,
        "n_init": n_init,
        "random_state": random_state,
        "frechet_threshold_nm": frechet_threshold,
        "noise_min_tracks": int(args.noise_min_tracks),
        "n_tracks": len(features.track_ids),
        "n_resample": features.n_resample,
    }
    report_path = write_report(
        output_dir=output_dir,
        config=config,
        gt_metadata=gt_metadata,
        selectors=selectors,
        metrics=metrics,
        scores=scores,
        selected_contact_sheets=contact_sheets,
    )
    summary = {
        "config": config,
        "ground_truth": {key: value for key, value in gt_metadata.items() if key != "manifest"},
        "selectors": selectors,
        "validity_metrics_path": (output_dir / "validity_metrics.csv").as_posix(),
        "match_scores_path": (output_dir / "match_scores.csv").as_posix(),
        "medoid_assignments_path": (output_dir / "medoid_assignments.csv").as_posix(),
        "pairwise_medoid_distances_path": (output_dir / "pairwise_medoid_distances.csv").as_posix(),
        "nearest_ground_truth_by_cluster_path": (output_dir / "nearest_ground_truth_by_cluster.csv").as_posix(),
        "metrics_chart_path": metrics_chart.as_posix(),
        "contact_sheet_paths": {str(k): path.as_posix() for k, path in sorted(contact_sheets.items())},
        "findings_path": report_path.as_posix(),
        "label_paths": {str(k): path.as_posix() for k, path in sorted(label_paths.items())},
        "medoid_paths": {str(k): path.as_posix() for k, path in sorted(medoid_paths.items())},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paper June clustering ablation against PPE ground-truth subcluster medoids."
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--run-dir", type=Path, default=None, help="Source PPE run directory; defaults from GT manifest.")
    parser.add_argument("--features", type=Path, default=None, help="Feature npz. Defaults from source run state.")
    parser.add_argument(
        "--resampled-tracks",
        type=Path,
        default=None,
        help="Resampled tracks parquet. Defaults from source run state.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--frechet-threshold-nm",
        type=float,
        default=None,
        help=f"Medoid Frechet match threshold in NM. Defaults to {DEFAULT_FRECHET_THRESHOLD_NM:.1f}.",
    )
    parser.add_argument("--noise-min-tracks", type=int, default=DEFAULT_NOISE_MIN_TRACKS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_ablation(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
