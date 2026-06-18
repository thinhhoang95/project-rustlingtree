"""Peak-detection baseline for paper-June intervention-window experiments."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
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
from scipy.signal import find_peaks, peak_widths

from ppe_evaluation.metrics import EvaluationReport, evaluate_run
from vlm_ppe.schemas import InterventionWindow


DEFAULT_DATASET_ROOT = Path("data/artifacts/ppe/2026-04-01")
DEFAULT_OUTPUT_SUBDIR = Path("paper-june/window-ablation")


@dataclass(frozen=True)
class PeakDetectionConfig:
    iou_threshold: float = 0.1
    min_normalized_height: float = 0.35
    min_normalized_prominence: float = 0.15
    min_distance_stations: int = 6
    width_rel_height: float = 0.5
    padding_stations: int = 2
    smoothing_window: int = 5
    max_windows_per_cluster: int = 3

    def params_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PeakCandidate:
    start_row_index: int
    end_row_index: int
    peak_row_index: int
    score: float
    confidence: float


@dataclass(frozen=True)
class SourceRunArtifacts:
    run_dir: Path
    run_id: str
    dataset_id: str
    medoids_path: Path
    residual_profiles_path: Path
    resampled_tracks_path: Path | None
    cluster_assignments_path: Path | None


@dataclass(frozen=True)
class BaselineRunResult:
    source_run_id: str
    baseline_run_dir: Path
    windows_path: Path
    evaluation_dir: Path
    report: EvaluationReport


def detect_peak_windows(
    residual_profiles: pd.DataFrame,
    config: PeakDetectionConfig = PeakDetectionConfig(),
) -> list[InterventionWindow]:
    _require_columns(
        residual_profiles,
        {
            "cluster_id",
            "station_index",
            "s_fraction",
            "s_nm",
            "residual_energy_nm2",
            "heading_dispersion",
        },
        label="residual profiles",
    )
    windows: list[InterventionWindow] = []
    for cluster_id, group in residual_profiles.groupby("cluster_id", sort=True):
        windows.extend(detect_peak_windows_for_cluster(group, int(cluster_id), config))
    return windows


def detect_peak_windows_for_cluster(
    profile: pd.DataFrame,
    cluster_id: int,
    config: PeakDetectionConfig = PeakDetectionConfig(),
) -> list[InterventionWindow]:
    ordered = profile.sort_values("station_index", kind="stable").reset_index(drop=True)
    if len(ordered) < 3:
        return []

    energy = ordered["residual_energy_nm2"].to_numpy(dtype=float)
    normalized = _normalize_curve(energy)
    if float(np.nanmax(normalized)) < float(config.min_normalized_height):
        return []

    smoothed = _moving_average(normalized, int(config.smoothing_window))
    peaks, properties = find_peaks(
        smoothed,
        height=float(config.min_normalized_height),
        prominence=float(config.min_normalized_prominence),
        distance=max(1, int(config.min_distance_stations)),
    )
    if len(peaks) == 0:
        return []

    widths = peak_widths(smoothed, peaks, rel_height=float(config.width_rel_height))
    candidates = _peak_candidates(peaks, properties, widths, len(ordered), config)
    candidates = _merge_overlapping_candidates(candidates)
    if config.max_windows_per_cluster > 0:
        candidates = sorted(candidates, key=lambda item: (-item.score, item.start_row_index))[
            : int(config.max_windows_per_cluster)
        ]
    candidates = sorted(candidates, key=lambda item: (item.start_row_index, item.end_row_index))

    windows: list[InterventionWindow] = []
    for sequence, candidate in enumerate(candidates, start=1):
        start = ordered.iloc[candidate.start_row_index]
        end = ordered.iloc[candidate.end_row_index]
        peak = ordered.iloc[candidate.peak_row_index]
        start_s_nm = float(start["s_nm"])
        end_s_nm = float(end["s_nm"])
        window_slice = ordered.iloc[candidate.start_row_index : candidate.end_row_index + 1]
        windows.append(
            InterventionWindow(
                cluster_id=int(cluster_id),
                window_id=f"C{int(cluster_id)}_P{sequence}",
                class_name="other",
                confidence=float(np.clip(candidate.confidence, 0.0, 1.0)),
                visual_reason="Class-agnostic peak in normalized residual-energy curve.",
                start_station_index=int(start["station_index"]),
                end_station_index=int(end["station_index"]),
                start_s_fraction=float(start["s_fraction"]),
                end_s_fraction=float(end["s_fraction"]),
                start_s_nm=start_s_nm,
                end_s_nm=end_s_nm,
                length_nm=max(0.0, end_s_nm - start_s_nm),
                peak_residual_energy_nm2=max(0.0, float(peak["residual_energy_nm2"])),
                peak_heading_dispersion=float(np.clip(window_slice["heading_dispersion"].max(), 0.0, 1.0)),
                track_ids=[],
            )
        )
    return windows


def run_peak_detection_for_run(
    source_run_dir: str | Path,
    *,
    ground_truth_dir: str | Path,
    output_dir: str | Path,
    config: PeakDetectionConfig = PeakDetectionConfig(),
) -> BaselineRunResult:
    source = resolve_source_run_artifacts(Path(source_run_dir))
    residual_profiles = pd.read_parquet(source.residual_profiles_path)
    windows = detect_peak_windows(residual_profiles, config)

    baseline_run_dir = Path(output_dir) / "intervention-window-baselines" / "peak_energy" / source.run_id
    residuals_dir = baseline_run_dir / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    windows_path = residuals_dir / "intervention_windows.parquet"
    _write_windows(windows, windows_path)
    _write_baseline_state(source, baseline_run_dir, windows_path)

    evaluation_dir = baseline_run_dir / "evaluation_class_agnostic"
    report = evaluate_run(
        baseline_run_dir,
        ground_truth_dir,
        evaluation_dir,
        window_iou_threshold=float(config.iou_threshold),
        class_aware=False,
    )
    return BaselineRunResult(
        source_run_id=source.run_id,
        baseline_run_dir=baseline_run_dir,
        windows_path=windows_path,
        evaluation_dir=evaluation_dir,
        report=report,
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = _resolve_existing_path(args.dataset_root, "dataset root")
    ground_truth_dir = _resolve_existing_path(args.ground_truth or dataset_root / "ground_truth", "ground truth")
    output_dir = Path(args.output_dir or dataset_root / DEFAULT_OUTPUT_SUBDIR)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PeakDetectionConfig(
        iou_threshold=float(args.iou_threshold),
        min_normalized_height=float(args.min_normalized_height),
        min_normalized_prominence=float(args.min_normalized_prominence),
        min_distance_stations=int(args.min_distance_stations),
        width_rel_height=float(args.width_rel_height),
        padding_stations=int(args.padding_stations),
        smoothing_window=int(args.smoothing_window),
        max_windows_per_cluster=int(args.max_windows_per_cluster),
    )
    source_runs = (
        [_resolve_existing_path(path, "run directory") for path in args.run_dir]
        if args.run_dir
        else discover_source_runs(dataset_root)
    )
    if not source_runs:
        raise ValueError(f"no valid source runs found under {dataset_root / 'runs'}")

    rows: list[dict[str, Any]] = []
    baseline_results: list[BaselineRunResult] = []
    for source_run in source_runs:
        ppe_report = evaluate_ppe_run(source_run, ground_truth_dir, config)
        rows.append(
            comparison_row_from_report(
                ppe_report,
                family="PPE",
                method=f"PPE {ppe_report.run_id}",
                algorithm="vlm_ppe",
                source_run_id=source_run.name,
                params="",
                source_summary=ppe_report.output_dir / "summary.json" if ppe_report.output_dir else None,
                windows_path=source_run / "residuals" / "intervention_windows.parquet",
            )
        )

        baseline = run_peak_detection_for_run(
            source_run,
            ground_truth_dir=ground_truth_dir,
            output_dir=output_dir,
            config=config,
        )
        baseline_results.append(baseline)
        rows.append(
            comparison_row_from_report(
                baseline.report,
                family="Classical",
                method=f"Peak energy ({baseline.source_run_id})",
                algorithm="scipy.find_peaks",
                source_run_id=baseline.source_run_id,
                params=config.params_json(),
                source_summary=baseline.evaluation_dir / "summary.json",
                windows_path=baseline.windows_path,
            )
        )

    comparison = sort_comparison(pd.DataFrame(rows))
    artifact_paths = write_comparison_artifacts(comparison, output_dir)
    return {
        "dataset_root": dataset_root.as_posix(),
        "ground_truth_dir": ground_truth_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "source_runs": [path.as_posix() for path in source_runs],
        "baseline_runs": [result.baseline_run_dir.as_posix() for result in baseline_results],
        "config": asdict(config),
        "artifacts": {key: path.as_posix() for key, path in artifact_paths.items()},
    }


def evaluate_ppe_run(
    source_run_dir: Path,
    ground_truth_dir: Path,
    config: PeakDetectionConfig,
) -> EvaluationReport:
    return evaluate_run(
        source_run_dir,
        ground_truth_dir,
        source_run_dir / "evaluation_class_agnostic",
        window_iou_threshold=float(config.iou_threshold),
        class_aware=False,
    )


def discover_source_runs(dataset_root: str | Path) -> list[Path]:
    runs_root = Path(dataset_root) / "runs"
    if not runs_root.exists():
        return []
    source_runs: list[Path] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir() and not path.name.startswith("._")):
        try:
            resolve_source_run_artifacts(run_dir)
        except (FileNotFoundError, ValueError):
            continue
        source_runs.append(run_dir)
    return source_runs


def resolve_source_run_artifacts(run_dir: Path) -> SourceRunArtifacts:
    run_dir = run_dir.resolve()
    state = _load_json(run_dir / "state.json")
    dataset_id = _dataset_id_from_state_or_path(state, run_dir)
    run_id = str(state.get("run_id") or run_dir.name)
    medoids_path = _resolve_artifact_path(state.get("medoids_path"), run_dir / "templates" / "cluster_medoids.parquet", run_dir)
    residual_profiles_path = _resolve_artifact_path(
        state.get("residual_profiles_path"),
        run_dir / "residuals" / "residual_profiles.parquet",
        run_dir,
    )
    resampled_tracks_path = _resolve_optional_artifact_path(state.get("resampled_tracks_path"), run_dir, run_dir / "processed" / "resampled_tracks.parquet")
    cluster_assignments_path = _resolve_optional_artifact_path(
        state.get("cluster_assignments_path"),
        run_dir,
        run_dir / "templates" / "chosen_cluster_assignments.csv",
    )
    return SourceRunArtifacts(
        run_dir=run_dir,
        run_id=run_id,
        dataset_id=dataset_id,
        medoids_path=medoids_path,
        residual_profiles_path=residual_profiles_path,
        resampled_tracks_path=resampled_tracks_path,
        cluster_assignments_path=cluster_assignments_path,
    )


def comparison_row_from_report(
    report: EvaluationReport,
    *,
    family: str,
    method: str,
    algorithm: str,
    source_run_id: str,
    params: str,
    source_summary: Path | None,
    windows_path: Path | None,
) -> dict[str, Any]:
    medoid = report.medoid_summary
    window = report.window_summary
    classification = report.window_classification_summary
    return {
        "family": family,
        "method": method,
        "algorithm": algorithm,
        "source_run_id": source_run_id,
        "run_id": report.run_id,
        "params": params,
        "iou_threshold": float(window["primary_iou_threshold"]),
        "class_aware": bool(window["class_aware"]),
        "gt_windows": int(window["gt_windows"]),
        "pred_windows": int(window["pred_windows"]),
        "window_tp": int(window["tp"]),
        "window_fp": int(window["fp"]),
        "window_tn": int(window.get("tn", 0)),
        "window_fn": int(window["fn"]),
        "window_precision": float(window["precision"]),
        "window_recall": float(window["recall"]),
        "window_f1": float(window["f1"]),
        "window_map": float(window.get("map", 0.0)),
        "window_map_at_0_5": float(window.get("map_at_0_5", 0.0)),
        "classification_tp": int(classification["tp"]),
        "classification_fp": int(classification["fp"]),
        "classification_tn": int(classification.get("tn", 0)),
        "classification_fn": int(classification["fn"]),
        "classification_precision": float(classification["precision"]),
        "classification_recall": float(classification["recall"]),
        "classification_f1": float(classification["f1"]),
        "classification_accuracy": float(classification["accuracy"]),
        "classification_overall_accuracy": float(classification["overall_accuracy"]),
        "localized_matches": int(classification["localized_matches"]),
        "classification_correct": int(classification["correct"]),
        "classification_incorrect": int(classification["incorrect"]),
        "medoid_gt": int(medoid["gt_medoids"]),
        "medoid_pred": int(medoid["pred_medoids"]),
        "medoid_precision": float(medoid["precision"]),
        "medoid_recall": float(medoid["recall"]),
        "medoid_f1": float(medoid["f1"]),
        "medoid_threshold_nm": float(medoid["frechet_threshold_nm"]),
        "source_summary": source_summary.as_posix() if source_summary else "",
        "windows_path": windows_path.as_posix() if windows_path else "",
    }


def sort_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    family_order = {"PPE": 0, "Classical": 1}
    ordered = comparison.assign(_family_order=comparison["family"].map(family_order).fillna(99))
    ordered = ordered.sort_values(
        ["_family_order", "window_f1", "window_recall", "window_precision", "method"],
        ascending=[True, False, False, False, True],
        kind="stable",
    )
    return ordered.drop(columns=["_family_order"]).reset_index(drop=True)


def write_comparison_artifacts(comparison: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "intervention_window_comparison.csv"
    json_path = root / "intervention_window_comparison.json"
    md_path = root / "intervention_window_comparison.md"
    notebook_path = root / "intervention_window_comparison.ipynb"
    metric_bars_path = root / "intervention_window_metric_bars.png"
    confusion_counts_path = root / "intervention_window_confusion_counts.png"

    comparison.to_csv(csv_path, index=False)
    comparison.to_json(json_path, orient="records", indent=2)
    md_path.write_text(_comparison_markdown(comparison), encoding="utf-8")
    _render_metric_bars(comparison, metric_bars_path)
    _render_confusion_counts(comparison, confusion_counts_path)
    write_comparison_notebook(notebook_path)
    return {
        "csv": csv_path,
        "json": json_path,
        "markdown": md_path,
        "notebook": notebook_path,
        "metric_bars": metric_bars_path,
        "confusion_counts": confusion_counts_path,
    }


def write_comparison_notebook(path: str | Path) -> Path:
    notebook_path = Path(path)
    cells = [
        _markdown_cell(
            "# Intervention Window Comparison\n\n"
            "This notebook compares PPE intervention-window localization with a classical "
            "`scipy.signal.find_peaks` baseline over residual-energy profiles."
        ),
        _markdown_cell(
            "Localization rows use the shared PPE evaluator with class-agnostic IoU matching "
            "at threshold `0.1`. Classification metrics are reported separately."
        ),
        _code_cell(
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "ARTIFACT_DIR = Path.cwd()\n"
            "if not (ARTIFACT_DIR / 'intervention_window_comparison.csv').exists():\n"
            "    for candidate in [Path.cwd(), *Path.cwd().parents]:\n"
            "        path = candidate / 'data/artifacts/ppe/2026-04-01/paper-june/window-ablation'\n"
            "        if (path / 'intervention_window_comparison.csv').exists():\n"
            "            ARTIFACT_DIR = path\n"
            "            break\n"
            "comparison = pd.read_csv(ARTIFACT_DIR / 'intervention_window_comparison.csv')\n"
            "comparison"
        ),
        _code_cell(
            "plot = comparison.copy()\n"
            "plot['display_method'] = plot['method']\n"
            "ax = plot.set_index('display_method')[['window_precision', 'window_recall', 'window_f1']].plot.barh(figsize=(8, 0.45 * len(plot) + 2))\n"
            "ax.set_xlabel('Score')\n"
            "ax.set_xlim(0, 1.05)\n"
            "ax.grid(axis='x', alpha=0.25)\n"
            "ax.set_title('Class-Agnostic Window Localization')\n"
            "plt.tight_layout()"
        ),
        _code_cell(
            "counts = comparison.set_index('method')[['window_tp', 'window_fp', 'window_fn']]\n"
            "ax = counts.plot.barh(stacked=True, figsize=(7, 0.45 * len(counts) + 2), color=['#16a34a', '#dc2626', '#f59e0b'])\n"
            "ax.set_xlabel('Count')\n"
            "ax.grid(axis='x', alpha=0.25)\n"
            "ax.set_title('Window Localization Counts')\n"
            "plt.tight_layout()"
        ),
        _markdown_cell(
            "## Window Match Diagnostics\n\n"
            "The sheets below render the primary IoU-threshold rows from each method's "
            "`window_matches.csv`. Each panel is one counted localization decision: "
            "`TP` for a matched model/ground-truth pair, `FP` for an unmatched model "
            "window, and `FN` for a ground-truth window that was missed. Solid gray "
            "paths are ground-truth medoids, dashed gray paths are model medoids, blue "
            "segments are ground-truth windows, and red segments are model windows."
        ),
        _code_cell(
            "import math\n"
            "import re\n"
            "from types import SimpleNamespace\n"
            "from textwrap import fill\n\n"
            "import numpy as np\n"
            "from IPython.display import display\n\n"
            "from ppe_evaluation.artifacts import load_ground_truth\n\n"
            "DATASET_ROOT = ARTIFACT_DIR.parents[1]\n"
            "CONTACT_SHEET_DIR = ARTIFACT_DIR / 'figures' / 'intervention_window_contact_sheets'\n"
            "CONTACT_SHEET_DIR.mkdir(parents=True, exist_ok=True)\n\n"
            "def _has_value(value):\n"
            "    if value is None:\n"
            "        return False\n"
            "    try:\n"
            "        return not bool(pd.isna(value))\n"
            "    except (TypeError, ValueError):\n"
            "        return True\n\n"
            "def _safe_float(value):\n"
            "    return float(value) if _has_value(value) else np.nan\n\n"
            "def _safe_int(value):\n"
            "    return int(float(value)) if _has_value(value) else None\n\n"
            "def _safe_text(value):\n"
            "    return str(value) if _has_value(value) and str(value) != 'nan' else ''\n\n"
            "def _cluster_from_gt_window_id(window_id):\n"
            "    match = re.match(r'C(\\d+)_', _safe_text(window_id))\n"
            "    return int(match.group(1)) if match else None\n\n"
            "def _slug(value):\n"
            "    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')\n\n"
            "def _evaluation_dir(row):\n"
            "    summary = Path(str(row['source_summary']))\n"
            "    if not summary.is_absolute():\n"
            "        summary = ARTIFACT_DIR / summary\n"
            "    return summary.parent\n\n"
            "def _primary_match_rows(row):\n"
            "    matches = pd.read_csv(_evaluation_dir(row) / 'window_matches.csv')\n"
            "    threshold = float(row['iou_threshold'])\n"
            "    primary = matches.loc[np.isclose(matches['iou_threshold'].astype(float), threshold)].copy()\n"
            "    order = {'matched': 0, 'unmatched_prediction': 1, 'missed_ground_truth': 2}\n"
            "    primary['_order'] = primary['reason'].map(order).fillna(99)\n"
            "    return primary.sort_values(['_order', 'gt_medoid_id', 'pred_cluster_id', 'gt_window_id', 'pred_window_id'], na_position='last').drop(columns=['_order'])\n\n"
            "def _run_dir(row):\n"
            "    return _evaluation_dir(row).parent\n\n"
            "def _prediction_artifacts(row):\n"
            "    medoids_path = DATASET_ROOT / 'runs' / str(row['source_run_id']) / 'templates' / 'cluster_medoids.parquet'\n"
            "    medoids_frame = pd.read_parquet(medoids_path)\n"
            "    medoids = []\n"
            "    for cluster_id, group in medoids_frame.groupby('cluster_id', sort=True):\n"
            "        ordered = group.sort_values('station_index', kind='stable')\n"
            "        medoids.append(SimpleNamespace(cluster_id=int(cluster_id), points=ordered[['x_nm', 'y_nm']].to_numpy(dtype=float)))\n"
            "    windows_path = Path(str(row['windows_path']))\n"
            "    if not windows_path.is_absolute():\n"
            "        windows_path = ARTIFACT_DIR / windows_path\n"
            "    windows = pd.read_parquet(windows_path) if windows_path.exists() else pd.DataFrame()\n"
            "    return SimpleNamespace(medoids=medoids, windows=windows)\n\n"
            "def _ground_truth_artifacts():\n"
            "    return load_ground_truth(DATASET_ROOT / 'ground_truth')\n\n"
            "def _accounting_label(match_row):\n"
            "    reason = str(match_row['reason'])\n"
            "    if reason == 'matched':\n"
            "        return 'TP', '#15803d'\n"
            "    if reason == 'unmatched_prediction':\n"
            "        return 'FP', '#b91c1c'\n"
            "    if reason == 'missed_ground_truth':\n"
            "        return 'FN', '#b45309'\n"
            "    return reason, '#374151'\n\n"
            "def _gt_medoids_by_key(truth):\n"
            "    medoids = {}\n"
            "    for medoid in truth.medoids:\n"
            "        medoids[(str(medoid.medoid_id), medoid.cluster_id)] = medoid\n"
            "        medoids.setdefault((str(medoid.medoid_id), None), medoid)\n"
            "    return medoids\n\n"
            "def _pred_medoids_by_cluster(predictions):\n"
            "    return {int(medoid.cluster_id): medoid for medoid in predictions.medoids if medoid.cluster_id is not None}\n\n"
            "def _records_by_key(frame, key):\n"
            "    return {str(item[key]): item for item in frame.to_dict('records')} if not frame.empty else {}\n\n"
            "def _pred_windows_by_key(frame):\n"
            "    if frame.empty:\n"
            "        return {}\n"
            "    return {(int(item['cluster_id']), str(item['window_id'])): item for item in frame.to_dict('records')}\n\n"
            "def _gt_medoid_for_window(window, gt_medoids):\n"
            "    if window is None:\n"
            "        return None\n"
            "    gt_medoid_id = _safe_text(window.get('gt_medoid_id'))\n"
            "    source_cluster = _cluster_from_gt_window_id(window.get('window_id'))\n"
            "    return gt_medoids.get((gt_medoid_id, source_cluster)) or gt_medoids.get((gt_medoid_id, None))\n\n"
            "def _window_segment(points, window):\n"
            "    if points is None or window is None or len(points) == 0:\n"
            "        return None\n"
            "    start = _safe_int(window.get('start_station_index'))\n"
            "    end = _safe_int(window.get('end_station_index'))\n"
            "    if start is None or end is None:\n"
            "        return None\n"
            "    start = max(0, min(len(points) - 1, start))\n"
            "    end = max(0, min(len(points) - 1, end))\n"
            "    start, end = sorted((start, end))\n"
            "    return points[start : end + 1]\n\n"
            "def _plot_path(axis, points, *, color, linestyle, linewidth, label, alpha=0.85):\n"
            "    if points is None or len(points) == 0:\n"
            "        return\n"
            "    axis.plot(points[:, 0], points[:, 1], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)\n"
            "    axis.scatter(points[0, 0], points[0, 1], color=color, s=14, marker='o', zorder=3)\n"
            "    axis.scatter(points[-1, 0], points[-1, 1], color=color, s=22, marker='x', zorder=3)\n\n"
            "def _plot_window_segment(axis, points, window, *, color, linewidth, label):\n"
            "    segment = _window_segment(points, window)\n"
            "    if segment is None or len(segment) == 0:\n"
            "        return\n"
            "    axis.plot(segment[:, 0], segment[:, 1], color=color, linewidth=linewidth, alpha=0.92, solid_capstyle='round', label=label)\n"
            "    axis.scatter(segment[0, 0], segment[0, 1], color=color, s=24, marker='o', zorder=4)\n"
            "    axis.scatter(segment[-1, 0], segment[-1, 1], color=color, s=28, marker='s', zorder=4)\n\n"
            "def _set_path_limits(axis, *point_sets):\n"
            "    valid = [points for points in point_sets if points is not None and len(points) > 0]\n"
            "    if not valid:\n"
            "        return\n"
            "    combined = np.vstack(valid)\n"
            "    x_min, y_min = combined.min(axis=0)\n"
            "    x_max, y_max = combined.max(axis=0)\n"
            "    span = max(float(x_max - x_min), float(y_max - y_min), 1.0)\n"
            "    pad = span * 0.08\n"
            "    x_mid = float((x_min + x_max) / 2.0)\n"
            "    y_mid = float((y_min + y_max) / 2.0)\n"
            "    half = span / 2.0 + pad\n"
            "    axis.set_xlim(x_mid - half, x_mid + half)\n"
            "    axis.set_ylim(y_mid - half, y_mid + half)\n\n"
            "def _plot_match_panel(axis, truth, predictions, match_row, threshold):\n"
            "    label, color = _accounting_label(match_row)\n"
            "    gt_windows = _records_by_key(truth.windows, 'window_id')\n"
            "    pred_windows = _pred_windows_by_key(predictions.windows)\n"
            "    gt_medoids = _gt_medoids_by_key(truth)\n"
            "    pred_medoids = _pred_medoids_by_cluster(predictions)\n"
            "    gt_window_id = _safe_text(match_row.get('gt_window_id'))\n"
            "    pred_window_id = _safe_text(match_row.get('pred_window_id'))\n"
            "    pred_cluster_id = _safe_int(match_row.get('pred_cluster_id'))\n"
            "    gt_window = gt_windows.get(gt_window_id)\n"
            "    pred_window = pred_windows.get((pred_cluster_id, pred_window_id)) if pred_cluster_id is not None else None\n"
            "    gt_medoid = _gt_medoid_for_window(gt_window, gt_medoids)\n"
            "    pred_medoid = pred_medoids.get(pred_cluster_id) if pred_cluster_id is not None else None\n"
            "    gt_points = np.asarray(gt_medoid.points, dtype=float) if gt_medoid is not None else None\n"
            "    pred_points = np.asarray(pred_medoid.points, dtype=float) if pred_medoid is not None else None\n"
            "    if gt_points is None and pred_points is None:\n"
            "        axis.text(0.5, 0.5, 'No medoid path for this row', ha='center', va='center', transform=axis.transAxes)\n"
            "        axis.set_axis_off()\n"
            "        return\n"
            "    _plot_path(axis, gt_points, color='#4b5563', linestyle='-', linewidth=1.25, label='GT medoid')\n"
            "    _plot_path(axis, pred_points, color='#9ca3af', linestyle='--', linewidth=1.15, label='model medoid')\n"
            "    _plot_window_segment(axis, gt_points, gt_window, color='#2563eb', linewidth=5.0, label='GT window')\n"
            "    _plot_window_segment(axis, pred_points, pred_window, color='#dc2626', linewidth=4.2, label='model window')\n"
            "    _set_path_limits(axis, gt_points, pred_points)\n"
            "    iou = _safe_float(match_row.get('iou'))\n"
            "    cluster_label = f\"model C{pred_cluster_id}\" if pred_cluster_id is not None else 'no model cluster'\n"
            "    axis.set_title(f\"{label} {cluster_label} | IoU {iou:.3f} | threshold {threshold:.2f}\", color=color, fontsize=10, loc='left')\n"
            "    gt_label = f\"GT {gt_window_id or '-'} {_safe_text(match_row.get('gt_class_name')) or '-'} [{_safe_text(match_row.get('gt_start_station_index')) or '-'}-{_safe_text(match_row.get('gt_end_station_index')) or '-'}]\"\n"
            "    pred_label = f\"Model {pred_window_id or '-'} {_safe_text(match_row.get('pred_class_name')) or '-'} [{_safe_text(match_row.get('pred_start_station_index')) or '-'}-{_safe_text(match_row.get('pred_end_station_index')) or '-'}]\"\n"
            "    reason = str(match_row.get('reason'))\n"
            "    note = 'matched above threshold' if reason == 'matched' else ('model window counted as FP' if reason == 'unmatched_prediction' else 'ground-truth window counted as FN')\n"
            "    axis.text(0.01, 0.99, fill(f\"{gt_label}\\n{pred_label}\\n{note}\", 58), transform=axis.transAxes, va='top', ha='left', fontsize=8, bbox={'facecolor': 'white', 'edgecolor': '#d1d5db', 'alpha': 0.92, 'boxstyle': 'round,pad=0.28'})\n"
            "    axis.set_xlabel('x (NM)')\n"
            "    axis.set_ylabel('y (NM)')\n"
            "    axis.grid(True, alpha=0.22)\n"
            "    axis.set_aspect('equal', adjustable='box')\n"
            "    axis.legend(loc='lower right', fontsize=7)\n\n"
            "def render_window_match_contact_sheet(row, *, show=True):\n"
            "    rows = _primary_match_rows(row)\n"
            "    truth = _ground_truth_artifacts()\n"
            "    predictions = _prediction_artifacts(row)\n"
            "    if rows.empty:\n"
            "        return None\n"
            "    all_rows = rows.to_dict('records')\n"
            "    n_cols = 2\n"
            "    n_rows = math.ceil(len(rows) / n_cols)\n"
            "    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.4 * n_cols, 6.4 * n_rows), squeeze=False)\n"
            "    threshold = float(row['iou_threshold'])\n"
            "    title = f\"{row['method']} - 2D window accounting at IoU {threshold:.2f} (blue GT, red model)\"\n"
            "    fig.suptitle(title, fontsize=14)\n"
            "    for axis, match_row in zip(axes.ravel(), all_rows, strict=False):\n"
            "        _plot_match_panel(axis, truth, predictions, match_row, threshold)\n"
            "    for axis in axes.ravel()[len(rows):]:\n"
            "        axis.axis('off')\n"
            "    fig.tight_layout(rect=(0, 0, 1, 0.96))\n"
            "    path = CONTACT_SHEET_DIR / f\"{_slug(row['method'])}_window_match_contact_sheet.png\"\n"
            "    fig.savefig(path, dpi=180, bbox_inches='tight')\n"
            "    if show:\n"
            "        display(fig)\n"
            "    plt.close(fig)\n"
            "    return path\n\n"
            "def window_match_detail_table(row):\n"
            "    rows = _primary_match_rows(row).copy()\n"
            "    rows.insert(0, 'method', row['method'])\n"
            "    rows.insert(1, 'accounting', rows['reason'].map({'matched': 'TP', 'unmatched_prediction': 'FP', 'missed_ground_truth': 'FN'}).fillna(rows['reason']))\n"
            "    return rows[[\n"
            "        'method', 'accounting', 'reason', 'gt_medoid_id', 'pred_cluster_id', 'gt_window_id', 'pred_window_id',\n"
            "        'gt_class_name', 'pred_class_name', 'gt_start_s_fraction', 'gt_end_s_fraction',\n"
            "        'pred_start_s_fraction', 'pred_end_s_fraction', 'iou'\n"
            "    ]]\n"
        ),
        _code_cell(
            "match_details = pd.concat([window_match_detail_table(row) for row in comparison.to_dict('records')], ignore_index=True)\n"
            "display(match_details)\n"
            "contact_sheet_paths = [render_window_match_contact_sheet(row) for row in comparison.to_dict('records')]\n"
            "contact_sheet_paths"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return notebook_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper-June intervention-window peak baseline.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--min-normalized-height", type=float, default=0.35)
    parser.add_argument("--min-normalized-prominence", type=float, default=0.15)
    parser.add_argument("--min-distance-stations", type=int, default=6)
    parser.add_argument("--width-rel-height", type=float, default=0.5)
    parser.add_argument("--padding-stations", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--max-windows-per-cluster", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2))
    return 0


def _peak_candidates(
    peaks: np.ndarray,
    properties: dict[str, np.ndarray],
    widths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_points: int,
    config: PeakDetectionConfig,
) -> list[PeakCandidate]:
    left_ips = widths[2]
    right_ips = widths[3]
    prominences = properties.get("prominences", np.zeros(len(peaks), dtype=float))
    heights = properties.get("peak_heights", np.ones(len(peaks), dtype=float))
    candidates: list[PeakCandidate] = []
    for index, peak in enumerate(peaks.tolist()):
        start = int(math.floor(float(left_ips[index]))) - int(config.padding_stations)
        end = int(math.ceil(float(right_ips[index]))) + int(config.padding_stations)
        start = max(0, min(n_points - 1, start))
        end = max(0, min(n_points - 1, end))
        if end <= start:
            start = max(0, int(peak) - 1)
            end = min(n_points - 1, int(peak) + 1)
        height = float(heights[index])
        prominence = float(prominences[index])
        candidates.append(
            PeakCandidate(
                start_row_index=start,
                end_row_index=end,
                peak_row_index=int(peak),
                score=height + prominence,
                confidence=max(height, prominence),
            )
        )
    return candidates


def _merge_overlapping_candidates(candidates: list[PeakCandidate]) -> list[PeakCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (item.start_row_index, item.end_row_index))
    merged: list[PeakCandidate] = [ordered[0]]
    for candidate in ordered[1:]:
        previous = merged[-1]
        if candidate.start_row_index > previous.end_row_index:
            merged.append(candidate)
            continue
        peak_holder = candidate if candidate.score > previous.score else previous
        merged[-1] = PeakCandidate(
            start_row_index=min(previous.start_row_index, candidate.start_row_index),
            end_row_index=max(previous.end_row_index, candidate.end_row_index),
            peak_row_index=peak_holder.peak_row_index,
            score=max(previous.score, candidate.score),
            confidence=max(previous.confidence, candidate.confidence),
        )
    return merged


def _normalize_curve(values: np.ndarray) -> np.ndarray:
    curve = np.asarray(values, dtype=float)
    finite = np.isfinite(curve)
    if not finite.any():
        return np.zeros_like(curve, dtype=float)
    cleaned = curve.copy()
    fill = float(np.nanmedian(cleaned[finite]))
    cleaned[~finite] = fill
    minimum = float(np.min(cleaned))
    maximum = float(np.max(cleaned))
    if maximum <= minimum:
        return np.zeros_like(cleaned, dtype=float)
    return (cleaned - minimum) / (maximum - minimum)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    width = int(window)
    if width <= 1 or len(values) < 3:
        return values
    if width % 2 == 0:
        width += 1
    width = min(width, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if width <= 1:
        return values
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _write_windows(windows: list[InterventionWindow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [window.model_dump() for window in windows]
    pd.DataFrame(rows, columns=list(InterventionWindow.model_fields)).to_parquet(path, index=False)


def _write_baseline_state(source: SourceRunArtifacts, baseline_run_dir: Path, windows_path: Path) -> None:
    state: dict[str, Any] = {
        "run_id": f"peak-energy-{source.run_id}",
        "run_dir": baseline_run_dir.as_posix(),
        "config": {
            "dataset_id": source.dataset_id,
            "baseline": "peak_energy",
            "source_run_id": source.run_id,
        },
        "medoids_path": source.medoids_path.as_posix(),
        "residual_profiles_path": source.residual_profiles_path.as_posix(),
        "intervention_windows_path": windows_path.as_posix(),
    }
    if source.resampled_tracks_path is not None:
        state["resampled_tracks_path"] = source.resampled_tracks_path.as_posix()
    if source.cluster_assignments_path is not None:
        state["cluster_assignments_path"] = source.cluster_assignments_path.as_posix()
    (baseline_run_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _comparison_markdown(comparison: pd.DataFrame) -> str:
    pretty = comparison.copy()
    for column in [
        "iou_threshold",
        "window_precision",
        "window_recall",
        "window_f1",
        "window_map",
        "window_map_at_0_5",
        "classification_precision",
        "classification_recall",
        "classification_f1",
        "classification_accuracy",
        "classification_overall_accuracy",
        "medoid_precision",
        "medoid_recall",
        "medoid_f1",
        "medoid_threshold_nm",
    ]:
        pretty[column] = pretty[column].map(lambda value: f"{float(value):.3f}" if pd.notna(value) else "")
    return (
        "# Intervention Window Comparison\n\n"
        "Localization uses class-agnostic PPE window matching with IoU threshold `0.1`. "
        "Classification performance is reported separately and is not mixed into localization scoring.\n\n"
        + _dataframe_to_markdown(pretty)
        + "\n"
    )


def _render_metric_bars(comparison: pd.DataFrame, path: Path) -> None:
    if comparison.empty:
        return
    plot = comparison.set_index("method")[["window_precision", "window_recall", "window_f1"]]
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.45 * len(plot) + 1.5)))
    plot.plot.barh(ax=ax)
    ax.set_xlabel("Score")
    ax.set_xlim(0.0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Class-Agnostic Intervention-Window Localization")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_confusion_counts(comparison: pd.DataFrame, path: Path) -> None:
    if comparison.empty:
        return
    counts = comparison.set_index("method")[["window_tp", "window_fp", "window_fn"]]
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.45 * len(counts) + 1.5)))
    bottom = None
    colors = {"window_tp": "#16a34a", "window_fp": "#dc2626", "window_fn": "#f59e0b"}
    for column in ["window_tp", "window_fp", "window_fn"]:
        ax.barh(counts.index, counts[column], left=bottom, label=column.removeprefix("window_").upper(), color=colors[column])
        bottom = counts[column] if bottom is None else bottom + counts[column]
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    ax.set_title("Intervention-Window Localization Counts")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": _cell_id("markdown", source), "metadata": {}, "source": source.splitlines(keepends=True)}


def _code_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "id": _cell_id("code", source), "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def _cell_id(kind: str, source: str) -> str:
    digest = hashlib.sha1(f"{kind}\n{source}".encode("utf-8")).hexdigest()[:12]
    return f"{kind[:4]}-{digest}"


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in frame.to_numpy(dtype=object)]
    widths = [len(column) for column in columns]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]
    header = "| " + " | ".join(column.ljust(width) for column, width in zip(columns, widths, strict=True)) + " |"
    divider = "| " + " | ".join("---".ljust(width) for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _resolve_existing_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} does not exist: {value}")


def _resolve_artifact_path(value: object, default: Path, run_dir: Path) -> Path:
    resolved = _resolve_optional_artifact_path(value, run_dir, default)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"required artifact is missing: {default}")
    return resolved


def _resolve_optional_artifact_path(value: object, run_dir: Path, default: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if value:
        path = Path(str(value))
        candidates.append(path)
        if not path.is_absolute():
            candidates.extend([PROJECT_ROOT / path, run_dir / path])
    if default is not None:
        candidates.append(default)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _dataset_id_from_state_or_path(state: dict[str, Any], run_dir: Path) -> str:
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    if config.get("dataset_id"):
        return str(config["dataset_id"])
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent.name
    raise ValueError(f"could not infer dataset_id for {run_dir}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} are missing columns: {sorted(missing)}")


if __name__ == "__main__":
    raise SystemExit(main())
