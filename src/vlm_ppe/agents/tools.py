from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vlm_ppe.clustering.features import build_shape_features, load_features, write_features
from vlm_ppe.clustering.kmeans_runner import load_labels, run_candidate_kmeans, write_clustering_runs
from vlm_ppe.clustering.medoid import compute_cluster_medoids, write_medoids
from vlm_ppe.diagnostics.plots_clustering import render_cluster_panels, render_metrics_chart
from vlm_ppe.diagnostics.plots_medoids import render_medoid_plot
from vlm_ppe.diagnostics.report import write_medoid_report
from vlm_ppe.io.adsb_loader import ingest_adsb_tracks
from vlm_ppe.io.parquet_store import read_parquet, write_parquet
from vlm_ppe.processing import resample_track_frame
from vlm_ppe.schemas import ClusterReview, GraphEvent, KMetric, PPEConfig, PPEState


def state_model(state: dict) -> PPEState:
    return PPEState.model_validate(state)


def config_model(state: dict) -> PPEConfig:
    return PPEConfig.model_validate(state_model(state).config)


def append_event(state: dict, node: str, status: str, message: str | None = None, payload: dict | None = None) -> None:
    current = state_model(state)
    event = GraphEvent(node=node, status=status, message=message, payload=payload or {})
    path = Path(current.run_dir) / "graph_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(event.model_dump_json() + "\n")


def ingest_tracks_tool(state: dict) -> dict:
    current = state_model(state)
    config = config_model(state)
    processed_dir = Path(current.run_dir) / "processed"
    tracks, track_index, coordinate_system = ingest_adsb_tracks(config)
    tracks_path = write_parquet(tracks, processed_dir / "tracks.parquet")
    index_path = write_parquet(track_index, processed_dir / "track_index.parquet")
    return {
        "tracks_path": tracks_path,
        "track_index_path": index_path,
        "coordinate_system": coordinate_system.model_dump(),
        "status": "tracks_ingested",
    }


def resample_tracks_tool(state: dict) -> dict:
    current = state_model(state)
    config = config_model(state)
    if current.tracks_path is None:
        raise ValueError("tracks_path is required before resampling")
    tracks = read_parquet(current.tracks_path)
    resampled = resample_track_frame(tracks, config.n_resample)
    path = write_parquet(resampled, Path(current.run_dir) / "processed" / "resampled_tracks.parquet")
    return {"resampled_tracks_path": path, "status": "tracks_resampled"}


def build_features_tool(state: dict) -> dict:
    current = state_model(state)
    if current.resampled_tracks_path is None:
        raise ValueError("resampled_tracks_path is required before feature construction")
    resampled = read_parquet(current.resampled_tracks_path)
    feature_set = build_shape_features(resampled)
    features_path, metadata_path = write_features(
        feature_set,
        Path(current.run_dir) / "processed" / "features.npz",
        Path(current.run_dir) / "processed" / "feature_metadata.parquet",
    )
    return {"features_path": features_path, "feature_metadata_path": metadata_path, "status": "features_built"}


def run_candidate_clustering_tool(state: dict) -> dict:
    current = state_model(state)
    config = config_model(state)
    if current.features_path is None:
        raise ValueError("features_path is required before clustering")
    features = load_features(current.features_path)
    runs = run_candidate_kmeans(
        features,
        k_min=config.k_min,
        k_max=current.k_max_current,
        n_init=config.kmeans_n_init,
        random_state=config.kmeans_random_state,
    )
    clustering_dir = Path(current.run_dir) / "clustering" / f"attempt_{current.retry_count:02d}"
    metrics_path, clustering_dir_path = write_clustering_runs(runs, features.track_ids, clustering_dir)
    metrics = [run.metric.model_dump() for run in runs]
    return {
        "k_metrics_path": metrics_path,
        "clustering_dir": clustering_dir_path,
        "candidate_k_values": [run.k for run in runs],
        "clustering_metrics": metrics,
        "status": "candidate_clustering_done",
    }


def render_evidence_pack_tool(state: dict) -> dict:
    current = state_model(state)
    if current.features_path is None or current.resampled_tracks_path is None or current.clustering_dir is None:
        raise ValueError("features, resampled tracks, and clustering_dir are required before rendering evidence")
    features = load_features(current.features_path)
    metrics_frame = pd.read_csv(Path(current.k_metrics_path))
    runs = []
    from vlm_ppe.clustering.kmeans_runner import ClusteringRun

    for row in metrics_frame.to_dict("records"):
        labels = pd.read_csv(Path(current.clustering_dir) / f"k_{int(row['k']):02d}_labels.csv")["cluster_id"].to_numpy()
        runs.append(ClusteringRun(k=int(row["k"]), labels=labels, metric=KMetric.model_validate(row)))
    resampled = read_parquet(current.resampled_tracks_path)
    evidence_dir = Path(current.run_dir) / "evidence" / "clustering" / f"attempt_{current.retry_count:02d}"
    evidence = render_cluster_panels(resampled, runs, features.track_ids, evidence_dir)
    evidence.append(render_metrics_chart(runs, evidence_dir))
    return {"evidence_images": [item.model_dump() for item in evidence], "status": "evidence_rendered"}


def validate_review_tool(state: dict) -> dict:
    current = state_model(state)
    if not current.vlm_reviews:
        raise ValueError("vlm review is required")
    review = ClusterReview.model_validate(current.vlm_reviews[-1])
    available = set(int(k) for k in state.get("candidate_k_values", []))
    if review.chosen_k not in available:
        raise ValueError(f"VLM chose unavailable K={review.chosen_k}; available={sorted(available)}")
    return {"chosen_k": review.chosen_k, "status": "review_validated"}


def retry_or_accept_tool(state: dict) -> dict:
    current = state_model(state)
    config = config_model(state)
    review = ClusterReview.model_validate(current.vlm_reviews[-1])
    retry_requested = review.retry_requested or review.suggested_action == "retry"
    can_retry = current.retry_count < config.max_retries
    if retry_requested and can_retry:
        requested = review.requested_k_max or current.k_max_current + 2
        next_k_max = min(config.max_k_expansion, max(current.k_max_current + 1, int(requested)))
        return {
            "retry_count": current.retry_count + 1,
            "k_max_current": next_k_max,
            "status": "retry_requested",
            "should_retry": True,
        }
    return {"status": "cluster_choice_accepted", "should_retry": False}


def compute_medoids_tool(state: dict) -> dict:
    current = state_model(state)
    if current.resampled_tracks_path is None or current.clustering_dir is None or current.chosen_k is None:
        raise ValueError("resampled tracks, clustering_dir, and chosen_k are required before medoid extraction")
    resampled = read_parquet(current.resampled_tracks_path)
    labels = load_labels(current.clustering_dir, current.chosen_k)
    medoids = compute_cluster_medoids(resampled, labels)
    templates_dir = Path(current.run_dir) / "templates"
    medoids_path, summary_path = write_medoids(medoids, templates_dir)
    chosen_labels_path = templates_dir / "chosen_cluster_assignments.csv"
    labels.to_csv(chosen_labels_path, index=False)
    return {
        "medoids_path": medoids_path,
        "medoid_summary_path": summary_path,
        "cluster_assignments_path": chosen_labels_path.as_posix(),
        "medoids": [item.model_dump() for item in medoids],
        "status": "medoids_computed",
    }


def render_medoid_report_tool(state: dict) -> dict:
    current = state_model(state)
    if current.resampled_tracks_path is None or not state.get("medoids") or not current.vlm_reviews:
        raise ValueError("resampled tracks, medoids, and review are required before report rendering")
    resampled = read_parquet(current.resampled_tracks_path)
    from vlm_ppe.schemas import ClusterMedoid

    medoids = [ClusterMedoid.model_validate(item) for item in state["medoids"]]
    review = ClusterReview.model_validate(current.vlm_reviews[-1])
    plot_path = render_medoid_plot(resampled, medoids, Path(current.run_dir) / "templates")
    report_path = write_medoid_report(
        output_dir=Path(current.run_dir) / "reports",
        run_id=current.run_id,
        review=review,
        medoids=medoids,
        medoid_plot_path=plot_path,
    )
    return {"medoid_report_path": report_path, "medoid_plot_path": plot_path, "status": "medoid_report_rendered"}


def export_state_tool(state: dict) -> dict:
    current_state = dict(state)
    current_state.pop("should_retry", None)
    current_state["status"] = "complete"
    path = Path(state_model(state).run_dir) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(current_state, stream, indent=2, default=str)
    return {"state_path": path.as_posix(), "status": "complete"}
