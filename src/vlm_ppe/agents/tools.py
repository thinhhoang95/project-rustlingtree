from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vlm_ppe.agents.prompts import subcluster_review_prompt
from vlm_ppe.agents.vlm_client import ClusterReviewClient, OpenRouterVLMClient
from vlm_ppe.clustering.community_detection import load_labels, run_candidate_community_detection, write_clustering_runs
from vlm_ppe.clustering.features import build_shape_features, load_features, subset_features, write_features
from vlm_ppe.clustering.medoid import compute_cluster_medoids, write_medoids
from vlm_ppe.clustering.residual_windows import (
    compute_cluster_residual_profiles,
    write_intervention_windows,
    write_residual_profiles,
)
from vlm_ppe.diagnostics.plots_clustering import render_cluster_panels, render_metrics_chart
from vlm_ppe.diagnostics.plots_medoids import render_medoid_plot
from vlm_ppe.diagnostics.plots_residuals import render_residual_window_diagnostics
from vlm_ppe.diagnostics.report import write_medoid_report
from vlm_ppe.io.adsb_loader import ingest_adsb_tracks
from vlm_ppe.io.parquet_store import read_parquet, write_parquet
from vlm_ppe.processing import resample_track_frame
from vlm_ppe.audit import log_graph_event, log_subcluster_vlm_request, log_subcluster_vlm_response
from vlm_ppe.schemas import (
    ClusterMedoid,
    ClusterReview,
    CommunityMetric,
    EvidenceImage,
    InterventionWindow,
    PPEConfig,
    PPEState,
    WindowReview,
)


def state_model(state: dict) -> PPEState:
    return PPEState.model_validate(state)


def config_model(state: dict) -> PPEConfig:
    return PPEConfig.model_validate(state_model(state).config)


def append_event(state: dict, node: str, status: str, message: str | None = None, payload: dict | None = None) -> None:
    current = state_model(state)
    log_graph_event(run_dir=current.run_dir, node=node, status=status, message=message, payload=payload or {})


def _match_threshold_candidate(
    metrics: list[CommunityMetric],
    threshold_nm: float,
) -> CommunityMetric | None:
    if not metrics:
        return None
    for metric in metrics:
        if abs(float(metric.threshold_nm) - float(threshold_nm)) <= 1e-6:
            return metric
    nearest = min(metrics, key=lambda item: abs(float(item.threshold_nm) - float(threshold_nm)))
    spacings = [
        abs(float(right.threshold_nm) - float(left.threshold_nm))
        for left, right in zip(metrics, metrics[1:], strict=False)
        if abs(float(right.threshold_nm) - float(left.threshold_nm)) > 1e-9
    ]
    spacing_tolerance = min(spacings) * 0.25 if spacings else 0.01
    tolerance = max(1e-6, min(0.01, spacing_tolerance))
    if abs(float(nearest.threshold_nm) - float(threshold_nm)) <= tolerance:
        return nearest
    return None


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
    extra_thresholds = []
    if current.chosen_threshold_override_nm is not None:
        extra_thresholds.append(float(current.chosen_threshold_override_nm))
    runs = run_candidate_community_detection(
        features,
        threshold_min_nm=config.cd_threshold_min_nm,
        threshold_max_nm=current.threshold_max_current_nm,
        threshold_steps=config.cd_threshold_steps,
        extra_thresholds_nm=extra_thresholds,
    )
    clustering_dir = Path(current.run_dir) / "clustering" / f"attempt_{current.retry_count:02d}"
    metrics_path, clustering_dir_path = write_clustering_runs(runs, features.track_ids, clustering_dir)
    metrics = [run.metric.model_dump() for run in runs]
    thresholds = [float(run.threshold_nm) for run in runs]
    return {
        "community_metrics_path": metrics_path,
        "clustering_dir": clustering_dir_path,
        "candidate_thresholds_nm": thresholds,
        "threshold_max_current_nm": max(thresholds),
        "clustering_metrics": metrics,
        "status": "candidate_clustering_done",
    }


def render_evidence_pack_tool(state: dict) -> dict:
    current = state_model(state)
    if current.features_path is None or current.resampled_tracks_path is None or current.clustering_dir is None:
        raise ValueError("features, resampled tracks, and clustering_dir are required before rendering evidence")
    features = load_features(current.features_path)
    metrics_frame = pd.read_csv(Path(current.community_metrics_path))
    runs = []
    from vlm_ppe.clustering.community_detection import CommunityDetectionRun

    for row in metrics_frame.to_dict("records"):
        candidate_id = int(row["candidate_id"])
        labels = pd.read_csv(Path(current.clustering_dir) / f"threshold_{candidate_id:02d}_labels.csv")[
            "cluster_id"
        ].to_numpy()
        runs.append(
            CommunityDetectionRun(
                candidate_id=candidate_id,
                threshold_nm=float(row["threshold_nm"]),
                labels=labels,
                metric=CommunityMetric.model_validate(row),
            )
        )
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
    metrics = [CommunityMetric.model_validate(item) for item in state.get("clustering_metrics", [])]
    candidate = _match_threshold_candidate(metrics, review.chosen_threshold_nm)
    if candidate is None:
        available = [float(metric.threshold_nm) for metric in metrics]
        raise ValueError(
            f"VLM chose unavailable threshold_nm={review.chosen_threshold_nm}; available={available}"
        )
    return {
        "chosen_threshold_nm": candidate.threshold_nm,
        "chosen_threshold_candidate_id": candidate.candidate_id,
        "status": "review_validated",
    }


def retry_or_accept_tool(state: dict) -> dict:
    current = state_model(state)
    config = config_model(state)
    review = ClusterReview.model_validate(current.vlm_reviews[-1])
    retry_requested = review.retry_requested or review.suggested_action == "retry"
    can_retry = current.retry_count < config.max_retries
    if retry_requested and can_retry:
        current_max = float(current.threshold_max_current_nm or config.cd_threshold_min_nm)
        requested = review.requested_threshold_max_nm or current_max * config.cd_threshold_retry_growth
        next_threshold_max = max(current_max, float(requested))
        return {
            "retry_count": current.retry_count + 1,
            "threshold_max_current_nm": next_threshold_max,
            "status": "retry_requested",
            "should_retry": True,
        }
    return {"status": "cluster_choice_accepted", "should_retry": False}


def refine_subclusters_tool(state: dict, *, vlm_client: ClusterReviewClient | None = None) -> dict:
    current = state_model(state)
    config = config_model(state)
    if current.features_path is None or current.resampled_tracks_path is None:
        raise ValueError("features and resampled tracks are required before subcluster refinement")
    if current.clustering_dir is None or current.chosen_threshold_candidate_id is None:
        raise ValueError("clustering_dir and chosen threshold candidate are required before subcluster refinement")

    base_labels = load_labels(current.clustering_dir, current.chosen_threshold_candidate_id)
    base_labels["flight_id"] = base_labels["flight_id"].astype(str)
    base_labels["cluster_id"] = base_labels["cluster_id"].astype(int)
    features = load_features(current.features_path)
    resampled = read_parquet(current.resampled_tracks_path)

    final_assignments: list[dict[str, object]] = []
    tree_nodes: list[dict[str, object]] = []
    review_paths: list[str] = []
    reviews: list[dict[str, object]] = []
    review_count = 0
    final_cluster_id = 0
    resolved_client: ClusterReviewClient | None = None

    def client() -> ClusterReviewClient:
        nonlocal resolved_client
        if resolved_client is None:
            resolved_client = vlm_client or OpenRouterVLMClient(
                model=config.vlm_model,
                reasoning_effort=config.vlm_reasoning_effort,
            )
        return resolved_client

    def lineage_label(lineage: list[int]) -> str:
        return ".".join(str(item) for item in lineage)

    def accept_leaf(node: dict[str, object], track_ids: list[str], reason: str) -> dict[str, object]:
        nonlocal final_cluster_id
        leaf_cluster_id = final_cluster_id
        final_cluster_id += 1
        node["status"] = "accepted_leaf"
        node["stop_reason"] = reason
        node["final_cluster_id"] = leaf_cluster_id
        for track_id in track_ids:
            final_assignments.append({"flight_id": track_id, "cluster_id": leaf_cluster_id})
        return node

    def inspect_node(track_ids: list[str], *, root_cluster_id: int, lineage: list[int], depth: int) -> dict[str, object]:
        nonlocal review_count
        node_id = f"node_{len(tree_nodes):04d}"
        node: dict[str, object] = {
            "node_id": node_id,
            "root_cluster_id": int(root_cluster_id),
            "lineage": [int(item) for item in lineage],
            "depth": int(depth),
            "n_tracks": len(track_ids),
            "track_ids": list(track_ids),
            "children": [],
        }
        tree_nodes.append(node)

        if not config.subcluster_review_enabled:
            return accept_leaf(node, track_ids, "subcluster review disabled")
        if depth >= 1:
            return accept_leaf(node, track_ids, "subcluster depth 1 reached")
        if len(track_ids) < config.subcluster_min_tracks:
            return accept_leaf(node, track_ids, "below subcluster minimum track count")
        if len(track_ids) < 2:
            return accept_leaf(node, track_ids, "fewer than two tracks")
        if review_count >= config.subcluster_max_reviews:
            return accept_leaf(node, track_ids, "subcluster review budget exhausted")

        local_features = subset_features(features, track_ids)
        runs = run_candidate_community_detection(
            local_features,
            threshold_min_nm=config.cd_threshold_min_nm,
            threshold_max_nm=None,
            threshold_steps=config.cd_threshold_steps,
            extra_thresholds_nm=None,
        )
        available_thresholds_nm = [float(run.threshold_nm) for run in runs]
        if len(available_thresholds_nm) <= 1:
            return accept_leaf(node, track_ids, "only one local threshold is available")

        clustering_dir = Path(current.run_dir) / "clustering" / "subclusters" / node_id
        metrics_path, clustering_dir_path = write_clustering_runs(runs, local_features.track_ids, clustering_dir)
        evidence_dir = Path(current.run_dir) / "evidence" / "subclustering" / node_id
        evidence = render_cluster_panels(resampled, runs, local_features.track_ids, evidence_dir)
        evidence = [
            image.model_copy(
                update={
                    "kind": "subcluster_panel",
                    "caption": (
                        f"Subcluster candidate {node_id} lineage {lineage_label(lineage)}: {image.caption}"
                    ),
                }
            )
            for image in evidence
        ]
        metrics_image = render_metrics_chart(runs, evidence_dir).model_copy(
            update={
                "kind": "subcluster_metrics_chart",
                "caption": (
                    f"Subcluster candidate {node_id} lineage {lineage_label(lineage)}: "
                    "community count and silhouette by threshold"
                ),
            }
        )
        evidence.append(metrics_image)
        metrics = [run.metric for run in runs]
        prompt = subcluster_review_prompt(
            metrics,
            available_thresholds_nm,
            root_cluster_id=root_cluster_id,
            lineage=lineage,
            depth=depth,
            n_tracks=len(track_ids),
            min_tracks=config.subcluster_min_tracks,
        )
        log_subcluster_vlm_request(
            run_dir=current.run_dir,
            node_id=node_id,
            root_cluster_id=root_cluster_id,
            lineage=lineage,
            depth=depth,
            model=config.vlm_model,
            prompt=prompt,
            evidence_images=evidence,
            metrics=metrics,
            available_thresholds_nm=available_thresholds_nm,
        )
        review_count += 1
        review = client().review_clusters(
            evidence_images=evidence,
            metrics=metrics,
            available_thresholds_nm=available_thresholds_nm,
            attempt=0,
            max_retries=0,
            prompt=prompt,
        )
        chosen_metric = _match_threshold_candidate(metrics, review.chosen_threshold_nm)
        if chosen_metric is None:
            raise ValueError(
                f"subcluster review for {node_id} chose unavailable threshold_nm={review.chosen_threshold_nm}; "
                f"available={available_thresholds_nm}"
            )
        review_dir = Path(current.run_dir) / "vlm_reviews" / "subclusters"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_dir / f"{node_id}.json"
        review_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
        log_subcluster_vlm_response(
            run_dir=current.run_dir,
            node_id=node_id,
            root_cluster_id=root_cluster_id,
            lineage=lineage,
            depth=depth,
            review=review,
            response_path=review_path.as_posix(),
        )
        review_paths.append(review_path.as_posix())
        reviews.append(
            {
                "node_id": node_id,
                "root_cluster_id": int(root_cluster_id),
                "lineage": [int(item) for item in lineage],
                "depth": int(depth),
                "review": review.model_dump(),
            }
        )
        node.update(
            {
                "status": "reviewed",
                "community_metrics_path": metrics_path,
                "clustering_dir": clustering_dir_path,
                "evidence_images": [image.model_dump() for image in evidence],
                "chosen_threshold_nm": float(chosen_metric.threshold_nm),
                "chosen_threshold_candidate_id": int(chosen_metric.candidate_id),
                "review_path": review_path.as_posix(),
            }
        )

        if review.suggested_action == "human_review":
            return accept_leaf(node, track_ids, "subcluster review requested human review")

        chosen_labels = load_labels(clustering_dir_path, int(chosen_metric.candidate_id))
        chosen_labels["flight_id"] = chosen_labels["flight_id"].astype(str)
        chosen_labels["cluster_id"] = chosen_labels["cluster_id"].astype(int)
        if chosen_labels["cluster_id"].nunique() <= 1:
            return accept_leaf(node, track_ids, "VLM chose a one-community threshold")

        child_node_ids: list[str] = []
        for child_cluster_id in sorted(chosen_labels["cluster_id"].unique()):
            child_track_ids = (
                chosen_labels.loc[chosen_labels["cluster_id"] == int(child_cluster_id), "flight_id"]
                .astype(str)
                .tolist()
            )
            child_node = inspect_node(
                child_track_ids,
                root_cluster_id=root_cluster_id,
                lineage=[*lineage, int(child_cluster_id)],
                depth=depth + 1,
            )
            child_node_ids.append(str(child_node["node_id"]))
        node["children"] = child_node_ids
        node["status"] = "split"
        return node

    for root_cluster_id in sorted(base_labels["cluster_id"].unique()):
        root_track_ids = (
            base_labels.loc[base_labels["cluster_id"] == int(root_cluster_id), "flight_id"].astype(str).tolist()
        )
        inspect_node(root_track_ids, root_cluster_id=int(root_cluster_id), lineage=[int(root_cluster_id)], depth=0)

    order_by_track_id = {track_id: index for index, track_id in enumerate(features.track_ids)}
    refined = pd.DataFrame(final_assignments)
    refined["_order"] = refined["flight_id"].map(order_by_track_id)
    refined = refined.sort_values("_order", kind="stable").drop(columns=["_order"])

    output_dir = Path(current.run_dir) / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "refined_cluster_assignments.csv"
    refined.to_csv(assignments_path, index=False)
    tree_path = output_dir / "subcluster_tree.json"
    tree_payload = {
        "enabled": bool(config.subcluster_review_enabled),
        "global_chosen_threshold_nm": float(current.chosen_threshold_nm or 0.0),
        "final_cluster_count": int(final_cluster_id),
        "review_count": int(review_count),
        "nodes": tree_nodes,
    }
    with tree_path.open("w", encoding="utf-8") as stream:
        json.dump(tree_payload, stream, indent=2)

    return {
        "cluster_assignments_path": assignments_path.as_posix(),
        "subcluster_tree_path": tree_path.as_posix(),
        "subcluster_reviews": reviews,
        "subcluster_review_paths": review_paths,
        "final_cluster_count": final_cluster_id,
        "status": "subclusters_refined",
    }


def compute_medoids_tool(state: dict) -> dict:
    current = state_model(state)
    if current.resampled_tracks_path is None:
        raise ValueError("resampled tracks are required before medoid extraction")
    resampled = read_parquet(current.resampled_tracks_path)
    if current.cluster_assignments_path is not None:
        labels = pd.read_csv(current.cluster_assignments_path)
    elif current.clustering_dir is not None and current.chosen_threshold_candidate_id is not None:
        labels = load_labels(current.clustering_dir, current.chosen_threshold_candidate_id)
    else:
        raise ValueError("cluster assignments or chosen clustering outputs are required before medoid extraction")
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


def compute_residual_profiles_tool(state: dict) -> dict:
    current = state_model(state)
    if current.resampled_tracks_path is None or current.cluster_assignments_path is None or not state.get("medoids"):
        raise ValueError(
            "resampled tracks, cluster assignments, and medoids are required before residual-profile computation"
        )
    resampled = read_parquet(current.resampled_tracks_path)
    labels = pd.read_csv(current.cluster_assignments_path)
    medoids = [ClusterMedoid.model_validate(item) for item in state["medoids"]]
    residual_profiles = compute_cluster_residual_profiles(
        resampled,
        labels,
        medoids,
    )
    profiles_path = write_residual_profiles(
        residual_profiles,
        Path(current.run_dir) / "residuals",
    )
    return {
        "residual_profiles_path": profiles_path,
        "status": "residual_profiles_computed",
    }


def render_window_diagnostics_tool(state: dict) -> dict:
    current = state_model(state)
    if (
        current.resampled_tracks_path is None
        or current.cluster_assignments_path is None
        or current.residual_profiles_path is None
        or not state.get("medoids")
    ):
        raise ValueError(
            "resampled tracks, cluster assignments, medoids, and residual profiles are required before diagnostics"
        )
    resampled = read_parquet(current.resampled_tracks_path)
    labels = pd.read_csv(current.cluster_assignments_path)
    medoids = [ClusterMedoid.model_validate(item) for item in state["medoids"]]
    residual_profiles = read_parquet(current.residual_profiles_path)
    evidence = render_residual_window_diagnostics(
        resampled,
        labels,
        medoids,
        residual_profiles,
        Path(current.run_dir) / "evidence" / "residual_windows",
    )
    return {
        "window_evidence_images": [item.model_dump() for item in evidence],
        "status": "window_diagnostics_rendered",
    }


def render_window_review_diagnostics(state: dict, review: WindowReview, *, proposal_attempt: int) -> list[EvidenceImage]:
    current = state_model(state)
    if (
        current.resampled_tracks_path is None
        or current.cluster_assignments_path is None
        or current.residual_profiles_path is None
        or not state.get("medoids")
    ):
        raise ValueError(
            "resampled tracks, cluster assignments, medoids, and residual profiles are required before review diagnostics"
        )

    cluster_id = int(review.cluster_id)
    medoids = [ClusterMedoid.model_validate(item) for item in state["medoids"]]
    selected_medoids = [medoid for medoid in medoids if int(medoid.cluster_id) == cluster_id]
    if not selected_medoids:
        raise ValueError(f"window review references unknown cluster_id={cluster_id}")

    resampled = read_parquet(current.resampled_tracks_path)
    labels = pd.read_csv(current.cluster_assignments_path)
    residual_profiles = read_parquet(current.residual_profiles_path)
    output_dir = (
        Path(current.run_dir)
        / "evidence"
        / "residual_windows"
        / f"cluster_{cluster_id:02d}_attempt_{proposal_attempt:02d}_highlighted"
    )
    return render_residual_window_diagnostics(
        resampled,
        labels,
        selected_medoids,
        residual_profiles,
        output_dir,
        windows_by_cluster={cluster_id: list(review.windows)},
        caption_suffix=f"with VLM-proposed windows highlighted from attempt {proposal_attempt + 1}",
    )


def validate_window_reviews_tool(state: dict) -> dict:
    current = state_model(state)
    if current.residual_profiles_path is None or current.cluster_assignments_path is None:
        raise ValueError("residual profiles and cluster assignments are required before window-review validation")
    residual_profiles = read_parquet(current.residual_profiles_path)
    labels = pd.read_csv(current.cluster_assignments_path)
    reviews = [WindowReview.model_validate(item) for item in current.window_reviews]
    windows: list[InterventionWindow] = []
    seen_ids: set[str] = set()
    for review in reviews:
        cluster_id = int(review.cluster_id)
        cluster_profile = residual_profiles.loc[residual_profiles["cluster_id"].astype(int) == cluster_id]
        if cluster_profile.empty:
            raise ValueError(f"window review references unknown cluster_id={cluster_id}")
        cluster_profile = cluster_profile.sort_values("station_index", kind="stable")
        station_ids = set(cluster_profile["station_index"].astype(int))
        track_ids = labels.loc[labels["cluster_id"].astype(int) == cluster_id, "flight_id"].astype(str).tolist()
        for proposal in review.windows:
            if proposal.window_id in seen_ids:
                raise ValueError(f"window_id={proposal.window_id} was proposed more than once")
            seen_ids.add(proposal.window_id)
            if proposal.start_station_index not in station_ids or proposal.end_station_index not in station_ids:
                raise ValueError(
                    f"window_id={proposal.window_id} station bounds are outside cluster {cluster_id} profile"
                )
            rows = cluster_profile.loc[
                (cluster_profile["station_index"].astype(int) >= proposal.start_station_index)
                & (cluster_profile["station_index"].astype(int) <= proposal.end_station_index)
            ]
            start_row = rows.iloc[0]
            end_row = rows.iloc[-1]
            windows.append(
                InterventionWindow(
                    cluster_id=cluster_id,
                    window_id=proposal.window_id,
                    class_name=proposal.class_name,
                    confidence=proposal.confidence,
                    visual_reason=proposal.visual_reason,
                    start_station_index=proposal.start_station_index,
                    end_station_index=proposal.end_station_index,
                    start_s_fraction=float(start_row["s_fraction"]),
                    end_s_fraction=float(end_row["s_fraction"]),
                    start_s_nm=float(start_row["s_nm"]),
                    end_s_nm=float(end_row["s_nm"]),
                    length_nm=float(end_row["s_nm"] - start_row["s_nm"]),
                    peak_residual_energy_nm2=float(rows["residual_energy_nm2"].max()),
                    peak_heading_dispersion=float(rows["heading_dispersion"].max()),
                    track_ids=track_ids,
                )
            )

    windows_path = write_intervention_windows(windows, Path(current.run_dir) / "residuals")
    return {
        "intervention_windows_path": windows_path,
        "intervention_windows": [window.model_dump() for window in windows],
        "status": "window_reviews_validated",
    }


def export_state_tool(state: dict) -> dict:
    current_state = dict(state)
    current_state.pop("should_retry", None)
    current_state["status"] = "complete"
    path = Path(state_model(state).run_dir) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(current_state, stream, indent=2, default=str)
    return {"state_path": path.as_posix(), "status": "complete"}
