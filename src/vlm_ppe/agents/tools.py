from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vlm_ppe.agents.prompts import subcluster_review_prompt
from vlm_ppe.agents.vlm_client import ClusterReviewClient, OpenRouterVLMClient
from vlm_ppe.clustering.features import build_shape_features, load_features, write_features
from vlm_ppe.clustering.kmeans_runner import load_labels, run_candidate_kmeans, write_clustering_runs
from vlm_ppe.clustering.medoid import compute_cluster_medoids, write_medoids
from vlm_ppe.clustering.polygon_capture import PolygonCaptureResult, assign_tracks_to_subcluster_polygons
from vlm_ppe.clustering.residual_windows import (
    compute_cluster_residual_profiles,
    write_intervention_windows,
    write_residual_profiles,
)
from vlm_ppe.diagnostics.plots_clustering import (
    render_cluster_panels,
    render_metrics_chart,
    render_subcluster_capture_prompt_panel,
    render_subcluster_capture_result_panel,
)
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
    EvidenceImage,
    InterventionWindow,
    KMetric,
    PPEConfig,
    PPEState,
    SubclusterReview,
    WindowReview,
)


def state_model(state: dict) -> PPEState:
    return PPEState.model_validate(state)


def config_model(state: dict) -> PPEConfig:
    return PPEConfig.model_validate(state_model(state).config)


def append_event(state: dict, node: str, status: str, message: str | None = None, payload: dict | None = None) -> None:
    current = state_model(state)
    log_graph_event(run_dir=current.run_dir, node=node, status=status, message=message, payload=payload or {})


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


def refine_subclusters_tool(state: dict, *, vlm_client: ClusterReviewClient | None = None) -> dict:
    current = state_model(state)
    config = config_model(state)
    if current.features_path is None or current.resampled_tracks_path is None:
        raise ValueError("features and resampled tracks are required before subcluster refinement")
    if current.clustering_dir is None or current.chosen_k is None:
        raise ValueError("clustering_dir and chosen_k are required before subcluster refinement")

    base_labels = load_labels(current.clustering_dir, current.chosen_k)
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

    def new_node(
        track_ids: list[str],
        *,
        root_cluster_id: int,
        lineage: list[int],
        depth: int,
    ) -> dict[str, object]:
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
        return node

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
        node = new_node(track_ids, root_cluster_id=root_cluster_id, lineage=lineage, depth=depth)
        node_id = str(node["node_id"])

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

        max_subclusters = min(int(config.subcluster_max_polygons), len(track_ids))
        if max_subclusters <= 1:
            return accept_leaf(node, track_ids, "only one polygon subcluster is available")

        evidence_dir = Path(current.run_dir) / "evidence" / "subclustering" / node_id
        evidence = [
            render_subcluster_capture_prompt_panel(
                resampled,
                track_ids,
                evidence_dir,
                root_cluster_id=root_cluster_id,
                node_id=node_id,
            )
        ]
        coordinate_bounds = _coordinate_bounds(resampled, track_ids)
        prompt = subcluster_review_prompt(
            root_cluster_id=root_cluster_id,
            lineage=lineage,
            depth=depth,
            n_tracks=len(track_ids),
            min_tracks=config.subcluster_min_tracks,
            max_subclusters=max_subclusters,
            coordinate_bounds=coordinate_bounds,
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
            max_subclusters=max_subclusters,
            coordinate_bounds=coordinate_bounds,
        )
        review_count += 1
        review = client().review_subclusters(
            root_cluster_id=root_cluster_id,
            evidence_images=evidence,
            n_tracks=len(track_ids),
            max_subclusters=max_subclusters,
            coordinate_bounds=coordinate_bounds,
            prompt=prompt,
        )
        review = _validate_subcluster_review(review, max_subclusters=max_subclusters, node_id=node_id)
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
                "evidence_images": [image.model_dump() for image in evidence],
                "subcluster_count": int(review.subcluster_count),
                "review_path": review_path.as_posix(),
            }
        )

        if review.suggested_action == "human_review":
            return accept_leaf(node, track_ids, "subcluster review requested human review")
        if int(review.subcluster_count) <= 1 or not review.subclusters:
            return accept_leaf(node, track_ids, "VLM proposed no polygon subcluster split")

        capture_result = assign_tracks_to_subcluster_polygons(resampled, track_ids, review.subclusters)
        result_image = render_subcluster_capture_result_panel(
            resampled,
            track_ids,
            capture_result,
            evidence_dir,
            root_cluster_id=root_cluster_id,
            node_id=node_id,
        )
        evidence.append(result_image)
        node["evidence_images"] = [image.model_dump() for image in evidence]
        capture_artifact_paths = _write_polygon_capture_artifacts(
            current.run_dir,
            node_id=node_id,
            review=review,
            capture_result=capture_result,
        )
        node.update(
            {
                "capture_polygons": [
                    {
                        "subcluster_id": capture.subcluster_id,
                        "label": capture.label,
                        "polygon": capture.polygon,
                        "n_tracks": len(capture.track_ids),
                        "track_ids": capture.track_ids,
                    }
                    for capture in capture_result.captures
                ],
                "uncaptured_track_ids": capture_result.uncaptured_track_ids,
                "overlapping_track_ids": capture_result.overlapping_track_ids,
                **capture_artifact_paths,
            }
        )

        nonempty_captures = [capture for capture in capture_result.captures if capture.track_ids]
        if not nonempty_captures:
            return accept_leaf(node, track_ids, "VLM polygons captured no tracks")

        child_node_ids: list[str] = []
        for capture in nonempty_captures:
            child_node = new_node(
                capture.track_ids,
                root_cluster_id=root_cluster_id,
                lineage=[*lineage, int(capture.subcluster_id)],
                depth=depth + 1,
            )
            child_node.update(
                {
                    "source_subcluster_id": int(capture.subcluster_id),
                    "source_label": capture.label,
                    "capture_polygon": capture.polygon,
                }
            )
            accept_leaf(child_node, capture.track_ids, "accepted VLM polygon capture")
            child_node_ids.append(str(child_node["node_id"]))

        if capture_result.uncaptured_track_ids:
            residual_node = new_node(
                capture_result.uncaptured_track_ids,
                root_cluster_id=root_cluster_id,
                lineage=[*lineage, -1],
                depth=depth + 1,
            )
            residual_node["source_label"] = "Uncaptured residual"
            if review.uncaptured_tracks_policy == "human_review":
                accept_leaf(residual_node, capture_result.uncaptured_track_ids, "uncaptured tracks require human review")
            else:
                accept_leaf(residual_node, capture_result.uncaptured_track_ids, "uncaptured by VLM polygons")
            child_node_ids.append(str(residual_node["node_id"]))

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
        "global_chosen_k": int(current.chosen_k),
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


def _coordinate_bounds(resampled: pd.DataFrame, track_ids: list[str]) -> dict[str, float]:
    ids = set(str(track_id) for track_id in track_ids)
    frame = resampled.loc[resampled["flight_id"].astype(str).isin(ids)]
    if frame.empty:
        raise ValueError("cannot compute coordinate bounds for an empty track set")
    return {
        "x_min_nm": float(frame["x_nm"].min()),
        "x_max_nm": float(frame["x_nm"].max()),
        "y_min_nm": float(frame["y_nm"].min()),
        "y_max_nm": float(frame["y_nm"].max()),
    }


def _validate_subcluster_review(
    review: SubclusterReview,
    *,
    max_subclusters: int,
    node_id: str,
) -> SubclusterReview:
    if int(review.subcluster_count) > int(max_subclusters):
        raise ValueError(
            f"subcluster review for {node_id} proposed {review.subcluster_count} subclusters; "
            f"maximum is {max_subclusters}"
        )
    for subcluster in review.subclusters:
        if int(subcluster.subcluster_id) > int(review.subcluster_count):
            raise ValueError(
                f"subcluster review for {node_id} returned subcluster_id={subcluster.subcluster_id} "
                f"outside subcluster_count={review.subcluster_count}"
            )
    return review


def _write_polygon_capture_artifacts(
    run_dir: str,
    *,
    node_id: str,
    review: SubclusterReview,
    capture_result: PolygonCaptureResult,
) -> dict[str, str]:
    output_dir = Path(run_dir) / "clustering" / "subclusters" / node_id
    output_dir.mkdir(parents=True, exist_ok=True)
    polygons_path = output_dir / "capture_polygons.json"
    payload = {
        "review": review.model_dump(),
        "captures": [
            {
                "subcluster_id": capture.subcluster_id,
                "label": capture.label,
                "polygon": capture.polygon,
                "track_ids": capture.track_ids,
            }
            for capture in capture_result.captures
        ],
        "uncaptured_track_ids": capture_result.uncaptured_track_ids,
        "overlapping_track_ids": capture_result.overlapping_track_ids,
    }
    with polygons_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)

    rows: list[dict[str, object]] = []
    overlaps = capture_result.overlapping_track_ids
    for capture in capture_result.captures:
        for track_id in capture.track_ids:
            rows.append(
                {
                    "flight_id": track_id,
                    "subcluster_id": capture.subcluster_id,
                    "label": capture.label,
                    "assignment": "captured",
                    "overlapping_subcluster_ids": json.dumps(overlaps.get(track_id, []), separators=(",", ":")),
                }
            )
    for track_id in capture_result.uncaptured_track_ids:
        rows.append(
            {
                "flight_id": track_id,
                "subcluster_id": None,
                "label": "Uncaptured residual",
                "assignment": "uncaptured",
                "overlapping_subcluster_ids": "[]",
            }
        )
    assignments_path = output_dir / "polygon_assignments.csv"
    pd.DataFrame(rows).to_csv(assignments_path, index=False)
    return {
        "capture_polygons_path": polygons_path.as_posix(),
        "polygon_assignments_path": assignments_path.as_posix(),
    }


def compute_medoids_tool(state: dict) -> dict:
    current = state_model(state)
    if current.resampled_tracks_path is None:
        raise ValueError("resampled tracks are required before medoid extraction")
    resampled = read_parquet(current.resampled_tracks_path)
    if current.cluster_assignments_path is not None:
        labels = pd.read_csv(current.cluster_assignments_path)
    elif current.clustering_dir is not None and current.chosen_k is not None:
        labels = load_labels(current.clustering_dir, current.chosen_k)
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
