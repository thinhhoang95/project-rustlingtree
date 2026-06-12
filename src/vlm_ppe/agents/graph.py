from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from vlm_ppe.agents.prompts import cluster_review_prompt, window_review_prompt
from vlm_ppe.agents.tools import (
    append_event,
    build_features_tool,
    compute_residual_windows_tool,
    compute_medoids_tool,
    export_state_tool,
    ingest_tracks_tool,
    render_evidence_pack_tool,
    render_window_diagnostics_tool,
    render_medoid_report_tool,
    resample_tracks_tool,
    retry_or_accept_tool,
    run_candidate_clustering_tool,
    state_model,
    validate_review_tool,
    validate_window_reviews_tool,
)
from vlm_ppe.agents.vlm_client import ClusterReviewClient, OpenRouterVLMClient
from vlm_ppe.audit import (
    log_vlm_request,
    log_vlm_response,
    log_window_vlm_request,
    log_window_vlm_response,
    setup_audit_logging,
)
from vlm_ppe.schemas import (
    ClusterMedoid,
    ClusterReview,
    EvidenceImage,
    InterventionWindow,
    KMetric,
    PPEConfig,
    PPEState,
    WindowClassification,
    WindowReview,
)


def _node(name: str, fn):
    def wrapped(state: dict) -> dict:
        append_event(state, name, "started")
        try:
            update = fn(state)
            merged = {**state, **update}
            append_event(merged, name, "completed", payload={key: update[key] for key in update if key != "medoids"})
            return merged
        except Exception as exc:
            append_event(state, name, "failed", message=str(exc))
            raise

    return wrapped


def _vlm_review_node(vlm_client: ClusterReviewClient | None):
    def node(state: dict) -> dict:
        current = state_model(state)
        config = PPEConfig.model_validate(current.config)
        evidence = [EvidenceImage.model_validate(item) for item in current.evidence_images]
        metrics = [KMetric.model_validate(item) for item in state.get("clustering_metrics", [])]
        available_k = [int(k) for k in state.get("candidate_k_values", [])]
        prompt = cluster_review_prompt(metrics, available_k, current.retry_count, config.max_retries)
        log_vlm_request(
            run_dir=current.run_dir,
            attempt=current.retry_count,
            model=config.vlm_model,
            prompt=prompt,
            evidence_images=evidence,
            metrics=metrics,
            available_k=available_k,
            offline_override=current.chosen_k_override is not None,
        )
        if current.chosen_k_override is not None:
            review = ClusterReview(
                chosen_k=current.chosen_k_override,
                confidence=1.0,
                rationale=["Offline/manual cluster count override supplied by --chosen-k."],
                rejected_alternatives=[],
                clusters_to_recheck=[],
                retry_requested=False,
                requested_k_max=None,
                suggested_action="accept",
            )
        else:
            client = vlm_client or OpenRouterVLMClient(model=config.vlm_model)
            review = client.review_clusters(
                evidence_images=evidence,
                metrics=metrics,
                available_k=available_k,
                attempt=current.retry_count,
                max_retries=config.max_retries,
                prompt=prompt,
            )
        review_dir = Path(current.run_dir) / "vlm_reviews"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_dir / f"attempt_{current.retry_count:02d}.json"
        review_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
        log_vlm_response(
            run_dir=current.run_dir,
            attempt=current.retry_count,
            review=review,
            response_path=review_path.as_posix(),
        )
        return {
            "vlm_reviews": [*current.vlm_reviews, review.model_dump()],
            "latest_vlm_review_path": review_path.as_posix(),
            "status": "vlm_reviewed_clusters",
        }

    return _node("vlm_review_clusters", node)


def _vlm_window_review_node(vlm_client: ClusterReviewClient | None):
    def node(state: dict) -> dict:
        current = state_model(state)
        config = PPEConfig.model_validate(current.config)
        medoids = [ClusterMedoid.model_validate(item) for item in state.get("medoids", [])]
        evidence = [EvidenceImage.model_validate(item) for item in current.window_evidence_images]
        windows = [InterventionWindow.model_validate(item) for item in current.intervention_windows]
        windows_by_cluster: dict[int, list[InterventionWindow]] = {}
        for window in windows:
            windows_by_cluster.setdefault(int(window.cluster_id), []).append(window)

        reviews: list[WindowReview] = []
        review_paths: list[str] = []
        review_dir = Path(current.run_dir) / "vlm_reviews" / "windows"
        review_dir.mkdir(parents=True, exist_ok=True)

        for medoid in sorted(medoids, key=lambda item: item.cluster_id):
            cluster_id = int(medoid.cluster_id)
            cluster_windows = windows_by_cluster.get(cluster_id, [])
            prompt = window_review_prompt(cluster_id, cluster_windows)
            cluster_evidence = _cluster_window_evidence(evidence, cluster_id)
            offline_override = current.chosen_k_override is not None or not cluster_windows
            log_window_vlm_request(
                run_dir=current.run_dir,
                cluster_id=cluster_id,
                model=config.vlm_model,
                prompt=prompt,
                evidence_images=cluster_evidence,
                windows=cluster_windows,
                offline_override=offline_override,
            )
            if not cluster_windows:
                review = WindowReview(
                    cluster_id=cluster_id,
                    windows=[],
                    outlier_notes=["No deterministic candidate intervention windows were detected."],
                    suggested_action="accept",
                )
            elif current.chosen_k_override is not None:
                review = WindowReview(
                    cluster_id=cluster_id,
                    windows=[
                        WindowClassification(
                            window_id=window.window_id,
                            class_name="other",
                            confidence=1.0,
                            visual_reason="Offline/manual run: candidate window detected, VLM classification bypassed.",
                        )
                        for window in cluster_windows
                    ],
                    outlier_notes=[],
                    suggested_action="accept",
                )
            else:
                client = vlm_client or OpenRouterVLMClient(model=config.vlm_model)
                review = client.review_windows(
                    cluster_id=cluster_id,
                    windows=cluster_windows,
                    evidence_images=cluster_evidence,
                    prompt=prompt,
                )
            review_path = review_dir / f"cluster_{cluster_id:02d}.json"
            review_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
            log_window_vlm_response(run_dir=current.run_dir, review=review, response_path=review_path.as_posix())
            reviews.append(review)
            review_paths.append(review_path.as_posix())

        return {
            "window_reviews": [review.model_dump() for review in reviews],
            "window_review_paths": review_paths,
            "status": "vlm_reviewed_windows",
        }

    return _node("vlm_classify_windows", node)


def _cluster_window_evidence(evidence: list[EvidenceImage], cluster_id: int) -> list[EvidenceImage]:
    marker = f"Cluster {int(cluster_id)} "
    selected = [image for image in evidence if image.caption.startswith(marker)]
    return selected or evidence


def _should_retry(state: dict) -> str:
    return "retry" if bool(state.get("should_retry")) else "accept"


def build_graph(vlm_client: ClusterReviewClient | None = None):
    graph = StateGraph(dict)
    graph.add_node("ingest_tracks", _node("ingest_tracks", ingest_tracks_tool))
    graph.add_node("resample_tracks", _node("resample_tracks", resample_tracks_tool))
    graph.add_node("build_features", _node("build_features", build_features_tool))
    graph.add_node("run_candidate_clustering", _node("run_candidate_clustering", run_candidate_clustering_tool))
    graph.add_node("render_evidence_pack", _node("render_evidence_pack", render_evidence_pack_tool))
    graph.add_node("vlm_review_clusters", _vlm_review_node(vlm_client))
    graph.add_node("validate_review", _node("validate_review", validate_review_tool))
    graph.add_node("retry_or_accept", _node("retry_or_accept", retry_or_accept_tool))
    graph.add_node("compute_cluster_medoids", _node("compute_cluster_medoids", compute_medoids_tool))
    graph.add_node("render_medoid_report", _node("render_medoid_report", render_medoid_report_tool))
    graph.add_node("compute_residual_windows", _node("compute_residual_windows", compute_residual_windows_tool))
    graph.add_node("render_window_diagnostics", _node("render_window_diagnostics", render_window_diagnostics_tool))
    graph.add_node("vlm_classify_windows", _vlm_window_review_node(vlm_client))
    graph.add_node("validate_window_reviews", _node("validate_window_reviews", validate_window_reviews_tool))
    graph.add_node("export_state", _node("export_state", export_state_tool))

    graph.add_edge(START, "ingest_tracks")
    graph.add_edge("ingest_tracks", "resample_tracks")
    graph.add_edge("resample_tracks", "build_features")
    graph.add_edge("build_features", "run_candidate_clustering")
    graph.add_edge("run_candidate_clustering", "render_evidence_pack")
    graph.add_edge("render_evidence_pack", "vlm_review_clusters")
    graph.add_edge("vlm_review_clusters", "validate_review")
    graph.add_edge("validate_review", "retry_or_accept")
    graph.add_conditional_edges(
        "retry_or_accept",
        _should_retry,
        {"retry": "run_candidate_clustering", "accept": "compute_cluster_medoids"},
    )
    graph.add_edge("compute_cluster_medoids", "render_medoid_report")
    graph.add_edge("render_medoid_report", "compute_residual_windows")
    graph.add_edge("compute_residual_windows", "render_window_diagnostics")
    graph.add_edge("render_window_diagnostics", "vlm_classify_windows")
    graph.add_edge("vlm_classify_windows", "validate_window_reviews")
    graph.add_edge("validate_window_reviews", "export_state")
    graph.add_edge("export_state", END)
    return graph.compile()


def initial_state(
    config: PPEConfig,
    *,
    run_id: str | None = None,
    chosen_k: int | None = None,
    require_api_key: bool = True,
) -> dict[str, Any]:
    resolved_run_id = run_id or uuid.uuid4().hex[:12]
    run_dir = Path(config.output_root) / "runs" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = PPEState(
        run_id=resolved_run_id,
        run_dir=run_dir.as_posix(),
        audit_log_path=(run_dir / "audit.log").as_posix(),
        graph_events_path=(run_dir / "graph_events.jsonl").as_posix(),
        vlm_interactions_path=(run_dir / "vlm_interactions.jsonl").as_posix(),
        config=config.model_dump(mode="json"),
        k_max_current=config.k_max,
        chosen_k_override=chosen_k,
    )
    if require_api_key and chosen_k is None and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required for VLM-led runs; use --chosen-k for offline/test runs")
    return state.model_dump(mode="json")


def run_graph(config: PPEConfig, *, chosen_k: int | None = None, run_id: str | None = None, vlm_client: ClusterReviewClient | None = None) -> dict:
    graph = build_graph(vlm_client=vlm_client)
    state = initial_state(config, run_id=run_id, chosen_k=chosen_k, require_api_key=vlm_client is None)
    setup_audit_logging(state["run_dir"], level=config.log_level, console=config.log_to_console)
    return graph.invoke(
        state
    )
