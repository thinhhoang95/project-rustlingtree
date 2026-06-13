from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from vlm_ppe.agents.prompts import cluster_review_prompt, window_cluster_selection_prompt, window_review_prompt
from vlm_ppe.agents.tools import (
    append_event,
    build_features_tool,
    compute_residual_profiles_tool,
    compute_medoids_tool,
    export_state_tool,
    ingest_tracks_tool,
    render_evidence_pack_tool,
    render_window_diagnostics_tool,
    render_window_review_diagnostics,
    render_medoid_report_tool,
    resample_tracks_tool,
    refine_subclusters_tool,
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
    log_window_cluster_selection_request,
    log_window_cluster_selection_response,
    log_window_vlm_request,
    log_window_vlm_response,
    setup_audit_logging,
)
from vlm_ppe.schemas import (
    ClusterMedoid,
    ClusterReview,
    EvidenceImage,
    KMetric,
    PPEConfig,
    PPEState,
    WindowProposal,
    WindowClusterSelection,
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
            client = vlm_client or OpenRouterVLMClient(
                model=config.vlm_model,
                reasoning_effort=config.vlm_reasoning_effort,
            )
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
        max_attempts = int(config.window_review_max_attempts)
        client = vlm_client or OpenRouterVLMClient(
            model=config.vlm_model,
            reasoning_effort=config.vlm_reasoning_effort,
        )

        reviews: list[WindowReview] = []
        review_paths: list[str] = []
        all_evidence: list[EvidenceImage] = [*evidence]
        review_dir = Path(current.run_dir) / "vlm_reviews" / "windows"
        review_dir.mkdir(parents=True, exist_ok=True)

        selection_prompt = window_cluster_selection_prompt(medoids)
        log_window_cluster_selection_request(
            run_dir=current.run_dir,
            model=config.vlm_model,
            prompt=selection_prompt,
            evidence_images=evidence,
        )
        selection = client.review_window_clusters(
            evidence_images=evidence,
            medoids=medoids,
            prompt=selection_prompt,
        )
        selection = _validate_window_cluster_selection(selection, medoids)
        selection_path = review_dir / "cluster_selection.json"
        selection_path.write_text(selection.model_dump_json(indent=2), encoding="utf-8")
        log_window_cluster_selection_response(
            run_dir=current.run_dir,
            selection=selection,
            response_path=selection_path.as_posix(),
        )

        selected_cluster_ids = set(selection.selected_cluster_ids)
        for medoid in sorted(medoids, key=lambda item: item.cluster_id):
            cluster_id = int(medoid.cluster_id)
            if cluster_id not in selected_cluster_ids:
                continue
            baseline_evidence = _cluster_window_evidence(evidence, cluster_id)
            accepted_windows: list[WindowProposal] = []
            outlier_notes: list[str] = []
            pattern_count: int | None = None
            all_patterns_identified = False
            final_action = "accept"
            global_attempt = 0
            max_patterns = int(config.window_review_max_patterns)

            while len(accepted_windows) < max_patterns:
                highlighted_evidence: list[EvidenceImage] = []
                previous_review_json: str | None = None
                final_pattern_review: WindowReview | None = None

                for attempt in range(max_attempts):
                    accepted_windows_json = _window_proposals_json(accepted_windows)
                    prompt = window_review_prompt(
                        cluster_id,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        previous_review_json=previous_review_json,
                        accepted_windows_json=accepted_windows_json,
                        pattern_index=len(accepted_windows) + 1,
                        pattern_count=pattern_count,
                    )
                    attempt_evidence = [*baseline_evidence, *highlighted_evidence]
                    logged_attempt = global_attempt
                    log_window_vlm_request(
                        run_dir=current.run_dir,
                        cluster_id=cluster_id,
                        attempt=logged_attempt,
                        model=config.vlm_model,
                        prompt=prompt,
                        evidence_images=attempt_evidence,
                        offline_override=False,
                    )
                    review = client.review_windows(
                        cluster_id=cluster_id,
                        evidence_images=attempt_evidence,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        previous_review_json=previous_review_json,
                        prompt=prompt,
                    )
                    if int(review.cluster_id) != cluster_id:
                        raise ValueError(f"window review returned cluster_id={review.cluster_id}; expected {cluster_id}")
                    review = _normalize_window_ids(cluster_id, review, accepted_windows)
                    if review.pattern_count is not None:
                        pattern_count = int(review.pattern_count)
                    outlier_notes.extend(review.outlier_notes)
                    all_patterns_identified = all_patterns_identified or review.all_patterns_identified
                    final_action = review.suggested_action
                    review_path = review_dir / f"cluster_{cluster_id:02d}_attempt_{logged_attempt:02d}.json"
                    review_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
                    log_window_vlm_response(
                        run_dir=current.run_dir,
                        attempt=logged_attempt,
                        review=review,
                        response_path=review_path.as_posix(),
                    )
                    review_paths.append(review_path.as_posix())
                    final_pattern_review = review
                    global_attempt += 1

                    should_confirm_highlight = attempt == 0 and bool(review.windows)
                    should_revise = review.suggested_action == "revise"
                    can_continue = attempt < max_attempts - 1
                    if (
                        not can_continue
                        or review.suggested_action == "human_review"
                        or not (should_confirm_highlight or should_revise)
                    ):
                        break

                    previous_review_json = review.model_dump_json()
                    highlighted_review = WindowReview(
                        cluster_id=cluster_id,
                        pattern_count=pattern_count,
                        windows=[*accepted_windows, *review.windows],
                        outlier_notes=review.outlier_notes,
                        all_patterns_identified=review.all_patterns_identified,
                        suggested_action=review.suggested_action,
                    )
                    highlighted_evidence = render_window_review_diagnostics(
                        state,
                        highlighted_review,
                        proposal_attempt=logged_attempt,
                    )
                    all_evidence.extend(highlighted_evidence)

                if final_pattern_review is None:
                    break
                if final_pattern_review.suggested_action == "human_review":
                    break
                if not final_pattern_review.windows:
                    all_patterns_identified = True
                    break

                accepted_windows.extend(final_pattern_review.windows)

                if pattern_count is None:
                    raise ValueError(f"window review for cluster_id={cluster_id} omitted pattern_count")
                if all_patterns_identified or len(accepted_windows) >= pattern_count:
                    all_patterns_identified = True
                    break

            if global_attempt == 0:
                raise ValueError(f"window review did not run for cluster_id={cluster_id}")
            reviews.append(
                WindowReview(
                    cluster_id=cluster_id,
                    pattern_count=pattern_count,
                    windows=accepted_windows,
                    outlier_notes=_dedupe_text(outlier_notes),
                    all_patterns_identified=all_patterns_identified,
                    suggested_action=final_action,
                )
            )

        return {
            "window_evidence_images": [item.model_dump() for item in all_evidence],
            "window_cluster_selection": selection.model_dump(),
            "window_cluster_selection_path": selection_path.as_posix(),
            "selected_window_cluster_ids": selection.selected_cluster_ids,
            "window_reviews": [review.model_dump() for review in reviews],
            "window_review_paths": review_paths,
            "status": "vlm_reviewed_windows",
        }

    return _node("vlm_classify_windows", node)


def _cluster_window_evidence(evidence: list[EvidenceImage], cluster_id: int) -> list[EvidenceImage]:
    marker = f"Cluster {int(cluster_id)} "
    selected = [image for image in evidence if image.caption.startswith(marker)]
    return selected or evidence


def _window_proposals_json(windows: list[WindowProposal]) -> str | None:
    if not windows:
        return None
    return "[" + ",".join(window.model_dump_json() for window in windows) + "]"


def _validate_window_cluster_selection(
    selection: WindowClusterSelection,
    medoids: list[ClusterMedoid],
) -> WindowClusterSelection:
    if selection.suggested_action == "human_review":
        raise ValueError("window cluster selection requested human review")

    known_ids = {int(medoid.cluster_id) for medoid in medoids}
    selected_ids: list[int] = []
    seen: set[int] = set()
    for cluster_id in selection.selected_cluster_ids:
        normalized = int(cluster_id)
        if normalized not in known_ids:
            raise ValueError(f"window cluster selection referenced unknown cluster_id={normalized}")
        if normalized not in seen:
            selected_ids.append(normalized)
            seen.add(normalized)

    for skipped in selection.skipped_clusters:
        if int(skipped.cluster_id) not in known_ids:
            raise ValueError(f"window cluster selection skipped unknown cluster_id={skipped.cluster_id}")

    return selection.model_copy(update={"selected_cluster_ids": sorted(selected_ids)})


def _normalize_window_ids(cluster_id: int, review: WindowReview, accepted_windows: list[WindowProposal]) -> WindowReview:
    used = {window.window_id for window in accepted_windows}
    normalized: list[WindowProposal] = []
    next_index = len(used) + 1
    for window in review.windows:
        if window.window_id in used:
            while f"C{cluster_id}_W{next_index}" in used:
                next_index += 1
            window = window.model_copy(update={"window_id": f"C{cluster_id}_W{next_index}"})
        used.add(window.window_id)
        normalized.append(window)
    if normalized == review.windows:
        return review
    return review.model_copy(update={"windows": normalized})


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        text = item.strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


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
    graph.add_node(
        "refine_subclusters",
        _node("refine_subclusters", lambda state: refine_subclusters_tool(state, vlm_client=vlm_client)),
    )
    graph.add_node("compute_cluster_medoids", _node("compute_cluster_medoids", compute_medoids_tool))
    graph.add_node("render_medoid_report", _node("render_medoid_report", render_medoid_report_tool))
    graph.add_node("compute_residual_profiles", _node("compute_residual_profiles", compute_residual_profiles_tool))
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
        {"retry": "run_candidate_clustering", "accept": "refine_subclusters"},
    )
    graph.add_edge("refine_subclusters", "compute_cluster_medoids")
    graph.add_edge("compute_cluster_medoids", "render_medoid_report")
    graph.add_edge("render_medoid_report", "compute_residual_profiles")
    graph.add_edge("compute_residual_profiles", "render_window_diagnostics")
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
    if require_api_key and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required for VLM-led runs")
    return state.model_dump(mode="json")


def run_graph(config: PPEConfig, *, chosen_k: int | None = None, run_id: str | None = None, vlm_client: ClusterReviewClient | None = None) -> dict:
    graph = build_graph(vlm_client=vlm_client)
    state = initial_state(config, run_id=run_id, chosen_k=chosen_k, require_api_key=vlm_client is None)
    setup_audit_logging(state["run_dir"], level=config.log_level, console=config.log_to_console)
    return graph.invoke(
        state
    )
