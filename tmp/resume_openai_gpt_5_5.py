from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from vlm_ppe.agents.graph import (
    _dedupe_text,
    _normalize_window_ids,
    _validate_window_pattern_count_review,
    _window_proposals_json,
)
from vlm_ppe.agents.prompts import window_pattern_count_prompt, window_review_prompt
from vlm_ppe.agents.tools import export_state_tool, render_window_review_diagnostics, validate_window_reviews_tool
from vlm_ppe.audit import (
    log_window_pattern_count_vlm_request,
    log_window_pattern_count_vlm_response,
    log_window_vlm_request,
    log_window_vlm_response,
    setup_audit_logging,
)
from vlm_ppe.config import load_config
from vlm_ppe.schemas import ClusterMedoid, EvidenceImage, WindowPatternCountReview, WindowProposal, WindowReview


ROOT = Path("data/artifacts/ppe/2026-04-01/runs/openai-gpt-5-5")
CONFIG_PATH = Path("configs/ppe_kdfw_arrivals_openai_gpt_5_5.yaml")


class TimeoutOpenRouterClient:
    def __init__(self, *, model: str, reasoning_effort: str | None, timeout_s: float = 240.0) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
            timeout=timeout_s,
            max_retries=1,
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.app_referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        self.app_title = os.environ.get("OPENROUTER_X_TITLE") or "project-rustlingtree/vlm-ppe"

    def review_window_pattern_count(
        self,
        *,
        cluster_id: int,
        evidence_images: list[EvidenceImage],
        max_patterns: int,
        prompt: str,
    ) -> WindowPatternCountReview:
        text = self._request_json_text(prompt=prompt, evidence_images=evidence_images)
        return WindowPatternCountReview.model_validate(json.loads(text))

    def review_windows(
        self,
        *,
        cluster_id: int,
        evidence_images: list[EvidenceImage],
        pattern_count: int,
        pattern_index: int,
        attempt: int,
        max_attempts: int,
        previous_review_json: str | None,
        prompt: str,
    ) -> WindowReview:
        text = self._request_json_text(prompt=prompt, evidence_images=evidence_images)
        return WindowReview.model_validate(json.loads(text))

    def _request_json_text(self, *, prompt: str, evidence_images: list[EvidenceImage]) -> str:
        contents: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in evidence_images:
            contents.append({"type": "text", "text": f"Image: {image.caption}"})
            contents.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(image.path))}})
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": contents}],
            "response_format": {"type": "json_object"},
            "extra_headers": _openrouter_headers(self.app_referer, self.app_title),
        }
        if self.reasoning_effort is not None:
            request["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if isinstance(content, str) and content.strip():
            return content
        raise ValueError("OpenRouter response did not contain text")


def _image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _openrouter_headers(app_referer: str | None, app_title: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if app_referer:
        headers["HTTP-Referer"] = app_referer
    if app_title:
        headers["X-OpenRouter-Title"] = app_title
    return headers


def load_medoids() -> list[ClusterMedoid]:
    medoid_rows = pd.read_parquet(ROOT / "templates" / "cluster_medoids.parquet")
    summary = json.loads((ROOT / "templates" / "cluster_summary.json").read_text(encoding="utf-8"))["clusters"]
    summary_by_cluster = {int(item["cluster_id"]): item for item in summary}
    medoids: list[ClusterMedoid] = []
    for cluster_id, group in medoid_rows.groupby("cluster_id", sort=True):
        ordered = group.sort_values("station_index", kind="stable")
        item = summary_by_cluster[int(cluster_id)]
        medoids.append(
            ClusterMedoid(
                cluster_id=int(cluster_id),
                medoid_track_id=str(item["medoid_track_id"]),
                n_tracks=int(item["n_tracks"]),
                mean_distance_nm=float(item["mean_distance_nm"]),
                max_distance_nm=float(item["max_distance_nm"]),
                template_points=[
                    (float(row.x_nm), float(row.y_nm))
                    for row in ordered.itertuples(index=False)
                ],
            )
        )
    return medoids


def baseline_evidence(cluster_id: int) -> list[EvidenceImage]:
    path = ROOT / "evidence" / "residual_windows" / f"cluster_{cluster_id:02d}_residual_windows.png"
    return [
        EvidenceImage(
            kind="residual_windows",
            path=path.as_posix(),
            caption=f"Cluster {cluster_id} residual diagnostics without window highlights",
        )
    ]


def load_json_model(path: Path, model_type):
    return model_type.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_existing_reviews(cluster_ids: list[int]) -> tuple[list[WindowPatternCountReview], list[str], list[WindowReview], list[str]]:
    count_reviews: list[WindowPatternCountReview] = []
    count_paths: list[str] = []
    reviews: list[WindowReview] = []
    review_paths: list[str] = []
    for cluster_id in cluster_ids:
        count_path = ROOT / "vlm_reviews" / "windows" / f"cluster_{cluster_id:02d}_pattern_count.json"
        if not count_path.exists():
            continue
        count = load_json_model(count_path, WindowPatternCountReview)
        count_reviews.append(count)
        count_paths.append(count_path.as_posix())
        attempt_paths = sorted(
            path
            for path in (ROOT / "vlm_reviews" / "windows").glob(f"cluster_{cluster_id:02d}_attempt_*.json")
            if not path.name.endswith("_request.json")
        )
        review_paths.extend(path.as_posix() for path in attempt_paths)
        if count.suggested_action == "human_review":
            reviews.append(WindowReview(cluster_id=cluster_id, windows=[], outlier_notes=count.outlier_notes, suggested_action="human_review"))
        elif int(count.pattern_count) == 0:
            reviews.append(WindowReview(cluster_id=cluster_id, windows=[], outlier_notes=count.outlier_notes, all_patterns_identified=True))
        elif attempt_paths:
            final = load_json_model(attempt_paths[-1], WindowReview)
            reviews.append(
                WindowReview(
                    cluster_id=cluster_id,
                    windows=final.windows,
                    outlier_notes=_dedupe_text([*count.outlier_notes, *final.outlier_notes]),
                    all_patterns_identified=True,
                    suggested_action=final.suggested_action,
                )
            )
    return count_reviews, count_paths, reviews, review_paths


def classify_missing_cluster(state: dict, cluster_id: int, client: TimeoutOpenRouterClient) -> tuple[WindowPatternCountReview, str, WindowReview, list[str], list[EvidenceImage]]:
    config = load_config(CONFIG_PATH)
    review_dir = ROOT / "vlm_reviews" / "windows"
    max_patterns = int(config.window_review_max_patterns)
    max_attempts = int(config.window_review_max_attempts)
    evidence = baseline_evidence(cluster_id)
    count_prompt = window_pattern_count_prompt(cluster_id, max_patterns=max_patterns)
    log_window_pattern_count_vlm_request(
        run_dir=ROOT,
        cluster_id=cluster_id,
        model=config.vlm_model,
        prompt=count_prompt,
        evidence_images=evidence,
    )
    count = client.review_window_pattern_count(
        cluster_id=cluster_id,
        evidence_images=evidence,
        max_patterns=max_patterns,
        prompt=count_prompt,
    )
    count = _validate_window_pattern_count_review(count, cluster_id, max_patterns)
    count_path = review_dir / f"cluster_{cluster_id:02d}_pattern_count.json"
    count_path.write_text(count.model_dump_json(indent=2), encoding="utf-8")
    log_window_pattern_count_vlm_response(run_dir=ROOT, review=count, response_path=count_path.as_posix())

    if count.suggested_action == "human_review":
        return count, count_path.as_posix(), WindowReview(cluster_id=cluster_id, windows=[], outlier_notes=count.outlier_notes, suggested_action="human_review"), [], evidence
    if int(count.pattern_count) == 0:
        return count, count_path.as_posix(), WindowReview(cluster_id=cluster_id, windows=[], outlier_notes=count.outlier_notes, all_patterns_identified=True), [], evidence

    accepted_windows: list[WindowProposal] = []
    outlier_notes = [*count.outlier_notes]
    review_paths: list[str] = []
    all_evidence = [*evidence]
    all_patterns_identified = False
    final_action = count.suggested_action
    global_attempt = 0

    while len(accepted_windows) < int(count.pattern_count):
        highlighted_evidence: list[EvidenceImage] = []
        previous_review_json: str | None = None
        final_pattern_review: WindowReview | None = None
        for attempt in range(max_attempts):
            prompt = window_review_prompt(
                cluster_id,
                attempt=attempt,
                max_attempts=max_attempts,
                pattern_count=int(count.pattern_count),
                previous_review_json=previous_review_json,
                accepted_windows_json=_window_proposals_json(accepted_windows),
                pattern_index=len(accepted_windows) + 1,
            )
            attempt_evidence = [*evidence, *highlighted_evidence]
            logged_attempt = global_attempt
            log_window_vlm_request(
                run_dir=ROOT,
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
                pattern_count=int(count.pattern_count),
                pattern_index=len(accepted_windows) + 1,
                attempt=attempt,
                max_attempts=max_attempts,
                previous_review_json=previous_review_json,
                prompt=prompt,
            )
            if int(review.cluster_id) != cluster_id:
                raise ValueError(f"window review returned cluster_id={review.cluster_id}; expected {cluster_id}")
            review = _normalize_window_ids(cluster_id, review, accepted_windows)
            outlier_notes.extend(review.outlier_notes)
            all_patterns_identified = all_patterns_identified or review.all_patterns_identified
            final_action = review.suggested_action
            review_path = review_dir / f"cluster_{cluster_id:02d}_attempt_{logged_attempt:02d}.json"
            review_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
            log_window_vlm_response(run_dir=ROOT, attempt=logged_attempt, review=review, response_path=review_path.as_posix())
            review_paths.append(review_path.as_posix())
            final_pattern_review = review
            global_attempt += 1

            should_confirm_highlight = attempt == 0 and bool(review.windows)
            should_revise = review.suggested_action == "revise"
            if attempt >= max_attempts - 1 or review.suggested_action == "human_review" or not (should_confirm_highlight or should_revise):
                break
            previous_review_json = review.model_dump_json()
            highlighted_review = WindowReview(
                cluster_id=cluster_id,
                windows=[*accepted_windows, *review.windows],
                outlier_notes=review.outlier_notes,
                all_patterns_identified=review.all_patterns_identified,
                suggested_action=review.suggested_action,
            )
            highlighted_evidence = render_window_review_diagnostics(state, highlighted_review, proposal_attempt=logged_attempt)
            all_evidence.extend(highlighted_evidence)

        if final_pattern_review is None or final_pattern_review.suggested_action == "human_review":
            break
        if not final_pattern_review.windows:
            raise ValueError(f"window review for cluster_id={cluster_id} returned no window")
        if len(final_pattern_review.windows) > 1:
            raise ValueError(f"window review for cluster_id={cluster_id} returned multiple windows")
        accepted_windows.extend(final_pattern_review.windows)
        if all_patterns_identified or len(accepted_windows) >= int(count.pattern_count):
            all_patterns_identified = True
            break

    return (
        count,
        count_path.as_posix(),
        WindowReview(
            cluster_id=cluster_id,
            windows=accepted_windows,
            outlier_notes=_dedupe_text(outlier_notes),
            all_patterns_identified=all_patterns_identified,
            suggested_action=final_action,
        ),
        review_paths,
        all_evidence,
    )


def main() -> None:
    config = load_config(CONFIG_PATH)
    setup_audit_logging(ROOT, level=config.log_level, console=True)
    medoids = load_medoids()
    cluster_ids = [int(item.cluster_id) for item in medoids]
    count_reviews, count_paths, reviews, review_paths = load_existing_reviews(cluster_ids)
    existing_count_ids = {int(item.cluster_id) for item in count_reviews}
    missing = [cluster_id for cluster_id in cluster_ids if cluster_id not in existing_count_ids]
    if missing and missing != [8]:
        raise RuntimeError(f"expected only cluster 8 to be missing; missing={missing}")

    state = {
        "run_id": "openai-gpt-5-5",
        "run_dir": ROOT.as_posix(),
        "audit_log_path": (ROOT / "audit.log").as_posix(),
        "graph_events_path": (ROOT / "graph_events.jsonl").as_posix(),
        "vlm_interactions_path": (ROOT / "vlm_interactions.jsonl").as_posix(),
        "config": config.model_dump(mode="json"),
        "k_max_current": config.k_max,
        "chosen_k": 4,
        "tracks_path": (ROOT / "processed" / "tracks.parquet").as_posix(),
        "track_index_path": (ROOT / "processed" / "track_index.parquet").as_posix(),
        "resampled_tracks_path": (ROOT / "processed" / "resampled_tracks.parquet").as_posix(),
        "features_path": (ROOT / "processed" / "features.npz").as_posix(),
        "feature_metadata_path": (ROOT / "processed" / "feature_metadata.parquet").as_posix(),
        "cluster_assignments_path": (ROOT / "templates" / "chosen_cluster_assignments.csv").as_posix(),
        "subcluster_tree_path": (ROOT / "clustering" / "subcluster_tree.json").as_posix(),
        "medoids_path": (ROOT / "templates" / "cluster_medoids.parquet").as_posix(),
        "medoid_summary_path": (ROOT / "templates" / "cluster_summary.json").as_posix(),
        "medoid_report_path": (ROOT / "reports" / "medoid_report.md").as_posix(),
        "residual_profiles_path": (ROOT / "residuals" / "residual_profiles.parquet").as_posix(),
        "window_evidence_images": [image.model_dump() for cid in cluster_ids for image in baseline_evidence(cid)],
        "window_pattern_count_reviews": [review.model_dump() for review in count_reviews],
        "window_pattern_count_review_paths": count_paths,
        "window_reviews": [review.model_dump() for review in reviews],
        "window_review_paths": review_paths,
        "medoids": [item.model_dump() for item in medoids],
        "status": "resume_before_cluster_08",
    }

    if missing:
        client = TimeoutOpenRouterClient(model=config.vlm_model, reasoning_effort=config.vlm_reasoning_effort)
        count, count_path, review, new_review_paths, new_evidence = classify_missing_cluster(state, 8, client)
        count_reviews.append(count)
        count_paths.append(count_path)
        reviews.append(review)
        review_paths.extend(new_review_paths)
    else:
        new_evidence = []
    state["window_pattern_count_reviews"] = [item.model_dump() for item in sorted(count_reviews, key=lambda item: item.cluster_id)]
    state["window_pattern_count_review_paths"] = count_paths
    state["window_reviews"] = [item.model_dump() for item in sorted(reviews, key=lambda item: item.cluster_id)]
    state["window_review_paths"] = review_paths
    state["window_evidence_images"] = [*state["window_evidence_images"], *[item.model_dump() for item in new_evidence]]
    state["status"] = "vlm_reviewed_windows"

    state.update(validate_window_reviews_tool(state))
    state.update(export_state_tool(state))
    print(json.dumps({"status": state["status"], "state_path": state["state_path"], "intervention_windows_path": state["intervention_windows_path"]}, indent=2))


if __name__ == "__main__":
    main()
