from __future__ import annotations

from vlm_ppe.agents.prompts import (
    cluster_review_prompt,
    subcluster_review_prompt,
    window_pattern_count_prompt,
    window_review_prompt,
)
from vlm_ppe.schemas import WindowReview


def test_cluster_review_prompt_uses_literal_array_examples() -> None:
    prompt = cluster_review_prompt(metrics=[], available_k=[2, 3, 4], attempt=0, max_retries=1)

    assert '"rationale": ["K=4 separates visually distinct trajectory families."]' in prompt
    assert '"rejected_alternatives": ["K=5 creates a small cluster that looks like noise."]' in prompt
    assert '"rationale": [string]' not in prompt
    assert "main repeated path patterns" in prompt


def test_subcluster_review_prompt_requests_manual_polygon_capture() -> None:
    prompt = subcluster_review_prompt(
        root_cluster_id=3,
        lineage=[3],
        depth=0,
        n_tracks=5,
        min_tracks=4,
        max_subclusters=3,
        coordinate_bounds={"x_min_nm": -8.0, "x_max_nm": -2.0, "y_min_nm": -1.0, "y_max_nm": 4.0},
    )

    assert "manual visual separation" in prompt
    assert "N=1 means the cluster is already one practical path pattern" in prompt
    assert "convex hull" in prompt
    assert "crosses, touches, or runs inside the convex polygon" in prompt
    assert "Do not split one trajectory family on spacing, sample density, or minor noisy variation" in prompt
    assert "split children stop at depth 1" in prompt
    assert '"subcluster_count": 3' in prompt
    assert '"polygon": [[-8.0, 2.5], [-6.8, 2.4], [-6.8, 3.3], [-8.0, 3.4]]' in prompt
    assert '"x_min_nm":-8.0' in prompt


def test_window_pattern_count_prompt_requests_count_only_with_rationale() -> None:
    prompt = window_pattern_count_prompt(2, max_patterns=8)

    assert "only to count how many distinct intervention patterns/windows are present" in prompt
    assert "Do not propose station boundaries, window IDs, or class labels" in prompt
    assert '"pattern_count": 2' in prompt
    assert '"confidence": 0.81' in prompt
    assert '"rationale": ["Two separated residual-energy peaks align with two visually distinct maneuver regions."]' in prompt
    assert "Maximum count allowed by configuration: 8" in prompt


def test_window_review_prompt_uses_literal_classification_example() -> None:
    prompt = window_review_prompt(2, pattern_count=2)

    assert '"class_name": "dogleg"' in prompt
    assert '"start_station_index": 32' in prompt
    assert '"end_station_index": 61' in prompt
    assert '"pattern_count": 2' not in prompt
    assert '"all_patterns_identified": false' in prompt
    assert '"outlier_notes": ["Two tracks appear visually different from the main window pattern."]' in prompt
    assert "Fixed pattern_count from prior count-only review: 2" in prompt
    assert "Do not revise pattern_count in this request" in prompt
    assert "Your task is to propose and classify tight intervention window boundaries" in prompt
    assert "The window must be tight" in prompt
    assert "unhighlighted residual diagnostics" in prompt


def test_window_review_prompt_describes_highlight_revision_attempt() -> None:
    prompt = window_review_prompt(2, attempt=1, max_attempts=3, pattern_count=2, previous_review_json='{"cluster_id":2}')

    assert "highlighted diagnostic image based on your previous proposal" in prompt
    assert 'suggested_action set to "revise"' in prompt
    assert "If the highlighted window range is tight" in prompt
    assert 'Previous window proposal JSON: {"cluster_id":2}' in prompt


def test_window_review_schema_accepts_v1_classes() -> None:
    for class_name in ["no_stretch", "dogleg", "trombone", "PMS", "other"]:
        review = WindowReview.model_validate(
            {
                "cluster_id": 1,
                "windows": [
                    {
                        "window_id": "C1_W1",
                        "start_station_index": 3,
                        "end_station_index": 6,
                        "class_name": class_name,
                        "confidence": 0.7,
                        "visual_reason": "ok",
                    }
                ],
                "outlier_notes": "single note",
                "suggested_action": "revise",
            }
        )

        assert review.windows[0].class_name == class_name
        assert review.outlier_notes == ["single note"]
        assert review.suggested_action == "revise"
