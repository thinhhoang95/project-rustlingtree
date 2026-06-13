from __future__ import annotations

from vlm_ppe.agents.prompts import cluster_review_prompt, window_review_prompt
from vlm_ppe.schemas import WindowReview


def test_cluster_review_prompt_uses_literal_array_examples() -> None:
    prompt = cluster_review_prompt(metrics=[], available_k=[2, 3, 4], attempt=0, max_retries=1)

    assert '"rationale": ["K=4 separates visually distinct trajectory families."]' in prompt
    assert '"rejected_alternatives": ["K=5 creates a small cluster that looks like noise."]' in prompt
    assert '"rationale": [string]' not in prompt


def test_window_review_prompt_uses_literal_classification_example() -> None:
    prompt = window_review_prompt(2)

    assert '"class_name": "dogleg"' in prompt
    assert '"start_station_index": 32' in prompt
    assert '"end_station_index": 61' in prompt
    assert '"outlier_notes": ["Two tracks appear visually different from the main window pattern."]' in prompt
    assert "Your task is to propose the intervention window boundaries" in prompt
    assert "unhighlighted residual diagnostics" in prompt


def test_window_review_prompt_describes_highlight_revision_attempt() -> None:
    prompt = window_review_prompt(2, attempt=1, max_attempts=3, previous_review_json='{"cluster_id":2}')

    assert "highlighted diagnostic image based on your previous proposal" in prompt
    assert 'suggested_action set to "revise"' in prompt
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
