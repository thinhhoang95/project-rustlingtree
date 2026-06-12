from __future__ import annotations

from vlm_ppe.agents.prompts import cluster_review_prompt, window_review_prompt
from vlm_ppe.schemas import InterventionWindow, WindowReview


def test_cluster_review_prompt_uses_literal_array_examples() -> None:
    prompt = cluster_review_prompt(metrics=[], available_k=[2, 3, 4], attempt=0, max_retries=1)

    assert '"rationale": ["K=4 separates visually distinct trajectory families."]' in prompt
    assert '"rejected_alternatives": ["K=5 creates a small cluster that looks like noise."]' in prompt
    assert '"rationale": [string]' not in prompt


def test_window_review_prompt_uses_literal_classification_example() -> None:
    window = InterventionWindow(
        cluster_id=2,
        window_id="C2_W1",
        start_station_index=3,
        end_station_index=6,
        start_s_fraction=0.3,
        end_s_fraction=0.6,
        start_s_nm=3.0,
        end_s_nm=6.0,
        length_nm=3.0,
        peak_residual_energy_nm2=9.0,
        peak_heading_dispersion=0.1,
        trigger_reasons=["residual_energy"],
        track_ids=["T1", "T2"],
    )

    prompt = window_review_prompt(2, [window])

    assert '"class_name": "dogleg"' in prompt
    assert '"outlier_notes": ["Two tracks appear visually different from the main window pattern."]' in prompt
    assert "Detected windows JSON:" in prompt


def test_window_review_schema_accepts_v1_classes() -> None:
    for class_name in ["no_stretch", "dogleg", "trombone", "PMS", "other"]:
        review = WindowReview.model_validate(
            {
                "cluster_id": 1,
                "windows": [
                    {
                        "window_id": "C1_W1",
                        "class_name": class_name,
                        "confidence": 0.7,
                        "visual_reason": "ok",
                    }
                ],
                "outlier_notes": "single note",
                "suggested_action": "accept",
            }
        )

        assert review.windows[0].class_name == class_name
        assert review.outlier_notes == ["single note"]
