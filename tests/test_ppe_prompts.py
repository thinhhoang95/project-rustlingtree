from __future__ import annotations

from vlm_ppe.agents.prompts import (
    cluster_review_prompt,
    subcluster_review_prompt,
    window_cluster_selection_prompt,
    window_review_prompt,
)
from vlm_ppe.schemas import ClusterMedoid, CommunityMetric
from vlm_ppe.schemas import WindowReview


def test_cluster_review_prompt_uses_literal_array_examples() -> None:
    prompt = cluster_review_prompt(metrics=[], available_thresholds_nm=[0.5, 1.0, 1.5], attempt=0, max_retries=1)

    assert (
        '"rationale": ["A 1.75 NM threshold separates visually distinct trajectory communities."]' in prompt
    )
    assert (
        '"rejected_alternatives": ["A 1.25 NM threshold creates tiny communities that look like noise."]'
        in prompt
    )
    assert '"rationale": [string]' not in prompt
    assert "main repeated path patterns" in prompt
    assert '"chosen_threshold_nm": 1.75' in prompt


def test_subcluster_review_prompt_explains_local_threshold_and_overdetail_guardrail() -> None:
    prompt = subcluster_review_prompt(
        metrics=[
            CommunityMetric(
                candidate_id=1,
                threshold_nm=0.8,
                community_count=2,
                edge_count=3,
                edge_density=0.3,
                silhouette=0.4,
                community_count_min=2,
                community_count_max=3,
                community_count_mean=2.5,
                singleton_count=0,
            )
        ],
        available_thresholds_nm=[0.4, 0.8],
        root_cluster_id=3,
        lineage=[3],
        depth=0,
        n_tracks=5,
        min_tracks=4,
    )

    assert "Choose the local community-detection distance threshold" in prompt
    assert "yields one community" in prompt
    assert "Require clean separation" in prompt
    assert "Outliers can hide real repeated geometry" in prompt
    assert "do not reject a lower threshold only because it contains tiny outlier/noise communities" in prompt
    assert "split children stop at depth 1" in prompt
    assert "will not keep looping" in prompt
    assert 'do not return "split"' in prompt
    assert '"clusters_to_recheck": []' in prompt
    assert '"chosen_threshold_nm": 0.85' in prompt


def test_window_cluster_selection_prompt_selects_only_meaningful_clusters() -> None:
    prompt = window_cluster_selection_prompt(
        [
            ClusterMedoid(
                cluster_id=0,
                medoid_track_id="T0",
                n_tracks=120,
                mean_distance_nm=1.2,
                max_distance_nm=5.4,
                template_points=[(0.0, 0.0), (1.0, 1.0)],
            )
        ]
    )

    assert "Select only clusters that should proceed to detailed intervention-window analysis" in prompt
    assert "outlier quarantines" in prompt
    assert '"selected_cluster_ids": [0, 2, 4]' in prompt
    assert '"skipped_clusters"' in prompt
    assert '"n_tracks":120' in prompt


def test_window_review_prompt_uses_literal_classification_example() -> None:
    prompt = window_review_prompt(2)

    assert '"class_name": "dogleg"' in prompt
    assert '"start_station_index": 32' in prompt
    assert '"end_station_index": 61' in prompt
    assert '"pattern_count": 2' in prompt
    assert '"all_patterns_identified": false' in prompt
    assert '"outlier_notes": ["Two tracks appear visually different from the main window pattern."]' in prompt
    assert "Your task is to count distinct intervention patterns" in prompt
    assert "The window must be tight" in prompt
    assert "unhighlighted residual diagnostics" in prompt


def test_window_review_prompt_describes_highlight_revision_attempt() -> None:
    prompt = window_review_prompt(2, attempt=1, max_attempts=3, previous_review_json='{"cluster_id":2}')

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
