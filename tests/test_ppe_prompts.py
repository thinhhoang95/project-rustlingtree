from __future__ import annotations

from vlm_ppe.agents.prompts import cluster_review_prompt


def test_cluster_review_prompt_uses_literal_array_examples() -> None:
    prompt = cluster_review_prompt(metrics=[], available_k=[2, 3, 4], attempt=0, max_retries=1)

    assert '"rationale": ["K=4 separates visually distinct trajectory families."]' in prompt
    assert '"rejected_alternatives": ["K=5 creates a small cluster that looks like noise."]' in prompt
    assert '"rationale": [string]' not in prompt
