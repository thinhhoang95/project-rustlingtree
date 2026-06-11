from __future__ import annotations

import json

from vlm_ppe.schemas import KMetric


def cluster_review_prompt(metrics: list[KMetric], available_k: list[int], attempt: int, max_retries: int) -> str:
    metrics_payload = [metric.model_dump() for metric in metrics]
    return (
        "You are the visual reviewer and orchestrator for VLM-PPE.\n"
        "Use only the provided ADS-B trajectory plots and numeric metrics. Do not infer from AIP charts.\n"
        "Choose the most plausible number of practical path clusters. Prefer fewer clusters when differences look like noise, "
        "and more clusters only when common geometry is visually distinct. You may request a bounded retry if the evidence is "
        "insufficient or K should be expanded.\n\n"
        f"Attempt: {attempt + 1}. Maximum retries: {max_retries}.\n"
        f"Available K values: {available_k}.\n"
        f"K metrics JSON: {json.dumps(metrics_payload, separators=(',', ':'))}\n\n"
        "Return strict JSON matching this example shape. Text-list fields must always be JSON arrays, "
        "even when there is only one item:\n"
        "{\n"
        '  "chosen_k": 4,\n'
        '  "confidence": 0.82,\n'
        '  "rationale": ["K=4 separates visually distinct trajectory families."],\n'
        '  "rejected_alternatives": ["K=5 creates a small cluster that looks like noise."],\n'
        '  "clusters_to_recheck": [1],\n'
        '  "retry_requested": false,\n'
        '  "requested_k_max": null,\n'
        '  "suggested_action": "accept"\n'
        "}\n"
    )
