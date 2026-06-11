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
        "Return strict JSON with this shape:\n"
        "{\n"
        '  "chosen_k": int,\n'
        '  "confidence": float between 0 and 1,\n'
        '  "rationale": [string],\n'
        '  "rejected_alternatives": [string],\n'
        '  "clusters_to_recheck": [int],\n'
        '  "retry_requested": boolean,\n'
        '  "requested_k_max": int or null,\n'
        '  "suggested_action": "accept" | "retry" | "human_review"\n'
        "}\n"
    )
