from __future__ import annotations

import json

from vlm_ppe.schemas import InterventionWindow, KMetric


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


def window_review_prompt(cluster_id: int, windows: list[InterventionWindow]) -> str:
    windows_payload = [window.model_dump() for window in windows]
    return (
        "You are reviewing one VLM-PPE trajectory cluster.\n"
        "Use only the provided ADS-B trajectory plots and numeric residual-window records. Do not infer from AIP charts.\n"
        "Classify each detected candidate intervention window. The deterministic tool created the windows; you may only "
        "assign class labels and note ambiguity.\n\n"
        "Class definitions:\n"
        "- no_stretch: no meaningful deviation program; max cross-track deviation and added path length appear small.\n"
        "- dogleg: one outward vector-like leg followed by one closure/rejoin leg.\n"
        "- trombone: two vector-like legs before closure/rejoin.\n"
        "- PMS: point-merge-like sequencing leg followed by direct-to common merge point.\n"
        "- other: holding, looping, direct shortcut, unclear, too rare, or too complex for v1.\n\n"
        f"Cluster ID: {cluster_id}.\n"
        f"Detected windows JSON: {json.dumps(windows_payload, separators=(',', ':'))}\n\n"
        "Return strict JSON matching this example shape. Text-list fields must always be JSON arrays, "
        "even when there is only one item:\n"
        "{\n"
        f'  "cluster_id": {cluster_id},\n'
        '  "windows": [\n'
        "    {\n"
        '      "window_id": "C2_W1",\n'
        '      "class_name": "dogleg",\n'
        '      "confidence": 0.82,\n'
        '      "visual_reason": "One clear outward vector and one closure leg back to common flow."\n'
        "    }\n"
        "  ],\n"
        '  "outlier_notes": ["Two tracks appear visually different from the main window pattern."],\n'
        '  "suggested_action": "accept"\n'
        "}\n"
    )
