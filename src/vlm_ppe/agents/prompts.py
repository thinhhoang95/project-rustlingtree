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


def window_review_prompt(
    cluster_id: int,
    *,
    attempt: int = 0,
    max_attempts: int = 3,
    previous_review_json: str | None = None,
    accepted_windows_json: str | None = None,
    pattern_index: int = 1,
    pattern_count: int | None = None,
) -> str:
    attempt_block = (
        f"Attempt {attempt + 1} of {max_attempts} for the current unconfirmed pattern. "
        "This first attempt uses unhighlighted residual diagnostics. First estimate how many distinct intervention "
        "patterns/windows exist in this cluster, then return the next unconfirmed pattern only. If you return a "
        "window, the graph will render your proposed range highlighted and send it back for confirmation or revision.\n"
        if attempt == 0
        else (
            f"Attempt {attempt + 1} of {max_attempts} for the current unconfirmed pattern. "
            "You are now seeing both the original residual diagnostics and a highlighted diagnostic image based on "
            "your previous proposal. If the highlighted window range is tight, return suggested_action set to "
            '"accept". If it starts too early, ends too late, misses part of the pattern, or includes quiet common-flow '
            'sections, return the adjusted current window with suggested_action set to "revise".\n'
        )
    )
    previous_block = f"Previous window proposal JSON: {previous_review_json}\n\n" if previous_review_json else ""
    accepted_block = f"Already accepted windows JSON: {accepted_windows_json}\n\n" if accepted_windows_json else ""
    count_block = (
        f"Previously estimated pattern_count: {pattern_count}. Return the same count unless the evidence clearly shows it was wrong.\n"
        if pattern_count is not None
        else "Estimate pattern_count before choosing boundaries. pattern_count is the total number of distinct intervention windows in this cluster.\n"
    )
    return (
        "You are reviewing one VLM-PPE trajectory cluster.\n"
        "Use only the provided ADS-B trajectory plots, medoid overlay, residual-energy curve, heading-dispersion curve, "
        "and station-index labels. Do not infer from AIP charts.\n"
        "Your task is to count distinct intervention patterns, then propose and classify tight intervention window boundaries.\n"
        "Use station indices shown on the diagnostic plot. Return no windows if the cluster has no meaningful intervention.\n\n"
        f"{attempt_block}"
        f"{previous_block}"
        f"{accepted_block}"
        f"{count_block}"
        f"Current unconfirmed pattern number: {pattern_index}.\n\n"
        "Class definitions:\n"
        "- no_stretch: no meaningful deviation program; max cross-track deviation and added path length appear small.\n"
        "- dogleg: one outward vector-like leg followed by one closure/rejoin leg.\n"
        "- trombone: two vector-like legs before closure/rejoin.\n"
        "- PMS: point-merge-like sequencing leg followed by direct-to common merge point.\n"
        "- other: holding, looping, direct shortcut, unclear, too rare, or too complex for v1.\n\n"
        f"Cluster ID: {cluster_id}.\n"
        "Window boundary rules:\n"
        "- start_station_index and end_station_index must be integers from the plotted station axis.\n"
        "- end_station_index must be greater than or equal to start_station_index.\n"
        "- The window must be tight: start exactly where the pattern begins and end exactly where it rejoins common flow.\n"
        "- Pick the smallest station interval that covers the visually meaningful maneuver, no longer and no shorter.\n"
        "- Do not pad the window to include quiet common-flow sections unless they are part of the pattern.\n"
        "- If the highlighted range is not tight enough, use this attempt to adjust the station range.\n"
        "- Do not duplicate already accepted windows.\n"
        "- Return only the current unconfirmed pattern in windows; accepted windows are provided only as context.\n"
        "- If all patterns have already been identified, return windows as [] and all_patterns_identified as true.\n\n"
        "Return strict JSON matching this example shape. Text-list fields must always be JSON arrays, "
        "even when there is only one item:\n"
        "{\n"
        f'  "cluster_id": {cluster_id},\n'
        '  "pattern_count": 2,\n'
        '  "windows": [\n'
        "    {\n"
        '      "window_id": "C2_W1",\n'
        '      "start_station_index": 32,\n'
        '      "end_station_index": 61,\n'
        '      "class_name": "dogleg",\n'
        '      "confidence": 0.82,\n'
        '      "visual_reason": "One clear outward vector and one closure leg back to common flow."\n'
        "    }\n"
        "  ],\n"
        '  "outlier_notes": ["Two tracks appear visually different from the main window pattern."],\n'
        '  "all_patterns_identified": false,\n'
        '  "suggested_action": "accept"\n'
        "}\n"
    )
