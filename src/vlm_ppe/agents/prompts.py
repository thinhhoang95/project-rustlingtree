from __future__ import annotations

import json

from vlm_ppe.schemas import ClusterMedoid, CommunityMetric


def cluster_review_prompt(
    metrics: list[CommunityMetric],
    available_thresholds_nm: list[float],
    attempt: int,
    max_retries: int,
) -> str:
    metrics_payload = [metric.model_dump() for metric in metrics]
    return (
        "You are the visual reviewer and orchestrator for VLM-PPE.\n"
        "Use only the provided ADS-B trajectory plots and numeric metrics. Do not infer from AIP charts.\n"
        "Choose the most plausible community-detection distance threshold in nautical miles. The threshold builds a graph "
        "where tracks are connected when their RMS trajectory distance is less than or equal to that value; connected "
        "components become practical path communities. Prefer larger thresholds when differences look like noise, and lower "
        "thresholds only when common geometry is visually distinct. Do not chase fine-grained one-off track differences; "
        "capture the main repeated path patterns with enough communities, but no more detail than the evidence supports. "
        "You may request a bounded retry if the evidence is insufficient or the threshold range should be expanded.\n\n"
        f"Attempt: {attempt + 1}. Maximum retries: {max_retries}.\n"
        f"Available threshold_nm values: {available_thresholds_nm}.\n"
        f"Community-detection metrics JSON: {json.dumps(metrics_payload, separators=(',', ':'))}\n\n"
        "Return strict JSON matching this example shape. Text-list fields must always be JSON arrays, "
        "even when there is only one item:\n"
        "{\n"
        '  "chosen_threshold_nm": 1.75,\n'
        '  "confidence": 0.82,\n'
        '  "rationale": ["A 1.75 NM threshold separates visually distinct trajectory communities."],\n'
        '  "rejected_alternatives": ["A 1.25 NM threshold creates tiny communities that look like noise."],\n'
        '  "clusters_to_recheck": [1],\n'
        '  "retry_requested": false,\n'
        '  "requested_threshold_max_nm": null,\n'
        '  "suggested_action": "accept"\n'
        "}\n"
    )


def subcluster_review_prompt(
    metrics: list[CommunityMetric],
    available_thresholds_nm: list[float],
    *,
    root_cluster_id: int,
    lineage: list[int],
    depth: int,
    n_tracks: int,
    min_tracks: int,
) -> str:
    metrics_payload = [metric.model_dump() for metric in metrics]
    lineage_text = " -> ".join(str(item) for item in lineage)
    return (
        "You are inspecting one accepted VLM-PPE trajectory cluster for possible subclusters.\n"
        "Use only the provided ADS-B trajectory plots and numeric metrics. Do not infer from AIP charts.\n"
        "Choose the local community-detection distance threshold for this cluster. A threshold that yields one community "
        "means the cluster is already one practical path pattern; a threshold that yields multiple communities means there "
        "are visually distinct, repeated subcluster patterns worth splitting.\n"
        "Require clean separation: every trajectory subcluster you keep as an operational pattern must have visibly "
        "distinct geometry from the others. Do not split one trajectory family on spacing, density, or minor noisy variation.\n"
        "Outliers can hide real repeated geometry. When comparing threshold values, first identify tiny, scattered, or one-off "
        "clusters as outlier/noise quarantine candidates, then judge whether the remaining non-outlier clusters expose "
        "cleanly distinct repeated trajectory families. A lower threshold is appropriate when it both quarantines outliers/noise "
        "and separates repeated geometries that higher thresholds leave merged. Do not choose a lower threshold only because "
        "it improves a metric, and do not reject a lower threshold only because it contains tiny outlier/noise communities.\n"
        "This is a single local split pass. If your chosen threshold yields multiple communities, the resulting child clusters become final leaves; "
        "the workflow will not keep looping until another VLM review green-lights them.\n\n"
        f"Root/global cluster ID: {root_cluster_id}.\n"
        f"Current lineage: {lineage_text}.\n"
        f"Current subcluster depth: {depth}. Maximum reviewed depth: 0; split children stop at depth 1.\n"
        f"Track count in this cluster: {n_tracks}. Minimum tracks for local review: {min_tracks}.\n"
        f"Available local threshold_nm values: {available_thresholds_nm}.\n"
        f"Local community-detection metrics JSON: {json.dumps(metrics_payload, separators=(',', ':'))}\n\n"
        "Return strict JSON matching this example shape. Text-list fields must always be JSON arrays, "
        "even when there is only one item. `chosen_threshold_nm` is the local threshold for this cluster. Set `clusters_to_recheck` to [] "
        "because this pass does not recursively review child clusters. "
        '`suggested_action` must be exactly one of "accept", "retry", or "human_review"; do not return "split". '
        "Choosing a threshold that yields multiple communities is how you request a split, while suggested_action remains "
        "\"accept\" when the local threshold choice is usable:\n"
        "{\n"
        '  "chosen_threshold_nm": 0.85,\n'
        '  "confidence": 0.78,\n'
        '  "rationale": ["A 0.85 NM threshold cleanly separates two repeated trajectory families."],\n'
        '  "rejected_alternatives": ["A 0.40 NM threshold only isolates minor spacing noise."],\n'
        '  "clusters_to_recheck": [],\n'
        '  "retry_requested": false,\n'
        '  "requested_threshold_max_nm": null,\n'
        '  "suggested_action": "accept"\n'
        "}\n"
    )


def window_cluster_selection_prompt(medoids: list[ClusterMedoid]) -> str:
    medoid_payload = [
        {
            "cluster_id": medoid.cluster_id,
            "n_tracks": medoid.n_tracks,
            "mean_distance_nm": medoid.mean_distance_nm,
            "max_distance_nm": medoid.max_distance_nm,
        }
        for medoid in sorted(medoids, key=lambda item: item.cluster_id)
    ]
    return (
        "You are triaging VLM-PPE trajectory clusters before residual-window classification.\n"
        "Use only the provided medoid/residual diagnostic images and cluster summary JSON. Do not infer from AIP charts.\n"
        "Select only clusters that should proceed to detailed intervention-window analysis.\n\n"
        "Select a cluster when it represents a coherent repeated trajectory family with enough supporting tracks to make "
        "a meaningful intervention-window judgment. Skip clusters that appear to be outlier quarantines, one-off tracks, "
        "tiny scattered groups, or visually incoherent mixtures where window classification would mostly describe noise. "
        "Small clusters may still be selected if the evidence shows a repeated, coherent operational pattern; large clusters "
        "may be skipped if they are visibly noncoherent.\n\n"
        f"Cluster summaries JSON: {json.dumps(medoid_payload, separators=(',', ':'))}\n\n"
        "Return strict JSON matching this example shape. Text-list fields must always be JSON arrays, "
        "even when there is only one item. `selected_cluster_ids` controls which clusters receive window-review prompts:\n"
        "{\n"
        '  "selected_cluster_ids": [0, 2, 4],\n'
        '  "rationale": ["Selected clusters have coherent repeated geometry and enough tracks for window analysis."],\n'
        '  "skipped_clusters": [\n'
        '    {"cluster_id": 1, "reason": "Tiny scattered outlier/noise group, not a stable trajectory family."}\n'
        "  ],\n"
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
