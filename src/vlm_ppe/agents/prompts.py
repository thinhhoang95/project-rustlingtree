from __future__ import annotations

import json

from vlm_ppe.schemas import KMetric


def cluster_review_prompt(metrics: list[KMetric], available_k: list[int], attempt: int, max_retries: int) -> str:
    metrics_payload = [metric.model_dump() for metric in metrics]
    return (
        "You are the visual reviewer and orchestrator for VLM-PPE.\n"
        "Use only the provided ADS-B trajectory plots and numeric metrics. Do not infer from AIP charts.\n"
        "Choose the most plausible number of practical path clusters. Prefer fewer clusters when differences look like noise, "
        "and more clusters only when common geometry is visually distinct. Do not chase fine-grained one-off track differences; "
        "capture the main repeated path patterns with enough clusters, but no more detail than the evidence supports.\n\n"
        f"Attempt: {attempt + 1}. Maximum retries: {max_retries}.\n"
        f"Available K values: {available_k}.\n"
        f"K metrics JSON: {json.dumps(metrics_payload, separators=(',', ':'))}\n\n"
        "Decision fields:\n"
        "- chosen_k must be one of the Available K values above.\n"
        "- suggested_action is your decision and must be exactly one of \"accept\", \"retry\", or \"human_review\".\n"
        "- Choose \"retry\" only when the evidence is insufficient or K should be expanded beyond the available values. "
        "When you choose \"retry\", set retry_requested to true; otherwise keep retry_requested false. The two must agree.\n"
        "- requested_k_max is the new maximum K you want explored on the retry. Set it only when retrying; leave it null "
        "otherwise. The pipeline clamps it to the configured expansion limit, so a retry always widens K by at least one.\n"
        "- clusters_to_recheck is an optional advisory list of chosen-K cluster indices you found visually borderline; "
        "leave it [] when nothing stands out.\n\n"
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


def subcluster_review_prompt(
    *,
    root_cluster_id: int,
    lineage: list[int],
    depth: int,
    n_tracks: int,
    min_tracks: int,
    max_subclusters: int,
    coordinate_bounds: dict[str, float],
) -> str:
    lineage_text = " -> ".join(str(item) for item in lineage)
    return (
        "You are inspecting one accepted VLM-PPE trajectory cluster for hidden practical path subclusters.\n"
        "Use only the provided ADS-B trajectory plot. Do not infer from AIP charts, route names, airport procedures, "
        "or density metrics. Your task is manual visual separation of the trajectory shapes in this plot.\n\n"
        "First decide N, the number of practical subclusters in this local cluster. N=1 means the cluster is already "
        "one practical path pattern and you must return no polygons. Choose N>1 only when the plot shows discrete, "
        "visually distinct, repeated path families with a clear gap or materially different maneuver geometry. "
        "If paths smoothly vary from one trajectory to the next, form a continuous fan, or differ only by gradual "
        "offsets, choose N=1.\n\n"
        "For N>1, return one capture polygon for each proposed subcluster. Use the x (NM) and y (NM) axes shown in "
        "the plot. Each polygon is a sequence of [x_nm, y_nm] vertices; the pipeline will compute the convex hull of "
        "those points. A flight path is assigned to a subcluster when any segment of that path crosses, touches, or "
        "runs inside the convex polygon. Assignment follows subcluster_id order; if a path is captured by more than "
        "one polygon, the first matching subcluster wins.\n\n"
        "Draw polygons as discriminating gates around geometry that only that family crosses, not as broad envelopes "
        "around entire routes. Do not split one trajectory family on spacing, sample density, smooth variation, "
        "or minor noisy variation. Small one-off or scattered tracks may remain uncaptured by your polygons.\n\n"
        f"Root/global cluster ID: {root_cluster_id}.\n"
        f"Current lineage: {lineage_text}.\n"
        f"Subcluster depth of this cluster: {depth}. This pipeline reviews depth 0 only; any subcluster you propose "
        "becomes a terminal leaf and is not reviewed again.\n"
        f"Track count in this cluster: {n_tracks}. Minimum tracks for local review: {min_tracks}.\n"
        f"Maximum allowed local subclusters: {max_subclusters}.\n"
        f"Coordinate bounds JSON: {json.dumps(coordinate_bounds, separators=(',', ':'))}\n\n"
        "Decision fields:\n"
        '- suggested_action must be exactly one of "accept" or "human_review". Set it to "human_review" only when the '
        "entire cluster is too ambiguous to split; that skips the split and sends the whole cluster to a human.\n"
        "- uncaptured_tracks_policy governs only the leftover tracks when you do split (N>1). "
        '"keep_as_residual" keeps the uncaptured tracks as a residual leaf; "human_review" sends just those leftover '
        "tracks to a human while still accepting your polygon subclusters. It is ignored when N=1.\n\n"
        "Return strict JSON matching this example shape. Text-list fields must always be JSON arrays, "
        "even when there is only one item. If N=1, set subcluster_count to 1 and subclusters to []:\n"
        "{\n"
        '  "subcluster_count": 3,\n'
        '  "confidence": 0.78,\n'
        '  "rationale": ["Three discrete path families have clear visual separation and different turn geometry."],\n'
        '  "subclusters": [\n'
        "    {\n"
        '      "subcluster_id": 1,\n'
        '      "label": "Subcluster 1",\n'
        '      "polygon": [[-8.0, 2.5], [-6.8, 2.4], [-6.8, 3.3], [-8.0, 3.4]],\n'
        '      "rationale": "Captures a visually separated northern path family."\n'
        "    },\n"
        "    {\n"
        '      "subcluster_id": 2,\n'
        '      "label": "Subcluster 2",\n'
        '      "polygon": [[-8.0, -1.8], [-6.8, -1.9], [-6.8, -0.9], [-8.0, -0.8]],\n'
        '      "rationale": "Captures a visually separated southern path family."\n'
        "    },\n"
        "    {\n"
        '      "subcluster_id": 3,\n'
        '      "label": "Subcluster 3",\n'
        '      "polygon": [[-5.5, 0.2], [-4.5, 0.2], [-4.5, 1.1], [-5.5, 1.1]],\n'
        '      "rationale": "Captures a visually separated central path family."\n'
        "    }\n"
        "  ],\n"
        '  "uncaptured_tracks_policy": "keep_as_residual",\n'
        '  "suggested_action": "accept"\n'
        "}\n"
    )


def window_pattern_count_prompt(cluster_id: int, *, max_patterns: int) -> str:
    return (
        "You are reviewing one VLM-PPE trajectory cluster before detailed window boundary selection.\n"
        "Use only the provided ADS-B trajectory plots, medoid overlay, residual-energy curve, heading-dispersion curve, "
        "and station-index labels. Do not infer from AIP charts.\n"
        "Your task in this request is only to count how many distinct trajectory-variation windows are present in this "
        "cluster and explain why. Do not propose station boundaries, window IDs, or class labels in this response.\n\n"
        f"Cluster ID: {cluster_id}.\n"
        f"Maximum count allowed by configuration: {max_patterns}.\n\n"
        "Count rules:\n"
        "- pattern_count is the total number of distinct trajectory-variation windows in this cluster.\n"
        f"- pattern_count must not exceed {max_patterns}; the pipeline rejects a higher count.\n"
        "- Count only meaningful local variation regions in the diagnostic region of interest.\n"
        "- Count doglegs, trombones, point-merge-like structures, loops, shortcuts, and other coherent maneuver regions only "
        "when they are localized ROI variations, not routine entry/exit geometry.\n"
        "- Count a trombone when the local geometry is paperclip-like: an outbound leg, rounded/base turn, and inbound "
        "return leg that is roughly parallel or anti-parallel to the outbound leg.\n"
        "- Do not dismiss a localized paperclip-like or foldback variation as routine turn-radius variation when it is "
        "visibly repeated by multiple tracks and supported by residual or heading-dispersion diagnostics.\n"
        "- Use 0 only when the cluster has no meaningful repeated variation window.\n"
        "- Do not count fanning, spreading, converging, or merging patterns at the far-upstream or far-downstream "
        "trajectory extremities, especially far from the airport/terminal region; those are outside the region of interest.\n"
        "- Do not count tiny one-off outliers, isolated noisy tracks, or minor jitter inside otherwise common flow.\n"
        "- Count only patterns supported by the provided cluster diagnostics.\n"
        "- If the evidence is too ambiguous for a defensible count, set suggested_action to human_review.\n\n"
        "Return strict JSON matching this example shape. Text-list fields must always be JSON arrays, "
        "even when there is only one item:\n"
        "{\n"
        f'  "cluster_id": {cluster_id},\n'
        '  "pattern_count": 2,\n'
        '  "confidence": 0.81,\n'
        '  "rationale": ["Two separated residual-energy peaks align with two visually meaningful variation regions."],\n'
        '  "outlier_notes": ["One track has a weak late variation but does not form a repeated window."],\n'
        '  "suggested_action": "accept"\n'
        "}\n"
    )


def window_review_prompt(
    cluster_id: int,
    *,
    attempt: int = 0,
    max_attempts: int = 3,
    pattern_count: int,
    previous_review_json: str | None = None,
    accepted_windows_json: str | None = None,
    pattern_index: int = 1,
) -> str:
    first_attempt_task = (
        f"The prior count-only review fixed pattern_count at {pattern_count}. Return the next unconfirmed pattern only. "
        "If you return a window, the graph will render your proposed range highlighted and send it back for confirmation "
        "or revision.\n"
    )
    attempt_block = (
        f"Attempt {attempt + 1} of {max_attempts} for the current unconfirmed pattern. "
        f"This first attempt uses unhighlighted residual diagnostics. {first_attempt_task}"
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
    count_block = f"Fixed pattern_count from prior count-only review: {pattern_count}. Do not revise pattern_count in this request.\n"
    return (
        "You are reviewing one VLM-PPE trajectory cluster.\n"
        "Use only the provided ADS-B trajectory plots, medoid overlay, residual-energy curve, heading-dispersion curve, "
        "and station-index labels. Do not infer from AIP charts.\n"
        "Your task is to propose and classify tight trajectory-variation window boundaries for the current unconfirmed pattern.\n"
        "A window is a localized station interval inside the diagnostic region of interest where the cluster shows "
        "meaningful repeated path variation.\n"
        "Do not propose windows for fanning, spreading, converging, or merging patterns at the far-upstream or "
        "far-downstream trajectory extremities, especially far from the airport/terminal region; those extremity patterns "
        "are outside the region of interest.\n"
        "Use station indices shown on the diagnostic plot.\n\n"
        f"{attempt_block}"
        f"{previous_block}"
        f"{accepted_block}"
        f"{count_block}"
        f"Current unconfirmed pattern number: {pattern_index}.\n\n"
        "Class definitions:\n"
        "- no_stretch: rarely applies here, because the prior count-only review already confirmed at least one pattern. "
        "If the current pattern shows no real repeated variation on close inspection, do not emit a no_stretch window; "
        'instead set suggested_action to "human_review" (see decision fields below).\n'
        "- dogleg: a simple angled detour with one offset/outbound leg and one closure/rejoin leg, usually a bent V or open "
        "triangle. It does not contain a sustained outbound-and-inbound pair of roughly parallel legs.\n"
        "- trombone: a paperclip-like sequencing extension: outbound leg, rounded/base turn, and inbound return leg that is "
        "roughly parallel or anti-parallel to the outbound leg before closure/rejoin. Use trombone only when the geometry "
        "visibly folds back on itself like a U-turn, racetrack, or elongated paperclip; otherwise prefer dogleg or other.\n"
        "- PMS: point-merge-like sequencing structure inside the region of interest; do not use PMS for ordinary far-upstream "
        "or far-downstream merging at trajectory extremities.\n"
        "- other: holding, looping, direct shortcut, unclear, or too complex for v1.\n\n"
        f"Cluster ID: {cluster_id}.\n"
        "Window boundary rules:\n"
        "- start_station_index and end_station_index must be integers from the plotted station axis.\n"
        "- end_station_index must be greater than or equal to start_station_index.\n"
        "- The window must be tight: start where the visible variation begins and end where the variation resolves.\n"
        "- Pick the smallest station interval that covers the visually meaningful variation, no longer and no shorter.\n"
        "- Do not pad the window to include quiet common-flow sections before or after the variation.\n"
        "- Do not place a window on far-upstream/far-downstream fan-in/fan-out or routine merge/rejoin sections outside the ROI.\n"
        "- If the highlighted range is not tight enough, use this attempt to adjust the station range.\n"
        "- Do not duplicate already accepted windows.\n"
        "- Return only the current unconfirmed pattern in windows; accepted windows are provided only as context.\n"
        "- Do not use this request to change the pattern count; focus on the current pattern's boundaries and class.\n\n"
        "Decision fields:\n"
        "- windows must contain exactly one entry for the current pattern. The pipeline requires a window for each "
        "counted pattern, so do not return an empty windows list to skip it.\n"
        '- If you cannot identify a defensible window for the current pattern, set suggested_action to "human_review"; '
        "that is the only way to return without a window.\n"
        '- suggested_action must be exactly one of "accept", "revise", or "human_review".\n'
        "- Set all_patterns_identified to true on the window you return when it is the last real pattern; the pipeline "
        "then stops early instead of asking for the remaining counted patterns.\n\n"
        "Return strict JSON matching this example shape. Text-list fields must always be JSON arrays, "
        "even when there is only one item:\n"
        "{\n"
        f'  "cluster_id": {cluster_id},\n'
        '  "windows": [\n'
        "    {\n"
        '      "window_id": "C2_W1",\n'
        '      "start_station_index": 32,\n'
        '      "end_station_index": 61,\n'
        '      "class_name": "dogleg",\n'
        '      "confidence": 0.82,\n'
        '      "visual_reason": "One clear offset leg and one closure leg back to common flow, without a paperclip-like return leg."\n'
        "    }\n"
        "  ],\n"
        '  "outlier_notes": ["Two tracks show weak variation but do not form another repeated window."],\n'
        '  "all_patterns_identified": false,\n'
        '  "suggested_action": "accept"\n'
        "}\n"
    )
