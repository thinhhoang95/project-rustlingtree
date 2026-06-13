from __future__ import annotations

from pathlib import Path

from vlm_ppe.schemas import ClusterMedoid, ClusterReview


def write_medoid_report(
    *,
    output_dir: str | Path,
    run_id: str,
    review: ClusterReview,
    medoids: list[ClusterMedoid],
    medoid_plot_path: str,
) -> str:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# VLM-PPE Medoid Report: {run_id}",
        "",
        "## VLM Cluster Review",
        "",
        f"- Chosen CD threshold: {review.chosen_threshold_nm:.6f} NM",
        f"- Confidence: {review.confidence:.3f}",
        f"- Suggested action: {review.suggested_action}",
        "",
        "### Rationale",
        "",
    ]
    lines.extend(f"- {item}" for item in review.rationale)
    if review.rejected_alternatives:
        lines.extend(["", "### Rejected alternatives", ""])
        lines.extend(f"- {item}" for item in review.rejected_alternatives)
    lines.extend(["", "## Medoid Templates", ""])
    for medoid in medoids:
        lines.extend(
            [
                f"### Cluster {medoid.cluster_id}",
                "",
                f"- Medoid track: `{medoid.medoid_track_id}`",
                f"- Tracks: {medoid.n_tracks}",
                f"- Mean distance: {medoid.mean_distance_nm:.3f} NM",
                f"- Max distance: {medoid.max_distance_nm:.3f} NM",
                "",
            ]
        )
    lines.extend(["## Diagnostic Plot", "", f"![Cluster medoids]({medoid_plot_path})", ""])
    path = root / "medoid_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.as_posix()
