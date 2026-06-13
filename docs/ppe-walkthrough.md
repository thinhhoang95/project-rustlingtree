# VLM-PPE Walkthrough

This document explains the Practical Procedure Extraction framework implemented
under `src/vlm_ppe`. It is written as a self-contained guide for running,
inspecting, and debugging the pipeline through residual-window classification.

The current implementation covers PPE ingestion through PPE plan Sections 7 and
8:

1. ingest ADS-B tracks,
2. project them into a common local coordinate frame,
3. resample each track by arc length,
4. build shape features,
5. run candidate KMeans clusterings,
6. render VLM evidence packs,
7. ask the VLM to choose or retry the cluster count,
8. validate the VLM decision,
9. compute one medoid trajectory per accepted cluster,
10. compute residual-energy and heading-dispersion profiles,
11. render residual evidence with station guides,
12. ask the VLM to propose windows and classify them into the v1 intervention classes,
13. render each non-empty proposal highlighted and let the VLM confirm or revise it,
14. validate and enrich the structured proposals,
15. write audit artifacts and a final state index.

Intervention fitting and procedure-program export are not implemented yet. Those
correspond to Section 9 and beyond in the PPE plan.

## Core Idea

VLM-PPE is intentionally agent-led but geometry-safe.

The VLM is the visible orchestrator:

- it receives rendered trajectory evidence,
- it sees metrics and plots,
- it chooses the practical cluster count,
- it can request bounded retries,
- it proposes residual-window boundaries and classifications,
- it explains its decision through a structured rationale.

The deterministic tools remain authoritative for all numeric and geometric
outputs:

- ADS-B loading,
- projection,
- resampling,
- KMeans labels,
- metric computation,
- artifact writing,
- medoid trajectory selection,
- residual-profile computation.

The important contract is:

```text
VLM reviews and chooses.
Tools compute and validate.
The VLM proposes window boundaries, and deterministic tools validate and enrich
those proposals before they become artifacts.
```

The implementation logs the model-visible prompt, the images prepared for model
review, and the model's structured response. It does not log hidden
chain-of-thought. The logged "deliberation" is the explicit `rationale` and full
schema-validated JSON returned by the model.

## Code Map

Primary files:

- `src/vlm_ppe/cli.py`: command-line entrypoint.
- `src/vlm_ppe/config.py`: YAML config loading and path resolution.
- `src/vlm_ppe/schemas.py`: Pydantic models for config, state, metrics, review,
  medoids, and audit events.
- `src/vlm_ppe/audit.py`: console and file audit logging.
- `src/vlm_ppe/agents/graph.py`: LangGraph state machine.
- `src/vlm_ppe/agents/tools.py`: deterministic tool wrappers used by graph
  nodes.
- `src/vlm_ppe/agents/vlm_client.py`: Gemini-backed VLM client.
- `src/vlm_ppe/agents/prompts.py`: cluster-review and window-review prompt
  builders.
- `src/vlm_ppe/io/adsb_loader.py`: manifest, catalog, and compressed ADS-B
  ingestion.
- `src/vlm_ppe/io/parquet_store.py`: Parquet read/write helpers.
- `src/vlm_ppe/geo/projection.py`: common local projection.
- `src/vlm_ppe/geo/polyline.py`: polyline length helpers.
- `src/vlm_ppe/geo/resample.py`: arc-length resampling.
- `src/vlm_ppe/processing.py`: track-frame resampling.
- `src/vlm_ppe/clustering/features.py`: shape feature construction.
- `src/vlm_ppe/clustering/kmeans_runner.py`: candidate KMeans runs.
- `src/vlm_ppe/clustering/medoid.py`: cluster medoid extraction.
- `src/vlm_ppe/clustering/residual_windows.py`: residual-energy and
  heading-dispersion profile computation.
- `src/vlm_ppe/diagnostics/plots_clustering.py`: cluster evidence plots.
- `src/vlm_ppe/diagnostics/plots_medoids.py`: medoid summary plot.
- `src/vlm_ppe/diagnostics/plots_residuals.py`: residual evidence plots for
  VLM window proposal.
- `src/vlm_ppe/diagnostics/report.py`: Markdown medoid report.

Default config:

- `configs/ppe_kdfw_arrivals.yaml`

Tests:

- `tests/test_ppe_geo.py`
- `tests/test_ppe_clustering.py`
- `tests/test_ppe_graph.py`
- `tests/test_ppe_cli.py`

## Dependency And Runtime Setup

The package is installed through the repository's `pyproject.toml`.

Important dependencies:

- `langgraph`: agent graph runtime.
- `google-genai`: Gemini API client.
- `pyarrow`: Parquet support for pandas.
- `pyproj`: local projected coordinate frame.
- `pydantic>=2`: schema validation.
- `pyyaml`: YAML config loading.
- `scikit-learn`: KMeans and silhouette metrics.
- `matplotlib`: diagnostic image rendering.

Install the project in editable mode:

```sh
python -m pip install -e .
```

For VLM-led runs, export an OpenRouter API key:

```sh
export OPENROUTER_API_KEY="..."
```

`--chosen-k` only bypasses the cluster-count review. Window proposal still
requires the configured VLM unless tests inject a fake review client.

## Running The Pipeline

Normal VLM-led run:

```sh
vlm-ppe run-through-medoid \
  --config configs/ppe_kdfw_arrivals.yaml
```

Offline/manual cluster-count run:

```sh
vlm-ppe run-through-medoid \
  --config configs/ppe_kdfw_arrivals.yaml \
  --chosen-k 2 \
  --run-id smoke-offline
```

Increase console/file logging verbosity:

```sh
vlm-ppe run-through-medoid \
  --config configs/ppe_kdfw_arrivals.yaml \
  --log-level DEBUG
```

Disable console audit logging while still writing audit files:

```sh
vlm-ppe run-through-medoid \
  --config configs/ppe_kdfw_arrivals.yaml \
  --quiet
```

The command prints a small JSON object at the end:

```json
{
  "status": "complete",
  "run_dir": ".../data/artifacts/ppe/2026-04-01/runs/<run_id>",
  "state_path": ".../state.json"
}
```

`run_dir` is the root for all run artifacts.

## Configuration

The default config is:

```yaml
dataset_id: "2026-04-01"
operation: "arrival"
runway: null
manifest_path: "data_manifest.json"
output_root: "data/artifacts/ppe/2026-04-01"
n_resample: 100
k_min: 1
k_max: 8
kmeans_n_init: 50
kmeans_random_state: 17
max_retries: 2
max_k_expansion: 12
vlm_model: "gemini-3.5-flash"
window_review_max_attempts: 3
track_filter_center_lat: 32.897102378968
track_filter_center_lon: -97.036547781746
track_filter_radius_nm: 60.0
log_level: "INFO"
log_to_console: true
```

Meaning:

- `dataset_id`: manifest key to read from `data_manifest.json`.
- `operation`: currently `arrival` or `departure`; default PPE run targets
  arrivals.
- `runway`: optional runway filter. `null` means all selected-operation tracks.
- `manifest_path`: resource manifest.
- `output_root`: directory where run folders are written.
- `n_resample`: number of equal arc-length samples per track.
- `k_min`, `k_max`: candidate KMeans cluster-count range.
- `kmeans_n_init`: number of KMeans initializations.
- `kmeans_random_state`: deterministic clustering seed.
- `max_retries`: maximum VLM-requested reclustering attempts.
- `max_k_expansion`: upper bound if the VLM requests a larger `Kmax`.
- `vlm_model`: Gemini model name.
- `window_review_max_attempts`: maximum VLM attempts per cluster for proposing,
  viewing the highlighted proposal, and revising residual-window bounds.
- `track_filter_center_lat`, `track_filter_center_lon`: optional center of the
  circular trajectory window. The default KDFW config uses the airport center.
- `track_filter_radius_nm`: optional circular trajectory window radius in
  nautical miles. The default KDFW config keeps only the ADS-B trajectory
  portion within 60 NM of the airport center.
- `log_level`: audit logger threshold.
- `log_to_console`: whether audit messages should also stream to stdout.

`manifest_path` and `output_root` may be relative. Relative paths are resolved
against the config file directory first, then against the repository root when
the config lives in `configs/`.

## Input Data Contract

The default manifest entry must provide:

```json
{
  "landings_and_departures": "data/adsb/catalogs/2026-04-01_landings_and_departures.csv",
  "adsb_compressed_trajectories": "data/adsb/compressed/adsb_compressed_flights.jsonl"
}
```

The catalog CSV is used for:

- `flight_id`,
- `operation`,
- optional `runway`,
- event coordinates,
- runway threshold coordinates.

The compressed ADS-B JSONL is used for the actual path geometry. Each line is
expected to include:

```json
{
  "flight_id": "...",
  "callsign": "...",
  "icao24": "...",
  "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
  "points": [
    [1775019659, 33.1198, -96.9128, 1562.1, 3]
  ]
}
```

Only `time`, `lat`, `lon`, and `geoaltitude_m` are currently consumed. The PPE
implementation uses 2D path geometry; altitude is carried in the normalized
track table but not used for clustering or medoid extraction.

## LangGraph Flow

The graph is built in `src/vlm_ppe/agents/graph.py`.

The current node sequence is:

```text
START
  -> ingest_tracks
  -> resample_tracks
  -> build_features
  -> run_candidate_clustering
  -> render_evidence_pack
  -> vlm_review_clusters
  -> validate_review
  -> retry_or_accept
       -> run_candidate_clustering, if retry requested and allowed
       -> compute_cluster_medoids, otherwise
  -> render_medoid_report
  -> compute_residual_profiles
  -> render_window_diagnostics
  -> vlm_classify_windows
  -> validate_window_reviews
  -> export_state
  -> END
```

Each node receives the shared graph state as a dictionary and returns an updated
state dictionary. The state is schema-compatible with `PPEState`.

Key state fields include:

- `run_id`
- `run_dir`
- `config`
- `retry_count`
- `k_max_current`
- `tracks_path`
- `resampled_tracks_path`
- `features_path`
- `k_metrics_path`
- `clustering_dir`
- `evidence_images`
- `vlm_reviews`
- `chosen_k`
- `cluster_assignments_path`
- `medoids_path`
- `medoid_summary_path`
- `medoid_report_path`
- `residual_profiles_path`
- `intervention_windows_path`
- `intervention_windows`
- `window_evidence_images`
- `window_reviews`
- `window_review_paths`
- `audit_log_path`
- `graph_events_path`
- `vlm_interactions_path`

The final `state.json` is the compact index of a run.

## Stage Details

### Stage 0: Track Ingestion

Implemented by:

- `ingest_tracks_tool()` in `agents/tools.py`
- `ingest_adsb_tracks()` in `io/adsb_loader.py`

Steps:

1. Load `data_manifest.json`.
2. Resolve the configured dataset entry.
3. Read the arrival/departure catalog.
4. Filter by `operation` and optional `runway`.
5. Load only matching `flight_id`s from compressed ADS-B JSONL.
6. Merge catalog metadata into the track points.
7. Select a projection origin. If a trajectory filter center is configured,
   that center is used; otherwise runway threshold coordinates are used, falling
   back to event coordinates.
8. Project all selected tracks into one common local coordinate frame.
9. If `track_filter_radius_nm` is configured, clip each projected track to the
   circular window and insert interpolated boundary points where segments cross
   the window edge.
10. Remove degenerate tracks.
11. Write normalized track tables.

Outputs:

- `processed/tracks.parquet`
- `processed/track_index.parquet`

The common coordinate frame is important. Clustering would be invalid if each
track were projected into its own local frame.

### Stage 1: Arc-Length Resampling

Implemented by:

- `resample_tracks_tool()` in `agents/tools.py`
- `resample_track_frame()` in `processing.py`
- `arc_length_resample_points()` in `geo/resample.py`

Each track is converted into exactly `n_resample` points at equal arc-length
stations from start to end.

The resampled table stores:

- `flight_id`
- `station_index`
- `s_fraction`
- `s_nm`
- `track_length_nm`
- `x_nm`
- `y_nm`

Output:

- `processed/resampled_tracks.parquet`

### Stage 2: Shape Features

Implemented by:

- `build_features_tool()` in `agents/tools.py`
- `build_shape_features()` in `clustering/features.py`

For each track, the resampled `(x_nm, y_nm)` points are flattened into:

```text
[x0, y0, x1, y1, ..., xR, yR]
```

The feature matrix is standardized column-wise:

```text
standardized = (raw - mean) / scale
```

Outputs:

- `processed/features.npz`
- `processed/feature_metadata.parquet`

The `.npz` file stores raw features, standardized features, track IDs, mean,
scale, and resample count.

### Stage 3: Candidate Clustering

Implemented by:

- `run_candidate_clustering_tool()` in `agents/tools.py`
- `run_candidate_kmeans()` in `clustering/kmeans_runner.py`

KMeans is run for every `K` from `k_min` through the current `k_max_current`.
For each candidate, the tool records:

- labels,
- inertia,
- silhouette score when valid,
- minimum cluster size,
- maximum cluster size,
- mean cluster size.

Outputs:

- `clustering/attempt_XX/k_metrics.csv`
- `clustering/attempt_XX/k_01_labels.csv`
- `clustering/attempt_XX/k_02_labels.csv`
- one label CSV per candidate K.

If the VLM requests a retry, a new attempt directory is created:

```text
clustering/attempt_01/
```

### Stage 4: Evidence Rendering

Implemented by:

- `render_evidence_pack_tool()` in `agents/tools.py`
- `render_cluster_panels()` in `diagnostics/plots_clustering.py`
- `render_metrics_chart()` in `diagnostics/plots_clustering.py`

For every candidate K, a cluster panel image is rendered. A metrics chart is
also rendered.

Outputs:

```text
evidence/clustering/attempt_XX/k_01/cluster_panel.png
evidence/clustering/attempt_XX/k_02/cluster_panel.png
...
evidence/clustering/attempt_XX/k_metrics.png
```

The graph state records these as `evidence_images`, each with:

- `kind`
- `path`
- `caption`

These are the images prepared for model review.

### Stage 5: VLM Review

Implemented by:

- `vlm_review_clusters` node in `agents/graph.py`
- `GeminiVLMClient` in `agents/vlm_client.py`
- `cluster_review_prompt()` in `agents/prompts.py`

The VLM receives:

- cluster panel images,
- metrics chart,
- compact metrics JSON,
- available K values,
- attempt number,
- retry budget.

The prompt asks the model to return strict JSON:

```json
{
  "chosen_k": 3,
  "confidence": 0.78,
  "rationale": [
    "K=2 merges two visually distinct downwind-extension groups."
  ],
  "rejected_alternatives": [
    "K=4 splits one coherent group without a meaningful new procedure."
  ],
  "clusters_to_recheck": [1],
  "retry_requested": false,
  "requested_k_max": null,
  "suggested_action": "accept"
}
```

The response is validated by `ClusterReview`.

Offline mode:

If `--chosen-k` is supplied, no Gemini call is made. The graph still creates a
synthetic `ClusterReview` and writes the same audit files.

### Stage 6: Review Validation And Retry

Implemented by:

- `validate_review_tool()` in `agents/tools.py`
- `retry_or_accept_tool()` in `agents/tools.py`

Validation checks that the VLM-chosen K is one of the candidate K values.

Retry behavior:

- retry if `retry_requested` is true or `suggested_action == "retry"`,
- only retry while `retry_count < max_retries`,
- expand `k_max_current` up to `max_k_expansion`,
- return to `run_candidate_clustering`.

The retry loop is deliberately bounded. There is no open-ended agent loop.

### Stage 7: Medoid Extraction

This corresponds to Section 6 of the PPE plan.

Implemented by:

- `compute_medoids_tool()` in `agents/tools.py`
- `compute_cluster_medoids()` in `clustering/medoid.py`

For each accepted cluster, the medoid is:

```text
argmin_i sum_j mean_r(||X_i[r] - X_j[r]||_2)
```

Where:

- `X_i` is one resampled track,
- `r` is the resampled station index,
- distances are in nautical miles.

The medoid is an actual observed track from the cluster, not an averaged path.
This avoids inventing a trajectory that no aircraft flew.

Outputs:

- `templates/chosen_cluster_assignments.csv`
- `templates/cluster_medoids.parquet`
- `templates/cluster_summary.json`
- `templates/cluster_medoids.png`

### Stage 8: Medoid Report

Implemented by:

- `render_medoid_report_tool()` in `agents/tools.py`
- `render_medoid_plot()` in `diagnostics/plots_medoids.py`
- `write_medoid_report()` in `diagnostics/report.py`

Output:

- `reports/medoid_report.md`

The report includes:

- VLM-selected K,
- VLM confidence,
- VLM rationale,
- rejected alternatives,
- medoid track ID per cluster,
- track counts,
- mean and maximum distance to the cluster medoid,
- medoid plot path.

### Stage 9: Residual-Profile Computation

This corresponds to Section 7 of the PPE plan.

Implemented by:

- `compute_residual_profiles_tool()` in `agents/tools.py`
- `compute_cluster_residual_profiles()` in `clustering/residual_windows.py`

For each accepted cluster, the tool computes:

- median squared residual energy against the cluster medoid,
- circular heading dispersion.

These profiles are evidence for the VLM; they are not thresholded into windows
by deterministic code.

Outputs:

- `residuals/residual_profiles.parquet`

### Stage 10: Window Diagnostics And Classification

This corresponds to Section 8 of the PPE plan.

Implemented by:

- `render_window_diagnostics_tool()` in `agents/tools.py`
- `render_residual_window_diagnostics()` in `diagnostics/plots_residuals.py`
- `vlm_classify_windows` node in `agents/graph.py`
- `window_review_prompt()` in `agents/prompts.py`

The VLM receives residual plots with station-index guides. The x-axis ticks are
dense enough to expose station choices without labeling every resampled point on
long templates. The VLM proposes `start_station_index`, `end_station_index`, and
a class label for each window:

```text
no_stretch, dogleg, trombone, PMS, other
```

For each non-empty first proposal, deterministic tooling renders the same
cluster diagnostics again with the proposed station span highlighted on the map,
residual-energy curve, and heading-dispersion curve. The next VLM attempt sees
both the unhighlighted and highlighted diagnostics and must either accept the
range or return a complete revised window list. This loop is bounded by
`window_review_max_attempts`; the default is three attempts per cluster.

`validate_window_reviews` checks that proposed station bounds exist for that
cluster, derives nautical-mile spans and peak metrics from the residual profile,
and writes `residuals/intervention_windows.parquet`.

## Audit And Logging

The audit system is implemented in `src/vlm_ppe/audit.py`.

It writes to console by default and always writes persistent files under
`run_dir`.

### Human-Readable Audit Log

Path:

```text
<run_dir>/audit.log
```

This is the easiest file to read first. It includes:

- graph node starts,
- graph node completions,
- important output paths,
- VLM attempt setup,
- image paths prepared for the model,
- model response summary,
- model rationale lines,
- errors if any node fails.

Example console/file lines:

```text
[vlm-ppe] run_candidate_clustering: completed | k_metrics_path='...' candidate_k_values=[1, 2, 3, 4, 5, 6, 7, 8]
[vlm-ppe] VLM image 02/09 kind=cluster_panel exists=True bytes=317330 path=.../k_02/cluster_panel.png caption=Cluster overlay panel for K=2
[vlm-ppe] VLM response attempt 00: chosen_k=2 confidence=1.000 action=accept retry=False response=.../attempt_00.json
```

### Structured Graph Events

Path:

```text
<run_dir>/graph_events.jsonl
```

Each line is one JSON object:

```json
{
  "timestamp_utc": "2026-06-11T11:52:57.434799Z",
  "node": "vlm_review_clusters",
  "status": "completed",
  "message": null,
  "payload": {
    "latest_vlm_review_path": ".../vlm_reviews/attempt_00.json",
    "status": "vlm_reviewed_clusters"
  }
}
```

Use this when you want to parse or compare runs programmatically.

### Structured VLM Interactions

Path:

```text
<run_dir>/vlm_interactions.jsonl
```

This records both request and response events.

The request event includes:

- attempt number,
- model name,
- offline or live mode,
- available K values,
- prompt path,
- image list,
- metrics.

The response event includes:

- response path,
- full validated `ClusterReview`.

### Per-Attempt VLM Files

Directory:

```text
<run_dir>/vlm_reviews/
```

Files:

```text
attempt_00_prompt.txt
attempt_00_request.json
attempt_00.json
attempt_01_prompt.txt
attempt_01_request.json
attempt_01.json
...
```

Meanings:

- `attempt_XX_prompt.txt`: exact text prompt built for the VLM.
- `attempt_XX_request.json`: request audit bundle, including prompt, metrics,
  and model-visible image metadata.
- `attempt_XX.json`: validated VLM response JSON.

### State File

Path:

```text
<run_dir>/state.json
```

This is the run index. It points to all important artifacts and records final
status.

Useful fields:

```json
{
  "status": "complete",
  "chosen_k": 2,
  "audit_log_path": ".../audit.log",
  "graph_events_path": ".../graph_events.jsonl",
  "vlm_interactions_path": ".../vlm_interactions.jsonl",
  "medoids_path": ".../templates/cluster_medoids.parquet",
  "medoid_report_path": ".../reports/medoid_report.md"
}
```

## Reading A Run Step By Step

Start with:

```sh
less <run_dir>/audit.log
```

Then inspect the model-visible evidence:

```sh
cat <run_dir>/vlm_reviews/attempt_00_prompt.txt
cat <run_dir>/vlm_reviews/attempt_00_request.json
open <run_dir>/evidence/clustering/attempt_00/k_02/cluster_panel.png
open <run_dir>/evidence/clustering/attempt_00/k_metrics.png
cat <run_dir>/vlm_reviews/attempt_00.json
```

Then inspect deterministic outputs:

```sh
python - <<'PY'
import pandas as pd
from pathlib import Path

run = Path("<run_dir>")
print(pd.read_csv(run / "clustering/attempt_00/k_metrics.csv"))
print(pd.read_parquet(run / "templates/cluster_medoids.parquet").head())
PY
```

Finally read:

```sh
cat <run_dir>/reports/medoid_report.md
```

## Debugging Common Problems

### Missing `OPENROUTER_API_KEY`

Symptom:

```text
OPENROUTER_API_KEY is required for VLM-led runs
```

Fix:

```sh
export OPENROUTER_API_KEY="..."
```

Or run in offline mode:

```sh
vlm-ppe run-through-medoid --config configs/ppe_kdfw_arrivals.yaml --chosen-k 2
```

### No Tracks Selected

Likely causes:

- wrong `dataset_id`,
- `operation` has no matching rows,
- `runway` filter is too narrow,
- manifest path points to the wrong catalog,
- compressed JSONL does not contain matching `flight_id`s.

Check:

```sh
cat data_manifest.json
head data/adsb/catalogs/2026-04-01_landings_and_departures.csv
```

### VLM Chooses An Unavailable K

The graph validates that `chosen_k` is in `candidate_k_values`.

If Gemini returns a value outside the available K range, the run fails at:

```text
validate_review
```

Inspect:

```sh
cat <run_dir>/vlm_reviews/attempt_00.json
cat <run_dir>/graph_events.jsonl
```

Possible fixes:

- improve the prompt,
- increase `k_max`,
- use `--chosen-k` for manual debugging,
- rerun with `--log-level DEBUG`.

### VLM Requests Retry Forever

It cannot retry forever.

The graph enforces:

- `max_retries`,
- `max_k_expansion`.

Once retry budget is exhausted, the current accepted choice is used.

Inspect:

```sh
grep retry <run_dir>/audit.log
find <run_dir>/clustering -maxdepth 2 -type f | sort
```

### Cluster Panels Look Blank Or Wrong

Check:

- `processed/resampled_tracks.parquet`,
- `processed/features.npz`,
- image byte sizes in `vlm_interactions.jsonl`,
- projection origin in `state.json`.

Useful command:

```sh
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

run = Path("<run_dir>")
state = json.loads((run / "state.json").read_text())
print(state["coordinate_system"])
tracks = pd.read_parquet(run / "processed/resampled_tracks.parquet")
print(tracks[["x_nm", "y_nm"]].describe())
print(tracks["flight_id"].nunique(), "tracks")
PY
```

### Medoid Looks Unrepresentative

The medoid is selected by pairwise average resampled distance. If it looks wrong:

- inspect cluster labels for the chosen K,
- check whether one cluster combines multiple visual flows,
- check whether the VLM should have chosen a different K,
- inspect outlier-sized clusters in `k_metrics.csv`,
- rerun with a different `--chosen-k` for comparison.

## Verification Commands

Focused PPE tests:

```sh
pytest tests/test_ppe_geo.py \
       tests/test_ppe_clustering.py \
       tests/test_ppe_graph.py \
       tests/test_ppe_cli.py -q
```

Static check:

```sh
python -m ruff check src/vlm_ppe \
  tests/test_ppe_geo.py \
  tests/test_ppe_clustering.py \
  tests/test_ppe_graph.py \
  tests/test_ppe_cli.py
```

Compile check:

```sh
find src/vlm_ppe tests -name '*.py' ! -name '._*' ! -name '.__*' -print0 \
  | xargs -0 python -m py_compile
```

Real-data offline smoke:

```sh
vlm-ppe run-through-medoid \
  --config configs/ppe_kdfw_arrivals.yaml \
  --chosen-k 2 \
  --run-id audit-smoke
```

## Current Limitations

- Only stages through residual-window classification are implemented.
- The VLM review is currently used for cluster-count choice, bounded retry
  orchestration, and residual-window classification.
- The final exported geometry is a medoid trajectory, not yet a procedure
  program with intervention tokens.
- No dogleg, trombone, or PMS fitting is implemented yet.
- Altitude is ingested but not used in clustering.
- The evidence packs contain cluster panels, a metrics chart, and residual
  diagnostics. Future versions should add fitted-reconstruction diagnostics.

## Expected Next Extensions

The natural next implementation step is Section 9:

1. fit dogleg and trombone candidates inside accepted windows,
2. add deterministic acceptance gates for fitted geometry,
3. render reconstruction diagnostics,
4. ask the VLM to review fitted reconstructions,
5. export reconstructable procedure-program JSON.

The audit machinery is already structured for that extension. New VLM review
nodes should use the same pattern:

```text
render evidence
log prompt and images
call VLM
validate structured JSON
write response
gate with deterministic checks
```
