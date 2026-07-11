# VLM-PPE Planner Walkthrough

This document describes the current Practical Procedure Extraction (PPE)
implementation under `src/vlm_ppe`.

The previous content in this file described SIMAP's coupled descent NLP
planner. That is a different subsystem. The PPE implementation is a
LangGraph-driven ADS-B trajectory analysis pipeline: deterministic tools ingest,
project, resample, cluster, and score trajectories, while a VLM reviews rendered
evidence and returns schema-validated clustering and window decisions.

For the longer companion guide, see `docs/ppe-walkthrough.md`.

## 1. Runtime Entry Point

The CLI entry point is `vlm-ppe` in `src/vlm_ppe/cli.py`.

The implemented command is:

```sh
vlm-ppe run-through-medoid --config configs/ppe_kdfw_arrivals.yaml
```

Useful options:

- `--chosen-k <int>`: bypasses the live cluster-count review only.
- `--run-id <id>`: chooses the artifact run directory name.
- `--log-level DEBUG|INFO|WARNING|ERROR`: changes audit verbosity.
- `--quiet`: disables console audit logging while keeping audit files.

Live VLM calls use `OpenRouterVLMClient` and require:

```sh
export OPENROUTER_API_KEY="..."
```

`--chosen-k` does not make the whole graph offline. Subcluster review and window
review still call the configured VLM unless tests inject a fake
`ClusterReviewClient`.

## 2. Configuration

Configuration is loaded by `load_config()` in `src/vlm_ppe/config.py` and
validated by `PPEConfig` in `src/vlm_ppe/schemas.py`.

Important fields:

- `dataset_id`, `operation`, `runway`: select ADS-B tracks from the manifest
  and catalog.
- `manifest_path`: points to the data manifest.
- `output_root`: root for run artifacts.
- `n_resample`: equal arc-length samples per track.
- `k_min`, `k_max`: candidate KMeans range.
- `kmeans_n_init`, `kmeans_random_state`: deterministic KMeans settings.
- `max_retries`, `max_k_expansion`: bounded VLM-requested cluster-count retry
  behavior.
- `subcluster_review_enabled`, `subcluster_min_tracks`,
  `subcluster_max_polygons`, `subcluster_max_reviews`: VLM-guided polygon
  subcluster review controls.
- `vlm_model`, `vlm_reasoning_effort`: OpenRouter model settings.
- `window_review_max_attempts`, `window_review_max_patterns`: bounded
  residual-window review controls.
- `track_filter_center_lat`, `track_filter_center_lon`,
  `track_filter_radius_nm`: optional circular trajectory filter.
- `log_level`, `log_to_console`: audit logging controls.

Relative `manifest_path` and `output_root` values are resolved against the
config file directory. If the config is under `configs/`, unresolved paths are
also tried relative to the repository root.

The schema default model is `openai/gpt-5.5`; the KDFW YAML config currently
sets `google/gemini-3.5-flash`, and
`configs/ppe_kdfw_arrivals_openai_gpt_5_5.yaml` sets `openai/gpt-5.5`.

## 3. Graph Shape

The graph is built in `src/vlm_ppe/agents/graph.py`.

Current node sequence:

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
       -> run_candidate_clustering, when retry is requested and allowed
       -> refine_subclusters, otherwise
  -> compute_cluster_medoids
  -> render_medoid_report
  -> compute_residual_profiles
  -> render_window_diagnostics
  -> vlm_classify_windows
  -> validate_window_reviews
  -> export_state
  -> END
```

Each node receives and returns a plain dictionary compatible with `PPEState`.
Node wrappers log `started`, `completed`, or `failed` graph events.

## 4. State And Artifacts

`initial_state()` creates:

```text
<output_root>/runs/<run_id>/
```

The state records:

- `run_id`, `run_dir`, `config`, `retry_count`, `k_max_current`
- audit file paths
- processed track paths
- feature paths
- candidate clustering paths and metrics
- VLM review paths and validated review payloads
- refined cluster assignments
- medoid paths and medoid report path
- residual profile path
- residual-window evidence and review paths
- final `intervention_windows_path`

`export_state_tool()` writes:

```text
<run_dir>/state.json
```

and sets final status to `complete`.

## 5. Deterministic Track Processing

Track ingestion is implemented by `ingest_tracks_tool()` and
`ingest_adsb_tracks()`.

It:

1. loads `data_manifest.json`,
2. reads the selected dataset's landing/departure catalog,
3. filters by `operation` and optional `runway`,
4. loads matching compressed ADS-B trajectories,
5. merges catalog metadata into track points,
6. chooses one common local azimuthal-equidistant projection,
7. optionally clips tracks to the configured circular filter,
8. removes degenerate tracks,
9. writes normalized Parquet tables.

Outputs:

```text
processed/tracks.parquet
processed/track_index.parquet
```

The common coordinate frame matters: clustering uses shared `x_nm` and `y_nm`
coordinates, not per-track local projections.

Resampling is implemented by `resample_track_frame()` and
`arc_length_resample_points()`. Each flight is sorted by `time` and `seq`, then
converted to exactly `n_resample` equal arc-length stations.

Output:

```text
processed/resampled_tracks.parquet
```

Shape features are built by flattening each resampled path:

```text
[x0, y0, x1, y1, ..., xR, yR]
```

and standardizing columns. Outputs:

```text
processed/features.npz
processed/feature_metadata.parquet
```

## 6. Candidate Clustering And VLM Cluster Review

`run_candidate_clustering_tool()` runs KMeans for every K from `k_min` through
the current `k_max_current`.

For each K, it records:

- labels,
- inertia,
- silhouette score when valid,
- minimum, maximum, and mean cluster size.

Outputs:

```text
clustering/attempt_XX/k_metrics.csv
clustering/attempt_XX/k_01_labels.csv
clustering/attempt_XX/k_02_labels.csv
...
```

`render_evidence_pack_tool()` renders one cluster panel per K plus a metrics
chart under:

```text
evidence/clustering/attempt_XX/
```

`vlm_review_clusters` sends the rendered evidence, metrics, available K values,
attempt number, and retry budget to the configured review client. The response
is validated as `ClusterReview`.

The accepted JSON includes:

- `chosen_k`
- `confidence`
- `rationale`
- `rejected_alternatives`
- `clusters_to_recheck`
- `retry_requested`
- `requested_k_max`
- `suggested_action`

`validate_review_tool()` checks that `chosen_k` is one of the available K
values.

`retry_or_accept_tool()` retries only when `retry_requested` is true or
`suggested_action == "retry"`, and only while `retry_count < max_retries`. A
retry expands `k_max_current` up to `max_k_expansion` and loops back to
candidate clustering.

## 7. Subcluster Refinement

After the global cluster count is accepted, `refine_subclusters_tool()` reviews
each chosen cluster for VLM-guided polygon subclusters.

Review is skipped or accepted as a leaf when:

- subcluster review is disabled,
- depth is already at 1,
- the track count is below `subcluster_min_tracks`,
- fewer than two tracks are present,
- the run has exhausted `subcluster_max_reviews`,
- only one polygon subcluster is possible.

For eligible depth-0 clusters, the graph renders a local trajectory overlay and
asks the VLM for a `SubclusterReview`.

If the VLM returns:

- `suggested_action == "discard"`: the whole node is dropped as noise.
- `subcluster_count <= 1`: the node is accepted as one final leaf.
- `subcluster_count > 1`: convex capture polygons are applied.

Polygon assignment is deterministic. A flight is assigned when its path crosses,
touches, or runs inside a polygon. If multiple polygons capture the same flight,
the first matching `subcluster_id` wins. Non-empty polygon children become
terminal depth-1 leaves. Uncaptured tracks are either retained as a residual leaf
or discarded according to `uncaptured_tracks_policy`.

Outputs:

```text
clustering/refined_cluster_assignments.csv
clustering/subcluster_tree.json
clustering/subclusters/node_XXXX/capture_polygons.json
clustering/subclusters/node_XXXX/polygon_assignments.csv
evidence/subclustering/node_XXXX/
vlm_reviews/subclusters/node_XXXX.json
```

Downstream stages use the refined integer cluster IDs, not the original global
KMeans labels.

## 8. Medoids And Residual Profiles

`compute_medoids_tool()` computes one medoid per final refined cluster from the
resampled trajectories and writes:

```text
templates/cluster_medoids.parquet
templates/cluster_summary.json
templates/chosen_cluster_assignments.csv
```

`render_medoid_report_tool()` renders:

```text
templates/cluster_medoids.png
reports/medoid_report.md
```

`compute_residual_profiles_tool()` compares cluster members against their
medoid template and writes:

```text
residuals/residual_profiles.parquet
```

The residual profile includes station-indexed residual energy and heading
dispersion values used as evidence for later VLM window review.

## 9. Residual Window Classification

`render_window_diagnostics_tool()` renders residual-window diagnostics under:

```text
evidence/residual_windows/
```

`vlm_classify_windows` then processes each medoid cluster independently.

For each cluster:

1. A count-only prompt asks for `WindowPatternCountReview`.
2. If the count review asks for human review, the cluster records a human-review
   result and stops.
3. If `pattern_count == 0`, the cluster records no windows and stops.
4. Otherwise the graph asks for one current unconfirmed window at a time.
5. A first non-empty proposal is rendered again with the proposed window
   highlighted.
6. The VLM sees the highlighted evidence and either accepts or revises the same
   current pattern.
7. Attempts are bounded by `window_review_max_attempts`.
8. The loop continues until the fixed count is reached or the review reports
   that all patterns have been identified.

The window classes are:

```text
no_stretch, dogleg, trombone, PMS, other
```

There is no deterministic candidate-window detector in the current graph.
Residual energy and heading dispersion are model-visible evidence; the VLM
supplies the count, station bounds, and class label.

`validate_window_reviews_tool()` validates that station bounds exist in the
cluster residual profile, rejects duplicate window IDs, enriches windows with
station fractions, nautical-mile spans, peak residual metrics, and cluster
track IDs, then writes:

```text
residuals/intervention_windows.parquet
```

## 10. Audit Files

Persistent audit files are always written under `run_dir`.

Human-readable log:

```text
audit.log
```

Structured graph events:

```text
graph_events.jsonl
```

Structured VLM request/response events:

```text
vlm_interactions.jsonl
```

Cluster review files:

```text
vlm_reviews/attempt_00_prompt.txt
vlm_reviews/attempt_00_request.json
vlm_reviews/attempt_00.json
```

Subcluster and window reviews are written under:

```text
vlm_reviews/subclusters/
vlm_reviews/windows/
```

The audit logs record prompts, image metadata, validated response JSON, and
explicit model rationale fields. They do not record hidden chain-of-thought.

## 11. Reading Order

Start with:

1. `src/vlm_ppe/cli.py`
2. `src/vlm_ppe/config.py`
3. `PPEConfig`, `PPEState`, and review schemas in `src/vlm_ppe/schemas.py`
4. `build_graph()` and `run_graph()` in `src/vlm_ppe/agents/graph.py`
5. deterministic node functions in `src/vlm_ppe/agents/tools.py`
6. prompts in `src/vlm_ppe/agents/prompts.py`
7. OpenRouter client behavior in `src/vlm_ppe/agents/vlm_client.py`
8. geometry and clustering helpers under `src/vlm_ppe/geo/` and
   `src/vlm_ppe/clustering/`

Focused tests:

```sh
/opt/homebrew/Caskroom/miniforge/base/envs/rustlingtree/bin/python -m pytest \
  tests/test_ppe_geo.py \
  tests/test_ppe_clustering.py \
  tests/test_ppe_residual_windows.py \
  tests/test_ppe_prompts.py \
  tests/test_ppe_adsb_loader.py \
  tests/test_ppe_graph.py \
  tests/test_ppe_cli.py \
  tests/test_vlm_client.py
```
