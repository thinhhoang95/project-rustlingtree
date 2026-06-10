# HLLRD: Hierarchical Localized Low-Rank Decomposition

This document describes the HLLRD implementation in `src/hllrd`. It explains
the end-to-end information flow, the algorithmic choices, and the motivation
behind each step.

The current implementation is the V1 version of HLLRD. It is intentionally
simple: it fits fixed-rank, localized trajectory-variation events from an
aligned residual matrix. It does not yet implement full hierarchical
split-merge search, full block-coordinate refits, roughness-penalized PCA, or
random interval search.

## Purpose

HLLRD is used to discover localized, interpretable modes of lateral trajectory
variation inside one arrival cluster.

The main question is:

> Along a common arrival stream, where do flights vary coherently from the
> typical path, and which flights participate in each localized variation?

For each detected event, HLLRD returns:

- a station interval, such as stations `136:176`;
- a rank-2 local basis supported only inside that interval;
- a two-dimensional per-flight score vector `Z = (Z1, Z2)`;
- an active mask saying which flights have meaningful event amplitude;
- reconstruction and residual matrices for diagnostic plots.

The rank-2 choice lets one local event capture two dominant degrees of freedom.
For lateral residuals, this often corresponds to shifts, bends, or spread
patterns inside a specific route segment.

## Files

The HLLRD code is organized as follows:

- `src/hllrd/data.py`: cluster filtering and optional merge-anchor trimming.
- `src/hllrd/matrix.py`: aligned normal-residual matrix construction.
- `src/hllrd/candidates.py`: residual-energy peak detection and local rank-2 candidate scoring.
- `src/hllrd/fit.py`: greedy localized low-rank fitting, coefficient refit, activation, transform, save/load.
- `src/hllrd/report.py`: summary CSV/JSON and plots.
- `src/hllrd/cli.py`: command-line pipeline entry points.
- `src/hllrd/examples/south-east.ipynb`: South-East cluster report notebook.
- `src/hllrd/examples/south-east-interactive.py`: interactive event/Z-score viewer for the South-East artifacts.

## Data Model

HLLRD operates on one matrix:

```text
X in R^(n x M)
```

where:

- `n` is the number of flights;
- `M` is the number of aligned stations along the common path;
- each row is one flight;
- each column is one station;
- each entry is a normal-direction lateral residual in meters.

The implementation starts with normal residuals only. It does not currently fit
tangential residuals, altitude residuals, speed residuals, or tensor-valued
features.

The model is:

```text
X_centered ~= sum_k Z_k V_k^T + E
```

For event `k`:

```text
I_k = [start_k, end_k)
V_k in R^(M x 2), zero outside I_k
Z_k in R^(n x 2)
```

`V_k` is the local rank-2 basis. `Z_k` contains per-flight coefficients. A
flight with large `||Z_ik||_2` has a strong instance of that event.

## End-to-End Information Flow

The complete pipeline is:

```text
SIMAP arrival artifact
        |
        v
select one cluster's flight IDs
        |
        v
load raw ADS-B tracks
        |
        v
filter tracks to selected cluster
        |
        v
optional merge-anchor trim
        |
        v
resample every flight onto M station fractions
        |
        v
build median reference path
        |
        v
project each flight onto reference normals
        |
        v
center columns to form X_centered
        |
        v
fit localized rank-2 events by greedy residual pursuit
        |
        v
refit all coefficients against fixed dictionary
        |
        v
threshold activation using quiet-window energy
        |
        v
write model artifacts, summaries, plots, reports, and interactive visualizations
```

Each stage exists to reduce one source of ambiguity before the low-rank model is
fit.

## Stage 1: Cluster Selection

HLLRD first selects flights from a SIMAP arrival artifact JSONL file:

```text
data/artifacts/simap_arrival_flights.jsonl
```

`load_cluster_flights()` reads each JSON line and keeps flights whose
`wait_atc_point.arrival_cluster` matches one of:

```text
NE, NW, SE, SW
```

Motivation:

- Arrival geometry differs strongly by stream.
- Mixing clusters would force one residual matrix to explain unrelated route
  structure.
- Fitting within one cluster makes the median path and localized events easier
  to interpret.

The raw ADS-B table is then filtered to those selected `flight_id` values.

## Stage 2: Optional Merge-Anchor Trim

The CLI `build-matrix` command builds a matrix from the selected tracks as
given. The South-East notebook adds an extra domain-specific step:
`trim_tracks_from_anchor()`.

The trim step:

1. Starts from a coarse merge anchor latitude/longitude.
2. Finds each flight's closest raw sample to that anchor.
3. Optionally refines the anchor to the median of nearby closest samples.
4. Keeps only the raw samples from that closest point to the end of the track.
5. Drops flights whose closest point is too far from the anchor or whose
   remaining track is too short.

Motivation:

- Upstream portions of an arrival stream can fan out widely before the route is
  under the same control logic.
- If that upstream fanout is included, HLLRD spends most of its capacity
  explaining broad initial geometry instead of approach-control variation.
- Trimming lets the analysis focus on the merge-to-threshold segment.

Important detail:

- The trimmed matrix is still resampled by fraction along each trimmed flight,
  not by exact runway-threshold distance.
- The South-East notebook labels this as distance from merge along the mean
  path for interpretation, but the core matrix stations are normalized
  fractions.

## Stage 3: Matrix Construction

`build_matrix_from_tracks()` converts filtered tracks into a matrix artifact.

### 3.1 Clean and Sort

Rows missing `flight_id`, `time`, `lat`, or `lon` are removed. Tracks are sorted
by `flight_id` and `time`.

Motivation:

- HLLRD assumes each row is one valid, ordered trajectory.
- Missing positions cannot be projected or resampled.

### 3.2 Local Projection

Latitude/longitude coordinates are projected to a local tangent plane using
`LocalProjection`. The projection uses a median origin from the input positions.

Motivation:

- HLLRD needs distances and residuals in meters.
- A local planar approximation is sufficient for the cluster-sized KDFW arrival
  geometry and is much simpler than operating directly on geodetic coordinates.

### 3.3 Initial Station Resampling

Every flight is resampled onto:

```text
stations = linspace(0, 1, station_count)
```

The default station count is `200`.

`resample_polyline_by_fraction()` computes cumulative path length for each raw
flight, normalizes it to `[0, 1]`, and interpolates `x, y` at common fractions.
This first resampling pass is used only to build the initial reference path.

Motivation:

- Raw ADS-B samples are irregular in time and count.
- The median reference path needs one approximate point from each flight at
  each nominal station.
- Fractional station alignment is a simple first-order way to initialize that
  reference before measuring residuals against reference stations.

### 3.4 Reference Path

The reference path is the station-wise median of all sampled flight positions:

```text
reference_xy[j] = median_i sampled_xy[i, j]
```

Motivation:

- The median is robust to outlier flights.
- It defines the typical trajectory for the selected cluster.
- Residuals become deviations from a common centerline rather than absolute
  coordinates.

### 3.5 Normal Residuals

HLLRD computes tangent and normal vectors along the reference path. Each
flight's residual is then measured at reference stations, not at the flight's
own path-fraction stations. For each reference station, the implementation
finds the closest point on the original trimmed ADS-B polyline and projects
that displacement onto the local reference normal:

```text
closest_xy[i, j] = closest point on raw flight polyline to reference_xy[j]
X[i, j] = dot(closest_xy[i, j] - reference_xy[j], normal_xy[j])
```

Motivation:

- Normal residuals represent lateral deviation from the stream centerline.
- Measuring at reference stations avoids confusing along-track station mismatch
  with lateral deviation.
- A single scalar per station keeps V1 tractable and interpretable.

### 3.6 Column Centering

The matrix is centered station by station:

```text
X_centered[:, j] = X[:, j] - center_j
```

where `center_j` is the median by default, or the mean if configured.

Motivation:

- HLLRD should model variation around the typical residual at each station.
- Without centering, the first event could spend capacity reconstructing a
  shared offset rather than explaining differences across flights.
- Column centering also makes `X_centered` compatible with PCA/SVD logic.

The saved `MatrixArtifact` includes both `X` and `X_centered`, the reference
path, normals, flight IDs, stations, center vector, and metadata.

## Stage 4: Noise and Energy Scales

Two different scales are used.

### 4.1 Adjacent-Difference Noise Scale

`estimate_noise_sigma()` estimates:

```text
sigma_hat = MAD(diff(X_centered, axis=station)) / (0.6745 * sqrt(2))
```

Motivation:

- This estimates high-frequency residual wiggle.
- It is used in the analytic fallback floor for candidate score.
- It is not used for activation in the current implementation.

Why not use it for activation?

- Real trajectories are smooth.
- Adjacent station differences can be small even when a whole trajectory is
  displaced by hundreds or thousands of meters.
- Activation should reflect meaningful window-scale event amplitude, not only
  local high-frequency noise.

### 4.2 Quiet-Window Activation Floor

The current activation rule uses a quiet-window energy floor:

```text
e_j = mean_i X_centered[i, j]^2
activation_energy_floor = min_a mean_{j in [a, a + L_min)} e_j
activation_rms_floor = sqrt(activation_energy_floor)
```

Motivation:

- The residual energy curve shows how much flight-to-flight variation exists at
  each station.
- The quietest window of at least `L_min` stations is used as a data-adaptive
  background variation level.
- This produces an activation threshold that is comparable to the `Z` norm.

For an event of length `L`, the activation threshold is:

```text
tau_z(L) = activation_scale * sqrt(L * activation_energy_floor)
```

This is "apple to apple" with `||Z_ik||_2` because the event basis columns are
orthonormal. `||Z_ik||_2` is the L2 norm of the rank-2 reconstructed event
contribution for flight `i` over that event's window. Dividing by `sqrt(L)`
turns it into an average RMS event amplitude over the window.

So the activation rule:

```text
||Z_ik||_2 > activation_scale * sqrt(L * activation_energy_floor)
```

means:

> Mark the event active for this flight if the event's average reconstructed
> amplitude exceeds `activation_scale` times the quiet-window RMS background.

## Stage 5: Candidate Length Grid

`length_grid()` builds a short set of candidate lengths:

```text
L_min = max(5, ceil(0.02 * M)) unless configured
L_max = ceil(0.25 * M) unless configured
lengths = L_min, 2 * L_min, 4 * L_min, ..., L_max
```

Values are capped at `M` and deduplicated.

Motivation:

- Searching every possible interval is expensive.
- A short geometric grid tests compact, medium, and broad windows.
- The V1 method prioritizes high-return candidate generation over exhaustive
  interval search.

Current behavior:

- For each peak, lengths are tested from shortest to longest.
- The first length with positive score is accepted.
- This is the "narrowest over threshold" rule.
- The start can be expanded upstream by peak-rise backtracking, so an accepted
  interval can be slightly longer than the nominal grid length.

This explains why many fitted event windows have exactly `L_min`.

## Stage 6: Residual-Energy Peak Detection

At each greedy iteration, HLLRD computes residual station energy:

```text
e_j = mean_i R[i, j]^2
```

where `R` is the current residual matrix.

The energy curve is smoothed with a small moving average. Local peaks are found
above:

```text
median(smoothed_energy) + kappa_peak * MAD(smoothed_energy)
```

Peaks must be separated by:

```text
min_peak_distance = ceil(0.5 * L_min)
```

unless configured otherwise.

Motivation:

- Events should be placed where residual variation is concentrated.
- The robust median/MAD threshold avoids being dominated by a few extreme
  stations.
- Minimum distance prevents many near-duplicate candidates around one broad
  peak.
- Recomputing peaks on the residual after each selected event allows weaker
  structures to appear after stronger structures are removed.

Important interpretation detail:

- The residual-energy plot in reports usually shows the initial centered
  matrix energy.
- The fitter uses a changing residual `R`, so later events may not correspond
  to obvious peaks in the initial energy curve.

### Peak-Rise Backtracking

For each detected peak, HLLRD optionally backtracks on the smoothed residual
energy curve to find the start of the peak's rising shoulder. The default rule
uses:

```text
rise_threshold = median(smoothed_energy)
                 + peak_backtrack_rise_fraction
                   * (peak_energy - median(smoothed_energy))
```

with `peak_backtrack_rise_fraction = 0.05`.

If this rising-shoulder index is earlier than the centered interval start, the
candidate start is expanded upstream while keeping the centered interval end.

Motivation:

- Some operational variations build gradually and peak late.
- A strictly peak-centered window can start after the variation has already
  begun.
- When the window starts too late, the local basis may reduce residual by
  creating an artificial sharp turn near the support boundary.
- Backtracking biases the event window to cover the ramp-up region instead of
  only the high-energy tail of the peak.

## Stage 7: Local Rank-2 Candidate Scoring

For each candidate interval `[start, end)`, HLLRD extracts:

```text
local = R[:, start:end]
```

It then computes a local SVD:

```text
local ~= U S W^T
```

The first two right singular vectors form the local basis:

```text
V_local = W[:2]^T
```

The full event basis `V` is zero outside the candidate interval.

Per-flight candidate coefficients are:

```text
Z = local @ V_local
```

The raw rank-2 gain is:

```text
raw_gain = sum(first two singular values^2)
```

Motivation:

- SVD gives the best rank-2 approximation to the local residual patch.
- Restricting the basis to one interval makes the event interpretable as a
  localized route deformation.
- Rank 2 captures two independent local shape coordinates without introducing a
  large number of degrees of freedom.

### Candidate Activation

The candidate `Z` rows are thresholded using the quiet-window activation rule:

```text
active_mask_i = ||Z_i||_2 > tau_z(length)
```

Inactive rows are set to zero before scoring:

```text
Z_i = 0 for inactive flights
active_gain = sum_i ||Z_i||_2^2
```

If active count is below `n_min`, all rows are treated as inactive.

Motivation:

- A localized event should represent a pattern shared by enough flights.
- Single-flight outliers should not become population-level events.
- Zeroing inactive coefficients keeps event gain tied to meaningful
  participation.

### Candidate Score

The score is:

```text
score = active_gain
        - null_threshold(length)
        - lambda_i * length
        - lambda_activation * active_count
```

The default `null_threshold(length)` is a length-specific empirical threshold
computed from quiet background windows. For each candidate length, HLLRD:

1. finds the quietest windows by rolling station energy;
2. scores those windows with the same SVD, quiet-window activation, and
   `n_min` rule;
3. takes the configured quantile of their active gains.

This calibrates the score against the same post-activation statistic used for
real candidates without treating shifted copies of real localized events as
background.

The analytic threshold remains a fallback floor:

```text
analytic_null_threshold = c_null * sigma_hat^2 * (n + length)
```

Motivation:

- `active_gain` rewards explained residual energy.
- The empirical null discourages selecting windows explainable by smooth
  background variation or activation-selection effects.
- The analytic null floor keeps behavior defined when empirical null repeats are
  disabled.
- `lambda_i` can penalize long windows.
- `lambda_activation` can penalize overly broad activation, although the
  default is zero.

Only candidates with positive score are eligible.

## Stage 8: Greedy Residual Pursuit

The fitter repeats the following loop:

1. Find residual-energy peaks in current residual `R`.
2. Optionally backtrack each peak to its rising shoulder.
3. For each peak, test candidate lengths from shortest to longest.
4. Keep the first positive-scoring length for that peak.
5. Optionally keep the next longer candidate if `keep_next_longer=True`.
6. Remove candidates that duplicate previously selected intervals.
7. Select the remaining candidate with largest score.
8. Optionally trim weak-energy endpoints.
9. Add the event.
10. Subtract its active reconstruction from `R`.
11. Stop if there are no candidates, if score is nonpositive, if `K_max` is
    reached, or if incremental gain falls below `epsilon_gain`.

Motivation:

- Greedy residual pursuit is simple and fast.
- Each event explains what previous events did not explain.
- The residual update lets HLLRD find a sequence of localized effects rather
  than one global PCA basis.

The endpoint trim step removes low-contribution edges when a candidate is
longer than `L_min`. It cannot shrink an interval below `L_min`.

## Stage 9: Dictionary Construction and Joint Coefficient Refit

After greedy event selection, HLLRD builds a global dictionary:

```text
dictionary = [V_0[:, 0], V_0[:, 1], V_1[:, 0], V_1[:, 1], ...]
```

It then solves a ridge-regularized least-squares problem for all flights:

```text
coefficients = argmin_C ||X_centered - C dictionary^T||_F^2 + ridge * ||C||_F^2
```

Motivation:

- Greedy coefficients are fit against the residual at selection time.
- Once all bases are fixed, overlapping or nearby events should be refit
  together.
- The final coefficients are more internally consistent than the sequential
  coefficients.

After refit, the quiet-window activation threshold is applied again, event by
event. Inactive coefficient blocks are set to zero.

The final reconstruction is:

```text
reconstruction = coefficients @ dictionary^T
residual = X_centered - reconstruction
explained_fraction = 1 - ||residual||_F^2 / ||X_centered||_F^2
```

## Stage 10: Model Artifact

`save_fit_result()` writes a compressed `.npz` model artifact containing:

- `dictionary`: full dictionary matrix, shape `M x (2K)`;
- `coefficients`: final per-flight coefficients, shape `n x (2K)`;
- `reconstruction`: fitted centered matrix;
- `residual`: final centered residual;
- `column_center`: center vector used to center raw `X`;
- `basis`: per-event basis stack, shape `K x M x 2`;
- `event_coefficients`: per-event coefficient stack, shape `K x n x 2`;
- `active_masks`: per-event active masks, shape `K x n`;
- `intervals`: `start`, `end`, and `peak_index` for each event;
- `flight_ids`: matrix row identifiers;
- `config`: JSON-encoded `HLLRDV1Config`;
- `metrics`: JSON-encoded summary metrics.

The metrics include:

- total explained fraction;
- `sigma_hat`;
- `activation_energy_floor`;
- event summaries;
- user metadata.

Motivation:

- The artifact is self-contained enough to transform new matrices with the same
  learned dictionary.
- It also stores the event-level data required for reporting and interactive
  visualization.

## Stage 11: Transforming New Data

`transform_with_model()` applies an existing fitted model to another matrix.

The flow is:

1. Center the new matrix using the model's saved `column_center`, unless the
   caller says the input is already centered.
2. Refit coefficients against the saved dictionary.
3. Apply the model's saved activation floor and activation scale.
4. Reconstruct and compute residuals.
5. Count active events per flight.

Motivation:

- Candidate detection and basis learning are training-time operations.
- Transform should answer: how strongly do new flights express the learned
  event dictionary?
- Using the training activation floor makes transform results comparable to the
  fitted model.

## Stage 12: Reports and Diagnostics

The report layer writes:

- `summary.json`;
- `events.csv`;
- residual-energy plot;
- reconstruction heatmap.

`summary.json` includes:

- number of events `K`;
- explained fraction;
- `sigma_hat`;
- activation energy and RMS floors;
- average active events per flight;
- per-event intervals, active counts, gains, scores, and explained fractions.

`events.csv` is a flat event table useful for quick inspection.

The residual-energy plot overlays selected windows on the station-wise initial
energy curve.

The heatmap shows:

1. centered input matrix;
2. HLLRD reconstruction;
3. final residual.

Motivation:

- The event table confirms where the algorithm placed windows.
- The energy plot shows whether windows align with large residual structure.
- The heatmap reveals whether reconstruction captures coherent row/column
  structure or only isolated outliers.

## South-East Notebook

`src/hllrd/examples/south-east.ipynb` is a worked analysis for the South-East
arrival cluster.

It adds:

- merge-anchor trim before matrix construction;
- geographic plots of the mean path and event windows;
- per-event trajectory examples;
- all-trajectory background plots;
- Z-score sweep plots, where one event is active at a time and `Z1`, `Z2` are
  interpolated from observed active minima to observed active maxima.

The notebook also detects stale model artifacts from the old `tau_z` activation
schema. If a saved model has `tau_z` in its config or lacks
`activation_energy_floor` in metrics, the notebook rebuilds the model.

Motivation:

- HLLRD events are easier to interpret on a map than in matrix coordinates.
- Showing representative trajectories connects `Z` values to visible path
  deformation.
- The Z sweep explains the role of each event basis independent of other
  events.

## South-East Interactive Viewer

`src/hllrd/examples/south-east-interactive.py` opens a Matplotlib widget UI for
the South-East artifacts.

It provides:

- event selection;
- one slider for `Z1`;
- one slider for `Z2`;
- one active event at a time;
- a deformed trajectory over faint raw ADS-B flight trajectories by default.

The grey background tracks are loaded from the raw ADS-B directory recorded in
the matrix artifact metadata, split using the same gap rule as matrix building,
filtered to the matrix flight IDs, and trimmed from the merge anchor when trim
metadata is present. Use `--background-source matrix` to show the old
matrix-derived background instead.

Run from the repository root:

```bash
PYTHONPATH=src python src/hllrd/examples/south-east-interactive.py
```

Motivation:

- Static sweeps are useful, but manual sliders make it easier to understand the
  continuous relationship between coefficients and trajectory deformation.

## CLI Usage

The package exposes these scripts through `pyproject.toml`:

```text
hllrd
hllrd-build-matrix
hllrd-candidates
hllrd-fit
hllrd-transform
hllrd-report
```

The combined pipeline is:

```bash
hllrd run \
  --cluster SE \
  --output-dir data/hllrd/example
```

Equivalent staged commands:

```bash
hllrd build-matrix \
  --cluster SE \
  --output data/hllrd/example/matrix_SE.npz

hllrd candidates \
  --matrix data/hllrd/example/matrix_SE.npz \
  --output data/hllrd/example/candidates.csv

hllrd fit \
  --matrix data/hllrd/example/matrix_SE.npz \
  --output data/hllrd/example/model.npz \
  --summary data/hllrd/example/summary.json \
  --events-csv data/hllrd/example/events.csv \
  --energy-plot data/hllrd/example/residual_energy.png \
  --heatmap data/hllrd/example/reconstruction_heatmap.png

hllrd report \
  --model data/hllrd/example/model.npz \
  --matrix data/hllrd/example/matrix_SE.npz \
  --output-dir data/hllrd/example/report
```

Useful tuning parameters:

- `--stations`: number of matrix columns.
- `--L-min`, `--L-max`: event length grid bounds.
- `--K-max`: maximum event count.
- `--kappa-peak`: residual-energy peak threshold.
- `--smoothing-window`: energy smoothing width.
- `--activation-scale`: multiplier for quiet-window activation threshold.
- `--n-min`: minimum active flight count for a candidate.
- `--c-null`: analytic null threshold multiplier.
- `--empirical-null-repeats`: quiet windows per length; set to `0` to use only
  the analytic threshold.
- `--empirical-null-quantile`: quantile of quiet-window active gains.
- `--lambda-i`: length penalty.
- `--lambda-activation`: active-flight penalty.
- `--epsilon-gain`: early stop threshold for incremental event gain.
- `--keep-next-longer`: keep one backup candidate per peak.
- `--no-peak-backtrack`: disable peak-rise backtracking.
- `--peak-backtrack-rise-fraction`: rising-shoulder threshold as a fraction of
  peak energy above baseline.
- `--local-simplifier`: enable local piecewise-linear event simplification.
- `--local-simplifier-max-relative-loss`: maximum allowed relative
  reconstruction loss before the fitter keeps the raw candidate basis.

## Interpretation of Outputs

### Event Window

The event window says where along the aligned station axis the localized basis
is nonzero. A window does not necessarily imply a real-world procedure boundary.
It is the interval where the residual matrix had enough coherent rank-2
structure to pass the scoring rule.

### Peak Index

The peak index is the residual-energy peak around which the candidate interval
was centered. It is useful for diagnosing event discovery but should not be
overinterpreted after endpoint trimming or final coefficient refit.

### Active Fraction

The active fraction is:

```text
active_count / n
```

under the quiet-window threshold. It estimates how population-wide an event is.

High active fraction means many flights express the event above the background
quiet-window level. Low active fraction means the event is concentrated in a
smaller subset of flights.

### Z Scores

Each event has two scores per flight:

```text
Z1, Z2
```

They are coordinates in the event's local rank-2 basis. They are signed and
basis-dependent. A positive `Z1` does not have a universal meaning across
events. Interpret `Z` by plotting trajectories or event response sweeps.

### Explained Fraction

The global explained fraction is the fraction of centered matrix energy
explained by the full sparse activated reconstruction.

Per-event explained fractions in the summary are based on each event's
individual reconstruction energy divided by total centered energy. If events
overlap, these per-event fractions should be treated as descriptive, not as
strict additive variance decomposition.

## Current Limitations

The V1 implementation is intentionally limited:

- It models normal residuals only.
- It uses fixed rank 2 for every event.
- It uses greedy residual pursuit, not global optimization.
- It searches peak-driven intervals from a short length grid, with optional
  upstream expansion from peak-rise backtracking.
- It accepts the first positive length for each peak, so event lengths often
  equal `L_min`.
- Endpoint peaks are only included by fallback behavior in the peak finder.
- The South-East merge trim is notebook-specific, not part of the generic CLI
  pipeline.
- Event bases are not smoothed, so jagged components may appear if data are
  noisy.
- The activation floor is learned from training data and reused in transform;
  this is good for comparability but may need recalibration if applying a model
  to very different data.

## Design Rationale Summary

HLLRD makes a sequence of pragmatic V1 choices:

- Use one cluster at a time so the reference path is meaningful.
- Align trajectories onto common station fractions so matrix methods apply.
- Model normal residuals to focus on lateral route shape.
- Center columns so events explain variation, not shared offsets.
- Use residual-energy peaks to avoid exhaustive interval search.
- Use local rank-2 SVD to get compact interpretable event bases.
- Use greedy residual pursuit to expose multiple localized structures.
- Refit all coefficients after basis selection for consistency.
- Use quiet-window activation so `Z` thresholding reflects event-scale
  amplitude rather than adjacent-sample wiggle.
- Provide notebook and interactive visualization because event coefficients are
  best understood geometrically.
