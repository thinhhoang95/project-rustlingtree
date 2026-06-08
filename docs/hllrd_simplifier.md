# HLLRD Local Deviation Simplifier

This document describes the local simplifier used by HLLRD. It is
self-contained, but it assumes the reader knows the basic HLLRD data model:
one arrival cluster is converted into an aligned matrix of lateral normal
deviations, then greedy localized low-rank pursuit finds local variation
events.

## Purpose

ADS-B lateral tracks are noisy. A single operational maneuver, such as a
dogleg, can appear as a wiggly local pattern after resampling and low-rank
decomposition. Without an additional simplicity bias, HLLRD can spend model
capacity explaining small oscillations that are not operationally meaningful.

The simplifier adds that bias. For each accepted local event, it asks:

> Can this local deviation pattern be represented as a small piecewise-linear
> detour without losing too much residual-error explanation?

The intended interpretation is:

- `0` approximation points: no local detour, just an endpoint-to-endpoint line.
- `1` approximation point: a simple dogleg.
- `2` approximation points: a trombone-like out-and-back shape.
- `3` or `4` approximation points: more complex local structure, such as a
  holding-like or multi-turn pattern.

The simplifier is deliberately local. It does not simplify the full flight
trajectory, the median/reference trajectory, or the matrix-building stage.

## Where It Runs

HLLRD first builds the normal-deviation matrix:

```text
X_centered in R^(n x M)
```

where each row is one flight and each column is an aligned station along the
arrival stream. HLLRD then finds localized events:

```text
event interval I = [start, end)
local residual block R[:, start:end]
```

The simplifier runs only after a candidate interval has been selected and
trimmed. In implementation terms, this happens in
`src/hllrd/fit.py::_simplify_candidate_for_commit()`.

The pursuit loop still uses the raw, unsimplified candidate for residual
bookkeeping:

```text
R <- R - raw_candidate_reconstruction
```

but it commits the simplified candidate basis into the final dictionary:

```text
dictionary <- simplified_event_basis
```

### Motivation

This separation is important. If HLLRD subtracts the simplified event from the
residual during greedy pursuit, the search can under-explain the original
matrix and stop too early. In the South-East cluster, that naive approach
collapsed the fit from six events to two events and dropped explained fraction
to roughly `0.66`. Keeping raw residual bookkeeping preserves event discovery,
while committing simplified bases keeps the final event representation easier
to interpret.

## One-Dimensional Series Simplification

The core primitive is in `src/hllrd/simplifier.py`:

```python
simplify_series_by_gain(
    y,
    min_gain_per_point_m2=...,
    max_approximation_points=...,
)
```

Here `y` is one local normal-deviation series over the selected event interval.
The endpoints are always retained and are not counted as approximation points.
Only interior retained points are counted.

The algorithm is greedy:

1. Start with only the two endpoints.
2. Fit the straight line between them.
3. Compute the residual sum of squared error, SSE.
4. Try adding each unused interior station as one new knot.
5. Pick the knot that reduces SSE the most.
6. Keep it only if the reduction is at least `min_gain_per_point_m2`.
7. Repeat until the gain test fails or `max_approximation_points` is reached.

### Example: Dogleg

Consider a clean local dogleg shape:

```text
station:  0   1    2   3   4
y:        0   5   10   5   0
```

With endpoints only, the approximation is:

```text
0   0    0   0   0
```

The SSE is:

```text
0^2 + 5^2 + 10^2 + 5^2 + 0^2 = 150
```

The best interior point is station `2`, because adding it gives:

```text
0   5   10   5   0
```

The residual SSE becomes `0`. The error reduction is `150`, so if the minimum
gain threshold is below `150`, this event uses:

```text
retained indices = [0, 2, 4]
approximation points = 1
```

### Motivation

This example captures the operational meaning of a dogleg: one local detour
point between entry and exit. The simplifier does not need a special dogleg
detector; the dogleg emerges from the error-reduction rule.

## Local Block Simplification

An HLLRD event affects many flights, not just one series. The block-level
simplifier is:

```python
simplify_local_deviation_block(
    local,
    active_mask=...,
    min_gain_per_point_m2=...,
    max_approximation_points=...,
)
```

`local` has shape:

```text
n_flights x event_length
```

Only active rows are simplified. Inactive rows are set to zero in the simplified
block.

### Motivation

The active mask matters because HLLRD events are localized both in station and
in participating flights. If a flight is not active in the event, its local
noise should not influence the simplified event shape. Zeroing inactive rows
keeps the rank-2 basis focused on the participating maneuver.

## Gain Threshold

The default threshold is controlled by:

```python
HLLRDV1Config.local_simplifier_gain_sigma = 128.0
```

At fit time, this becomes:

```text
min_gain_per_point_m2 = (local_simplifier_gain_sigma * sigma_hat)^2
```

where `sigma_hat` is HLLRD's estimated noise scale from the centered residual
matrix.

For the South-East merge-trim matrix, `sigma_hat` is about `26.9 m`, so the
default threshold is approximately:

```text
(128 * 26.9)^2 ~= 11.9 million m^2
```

### Motivation

The threshold is expressed relative to estimated noise so it scales with data
quality. A fixed meter-squared threshold would be too strict for noisy clusters
and too loose for clean clusters.

The threshold is intentionally a marginal-gain threshold. It asks whether the
next approximation point is worth its complexity. This directly encodes the
tradeoff:

```text
more approximation points -> lower residual error
fewer approximation points -> simpler operational interpretation
```

## Maximum Point Count

The default maximum is:

```python
HLLRDV1Config.local_simplifier_max_points = 4
```

The simplifier can stop earlier if the marginal gain threshold fails.

### Motivation

The maximum prevents the simplified event from turning back into a noisy
polyline. The expected operational range is small:

- one point for a dogleg;
- two points for a trombone-like shape;
- three or four points for more complex local behavior.

If an event wants more than four interior points, HLLRD should usually represent
that as multiple events or the analyst should inspect whether the interval is
too broad.

## Final Basis Construction

After simplifying active local reconstructions, HLLRD computes a rank-2 basis
from the simplified block:

```text
simplified local block -> SVD -> rank-2 local basis
```

The final global dictionary still has two columns per event, and downstream
transform behavior remains unchanged:

```text
X_centered ~= coefficients @ dictionary.T
```

### Motivation

Keeping the existing rank-2 dictionary contract avoids changing the transform
API, model artifact shape, and interactive Z-score workflow. The simplifier
regularizes the event shape without replacing HLLRD's low-rank model.

## Diagnostics

Each event stores simplifier diagnostics in the model metrics and event
summary. Important fields include:

- `local_simplifier_enabled`: whether simplification was used.
- `local_simplifier_active_count`: number of active rows used by the
  simplifier.
- `local_simplifier_min_gain_per_point_m2`: marginal-gain threshold.
- `local_simplifier_mean_points`: mean number of interior points across active
  rows.
- `local_simplifier_median_points`: median number of interior points across
  active rows.
- `local_simplifier_max_points`: maximum observed active-row point count.
- `local_simplifier_point_count_histogram`: count of active rows by point
  count.
- `local_simplifier_initial_error_m2`: endpoint-only SSE summed over active
  rows.
- `local_simplifier_residual_error_m2`: simplified SSE summed over active rows.
- `local_simplifier_reduced_error_m2`: total SSE reduction.

### Example Diagnostic

For the South-East model's Event 1 after the default simplified fit:

```text
active_count: 86
median_approximation_points: 1
mean_approximation_points: 0.83
point_count_histogram: {"0": 32, "1": 47, "2": 0, "3": 4, "4": 3}
```

This means the typical active-row local pattern is a one-point detour, but
there is variation across flights. Some active rows are nearly straight in that
event window, while a few need three or four points.

### Motivation

The histogram is as important as the median. A median of one point supports the
dogleg interpretation, but a long tail at three or four points warns that the
event may mix multiple local behaviors.

## CLI Controls

The simplifier is enabled by default in the HLLRD fit CLI.

Disable it:

```bash
hllrd-fit \
  --matrix data/hllrd/south-east/matrix_SE_from_merge.npz \
  --output data/hllrd/south-east/model_raw.npz \
  --L-min 40 \
  --K-max 6 \
  --no-local-simplifier
```

Change the gain threshold:

```bash
hllrd-fit \
  --matrix data/hllrd/south-east/matrix_SE_from_merge.npz \
  --output data/hllrd/south-east/model_simplified.npz \
  --L-min 40 \
  --K-max 6 \
  --local-simplifier-gain-sigma 128
```

Change the maximum number of interior points:

```bash
hllrd-fit \
  --matrix data/hllrd/south-east/matrix_SE_from_merge.npz \
  --output data/hllrd/south-east/model_simplified.npz \
  --L-min 40 \
  --K-max 6 \
  --local-simplifier-max-points 4
```

## South-East Cluster Behavior

Using the South-East merge-trim matrix:

```text
data/hllrd/south-east/matrix_SE_from_merge.npz
```

with:

```text
L_min = 40
K_max = 6
local_simplifier_gain_sigma = 128
```

the simplified fit keeps six events and explains about `0.9766` of centered
residual energy. The raw unsimplified fit explains about `0.9855`.

### Motivation

The simplified model gives up a small amount of residual explanation to obtain
more interpretable local event shapes. This is the intended tradeoff. A model
that maximizes explained energy alone is more likely to encode ADS-B wiggles as
event structure.

## Interactive Viewer

The South-East viewer is:

```text
src/hllrd/examples/south-east-interactive.py
```

It loads:

```text
data/hllrd/south-east/model_from_merge_L40_K6.npz
```

by default.

The viewer has two relevant behaviors:

1. The model stores rank-2 event bases.
2. The display reconstructs directly from the stored event basis and selected
   `Z` scores, with no extra simplification.

The event response shown in red is:

```text
mean trajectory + (event.basis @ Z) * normals
```

The faint grey background trajectories are raw ADS-B tracks by default. They
are loaded from the raw ADS-B directory recorded in the matrix metadata, split
with the same gap rule used during preprocessing, filtered to the matrix flight
IDs, and trimmed from the merge anchor when trim metadata is available.

Use this flag to show the older matrix-derived grey background:

```bash
python src/hllrd/examples/south-east-interactive.py --background-source matrix
```

### Motivation

The viewer intentionally shows the raw model response. This keeps the visual
diagnostic honest: if a rank-2 basis response is still wiggly, the analyst sees
that directly. The simplifier regularizes the committed model basis during fit,
but the viewer does not add another trajectory simplification layer on top of
the saved model.

Using raw ADS-B grey tracks makes the background easier to compare with the
original data. The matrix-derived background remains available for debugging
because it shows exactly what HLLRD saw after resampling into normal-deviation
matrix space.

## Design Summary

The simplifier's design choices are:

- Local only: prevents changes to the median/reference trajectory and avoids
  distorting unrelated parts of the flight.
- Active rows only: keeps event shape driven by participating flights.
- Marginal gain per point: directly balances complexity and residual error.
- Noise-scaled default threshold: adapts to ADS-B data quality.
- Small maximum point count: preserves operational interpretability.
- Raw pursuit, simplified commit: preserves event discovery while regularizing
  the final event dictionary.
- Explicit diagnostics: exposes whether an event is truly dogleg-like or hides
  mixed local behaviors.
- Raw interactive reconstruction: keeps the visualized event response faithful
  to the saved model instead of adding a second simplification layer.
