# Phase 0 and Phase 1: ADS-B traffic, segment pairs, and evidence

The old two-aircraft factorial benchmark is intentionally gone. Its controlled
factor premises did not match the operational question, so the implementation
does not preserve `FactorialScenario`, correlation gates, or compatibility
adapters.

Phase 0 now asks a narrower, defensible question:

> On held-out naturalistic ADS-B demand, does a frozen policy improve the
> outcome of its bound route-segment pair without degrading whole-snapshot
> conflict or runway safety?

Phase 1 keeps the same traffic and pairing machinery, applies one global demand
scale, and restores downstream-trailer credit.

## Traffic contract

`TrafficScenarioBuilder` reconstructs independent snapshots from 50-NM
terminal-entry events.

- Windows are `[start, start + 3600)` and start every 1200 seconds.
- All valid arrivals and all represented runway labels are retained.
- Cluster identity is exact: `(airport, runway, arrival_cluster)`.
- Phase 0 requires `global_scale=1.0`, preserving every observed ID, timestamp,
  and per-cluster count.
- Phase 1 supplies one `TrafficScaleConfig.global_scale` for the whole batch.

For each window/runway/cluster, the target is rounded half up:

```text
target = floor(global_scale * observed + 0.5)
```

The runway target is the sum of its cluster targets. There is no second runway
rounding or apportionment step. Above scale one, baseline flights remain and
exact additions receive terminal-entry time from the conditioned empirical
intensity. Speed and altitude are copied jointly from a same-cluster donor and
the donor ID is recorded. Below scale one, seeded thinning selects the exact
target without replacement.

The two reproducible build boundaries are:

```bash
hailmary-build-route-graph --help
hailmary-build-traffic-batch --help
```

A complete runnable command chain, a real-data verification script, and its
checked result are in [`examples/hailmary_phase01`](../../examples/hailmary_phase01).

Both emit canonical hashes and audit JSON.

## Pair contract

Runway-threshold order is not a pair definition. A pair must be adjacent on a
shared, directed `RouteSegment` inferred from continuous medoid geometry.

At every real decision epoch, `build_current_segment_anchors()` constructs one
queue per segment:

1. Ignore inactive flights and flights that crossed the segment exit.
2. Put current occupants first, ordered by physical progress downstream to
   upstream on their active trajectory variant.
3. Put committed future entrants next, ordered by current segment-entry ETA
   with stable flight-ID ties.
4. Emit only adjacent leader/follower edges.
5. Bind a duplicate pair to its earliest unpassed common segment.

Exit ETA evaluates spacing; it never changes established physical order. A
trailing occupant with an earlier predicted exit is explicitly marked
`catch_up=True`.

Speed and upstream path-stretch actions cause the next real epoch to rebuild
entry ETAs and may change planned merge order. Segment membership remains the
same because current actions must rejoin before the shared segment. During a
common-root experiment, the original anchor and outcome cohort remain frozen
across rollout arms.

## Exact learning scope

Feature/rule schema v2 separates continuous values from exact categories:

```text
airport
runway
segment
leader_cluster
follower_cluster
```

There is no numeric `cluster_index`. A rule for cluster `2` cannot accidentally
match cluster `20` through an arbitrary numeric interval.

The learner still uses common-root selected, rival, and no-op arms. Evolution
evidence is rival-grounded; deployment evidence is no-op-grounded. Certified
rules are copied into a detached `FrozenRulebookPolicy`.

## Phase contracts

| Property | Phase 0 | Phase 1 |
| --- | --- | --- |
| Input | `TrafficScenarioBatch` | `TrafficScenarioBatch` |
| Traffic scale | exactly `1.0` | one configured global value |
| Simulated context | complete multi-runway snapshot | complete multi-runway snapshot |
| Pair credit | bound pair only (`trailer_count=0`) | pair plus configured downstream trailers |
| Safety | whole-snapshot conflict and runway gates | same |
| Scientific claim | benefit on naturalistic held-out demand | scaled-demand behavior |

`Phase0ExperimentRunner` enforces scale one even if a caller supplies a
different `OutcomeConfig`: its internal outcome contract always sets
`trailer_count=0`. A scenario with no actionable shared-segment pair remains in
traffic-fidelity reporting and is recorded as
`no_actionable_shared_segment_pair`; it is not fabricated into a training case.

Training and held-out batches must use disjoint source partitions or contiguous
time blocks separated by at least one full window. Flight IDs and material
scenario fingerprints must not overlap. Intensity and clustering fallbacks are
fit only from the permitted training corpus.

## Acceptance evidence

Phase 0 is evidence-producing, not pass-forcing. Its held-out claim requires:

- exact scale-one traffic count and timestamp fidelity;
- valid segment adjacency for every evaluated anchor;
- a paired block-bootstrap 95% lower bound above zero against permanent no-op;
- no degradation in whole-snapshot conflict or runway safety; and
- reproducible runtime, rulebook, outcome-plan, and traffic hashes.

Without the removed factorial intervention, Phase 0 does **not** claim that a
deliberately decorrelated causal predicate was identified. It claims only what
the naturalistic paired rollouts and held-out evidence support.

## Step-by-step verification

Run the compact examples before attempting the full corpus:

```bash
./.venv/bin/python examples/hailmary_phase01/01_demand_and_scaling.py
./.venv/bin/python examples/hailmary_phase01/02_route_graph_and_pairing.py
```

Their checked-in JSON results are in
`examples/hailmary_phase01/results`. The immutable real-data fixture is
`tests/fixtures/hailmary/phase01_rw18r_20260401_1020.json`; its SVG companion
shows the five medoids, reviewed segments, accepted pairs, and rejected
threshold-only adjacency. Source hashes and registered topology thresholds are
part of the test contract, so golden drift requires an explicit fixture-version
update.
