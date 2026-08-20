# Hailmary simulator

The implementation of `hailmary_scenario_simulator_design.md` lives in
`src/hailmary`. It is independent of the mutable scenario manager and uses
public SIMAP APIs only through `hailmary.adapters`.

Maintainer-oriented architecture, end-to-end flow, and package documentation
are indexed in [`technicals/README.md`](technicals/README.md).

## Offline pipeline

The main artifact flow is:

```text
catalog + raw ADS-B
  -> runway-partitioned 50-NM tracks and observed releases
  -> deterministic HDBSCAN/KMeans cluster library
  -> observed medoids
  -> immutable, SIMAP-validated trajectory templates
```

The Python entry points are:

- `build_cluster_library_from_adsb(...)` in `hailmary.clustering` for raw data;
- `TemplateCompiler` and `TemplateStore` in `hailmary.templates`; and
- `SIMAPAdapter` in `hailmary.adapters` for A320 envelopes and simplified
  `ReferencePath` compilation/validation.

Cluster artifacts retain the standardized training matrix needed to
deterministically reconstruct HDBSCAN/KMeans membership prediction. Held-out
flights use that predictor first, then the declared nearest-medoid fallback for
noise, preserving membership probability, distance, and OOD provenance. Raw
ground speed is derived before an exact runway-endpoint alignment is applied,
so snapping a sparse final ADS-B point cannot manufacture a terminal speed
spike.

`SIMAPAdapter.compile(...)` reconstructs bounded thrust and flight-path-angle
commands from the dense Hailmary CAS/altitude reference, then runs SIMAP's
public coupled time-domain replay. Physical CAS/TAS/ground-speed, altitude, and
elapsed-time arrays come from that replay; turn/bank demand, cross-track error,
threshold error, envelope margins, timing, and thrust saturation are persisted
in diagnostics. SIMAP's nonlinear planner does not accept an arbitrary dense
historical CAS profile as a fixed command, so this boundary is explicitly
labelled `simap_optimizer_used=false` rather than claiming optimizer provenance.
Custom aircraft configurations require their matching performance backend;
without one the adapter performs only the static envelope/path checks and
records that limitation.

Template quality gates remain strict: a historical medoid is rejected when its
zero-wind CAS approximation needs more than 5 kt of envelope correction, more
than 5% of stations need correction, or replay timing misses either the 10% or
30-second bound. Rejections report the measured excursion/direction/station;
they are data-quality findings, not silently relaxed profiles.

The adapter keeps two numerical corrections explicit and local to Hailmary;
legacy SIMAP code is not changed. It recomputes curvature from SIMAP's public
unwrapped track array so a westbound `-pi`/`+pi` angle wrap cannot create a
false turn. Bank feasibility is measured by contiguous over-limit distance and
traversal time on a distance-bounded path grid, rather than by a
sampling-density-dependent percentile. For curved paths, a deterministic
second replay pass compensates the measured first-pass along-track threshold
overshoot; the integrated station axis is then mapped to the physical release
and runway endpoints before immutable executable arrays are stored. Both-pass
threshold errors, the calibrated stop tolerance, station mapping, and bank
persistence are recorded in diagnostics.

For artifact-oriented workflows, the installed commands are:

```bash
hailmary-build-clusters --help
hailmary-build-templates --help
hailmary-build-offline-corpus --help
hailmary-build-route-graph --help
hailmary-build-traffic-batch --help
hailmary-simulate --help
hailmary-visualize-event-queue --help
```

Inspect a scaled traffic window against the production event queue in a local
browser GUI:

```bash
hailmary-visualize-event-queue \
  --window-start "2026-04-01 09:00" \
  --timezone UTC \
  --scale 1.25
```

The GUI shows observed and scaled flight counts, every pending queue item, the
equal-time batch most recently processed, live catalog action eligibility, and
all aircraft positions. The time scrubber and Previous/Next Event buttons move
only across states produced by `Simulator.advance_next()`. If no compiled route
graph is supplied, the CLI discovers `route_graph.json` next to the corpus or
compiles a sibling `route_graph_input.json` in memory.

The step-by-step visual checks requested by the design are in
`notebooks/hailmary/01_clusters.ipynb` and
`notebooks/hailmary/02_templates.ipynb`.

## Runtime flow

Build a `ScenarioDefinition` with `ScenarioGenerator`, then create a
`Simulator`. Physical events at the same timestamp are batched before one
decision epoch. An `ActionCatalog` exposes epoch-bound `no_op`, slowdown, and
path-stretch candidates. `apply_action` compiles a branch-local variant and
reschedules only the bound flight.

Live replacements keep the true release time plus a separate trajectory-clock
origin. They preserve the already-flown physical prefix exactly, map pending
stations/resources onto a stretched child path, and use replay-compiled
profiles only downstream of the splice. The default path-stretch realization
compares smooth short/medium/long runway-away candidates from identical child
states over one frozen semi-local horizon. Later interventions are suppressed;
exogenous events remain coupled; conflicts are accumulated interval-by-interval
so a later disturbance cannot rewrite earlier history. The geometry-only
clearance selector remains available as the required ablation.

ADS-B traffic generation creates independent half-open one-hour snapshots every
20 minutes. Phase 0 replays scale-one terminal-entry identities, timestamps,
runways, and cluster counts exactly. Phase 1 applies one batch-wide scale, with
half-up rounding per airport/runway/cluster and exact seeded additions or
thinning.

The canonical runtime helpers are:

- `build_current_segment_anchors(...)`;
- `simulator_state_vector(...)`;
- `simulator_outcome_plan(...)`; and
- `paired_simulator_rollout(...)`.

Forks share frozen trajectory arrays but have independent lineage, queues,
RNG state, metrics, and action logs. Content hashes identify identical dynamic
states; state IDs retain branch provenance.

## Version-1 assumptions

- OpenAP A320 with a 12,000 kg payload;
- zero wind for historical ground-speed-to-CAS derivation;
- segment-specific required intervals, initially 90 seconds;
- 16 slowdown stations and 8 path-stretch locations outside the last 4 NM;
- at most two slowdown actions and one path stretch per aircraft; and
- materialized exogenous events for coupled paired rollouts.

These assumptions are serialized in artifacts or configuration rather than
being hidden in learning code.
