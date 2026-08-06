# Hailmary Scenario Simulator: Repository Findings and Implementation Plan

To be referenced with `docs/hailmary/SEQD_Complete_Tutorial.md`.

Status: historical Phase-0 design, based on a repository scan on 2026-07-11.
Its factorial and single-threshold assumptions were superseded by the ADS-B
multi-runway Phase 0/1 design. For the implemented contract, use
[`phase0.md`](phase0.md) and the technical architecture; do not implement the
factorial sections below.

The approved implementation scope covers all phases in Section 12. The
end-to-end slice in Section 15 remains the first integration checkpoint, not
the stopping point for the implementation.

This document is the reusable technical basis for the first `hailmary` milestone:

- assign every arrival to a trajectory cluster;
- use the cluster medoid's path and speed profile as the default plan;
- advance a multi-aircraft scenario through decision events;
- apply slowdown and path-stretch actions;
- fork simulator state cheaply and exactly for paired rollouts; and
- derive a stable, accurate feature vector for a role-bound anchor.

The intended boundary is important: `hailmary` is a new package that lives in this repository and uses selected low-level Rustlingtree capabilities. It is not another mode of `mcp_tools.scenario_manager`, and it must not depend on that package's mutable diff, drafts, HTTP models, evaluators, or advisors.

## 1. Executive conclusions

The repository has useful aircraft and trajectory primitives, but it does not yet have the simulator that SEQD needs.

1. `simap` is the strongest reusable component. It has immutable request/state dataclasses, aircraft speed bounds, route geometry, reduced-order longitudinal dynamics, lateral replay, and CAS-bearing results. It simulates one aircraft/request at a time; it does not own a multi-aircraft world, event queue, snapshot, fork, or role-anchored feature vector.
2. `vlm_ppe` already implements projection, arc-length resampling, standardized 2-D shape features, KMeans clustering, and true medoid selection. An experimental `paper-june` module includes an HDBSCAN sweep. These are good algorithm references, but `hailmary` should implement its own small clustering pipeline rather than depend on VLM/LangGraph workflow code.
3. `mcp_tools.scenario_manager` loads schedules and can regenerate an individual trajectory after an interactive speed or route edit. Its state is a mutable process-wide `diff` plus draft dictionaries. This is not branchable simulation state. Its interventions replace a served artifact and are materially different from the action abstraction requested for `hailmary`.
4. Existing evaluators are read-only consumers of an `arrival_schedule()` protocol. Their math is reusable through an adapter, especially continuous segment conflict detection. They are not the source of truth for the simulator state or the SEQD anchored vector.
5. There is no existing implementation of SEQD's minimum-time map, leader/follower graph, spacing deviation, speed/path capacity, pressure, commitment, twin rollouts, or exact state cloning. Those belong in `hailmary`.
6. The current data has a material speed-profile limitation. Compressed historical ADS-B records contain time, position, and altitude but no airspeed. Raw records contain heading but no speed. SIMAP arrival artifacts contain dense CAS profiles, but their routes are generated base routes rather than the observed historical medoid paths, and only 235 of 787 generated artifacts in the current manifest report successful simulation. The template compiler must therefore make speed provenance explicit and must not silently call finite-difference ground speed “CAS.”

The recommended architecture is a deterministic, event-driven kinematic scenario engine over immutable, precompiled cluster templates (i.e., the trajectories are precomputed). SIMAP is used offline or on cache misses to validate/compile feasible per-aircraft trajectory variants (but the way the problem is phrased, there is nothing wrong with using SIMAP offline for precompute). The hot fork/rollout path shares immutable arrays and copies only small dynamic records, the event heap, and RNG state.

## 2. What SEQD requires from this milestone

The supplied `SEQD_Complete_Tutorial.md` defines a larger learning system. This milestone should implement the environment contract that later learning code can depend on without embedding XCS into the simulator.

At each decision epoch, the eventual learner needs the environment to:

1. expose live role instances such as leader-follower edges and aircraft-resource pairs;
2. bind a candidate anchor and derive its feature vector;
3. enumerate feasible actions at that anchor;
4. fork the exact current world into two or more temporary branches;
5. apply different initial actions while keeping exogenous randomness coupled;
6. roll each branch forward under a supplied frozen policy;
7. score a semi-local outcome; and
8. discard the branches, leaving the parent unchanged.

The action exposed upward should remain compact:

```text
(anchor, lever, band)
```

The simulator/realization layer owns coordinate geometry, speed-envelope enforcement, feasibility, and the short/medium/long search hidden behind the one path-stretch lever.

This separation keeps later XCS code out of aircraft dynamics and keeps `hailmary` usable without XCS.

## 3. Repository map and relevance

### 3.1 Packaging and runtime

#### `pyproject.toml`

The project is a `src`-layout Python package. Relevant installed dependencies already include NumPy, pandas, SciPy, scikit-learn, `hdbscan`, OpenAP, PyProj, and Pydantic. No new clustering dependency is required for the proposed design.

`hailmary` should be discovered automatically when added below `src/hailmary`. It should expose its own CLI entry points later, rather than adding routes to the scenario-manager API.

### 3.2 Data and scenario resources

#### `data_manifest.json`

This is the current indirection from a dataset date to event catalogs, fix sequences, SIMAP artifacts, compressed ADS-B, and fix resources. `hailmary` can read the same manifest format through its own loader, but should convert it immediately into typed, immutable input records.

#### `data/adsb/catalogs/2026-04-01_landings_and_departures.csv`

This is the schedule and runway source. It includes `flight_id`, operation, runway, event time, event position, and threshold coordinates. The scanned file has 2,046 data rows.

For `hailmary`, arrival release time is a separate field from predicted
threshold time. Version 1 reconstructs it from the observed raw-track crossing
of the 50 NM terminal boundary. If a track has samples on both sides of the
boundary, interpolate the crossing time along the crossing segment. Reject the
flight from observed-release replay if the crossing cannot be reconstructed;
do not substitute the catalog threshold event. Traffic scenarios retain the
observed crossing exactly at scale one and record explicit rejection counts.

#### `data/adsb/catalogs/2026-04-01_fix_sequences.csv`

This maps arrivals to first/last times and named fix sequences. It is useful for filtering, audit, and fallback routing, but the requested default geometry will come from the assigned medoid, not each flight's catalog route.

#### `data/adsb/raw/2026-04-01/*.csv`

`scenario.demand_opensky.adsb_catalog_common.RAW_COLUMNS` defines the seven raw columns as:

```text
time, icao24, lat, lon, heading, callsign, geoaltitude
```

There is no observed speed column. Speed can only be estimated from displacement over time unless a richer source is introduced.

#### `data/adsb/compressed/adsb_compressed_flights.jsonl`

There are 2,017 compressed trajectories. Each point has time, latitude, longitude, altitude, and a breakpoint mask. Compression guarantees bounded lateral/altitude reconstruction error, but it removes intermediate samples and stores no speed.

Use compressed tracks for exploratory clustering and UI output. Prefer raw tracks for final medoid time/speed reconstruction because finite differencing compressed breakpoint-only paths can distort speed.

#### `data/artifacts/simap_arrival_flights.jsonl`

There are 787 generated arrival artifacts. They include compressed 4-D points, a dense `cas_profile`, a base route, wait point, and simulation diagnostics. They are useful examples of the desired artifact shape and can seed tests.

They are not a clean medoid-template source:

- the geometry is a generated `original prefix -> final fix -> runway` route, not necessarily the observed trajectory;
- artifacts include failures and are still served;
- the current manifest reports 235 successful and 552 failed simulations; and
- not every catalog arrival has an artifact.

#### `data/artifacts/manifest.json`

This is valuable for provenance and quality-gate design. The `hailmary` artifact builder should produce a similar manifest, but should fail a template build when its reference profile is infeasible rather than merely recording the failure and using it silently.

### 3.3 Historical track preparation and clustering

#### `src/scenario/trajectory_compressor/io.py`

Loads raw ADS-B, normalizes callsigns/IDs, splits tracks at long gaps, and creates per-flight compression tasks. This is useful ingestion logic. Its output task deliberately drops heading and retains only time/lat/lon/altitude.

Decision: use independence option: implement a small `hailmary.data.adsb` loader with the same raw contract. The latter avoids inheriting flight-ID mutation and compression concerns in the simulator domain.

#### `src/scenario/trajectory_compressor/algorithms.py`

Implements local projection, Douglas-Peucker lateral simplification, altitude-as-time-series simplification, and breakpoint union. This is appropriate for output serialization, not for the internal simulation clock. Internal templates should retain a dense normalized station grid and only compress at export boundaries.

#### `src/vlm_ppe/io/adsb_loader.py`

The PPE loader performs the most relevant existing clustering preparation:

- selects catalog flights by operation/runway;
- loads compressed trajectory points;
- chooses a local azimuthal-equidistant projection;
- optionally clips to a radius;
- removes duplicate neighbors; and
- computes cumulative track length.

It currently carries altitude but clusters only on lateral geometry.

Important caution: this module is embedded in a PPE/VLM workflow and takes `PPEConfig`. `hailmary` should reproduce the small deterministic subset behind its own types rather than import the workflow layer.

#### `src/vlm_ppe/geo/projection.py`

Defines a PyProj azimuthal-equidistant `LocalProjection` with meter/NM conversion and inverse projection. This is the right projection class conceptually. A `hailmary.geometry.LocalFrame` can use the same equations/API while making the runway threshold the explicit origin.

#### `src/vlm_ppe/geo/polyline.py`

Provides cumulative length, length, and duplicate-neighbor removal. These small pure functions are directly reusable or easy to re-home.

#### `src/vlm_ppe/geo/resample.py`

Resamples a polyline at equally spaced fractions of its own arc length. This normalization is essential: trajectories must have the same station count before clustering and medoid distance calculation.

For `hailmary`, station direction should be standardized as distance-to-go, with `s=0` at the runway and increasing upstream. All input tracks must be reversed when necessary before resampling.

#### `src/vlm_ppe/clustering/features.py`

Flattens each resampled `(x, y)` track into `[x0,y0,...]`, standardizes features per column, and records normalization statistics. This is a good baseline feature representation.

For production clustering, add heading/tangent features or station weighting only after the pure geometry baseline is tested. Altitude and speed should remain attached template profiles rather than silently changing the initial cluster metric.

#### `src/vlm_ppe/clustering/kmeans_runner.py`

Sweeps KMeans over candidate `K`, records inertia and silhouette metrics, and exports labels. It requires a chosen K and assigns every point, which is convenient but not the requested density-based default.

Keep KMeans as a deterministic comparison/fallback, not as the primary clustering method.

#### `src/paper-june/clustering/clustering_ablation_density_algorithms.py`

Contains experimental DBSCAN, HDBSCAN, and Leiden parameter sweeps. The HDBSCAN sweep varies `min_cluster_size`, `min_samples`, and `cluster_selection_method`, and explicitly measures the noise fraction.

The useful part is the sweep logic and label metrics. The module is an experiment script under a hyphenated directory and should not become a runtime dependency.

#### `src/vlm_ppe/clustering/medoid.py`

Implements a true medoid: for a cluster tensor `(track, station, xy)`, it computes the mean Euclidean station distance for every pair and selects the observed track with the smallest distance sum. This matches the requested “cluster medoid,” not a centroid or averaged path.

Its current distance is an unweighted 2-D mean over normalized stations. `hailmary` should retain that definition for version 1 and store the metric/version in artifact metadata.

#### `src/paper-june/clustering/clustering_ablation_density.py`

Shows the needed HDBSCAN noise handling before medoid extraction: labels below zero are excluded. That is insufficient for the new requirement because every scenario flight must receive a usable default template. `hailmary` needs an explicit post-assignment policy for HDBSCAN noise.

#### `src/vlm_ppe/agents/*`, diagnostics, residual windows, and `ppe_evaluation/*`

These orchestrate VLM review, plots, subclusters, residual-pattern windows, and evaluation reports. They are valuable offline analysis tools but are outside the scenario simulator. `hailmary` clustering must be headless and deterministic; it must not require LangGraph, a VLM, or manual K review.

### 3.4 SIMAP aircraft and path dynamics

#### `src/simap/config.py`

Defines immutable `ModeConfig` and `AircraftConfig`, selects clean/approach/final mode by remaining distance, and computes planned CAS bounds. The lower bound includes a mass-adjusted stall margin; the upper bound uses mode limits. Bank limits combine comfort, procedure, and stall limits.

This is the authoritative reusable speed-envelope logic for a SIMAP-backed template compiler. `hailmary` must additionally impose its own “no speed-up command” invariant relative to the medoid reference profile.

#### `src/simap/calibration.py` and `src/simap/openap_adapter.py`

Load OpenAP data, choose an approach mass, and build calibrated aircraft/performance configuration. The current scenario paths commonly assume A320 and a default payload. The source catalogs do not provide typecode, so the first `hailmary` artifact version must declare its aircraft-model assumption instead of implying per-flight fidelity.

#### `src/simap/backends.py`, `aero.py`, and `units.py`

`backends.py` defines the performance protocol and effective-polar drag model used by the planners. `aero.py` exposes OpenAP atmosphere/CAS/TAS functions with a small fallback, and `units.py` centralizes feet/meters, knots/m/s, and vertical-speed conversions. `hailmary` should use these conversions through the SIMAP adapter and keep unit suffixes on all domain fields; it should not duplicate numeric conversion constants across action/feature modules.

#### `src/simap/path_geometry.py`

`ReferencePath` turns ordered geographic waypoints into a continuous local path with straight legs and fly-by turn arcs. It stores position, track, curvature, total length, and a strictly decreasing remaining-distance coordinate `s_m` where zero is the threshold.

This is the best reusable geometric spine. It also provides the station convention that `hailmary` should adopt everywhere.

One caveat: the builder treats every supplied point as a waypoint and synthesizes fly-by arcs. A densely sampled medoid must first be simplified to meaningful control points; passing all raw points would overfit and can create pathological turns.

#### `src/simap/lateral_dynamics.py`

Computes nonlinear lookahead guidance, wind-adjusted ground motion, cross-track/track error, required curvature, and bank limits. Use it for offline variant verification and high-fidelity replay, not for reconstructing a whole multi-aircraft state on every feature query.

#### `src/simap/fms/datatypes.py` and `src/simap/fms/core.py`

Define immutable FMS requests/results, managed speed targets, and `ATCSpeedSegment(s_from_m, cas_mps)`. The planner builds level-to-TOD plus managed descent profiles and records CAS, TAS, altitude, phase, thrust, and timing.

Existing speed interventions map naturally to `ATCSpeedSegment`, but current validation only checks positive CAS and station range. `hailmary` must validate lower/upper envelope and the no-speed-up rule before creating a segment.

#### `src/simap/fms/holds.py`

Adds altitude-hold instructions, hold-speed capture, and a hold-aware FMS planner. It is outside the requested first action set, but its immutable request-wrapping pattern is a useful precedent for adding a future `hold` lever without changing core FMS types.

#### `src/simap/fms_bichannel/core.py`

`FMSBiChannelState` contains time, remaining distance, altitude, TAS, east/north, heading, and bank. `FMSBiChannelResult` combines the longitudinal result with lateral map state and diagnostics. `plan_fms_bichannel()` plans the longitudinal profile and replays the lateral channel on the same time grid.

This is a single-flight batch function. Its frozen dataclasses make inputs safe to share, but calling it does not create a resumable or forkable scenario. `hailmary` should compile its results into immutable `TrajectoryVariant` arrays and make those arrays branch-shareable.

#### `src/simap/nlp_colloc/*`

The collocation stack supports constraint envelopes, coupled path/descent planning, a time-integrated replay, and rich diagnostics. It is higher fidelity and more expensive than the FMS path. It should remain an optional template-validation backend, not the initial hot-loop simulator.

`nlp_colloc.replay.State` is another immutable single-aircraft state, and `simulate_plan()` advances it internally to completion. It does not expose a world snapshot/fork abstraction.

`src/simap/simulator.py` and `src/simap/longitudinal_profiles.py` are compatibility re-export modules for replay and profile types. New `hailmary` code should import the owning public APIs from `simap`/their concrete packages rather than treating these shims as new simulator implementations.

#### `src/simap/nlp_colloc/tactical/*`

Resolves named fixes or coordinate tokens, builds `ReferencePath`, creates an A320/OpenAP performance model, and constructs constraint envelopes. This is useful when compiling a medoid polyline or dogleg variant into a physically checked SIMAP request.

The builder's public command starts at the first route waypoint and ends at a runway waypoint. A `hailmary.adapters.simap` layer should hide these tactical details from the scenario engine.

#### `src/simap/weather.py`

Defines a weather protocol and constant-weather implementation. Forked branches must share the same immutable weather realization. Version 1 can use constant/zero wind, but the artifact and state hash must include the weather model/seed.

### 3.5 Existing scenario manager and actions

#### `src/mcp_tools/scenario_manager/models.py`

Defines resource config and permissive HTTP response models. Its arrival payload is artifact-oriented (`columns` plus compressed `points`) rather than simulator-state-oriented. Do not use these Pydantic models as `hailmary` domain types.

#### `src/mcp_tools/scenario_manager/resources.py`, `api.py`, and `server.py`

`resources.py` validates/loads CSV, JSON, and JSONL resources. `api.py` wires one eagerly loaded manager into FastAPI endpoints, and `server.py` starts Uvicorn. They demonstrate a compatibility service boundary but add no simulation semantics. A future `hailmary` service should wrap the package's public API separately rather than share this process-global manager.

#### `src/mcp_tools/scenario_manager/manager.py`

Eagerly loads catalog/resources and serves arrival/departure schedules. Runtime mutable state is:

```text
diff: list[dict]
path_stretch_drafts: dict
speed_intervention_drafts: dict
```

`arrival_schedule()` reconstructs output payloads and applies the active diff. `_apply_diff()` now delegates to `apply_path_stretch_diff`; therefore the older statement in `docs/scenario_manager_and_precompute_artifact.md` that diff application is only a placeholder is stale.

This class is useful as a source adapter or compatibility output target. It is unsuitable as the `hailmary` world because it has one global mutable timeline, no clock/event heap, no snapshot lineage, and no cheap fork.

#### `src/mcp_tools/scenario_manager/precompute_artifact.py`

This is the most complete end-to-end example of building a SIMAP artifact:

```text
catalog + fix sequence + raw ADS-B seed
  -> route and final-fix selection
  -> tactical/FMS request
  -> plan_fms_bichannel
  -> compressed trajectory + dense CAS profile + diagnostics
```

Reuse its ideas and public low-level SIMAP calls. Do not import private helpers such as `_build_request` or `_payload_from_result` into `hailmary`; their leading underscores signal an unstable boundary and the current action modules are already tightly coupled to them.

#### `src/mcp_tools/scenario_manager/served_profile.py`

Reconstructs a SIMAP request from a currently served arrival and merges prior speed advisories. It exists to support interactive edit composition. This is not equivalent to replaying an event-driven state from an arbitrary epoch.

#### `src/mcp_tools/scenario_manager/speed_intervention.py`

Accepts arbitrary station/CAS advisories, simulates baseline and modified trajectories, stores a draft, and on save replaces the active per-flight intervention diff. It does not enforce “slowdown only,” does not offer fixed decision stations, and does not create branch-local state.

It is a useful SIMAP integration example, but `hailmary.actions.speed` should be independent.

#### `src/mcp_tools/scenario_manager/path_stretching.py`

Accepts interactive handles or a complete edited route, reruns FMS bichannel planning, computes elapsed/distance deltas, and stores/saves a replacement artifact. It can preserve active speed advisories.

It is intended for a user-authored route edit. The requested `hailmary` action instead chooses one action at a bound station, generates three deterministic doglegs, and internally retains the best feasible variant.

#### `src/mcp_tools/sensory/vector_assist.py` and `operational_space.py`

The existing vector assist is closer to, but still different from, the requested behavior. It:

- searches identified fixes and grid points in a north/south polygon mask;
- inserts or replaces a waypoint;
- rejects duplicate/tight-turn candidates;
- estimates time gain from added length;
- exactly simulates a bounded candidate set; and
- recommends the candidate closest to a requested time gain.

It does not use clearance from all cluster medoids to define free space and its action is targeted by requested time gain. `hailmary` can reuse its geometric lessons (candidate projection, turn validation, exact scoring), but should own a medoid-clearance field and fixed short/medium/long semantics.

#### `src/mcp_tools/scenario_manager/wait_atc_point.py`

Classifies arrivals into NE/NW/SE/SW and finds a decision point around a 50 NM gate. This may be useful as an input filter or fallback cluster label, but it is too coarse to replace data-driven trajectory clusters.

### 3.6 Existing evaluators and advisors

#### `src/mcp_tools/evaluators/feasible.py`

Reads `simulation.success` and final threshold error from served artifacts. It detects previously recorded failure; it does not independently establish dynamic feasibility.

#### `src/mcp_tools/evaluators/conflict.py`

This is the most reusable evaluator. It projects all trajectories, treats them as piecewise-linear 4-D segments, finds overlapping time intervals, solves continuous lateral/vertical separation inequalities, and merges hits. It accounts for compression tolerance.

Use it initially through a `HailmaryScheduleView` adapter for regression/cross-checks. The hot SEQD outcome scorer should consume native immutable trajectory variants and state times directly, avoiding repeated JSON/Pydantic conversion.

The existing threshold is 5 NM lateral and 1,000 ft vertical. Arrival runway spacing and wake separation are separate concepts and still need their own table.

#### `src/mcp_tools/evaluators/runway_overlap.py`

Builds fixed runway occupancy windows (60 seconds for arrivals, 90 for departures) and detects overlap on physical reciprocal-runway keys. It is useful as a residual safety/capacity term, not as leader-follower spacing logic.

#### `src/mcp_tools/utils/ground_distance.py`

Builds an A320 FMS descent on a synthetic long path to estimate required ground distance. It can inform feasibility diagnostics, but a per-anchor speed/path capacity must be computed on the active template, not on this synthetic path.

#### `src/mcp_tools/advisors/*`

Advisors wrap scenario-manager schedules, rerun what-if simulations, and return human-facing recommendations. They should remain external consumers. `hailmary` should expose pure protocols/artifacts that an advisor could call, not import advisor logic.

### 3.7 Other packages and tests

#### `src/scenario/cifp_parser` and `src/scenario/adip_charts`

These extract navigation/procedure/chart resources. They may later supply merge resources, airspace boundaries, or procedure constraints. The first simulator should use explicit configured resources and the existing fix catalog; it should not make chart parsing part of a rollout.

#### `src/hllrd_to_be_deleted`

This is explicitly marked for deletion and concerns trajectory simplification/FPCA experiments. It should not be a dependency of a new package.

#### `src/vlm_ppe`, `src/ppe_evaluation`, and `src/paper-june`

Beyond the pure clustering pieces called out above, these are research workflows, diagnostics, VLM orchestration, and ablations. They should consume or compare `hailmary` artifacts later, not own simulator state.

#### `tests/`

The strongest reusable testing patterns are in `test_ppe_clustering.py` (known medoid fixtures), `test_fms_bichannel.py` and `test_coupled_simulator.py` (state/path invariants), `test_speed_intervention.py`/`test_path_stretching.py` (what-if composition), and the evaluator tests (continuous-time safety edge cases). New tests should live under `tests/hailmary/` so the independent boundary is visible.

## 4. Proposed package boundary

Create a new top-level package:

```text
src/hailmary/
  __init__.py
  config.py
  errors.py
  ids.py
  data/
    manifest.py
    catalog.py
    adsb.py
  geometry/
    frame.py
    polyline.py
    dogleg.py
  clustering/
    features.py
    hdbscan_runner.py
    medoid.py
    assignment.py
    artifact.py
  templates/
    models.py
    speed.py
    compiler.py
    store.py
  scenario/
    models.py
    generator.py
  simulator/
    state.py
    events.py
    engine.py
    interpolation.py
    hashing.py
  actions/
    models.py
    catalog.py
    speed.py
    stretch.py
  features/
    anchors.py
    minimum_time.py
    state_vector.py
    schema.py
  evaluation/
    spacing.py
    conflict.py
    outcome.py
  rollout/
    policy.py
    paired.py
  adapters/
    simap.py
    scenario_manager.py
  cli/
    build_clusters.py
    build_templates.py
    simulate.py
```

Rules for independence:

- `hailmary` may import public `simap` APIs behind `adapters/simap.py`.
- Core modules must not import `mcp_tools.*`, `vlm_ppe.agents.*`, FastAPI, advisors, or existing scenario-manager models.
- Small pure geometry/clustering algorithms may be reimplemented with tests and attribution instead of importing workflow modules.
- Compatibility with existing evaluators is one-way through an adapter implementing `arrival_schedule()`.
- Core state uses frozen dataclasses/NumPy arrays, not permissive dictionaries.

## 5. Domain and artifact model

### 5.1 Immutable offline artifacts

`ClusterLibrary`:

```text
schema_version
dataset_id
airport/runway partition
projection definition
resample station count
feature normalization
HDBSCAN configuration and library versions
cluster records
flight assignments, probabilities, and original-noise flags
artifact content hash
```

`ClusterTemplate`:

```text
cluster_id
medoid_flight_id
member count and dispersion
station s_m (strictly increasing upstream, s=0 at threshold)
lat/lon and local east/north
altitude_m
reference ground_speed_mps
reference tas_mps/cas_mps plus provenance
lower/upper CAS envelope
monotone command profile
16 speed-action stations
8 path-stretch locations
precompiled dogleg variants or cache keys
quality diagnostics
```

`TrajectoryVariant`:

```text
variant_id/content hash
cluster/template ID
action provenance
station grid
geometry and altitude arrays
CAS/TAS/ground-speed arrays
relative elapsed-time array
resource crossing offsets
feasibility diagnostics
```

All NumPy arrays should be contiguous `float64`, marked non-writeable after construction, and validated for finiteness and monotonicity. Arrays are shared across forks.

### 5.2 Immutable scenario definition

`ScenarioDefinition` contains everything that never changes during a rollout:

- scenario ID and seed;
- release schedule;
- resource/runway definitions and separation table;
- per-flight identity, assigned cluster, and baseline template;
- weather realization;
- decision/action station configuration; and
- references to immutable trajectory variants.

### 5.3 Dynamic branch state

`SimulationState` contains only branch-local values:

```text
state_id and parent_state_id
sim_time_s
event_sequence counter
per-flight lifecycle and current variant ID
per-flight release time, station cursor, and action history
predicted resource-crossing times
persistent/copy-on-write event heap
RNG bit-generator state
accumulated metrics and intervention cost
```

No mutable object reachable from a child may also be mutable in its parent.

The default implementation can copy tuples/small frozen flight records and `heapq` lists. Optimize to a persistent map/heap only after profiling; immutable trajectory arrays dominate size and are already shared.

## 6. Offline cluster and medoid pipeline

### 6.1 Track cohort

Cluster arrivals separately by airport and landing runway direction. A medoid path ending at a different threshold must never become another flight's default.

For the first KDFW experiment:

1. select arrivals with valid threshold coordinates and at least two unique points;
2. clip each track to a common terminal radius, initially 50 NM;
3. ensure order is upstream to threshold;
4. append/interpolate the exact threshold endpoint if the observed track ends within a configured capture radius;
5. reject tracks that cannot be aligned to the runway; and
6. project around that runway threshold in an azimuthal-equidistant frame.

### 6.2 Normalization and features

Resample every clipped track to 128 equally spaced fractions of arc length for clustering. Store the original dense/raw samples separately.

Clustering arrays use normalized progress `u=0` at terminal-area entry and `u=1` at the threshold. Executable template arrays are then reordered/indexed by remaining distance `s_m`, ascending from `s=0` at the threshold to the upstream endpoint; aircraft motion always decreases `s`. Keeping both conventions explicit avoids accidentally reversing profiles during medoid compilation.

Version-1 feature vector:

```text
[x_0, y_0, ..., x_127, y_127]
```

Standardize each column using training-cohort mean and scale. Persist those statistics; online/new-flight assignment must use the training transform, never refit it.

Potential later additions are tangent/heading or station weights. Do not include altitude/speed until their effect on cluster meaning is deliberately evaluated.

### 6.3 HDBSCAN selection

Use HDBSCAN as the default with a deterministic parameter-selection artifact:

- sweep `min_cluster_size` over values appropriate to cohort size, beginning with 4, 8, 12, 16, and 24;
- sweep `min_samples` over `None`, 3, 5, and 8;
- compare `eom` and `leaf` selection;
- record cluster count, clustered/noise fractions, size distribution, silhouette on non-noise points, and persistence;
- reject configurations with implausibly high noise or singleton-like fragmentation; and
- select by a declared composite score, with deterministic tie-breaking by serialized parameters.

Do not choose parameters with a VLM in the runtime build.

Version 1 must use a fixed scored sweep rather than a manual notebook choice.
The score weights, rejection thresholds, and tie-break order are configuration
schema fields and are serialized into the artifact. If every HDBSCAN candidate
is rejected, run the declared deterministic KMeans fallback and mark the
artifact accordingly. The precise balance among silhouette, persistence,
coverage, and fragmentation remains one of the unresolved choices in Section
14; it must be frozen before Phase 1 is considered complete.

### 6.4 Final assignment for every flight

HDBSCAN label `-1` is an intermediate result, not a final simulator assignment.

For each training or new flight:

1. use HDBSCAN membership prediction when available;
2. if noise, compute distance to every medoid in the unstandardized station-space metric;
3. assign the nearest medoid deterministically;
4. retain `was_hdbscan_noise=true`, membership probability, medoid distance, and an out-of-distribution flag; and
5. if distance exceeds the cluster's calibrated acceptance radius, still assign a fallback medoid to meet the simulator contract, but exclude that record from template training and flag it for analysis.

This satisfies “every flight has a cluster/default path” without pretending that HDBSCAN considered every assignment in-distribution.

### 6.5 Medoid

Within each non-noise training cluster, choose the actual observed track minimizing:

```text
d(i,j) = mean_k ||P_i(k) - P_j(k)||_2
medoid = argmin_i sum_j d(i,j)
```

Tie-break by stable `flight_id`. Persist pairwise summary statistics and verify that the selected medoid is a cluster member.

Version 1 deliberately retains this unweighted 2-D mean station-distance
metric. It does not weight terminal stations or add tangent/heading terms.

### 6.6 Medoid altitude and speed profile

Geometry, altitude, and speed must all come from the same medoid flight and normalized station coordinate.

Recommended version-1 process:

1. go back to that medoid's raw time/lat/lon/altitude samples;
2. project and compute cumulative distance;
3. derive interval ground speed from distance/time;
4. remove invalid gaps and robustly smooth speed versus station;
5. interpolate altitude and ground speed to the template station grid;
6. construct a non-increasing commanded-speed reference using constrained isotonic regression in the downstream direction;
7. under version-1 zero-wind assumptions, treat smoothed ground speed as TAS only for conversion, then convert TAS/altitude to CAS with OpenAP;
8. compare CAS with the configured aircraft envelope, clamp only excursions of
   at most 5 kt, and reject the template if any excursion is larger than 5 kt
   or if clamping affects more than 5% of stations; and
9. run the compiled path/profile through SIMAP for feasibility and record the deviation from observed timing.

The artifact must label the source explicitly, for example:

```text
speed_source = finite_difference_ground_speed
wind_model = zero_wind
cas_derivation = ground_speed_as_tas_then_openap
```

This is an approximation, not observed CAS. A richer ADS-B/wind source should
replace it when available. The SIMAP-compiled profile is the version-1
executable source of truth. The smoothed historical medoid remains the command
reference, provenance record, and timing-validation target. Accept a compiled
template only when its absolute 50-NM traversal-time error is no more than both
10% of the observed duration and 30 seconds. Report the signed and absolute
errors even for accepted templates.

Every aircraft assigned to the cluster receives this same geometry and
reference profile. Only release time, identity, and later interventions differ.
Version 1 uses OpenAP `A320`, a 12,000 kg payload, and the approach mass returned
by `suggest_approach_mass_kg`; the resolved engine and mass are artifact
provenance rather than implicit defaults.

## 7. Scenario generation

`ScenarioGenerator` remains a generic immutable-definition primitive.
`TrafficScenarioBuilder` is the experiment boundary. It creates independent
one-hour ADS-B windows every 20 minutes, includes all valid runway arrivals,
preserves the observed 50-NM entry schedule exactly at Phase-0 scale one, and
uses one global scale in Phase 1. Route-graph segment traversals and gate
resources are explicit in scenario schema v2. No factorial schedule mutation
or correlation gate remains.

All exogenous events—flight releases, weather changes, injected disturbances—
come from named RNG streams and carry stable event identities. Whichever
coupling strategy is selected in Section 14 must guarantee identical exogenous
realizations in paired branches even after their endogenous action histories
diverge; merely sharing a master seed is insufficient.

## 8. Event-driven simulation and exact forks

### 8.1 Event types

Start with:

- `FLIGHT_RELEASED`;
- `ACTION_STATION_CROSSED`;
- `RESOURCE_CROSSED` (merge/final/threshold);
- `FLIGHT_COMPLETED`;
- optional `EXOGENOUS_DISTURBANCE`; and
- `DECISION_EPOCH` derived from configured trigger events.

Event ordering key:

```text
(event_time_s, event_priority, flight_id, station_index, insertion_sequence)
```

This makes ties deterministic and dynamic-content hashes reproducible.

Process all physical events with the same timestamp as one batch in declared
priority order:

```text
0 EXOGENOUS_DISTURBANCE
1 FLIGHT_RELEASED
2 ACTION_STATION_CROSSED
3 RESOURCE_CROSSED
4 FLIGHT_COMPLETED
```

Apply every state transition in the batch, then emit at most one
`DECISION_EPOCH` from the resulting state if any processed event is a configured
trigger. Version 1 triggers on disturbance, release, action-station, and
resource-crossing events; completion alone adds no second epoch after its
threshold-resource crossing. Arbitration never runs between two physical
events at the same timestamp. `DECISION_EPOCH` is derived and is not inserted
back into the physical-event heap.

### 8.2 Kinematic hot loop

Each trajectory variant stores a monotone mapping between station and relative elapsed time. At absolute scenario time `t`, flight state is found by monotone interpolation/inversion, not by integrating every aircraft at a global small time step.

This provides:

- exact template event times up to interpolation tolerance;
- fast skipping between meaningful epochs;
- cheap recomputation after swapping one flight's variant; and
- stable fork behavior.

SIMAP remains the compiler/validator for variants. The event engine consumes compiled arrays.

### 8.3 Fork API

Suggested API:

```python
parent = simulator.state
left = simulator.fork(label="selected")
right = simulator.fork(label="contender")

left.apply(action_a)
right.apply(action_b)
left.run_until(horizon, policy=frozen_policy)
right.run_until(horizon, policy=frozen_policy)
```

Required guarantees:

1. fork is pure with respect to the parent;
2. parent and child share immutable definitions/templates;
3. dynamic flight records, event queues, metrics, and RNG state are logically independent;
4. identical child actions/policies yield identical dynamic-content hashes;
5. different initial actions may produce different later policy actions, as required by SEQD;
6. learning, covering, policy mutation, and shared global counters cannot occur inside a rollout unless passed as explicitly branch-local objects; and
7. discarding children requires no rollback.

### 8.4 State hashing and provenance

Maintain two identities:

- `dynamic_content_hash` includes schema version, definition hash, simulation
  time, sorted flight dynamic records, sorted pending events, RNG state, metrics,
  and accumulated action log; and
- `state_id` is a provenance identity derived from the content hash plus parent
  state ID, branch label, and lineage metadata.

Neither hash includes Python object IDs or noncanonical dictionary ordering.
Identical branches have the same dynamic-content hash but distinct state IDs.

Hashes make paired-rollout bugs diagnosable and allow tests to prove that the real parent timeline was untouched.

## 9. Action model

### 9.1 Shared action schema

```text
ActionCandidate
  action_id
  anchor_id and bound flight_id
  lever: no_op | speed | path_stretch
  band
  station_index / s_m
  feasibility status and reason
  deterministic realization metadata
```

Actions are only offered when their bound aircraft is active and the configured
station-crossing event is in the current event batch. Stations are indexed from
terminal entry toward the 4-NM final gate. Eligibility lasts for that decision
epoch only; a passed station expires and an action is never snapped to a nearby
or later station. An action applied to a stale state version must fail rather
than silently retarget.

`no_op` is first-class.

### 9.2 Speed-action stations and path-stretch locations

Speed adjustments use 16 station fractions per cluster template over the controllable portion of flight, not over the runway flare:

```text
f_i = i / 17, i=1..16
```

Path stretch uses exactly 8 dedicated locations over the same controllable interval:

```text
g_j = j / 9, j=1..8
```

Map both sets of fractions onto the interval from the entry point to a configurable final commitment gate, initially 4 NM before threshold, preserving entry-to-final index order even though executable `s_m` decreases during flight. Snap each set to the dense template grid and deduplicate. Persist the actual `s_m`, coordinates, location type, and index.

Every valid template must provide 16 distinct speed-action stations and 8 distinct path-stretch locations. If the path or grid is too short to provide either required count, fail template validation; do not emit duplicate locations or silently reduce the declared count.

### 9.3 Speed adjustment

Expose three coarse bands for the initial SEQD phase, consistent with the tutorial's warning about data starvation:

- `light`: 10 kt below the medoid command profile;
- `medium`: 15 kt below the medoid command profile; and
- `heavy`: 20 kt below the medoid command profile.

Make these config values, not hard-coded domain facts.

At anchor station `s_a`, construct a pointwise commanded profile:

```text
v_new(s) = min(v_current_command(s), v_medoid(s) - delta_band)
v_new(s) = max(v_new(s), lower_envelope(s))
```

The reduction remains active until the downstream medoid profile naturally falls to the commanded value, after which the aircraft rejoins the medoid profile without a commanded acceleration. Reject the action when clamping leaves less than a small effective reduction (initially 2 kt).

Hard invariants:

- no intervention command exceeds the medoid profile at any station;
- no later intervention command exceeds the branch's current commanded profile;
- all commands remain within the SIMAP/OpenAP envelope (to prevent extending below stall speed)
- the upper envelope and 250 kt below 10,000 ft still apply; and
- action feasibility and realized delay are deterministic.

“Never speed up” is interpreted as no planned/commanded speed increase. A high-fidelity physical replay may show small transient CAS increases; record these separately. If the research requirement instead forbids any positive physical acceleration, use that stricter check as a template/action rejection criterion.

Each aircraft may receive at most two non-no-op speed changes over its lifetime.
After the second, speed actions are no longer enumerated. Sequential commands
apply only downstream of the current station and must preserve position,
altitude, elapsed time, and commanded-speed continuity at the splice.

Compile/cache a modified variant by `(current_variant_hash, anchor_index, band,
aircraft_config_hash, weather_hash)`, not by the original template hash. This is
required because the second speed action is composed with the first.

### 9.4 Path-stretch geometry and free space

At one of the 8 path-stretch locations, define a downstream rejoin point and create a triangular/fly-by dogleg whose apex lies on the side of the base segment with the greatest free space. Path stretch is not offered outside these eight locations.

Free space is defined from the cluster library, as requested:

1. create candidate offset directions around the midpoint between action and rejoin station;
2. require the apex to be farther from the runway than the unmodified midpoint;
3. construct the candidate dogleg corridor;
4. compute its minimum distance to every *other* cluster medoid polyline over the affected terminal region;
5. optionally include configured airspace-boundary clearance;
6. reject self-intersection, excessive turn, envelope/bank failure, or final-gate intrusion; and
7. choose the direction maximizing minimum medoid clearance, then runway-away displacement, then boundary clearance, with azimuth as deterministic final tie-break.

Do not use other aircraft's instantaneous positions to define the default direction. That would make the action meaning vary with traffic and confound the “one path-stretch action” abstraction. Traffic conflicts are evaluated after generating variants.

### 9.5 Short, medium, and long variants

Use target added distance rather than arbitrary coordinate offsets. Initial values:

| Variant | Base rejoin span | Target added distance | Approximate delay at 210 kt |
|---|---:|---:|---:|
| short | 10 NM | 2 NM | 34 s |
| medium | 14 NM | 5 NM | 86 s |
| long | 18 NM | 9 NM | 154 s |

For a symmetric triangular construction with original span `L` and target extra distance `delta`, initial lateral apex offset is:

```text
offset = 0.5 * sqrt(delta * (2*L + delta))
```

The final fly-by geometry will change length slightly, so solve/refine the offset and require added-distance error within a configured tolerance (initially 0.25 NM). Near the final gate, shorten the rejoin span only if the same safety/length requirements remain possible; otherwise mark the variant infeasible.

Each variant inherits the cluster medoid speed/altitude reference and is recompiled/validated through SIMAP. Geometry length, not an arbitrary speed change, should create the primary delay.

### 9.6 One path-stretch action from three variants

The learner sees one `path_stretch` action at each of the 8 path-stretch locations. At an eligible location, its realization layer evaluates short, medium, and long from identical child state and retains the best feasible candidate. This produces up to 24 path-stretch trajectory variants per cluster template (8 locations times 3 internal variants), while exposing only one abstract `path_stretch` action at a given location.

Each aircraft may receive at most one path stretch over its lifetime. After it
is applied, later path-stretch actions are not enumerated.

Default deterministic internal selection evaluates each feasible variant with
the full semi-local outcome from Section 11 over the same available-trailer
horizon. Later interventions are suppressed during this inner comparison; the
outer paired rollout subsequently evaluates the chosen realization under the
frozen downstream policy. This prevents recursive path-stretch realization
while retaining pair, propagation, parsimony, and throughput terms. Selection
then uses:

1. infeasible variants lose to feasible variants;
2. any loss of required separation or new conflict is a graded penalty;
3. improvement of the bound leader-follower spacing error is primary;
4. downstream propagation across the next three trailers is included;
5. intervention magnitude is a smaller penalty; and
6. ties prefer short, then medium, then long.

This inner selection is frozen configuration, not a learning policy, and it must be logged (`candidate_scores`, failures, chosen variant). It compares only the three realizations of the same abstract action. The outer paired rollout still compares the resulting path-stretch action with speed or no-op.

Because this is an oracle-like macro action, experiments must report it. An ablation that chooses variants by geometry-only clearance is advisable to measure how much performance comes from internal lookahead.

The geometry-only selector is a required reported ablation. It is not the
default realization policy.

## 10. Anchors and the state vector

### 10.1 Anchor identity

An anchor is a typed, immutable role binding, not a raw vector index:

```text
LeaderFollowerAnchor(resource_id, leader_id, follower_id)
AircraftResourceAnchor(resource_id, aircraft_id)
FlowAnchor(resource_id, ordered_flight_ids)
```

`anchor_id` is derived from type, resource, bound IDs, scenario state version, and epoch. This prevents applying a feature/action computed for an obsolete ordering.

### 10.2 Canonical trajectory queries

All features must derive from a single query layer over `TrajectoryVariant`:

- position/altitude/CAS at absolute time;
- station at absolute time;
- ETA at merge/final/threshold;
- earliest feasible ETA under envelope;
- maximum-delay ETA under speed-only action;
- maximum-delay ETA under allowed stretch; and
- future crossing order.

Never compute one feature from compressed JSON points and another from a dense SIMAP array. That would introduce internally inconsistent anchors.

### 10.3 Minimum-time and capacity maps

For each live aircraft/resource, cache a `ReachabilityMap` keyed by state/variant/action-envelope hash:

```text
eta_nominal_s
eta_earliest_s
eta_latest_speed_s
eta_latest_path_s
speed_capacity_s = eta_latest_speed - eta_nominal
path_capacity_s = eta_latest_path - eta_nominal
```

Version-1 capacities are local action capacities. At a station-crossing epoch,
`eta_latest_speed_s` is the latest ETA among feasible speed bands available at
that station, and `eta_latest_path_s` is the ETA of the selected best feasible
short/medium/long realization available at that station. Capacity is zero when
the lever is ineligible, exhausted, or has no feasible candidate. Do not include
actions at later stations or combine multiple remaining interventions in one
capacity value.

The earliest map uses the maximum feasible profile but never violates configured speed caps. Even though action commands cannot speed the aircraft up, earliest feasible time is still needed to define slack and commitment. It is a counterfactual reachability quantity, not an available speed-up action.

Version 1 can compute times by numerical integration over the station grid:

```text
dt = integral(ds / max(v_ground_alongtrack(s), epsilon))
```

SIMAP-compiled time arrays are preferred when available. Cache every result; state-vector derivation must not rerun a nonlinear optimizer.

### 10.4 Leader-follower ordering and spacing

Version 1 implements the resource graph interface but configures only the
landing-runway threshold as a sequencing resource. Merge and
procedure-derived resources are later graph extensions and must not be inferred
implicitly from the medoid geometry.

For each resource, sort live arrivals by the active current variant's predicted
ETA with stable `flight_id` tie-breaking. In this section, `nominal ETA` means
that current no-further-action prediction; it does not mean the original
unmodified cluster-template ETA. Create edges between adjacent flights.

For edge `(L,F)`:

```text
predicted_interval_s = ETA_F - ETA_L
required_interval_s = separation_table(L,F,resource)
spacing_deviation_s = predicted_interval_s - required_interval_s
required_delay_s = max(0, -spacing_deviation_s)
```

Version 1 uses a configurable homogeneous 90-second threshold interval for both
tests and initial experiments, equivalent to a nominal service rate of 40
arrivals/hour. Label the homogeneous-A320 assumption in every artifact and
report. A wake-category pair table replaces it only after trustworthy aircraft
type/wake data is introduced.

### 10.5 Feature schema

Create a versioned `FeatureSchema` with ordered names, units, normalization, bounds, and missingness masks. Return both a named record for debugging and a `float64` vector for learning.

Initial leader-follower vector:

```text
spacing_deviation_s
abs_spacing_deviation_s
required_delay_s
predicted_interval_s
required_interval_s
follower_time_to_resource_s
leader_time_to_resource_s
follower_distance_to_resource_m
leader_distance_to_resource_m
follower_cas_kts
follower_cas_lower_kts
follower_cas_margin_kts
speed_capacity_s
path_capacity_s
required_delay_over_speed_capacity
required_delay_over_path_capacity
commitment_fraction
intercept_or_final_gate_flag
remaining_action_station_fraction
local_flow_count
pressure_ratio
trailing_min_spacing_margin_s
cluster_id_embedding/index (kept categorical outside XCSR intervals if possible)
```

Use explicit masks/flags for undefined ratios; do not encode infinity or NaN
into the learning vector. Version 1 divides by `max(capacity_s, 1.0 s)`, clips
capacity ratios to `[0, 10]`, and sets the corresponding undefined mask when
raw capacity is at most zero. The named diagnostic record preserves raw values.

`pressure_ratio` uses a half-open 10-minute window `[t, t + 600 s)`: count live,
uncompleted arrivals with nominal threshold ETA in that window and divide by
`600 / required_interval_s`. Under the homogeneous 90-second default, the
denominator is `600 / 90` slots. Preserve the raw count, capacity, and window in
the named diagnostic record.

`commitment_fraction` is a fixed transparent index in `[0,1]`. Its schema stores
the weights and normalization constants and exposes all primitive components:

```text
time_component = clip(1 - time_to_threshold_s / 1200, 0, 1)
freedom_remaining = 0.5 * remaining_station_fraction
                  + 0.5 * remaining_intervention_budget_fraction
freedom_component = 1 - freedom_remaining
gate_component = intercept_or_final_gate_flag
commitment_fraction = (time_component + freedom_component + gate_component) / 3
```

Here `remaining_station_fraction` is unused eligible action locations ahead
divided by the template's total action locations, and
`remaining_intervention_budget_fraction` is remaining non-no-op budget divided
by three (two speed changes plus one stretch). Values and masks are computed
from canonical state, not inferred from the action log by learning code.

### 10.6 Accuracy contract

“Accurate” means reproducible from canonical state and bounded against independent or analytic references:

- positions and profiles interpolate on one station/time mapping;
- ETA integration error is bounded on constant-speed analytic fixtures;
- leader/follower order matches brute-force crossing times;
- capacity maps equal explicit best-current-action rollouts within tolerance;
- features are invariant under fork before either branch changes;
- the parent vector is unchanged after child rollout;
- vector ordering/schema hash is stable; and
- every raw feature includes unit/provenance diagnostics.

## 11. Evaluation and paired rollout support

Implement a native semi-local outcome matching the tutorial:

```text
y = w1 * bound-pair spacing score
  + w2 * next-k-trailers propagation score
  + w3 * intervention count/magnitude penalty
  + w4 * small global throughput/runway residual
```

Initial defaults: `k=3` and weights `1 : 1 : 0.3 : 0.1`.

Separation scoring uses a bounded, continuous piecewise function of
`r = predicted_interval_s / required_interval_s`:

```text
r <= 0.00          -> -1.0
0.00 < r < 1.00    -> -1.0 + 2.0*r
1.00 <= r <= 1.25  -> +1.0
1.25 < r < 2.00    -> +1.0 - 2.0*(r - 1.25)
r >= 2.00          -> -0.5
```

Thus compression is graded through violation and margin erosion, a 0-25%
surplus receives full credit, and large inefficient gaps receive a bounded
penalty. A go-around or dynamically infeasible completion receives `-1.0` for
that pair. Pair and propagation terms use this same scale before applying the
declared weights. This avoids a catastrophic unbounded constant while making
the weight ratio meaningful.

The rollout normally ends one 90-second nominal slot after the third trailer's
threshold crossing. If fewer than three trailers exist, use every available
trailer and end one slot after the last one's crossing; if none exists, end one
slot after the bound follower crosses. Record the effective trailer count and
horizon in the outcome trace.

`rollout.paired` should:

1. accept a parent, selected action, contender, frozen policy, and horizon;
2. fork after action feasibility/covering decisions are complete;
3. couple RNG/exogenous streams;
4. apply one different initial action in each arm;
5. allow the same frozen policy to choose different later actions in diverged states;
6. forbid policy learning/structural mutation in children;
7. score both arms and return delta plus audit traces; and
8. leave committing the selected first action to the caller.

The scenario simulator should not itself update XCS rule statistics.

## 12. Detailed implementation sequence

### Phase 0: contracts and golden fixtures

1. Add `src/hailmary` and typed config/error/ID modules.
2. Define artifact schema versions, hashes, immutable array validation, and units.
3. Create tiny synthetic straight/merge fixtures with analytic crossing times.
4. Add architecture tests prohibiting core imports from `mcp_tools` and PPE agent code.

Exit criteria: artifacts round-trip; dynamic-content hashes and provenance state
IDs are deterministic; package boundary tests pass.

### Phase 1: clustering artifact

1. Implement manifest/catalog/raw ADS-B readers.
2. Normalize runway-relative direction and 50 NM clipping.
3. Project and resample to 128 stations.
4. Build standardized geometry features.
5. Implement deterministic HDBSCAN sweep/selection and final noise reassignment.
6. Implement true medoids and artifact export.
7. Compare results against existing PPE medoid code on shared fixtures.

Exit criteria: every accepted arrival has one final cluster; every medoid is an observed member; repeated builds with the same inputs/config have identical assignments and hashes.

> Also create a notebook to visualize the results - this will help the user verify the results by step.

### Phase 2: cluster templates

1. Rehydrate medoid raw tracks.
2. build station-aligned geometry, altitude, and smoothed speed;
3. create monotone commanded-speed and CAS provenance;
4. derive SIMAP/OpenAP envelopes;
5. simplify medoid geometry into stable reference-path control points;
6. compile/validate baseline through the SIMAP adapter; and
7. persist 16 speed-action stations, 8 path-stretch locations, and diagnostics.

Exit criteria: each cluster has one feasible template; endpoint/monotonic/envelope checks pass; historical-profile clamping stays within the 5 kt/5% budget; and observed-versus-compiled 50-NM timing error satisfies both the 10% and 30-second limits.

> Create a notebook to visualize some of the results so the user can verify everything is sound.

### Phase 3: deterministic scenario engine

1. Implement scenario definition/generator and named RNG streams.
2. Implement trajectory interpolation/inversion.
3. Implement release/station/resource event scheduling.
4. Implement immutable dynamic records, fork, and canonical hash.
5. Implement state export and optional scenario-manager schedule adapter.

Exit criteria: parent isolation, identical-fork determinism, event tie determinism, and analytic ETA tests pass.

### Phase 4: speed actions

1. Enumerate light/medium/heavy actions at 16 stations.
2. Enforce no-speed-up and envelope invariants pointwise.
3. Compile/cache modified variants from the current variant hash.
4. Update only the bound flight and its future events in a child branch.
5. Add action audit/provenance.

Exit criteria: property tests never find a command above baseline/current profile or outside the envelope; delay is nonnegative; stale actions fail; a third speed change is unavailable; and a composed second change is continuous at its splice.

### Phase 5: path-stretch actions

1. Build the all-medoid clearance field in runway-local coordinates.
2. Generate runway-away directions and short/medium/long doglegs at each of the 8 path-stretch locations.
3. solve target added distance, validate turns/bounds, and compile via SIMAP;
4. implement deterministic three-candidate inner selection; and
5. cache variants per template/anchor/config/weather.

Exit criteria: every valid template has exactly 8 distinct path-stretch locations; feasible variants meet length tolerance; chosen apex is runway-away; clearance score matches brute force; no path self-intersection; the abstract action returns exactly one logged realization at an eligible location; and a second stretch is unavailable.

### Phase 6: anchors and feature vectors

1. Implement resource ETA/reachability caches.
2. Build leader-follower/aircraft-resource/flow anchors.
3. Implement spacing, capacities, ratios, pressure, and commitment primitives.
4. Add versioned feature schemas and masks.
5. Cross-check every feature against brute-force reference implementations.

Exit criteria: analytic and randomized feature-oracle tests pass within declared tolerances; vectors are fork-stable and finite.

### Phase 7: outcomes and paired rollouts

1. Implement graded spacing and propagation scoring.
2. Implement frozen-policy protocol and later-epoch action loop.
3. Implement coupled twin/multi-candidate rollouts.
4. Add no-op and heavy-speed-contender examples from the tutorial.
5. Add trace artifacts with state/action hashes.

Exit criteria: same initial action produces zero paired delta; deliberately beneficial speed/stretch fixtures have expected delta sign; no child modifies parent or policy.

### Phase 8: validation, performance, and research audit

1. Regression-cross-check native conflicts with existing `ConflictEvaluator` through the adapter.
2. Run clustering stability and held-out assignment studies.
3. Audit generator feature correlations.
4. Benchmark fork, anchor-vector, and rollout throughput.
5. Profile caches before introducing persistent collections or multiprocessing.

Suggested initial performance targets on a development laptop:

- fork under 1 ms for 100 active flights after templates are loaded;
- anchored vector under 2 ms on cache hit;
- no SIMAP optimizer call during ordinary feature extraction; and
- deterministic results across process counts for offline artifact builds.

## 13. Test matrix

### Geometry and clustering

- projection round-trip and runway-origin checks;
- resampling endpoints and monotone stations;
- reversed-track normalization;
- HDBSCAN repeated-build determinism;
- noise reassignment and OOD flags;
- medoid is a member and minimizes the declared distance;
- cluster/runway partitions never cross thresholds.

### Profile and envelope

- finite-difference speed on constant-speed tracks;
- gaps/duplicates/outlier rejection;
- downstream commanded profile is non-increasing;
- CAS conversion provenance and zero-wind assumption;
- lower/upper envelope enforcement;
- baseline SIMAP feasibility and timing-error budget.

### Fork and engine

- fork dynamic-content hash initially equals the parent's while its provenance state ID differs;
- child mutation leaves parent byte-equivalent;
- two identical branches finish with equal dynamic-content hashes/traces;
- equal-time physical events are fully batched before one decision epoch;
- exogenous streams satisfy the coupling contract selected in Section 14;
- serialization/resume preserves future events exactly.

### Actions

- no-op identity;
- stale anchor rejection;
- station eligibility exists for the crossing epoch only and expires afterward;
- speed action never commands an increase;
- speed delay is nonnegative and heavier bands are not less delaying than lighter bands when the compared actions are feasible;
- no aircraft receives more than two speed changes or one path stretch;
- second speed-action cache identity and output depend on the current variant;
- every valid template has exactly 8 distinct path-stretch locations and offers no path-stretch action elsewhere;
- dogleg apex is farther from runway than base midpoint;
- short/medium/long added-distance tolerances;
- free-space side maximizes minimum other-medoid clearance;
- infeasible candidate cannot win; deterministic tie order;
- one path-stretch action commits only one variant.

### Anchored vector

- exact constant-speed ETA;
- spacing deviation sign convention;
- leader/follower reorder after delay;
- speed/path capacity equals the explicit best feasible action at the current station;
- zero-capacity ratios use mask/clip, never NaN/inf;
- pressure window boundary cases;
- commitment primitives and fixed composite reproduce the schema formula;
- vector schema order/hash stability;
- branch independence after divergent actions.

### Paired rollout

- identical action arms produce delta zero;
- only initial action differs at fork point;
- frozen policy object is not mutated;
- later actions may diverge with state;
- only the selected first real action is committed outside temporary arms;
- short flows use the last available trailer, or the bound follower when none exists, to set the horizon;
- parent state remains unchanged after rollout/scoring.

## 14. Decisions, assumptions, and open questions

Decisions made in this plan:

- implementation covers all eight phases, with Section 15 as the first integration checkpoint;
- package independence is enforced at imports and domain types;
- observed 50-NM crossings define exact scale-one releases; Phase 1 scaling uses recorded seeded additions or thinning;
- HDBSCAN uses a fixed scored sweep, with KMeans as a deterministic fallback;
- every flight receives a final medoid assignment, with noise/OOD provenance retained;
- medoid means an observed track minimizing unweighted 2-D pairwise station distance;
- all assigned flights use the same cluster path, altitude, and reference speed template;
- the SIMAP-compiled A320 profile is executable truth and must pass the 10% plus 30-second timing gate;
- the hot simulator is event-driven over compiled immutable variants;
- SIMAP is a template compiler/validator, not the multi-aircraft state container;
- equal-time physical events are batched before at most one decision epoch;
- state provenance identity is separate from reproducible dynamic-content hashing;
- the initial resource graph contains the runway threshold only, with homogeneous 90-second separation;
- 16 speed-action stations and 8 path-stretch locations exclude the last 4 NM by default;
- action eligibility lasts only for the exact station-crossing epoch;
- each aircraft may receive at most two speed changes and one path stretch;
- speed begins with light/medium/heavy reduction bands of 10/15/20 kt, respectively;
- no-speed-up applies to commanded profiles, while physical transients are logged;
- stretch begins with 2/5/9 NM added-distance targets;
- the three stretch variants form one full-local-outcome oracle-like macro action;
- capacity means the best feasible action available at the current station;
- pressure uses a 10-minute threshold window and commitment uses the fixed transparent composite;
- spacing outcome uses the bounded piecewise ratio score and available-trailer horizon; and
- anchored features are versioned, named, finite, and derived from one canonical query layer.

Assumptions that must be visible in every experiment:

- OpenAP A320 performance with 12,000 kg payload until type/wake data exists;
- zero/constant wind in version 1;
- CAS is derived, not observed, when built from the historical medoid;
- runway separation is homogeneous at 90 seconds in version 1; and
- generated default templates intentionally remove within-cluster path/speed variation.

Remaining choices not settled in the clarification pass:

1. Should the HDBSCAN composite emphasize balanced quality, coverage, or stability, and what exact noise/fragmentation ceilings should reject a sweep candidate?
2. Should exogenous randomness be materialized in the immutable scenario, generated by counter-keyed draws, or coupled only by cloned named RNG streams?
3. May the two speed changes and one path stretch occur in any chronological order, or must the stretch come before/after all speed changes?
4. What wake/separation table will replace the homogeneous A320 assumption when trustworthy type data becomes available?
5. What exact normalization should be used for intervention magnitude and the small global throughput residual in the outcome score?

None of these require integrating `hailmary` into existing advisors or scenario-manager state.

## 15. First integration slice

The smallest end-to-end slice that validates the architecture is:

1. one runway partition;
2. HDBSCAN clusters plus medoids;
3. two cluster templates with shared medoid profiles;
4. one leader and one follower on a threshold resource;
5. 16 speed-action stations and 8 path-stretch locations;
6. actions `{no_op, speed_light, speed_medium, speed_heavy, path_stretch}`;
7. forkable event-driven state;
8. leader-follower anchored vector with spacing and capacity ratios; and
9. paired rollout to the available-trailer outcome horizon.

This exercises every risky foundation—data alignment, medoid templates, action realization, state forking, ETA accuracy, and paired scoring—before scaling traffic and completing the later validation phases. It is an integration checkpoint within the approved eight-phase implementation, not the final delivery boundary. XCS, swap, hold, full-airport traffic, and advisor/API integration remain outside this simulator plan.
