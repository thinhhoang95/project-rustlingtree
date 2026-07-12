# Hailmary component reference

## 1. Purpose and package rule

This reference maps the implementation under `src/hailmary` to its role in the
system. It complements the architecture document by answering four practical
questions for each component: what it owns, which contract it exposes, what it
depends on, and who consumes it.

The package is intentionally independent of the legacy mutable scenario
manager. External-system knowledge belongs in `hailmary.adapters`; core models
and the simulator must remain usable without importing legacy manager code.

## 2. Cross-cutting foundation

| Component | Responsibility | Main consumers |
| --- | --- | --- |
| `config.py` | Frozen, validated defaults for clustering, templates, stretch geometry, scenarios, features, and outcomes | All pipeline and runtime layers |
| `errors.py` | Typed package failures for artifacts, configuration, actions, simulation, correlation gates, and empty clustering results | All layers and callers |
| `ids.py` | Canonical data conversion, JSON encoding, content hashes, stable IDs, and provenance-aware state IDs | Artifacts, actions, scenarios, rollouts |
| `_arrays.py` | Defensive conversion to read-only numeric arrays plus length and monotonicity validation | Template and artifact models |

`HailmaryConfig` groups the version-1 configuration. Code should accept a
specific sub-configuration when that is all it needs; this keeps dependencies
and serialized provenance narrow.

## 3. `hailmary.data`

### `manifest.py`

`DataManifest` and `DatasetResources` resolve a named dataset's catalog and raw
ADS-B paths. `load_manifest` validates the manifest, while
`resolve_dataset_resources` selects and resolves one dataset. This is the file
discovery boundary; downstream code should receive resolved paths or typed
records.

### `catalog.py`

`CatalogArrival` is the authoritative arrival record. It carries normalized
runway/identity/time information. `load_arrival_catalog` reads and validates the
catalog; `normalize_runway` makes runway partition IDs stable.

### `adsb.py`

`RawADSBTrack` stores chronological source observations. The loaders group CSV
rows and can restrict them to catalog flights. `reconstruct_terminal_entry`
interpolates a track's terminal-radius crossing. The clustering pipeline adds
the full runway-aware preparation and rejection audit.

## 4. `hailmary.geometry`

### `frame.py`

`LocalFrame` owns the runway-centered geographic projection used by clustering,
templates, and stretch geometry. Persist its projection parameters with
artifacts; never assume local XY coordinates are portable without the frame.

### `polyline.py`

This module provides duplicate removal, cumulative arc length, clipping,
orientation, fixed-count resampling, terminal alignment, and conversion between
historical and executable station order. `ExecutablePolyline` and
`ResampledPolyline` make orientation explicit.

### `dogleg.py`

`construct_runway_away_dogleg` creates smooth stretch paths. Supporting
functions enforce segment clearance, turn geometry, forward progress,
self-intersection avoidance, and exact added-distance targets. The action layer
owns candidate selection; this module owns geometry only.

## 5. `hailmary.clustering`

### `features.py`

`ShapeFeatureTransform` stores the fitted normalization contract.
`fit_shape_features` creates training features; `transform_shape_features`
applies the same transform to held-out tracks. The transform's station and
feature counts are artifact invariants.

### `hdbscan_runner.py`

`run_hdbscan_sweep` evaluates deterministic HDBSCAN candidates and returns a
`ClusteringSelection` with labels, probabilities, candidate metrics, selected
parameters, library versions, and fallback status. `HDBSCANSelectionConfig`
adapts the public clustering configuration. KMeans fallback lives here because
it is part of model selection, not assignment.

### `medoid.py`

`MedoidRecord` stores observed medoid identity, path, membership count,
dispersion, and acceptance radius. `compute_cluster_medoids` selects one record
per non-noise cluster using station-wise path distance.

### `assignment.py`

`FlightAssignment` records cluster ID, probability, medoid distance, OOD flag,
and assignment method. Training assignments use `assign_all_flights`. Held-out
assignment reconstructs membership when portable training data exists and falls
back to nearest medoid for noise or legacy artifacts.

### `artifact.py`

`ClusterLibrary` is the canonical runway-partitioned artifact.
`PredictionTrainingData` retains portable estimator reconstruction inputs.
`build_cluster_library` orchestrates features, model selection, canonical
labels, medoids, assignments, and content hashing. `read`, `write`, and JSON
methods are the stable persistence boundary.

### `pipeline.py`

This is the raw ADS-B orchestration layer. Important types are
`PreparedADSBTrack`, `TrackRejection`, `ADSBPreparationResult`, and
`ADSBClusterBuildResult`. `prepare_adsb_tracks_for_clustering` performs the
audited observation-to-shape transformation;
`build_cluster_library_from_adsb` adds the cluster artifact.

## 6. `hailmary.templates`

### `models.py`

The core artifact types are:

- `ResourceCrossing`: station and time for a shared resource;
- `VariantDiagnostics`: feasibility and compilation measurements;
- `ActionProvenance`: parent and realized action metadata;
- `TrajectoryVariant`: complete immutable executable path/profile;
- `ActionStation`: a preselected operational location; and
- `ClusterTemplate`: baseline variant, cluster context, and action stations.

Model constructors enforce array length, station/time monotonicity, speed
envelopes, crossing consistency, unique metadata, and content IDs. Prefer
constructing a new variant over bypassing these checks.

### `speed.py`

This module converts raw position/time data to robust ground speed, converts
CAS/TAS, evaluates envelopes, clamps references under explicit quality gates,
forms monotone commands, and integrates elapsed time. It is compilation logic,
not runtime intervention logic.

### `compiler.py`

`MedoidTrack` is the observed profile input. `TemplateCompiler` converts it into
a dense `ClusterTemplate`, delegating aircraft/path validation through the
`VariantValidator` protocol. `select_action_stations` implements the fixed
16-speed/8-stretch station contract. `compile_kinematic_template` is the
validator-free convenience boundary used in tests or controlled ablations.

### `store.py`

`TemplateStore` is a deterministic in-memory registry. It supports lookup and
selection without making artifact models mutable.

## 7. `hailmary.adapters`

### `simap.py`

`SIMAPAdapter` is the physical-model boundary. It caches the resolved A320
context/envelope, simplifies a dense path to SIMAP's public reference-path
contract, reconstructs commands, runs coupled replay, performs threshold
calibration, and returns a physically compiled `TrajectoryVariant` with
diagnostics.

The adapter also owns two Hailmary-local corrections: wrap-safe curvature and
distance/time-persistent bank-limit evaluation. Custom aircraft configurations
must supply a matching backend; otherwise diagnostics disclose that only static
checks were possible.

### `scenario_manager.py`

`HailmaryScheduleView` translates current Hailmary state into the arrival
schedule shape expected by compatibility consumers. It is read-only and should
not acquire simulation authority.

## 8. `hailmary.scenario`

### `models.py`

`ResourceDefinition`, `ActionStationDefinition`, `ResourceCrossingDefinition`,
`FlightDefinition`, `MaterializedExogenousEvent`, and `ScenarioDefinition`
define the immutable experiment input. `freeze_variant`, `freeze_weather`, and
`freeze_payload` defensively snapshot caller-owned data. Referential integrity
and the definition hash are established during construction.

### `generator.py`

`ScenarioGenerator` converts `FlightGenerationSpec` records and templates into
a definition. The same module implements factorial conditions, physical factor
realization, provisional commitment measurement, and correlation audits.
`build_scenario_definition` is the functional convenience entry point.

## 9. `hailmary.simulator`

### `events.py`

`EventKind`, `ScheduledEvent`, `DecisionEpoch`, and `EventBatchResult` define the
event protocol. The priority table and `sort_key` are part of deterministic
semantics; changing them can change controller-visible states.

### `interpolation.py`

`MonotoneTrajectory` wraps the inverted station/time arrays and exposes sampling
and station/time conversion. `TrajectorySample` is the physical state returned
to runtime queries. All runtime code should use this layer rather than ad hoc
`numpy.interp` with potentially reversed axes.

### `state.py`

`FlightDynamic` stores lifecycle, current variant, clock origins, predicted and
actual crossings, counters, and commitment controls. `SimulationState` stores
the whole branch. `make_initial_state`, `evolve_state`, and `fork_state` are the
only normal state-construction paths. RNG helpers serialize generator state for
snapshots and exact forks.

### `hashing.py`

This module canonicalizes scenario and dynamic state for definition hashes,
dynamic content hashes, and provenance state IDs. Keep derived caches and
lineage-only fields out of content equivalence unless they change experiment
behavior.

### `engine.py`

`Simulator` owns event advancement, batching, lifecycle transitions,
disturbance application, decision epochs, snapshots/resume, sampling, forks,
policy-driven execution, branch-local variant installation, causal replacement,
action recording, and metrics. Helper event construction maps a variant's
station/time contract into absolute scenario events.

## 10. `hailmary.actions`

### `models.py`

`ActionLever` identifies no-op, speed, and path-stretch levers.
`ActionCandidate` is bound to an epoch and validates freshness before use.
`ActionRealization` is the audited outcome with variant ID, delay, magnitude,
and diagnostic metadata.

### `catalog.py`

`ActionCatalog` enumerates feasible candidates from the current event batch.
`apply_action` validates freshness and budgets, dispatches realization, installs
and splices the child, increments counters, and records the action. It also
builds the default coupled stretch oracle and automatic SIMAP validator when
the current variant provenance supports them.

### `speed.py`

`realize_speed_variant` applies an anchored downstream command reduction,
validates it, and preserves action provenance. It composes with an existing
speed action and respects the reference/envelope constraints.

### `stretch.py`

`PathStretchRealizer` builds, compiles, evaluates, and selects short/medium/long
variants. `StretchCandidateResult`, `StretchOutcomeEvaluation`, and
`StretchRealization` retain candidate-level diagnostics. `medoid_polylines`
extracts avoidance references from other templates.

### `splice.py`

`preserve_compiled_live_prefix` combines the exact already-flown physical
prefix with the replay-compiled child suffix. It exists because replaying a
whole action path can slightly alter the aircraft state at the live action
instant, which would violate causal continuity.

## 11. `hailmary.features`

### `anchors.py`

This module predicts active resource arrivals and derives stable
leader/follower, aircraft/resource, and flow anchors. Resource helpers locate a
crossing station and ETA on the current variant.

### `minimum_time.py`

`ReachabilityMap` describes nominal ETA plus speed/path delay capacity.
Simulator helpers derive capacity from the remaining action catalog without
running the action optimizer.

### `schema.py`

`FeatureField`, `FeatureSchema`, and `FeatureVector` make feature order, masks,
and version explicit. `leader_follower_feature_schema` is the canonical
version-1 schema factory.

### `state_vector.py`

This module computes spacing/capacity ratios, pressure, commitment components,
and the complete leader/follower vector. `simulator_state_vector` is the runtime
entry point; `derive_leader_follower_state_vector` is the pure calculation
boundary.

## 12. `hailmary.evaluation`

### `spacing.py`

`HomogeneousSeparationTable` supplies the version-1 90-second rule.
`compute_spacing` produces interval, margin, and required delay;
`spacing_ratio_score` converts that relationship into a bounded score.

### `conflict.py`

`TimedTrajectory` is an absolute-time path. `detect_pair_conflicts` and
`detect_conflicts` solve continuous-time overlap on piecewise-linear segments,
then return normalized `ConflictRecord` and `ConflictSummary` objects.

### `outcome.py`

`OutcomeCohort` freezes the pair and trailers. `SimulatorOutcomePlan` freezes
baseline crossings and horizon. `score_semi_local_outcome` combines pair,
propagation, intervention, and throughput terms; simulator helpers gather the
branch-local inputs and intervention summary.

## 13. `hailmary.rollout`

### `policy.py`

`FrozenPolicy` is the policy protocol. `NoOpPolicy` and
`CallableFrozenPolicy` are standard implementations. `policy_fingerprint`
supports mutation checks during counterfactual comparisons.

### `paired.py`

`paired_rollout` is the engine-agnostic fork/apply/run/score coordinator.
`paired_simulator_rollout` binds it to Hailmary's action and outcome helpers.
`PairedArmTrace` records each branch; `PairedRolloutResult` carries both arms,
their score delta, common policy fingerprint, parent hash, and horizon.

## 14. `hailmary.cli`

| Command | Module | Input | Output |
| --- | --- | --- | --- |
| `hailmary-build-clusters` | `build_clusters.py` | Prepared NPZ/JSON XY tracks plus partition/projection | Canonical cluster-library JSON |
| `hailmary-build-templates` | `build_templates.py` | Cluster JSON plus raw medoid profiles | Template manifest and variant NPZ files |
| `hailmary-simulate` | `simulate.py` | Scenario JSON referencing variant NPZ files | Deterministic simulation trace JSON |

The CLI persistence helpers use deterministic JSON/NPZ forms and validate data
again when reading. Python APIs expose richer results, especially raw ADS-B
rejection audits and in-process factorial/rollout workflows.

## 15. Tests and executable documentation

Focused tests live in `tests/hailmary` and are organized around the package
boundaries above. The most important cross-layer suites cover architecture
independence, raw ADS-B preparation, cluster artifact round trips, SIMAP replay,
template gates, event ties, forks, actions, live splices, exogenous shifts,
anchors, state vectors, conflicts, outcomes, and paired rollouts.

`notebooks/hailmary/01_clusters.ipynb` and
`notebooks/hailmary/02_templates.ipynb` provide visual inspection of the two
offline artifacts. They are validation aids, not alternate implementations.

