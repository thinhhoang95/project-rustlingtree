# Scenario Manager

The scenario manager exposes KDFW arrival and departure demand as API-ready schedules. It combines catalog data, compressed ADS-B tracks, and precomputed SIMAP arrival artifacts into a stable runtime view for downstream agents and clients.

The current design is centered on this flow:

1. Catalog generation creates arrival/departure events and route fix sequences.
2. ADS-B compression creates compact historical trajectories for departures.
3. `scenario-manager-precompute-artifact` creates SIMAP base-route trajectories for arrivals.
4. `ScenarioManager` loads the configured resources and serves `/health`, `/departures`, `/arrivals`, and `/diff`.
5. Future intervention logic is expected to keep the base trajectory immutable and apply an in-memory diff of agent interventions on top of selected flights.

## Important Current State

`ScenarioManager` already has a `diff` field and a `/diff` endpoint, and `arrival_schedule()` already calls `_apply_diff()` before it normalizes each arrival payload.

However, intervention patching is not implemented yet. `_apply_diff()` currently returns the base artifact unchanged, and `diff` starts as an empty in-memory list. So the answer to "is there a diff structure for interventions?" is:

Yes, the structure and API hook exist, but they are placeholders today. No current code overrides flight trajectories from `diff`.

This matters because precomputed arrival artifacts are base trajectories. They are not guaranteed to be orderly, safe, or vertically feasible. The artifact payload includes `simulation.success` and `simulation.message`, and the artifact manifest summarizes simulation failures. The intended intervention layer should let an agent add commands that replace or adjust selected flight trajectories while preserving the base artifact as the baseline.

## Code Map

Primary files:

- `src/mcp_tools/scenario_manager/manager.py`: runtime resource loader and schedule assembler.
- `src/mcp_tools/scenario_manager/api.py`: FastAPI route definitions.
- `src/mcp_tools/scenario_manager/server.py`: local uvicorn entrypoint.
- `src/mcp_tools/scenario_manager/models.py`: resource configuration and response models.
- `src/mcp_tools/scenario_manager/resources.py`: CSV, JSON, and JSONL loaders.
- `src/mcp_tools/scenario_manager/precompute_artifact.py`: SIMAP arrival artifact generator.
- `src/mcp_tools/scenario_manager/wait_atc_point.py`: ATC wait/decision point detection.
- `data_manifest.json`: default resource manifest consumed by `ScenarioResourceConfig.default()`.

For the detailed SIMAP request construction and FMS bichannel replay flow used
by `scenario-manager-precompute-artifact`, see
`docs/simap/fms_bichannel_artifact_flow.md`.

The installed command entrypoints are defined in `pyproject.toml`:

```sh
scenario-manager-api
scenario-manager-precompute-artifact
adsb-trajectory-compress
```

## Resource Configuration

`ScenarioResourceConfig.default()` loads `data_manifest.json` from the project root and selects the single entry with `"default": true`.

The active manifest entry must provide:

- `landings_and_departures`: CSV catalog of arrival and departure events.
- `fix_sequences`: CSV catalog of arrival route fix sequences.
- `simap_arrival_trajectories`: JSONL file produced by `scenario-manager-precompute-artifact`.
- `adsb_compressed_trajectories`: JSONL file produced by ADS-B compression.

Optional manifest fields:

- `simap_arrival_artifact_manifest`: JSON summary produced next to the SIMAP arrival artifact.
- `adsb_compressed_metadata`: JSON metadata produced by ADS-B compression.

The current default manifest points to:

```json
{
  "landings_and_departures": "data/adsb/catalogs/2026-04-01_landings_and_departures.csv",
  "fix_sequences": "data/adsb/catalogs/2026-04-01_fix_sequences.csv",
  "simap_arrival_trajectories": "data/artifacts/simap_arrival_flights.jsonl",
  "simap_arrival_artifact_manifest": "data/artifacts/manifest.json",
  "adsb_compressed_trajectories": "data/adsb/compressed/adsb_compressed_flights.jsonl",
  "adsb_compressed_metadata": "data/adsb/compressed/metadata.json"
}
```

Only one manifest entry may be marked as default.

## Input Resources

### Events CSV

Loaded by `load_events()`.

Required columns:

- `flight_id`
- `callsign`
- `icao24`
- `operation`
- `runway`
- `event_time`
- `event_time_utc`

Normalization:

- `flight_id`, `callsign`, `icao24`, and `runway` are stripped.
- `operation` is lowercased.
- `event_time` is parsed as an integer Unix timestamp.

`operation == "arrival"` drives `/arrivals`; `operation == "departure"` drives `/departures`.

### Fix Sequences CSV

Loaded by `load_fix_sequences()`.

Required columns:

- `flight_id`
- `first_time`
- `last_time`
- `fix_sequence`
- `fix_count`

Normalization:

- `flight_id` is stripped.
- `fix_sequence` is filled with an empty string when missing.
- `first_time`, `last_time`, and `fix_count` are parsed as integers.

At runtime the manager builds an index by `flight_id`.

### Compressed Flight JSONL

Loaded by `load_compressed_flights()`.

Each non-empty line must be a JSON object with a non-empty `flight_id`. The loader returns a dictionary keyed by `flight_id`. If duplicate flight IDs exist, the last line wins.

The manager loads two JSONL resources:

- `simap_arrival_flights`: precomputed SIMAP base-route arrival artifacts.
- `adsb_compressed_flights`: compressed ADS-B tracks used for departures.

## Precomputed Arrival Artifacts

`scenario-manager-precompute-artifact` writes SIMAP arrival base trajectories to:

```text
data/artifacts/simap_arrival_flights.jsonl
```

and writes a summary manifest to:

```text
data/artifacts/manifest.json
```

The artifact type is:

```text
simap_fms_bichannel_base_route_arrivals
```

### Generation Inputs

The precompute step uses:

- arrival rows from the events catalog,
- route fix sequences from the fix sequence catalog,
- raw ADS-B tracks,
- airport-related fixes from `data/kdfw_procs/airport_related_fixes.csv`,
- SIMAP tactical/FMS planning,
- trajectory compression tolerances.

Default generation parameters include:

- lateral tolerance: `100.0` meters
- altitude tolerance: `50.0` meters
- final fix target distance: `7.0` NM
- final fix cross-track tolerance: `0.15` NM
- FMS time step: `2.0` seconds
- top-of-descent tolerance: `25.0` meters
- max top-of-descent iterations: `24`
- raw ADS-B split gap: `1500` seconds

### Base Route Construction

For each arrival, the precompute step:

1. Reads the catalog fix sequence and appends the normalized runway token when needed.
2. Detects an ATC wait/decision point using the arrival route, runway, and KDFW capture polygons.
3. Keeps the original route prefix through that ATC point.
4. Selects a final fix aligned with the extended runway centerline, near the configured target distance.
5. Builds a base route:

```text
original route prefix -> selected final fix -> runway
```

This route is saved as both top-level arrival metadata and as the nested `base_route` object.

### ATC Wait Point Detection

`detect_wait_atc_point()` classifies a route into one of four arrival clusters:

- `NE`
- `NW`
- `SW`
- `SE`

It tries to find the route crossing at the configured gate radius, currently 50 NM by default. If no gate crossing is found, it falls back to classifying from the first route fix. It then selects the last route fix inside that cluster's capture polygon.

When successful, the wait point payload includes:

- `source`
- `identifier`
- `lat`
- `lon`
- `lateral_path_token`
- `route_index`
- `distance_nm`
- `selection_method`
- `arrival_cluster`
- `gate_cluster`
- `gate_radius_nm`
- `gate_lat`
- `gate_lon`
- `gate_classification_fallback`
- `capture_margin_nm`

If no ATC decision point can be detected, the arrival artifact is skipped with reason `missing ATC decision point`.

### Final Fix Selection

The final fix selector:

- normalizes the runway as `RW##` plus optional `L`, `C`, or `R`;
- derives true runway heading from the reciprocal runway when available;
- excludes runway fixes as candidates;
- keeps only fixes in front of the runway along the extended centerline;
- filters by cross-track tolerance;
- chooses the candidate closest to the configured target distance, then closest to centerline.

If no final fix is found, the arrival is skipped with a `ValueError` reason.

### Simulation and Compression

The precompute step seeds each simulation from the raw ADS-B point closest to the first base-route fix. It then builds a SIMAP tactical request and runs `plan_fms_bichannel()`.

The simulated trajectory is compressed into `points` using lateral and altitude breakpoints. The point layout is:

```json
["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"]
```

`breakpoint_mask` uses:

```json
{
  "lateral": 1,
  "altitude": 2
}
```

A point with mask `3` is both a lateral and altitude breakpoint.

### Artifact Payload Shape

Each generated arrival artifact contains:

- `flight_id`
- `callsign`
- `icao24`
- `runway`
- `route_type`
- `fix_sequence`
- `fix_count`
- `columns`
- `breakpoint_mask_bits`
- `points`
- `lateral_breakpoint_times`
- `altitude_breakpoint_times`
- `wait_atc_point`
- `baseline_final_fix`
- `base_route`
- `first_time`
- `last_time`
- `raw_point_count`
- `compressed_point_count`
- `lateral_tolerance_m`
- `altitude_tolerance_m`
- `simulation`

The nested `simulation` object contains:

- `success`
- `message`
- `max_abs_cross_track_m`
- `max_abs_track_error_rad`
- `final_threshold_error_m`
- `lateral_guidance`

`simulation.success == false` does not prevent the artifact from being served. It means the base route was generated but the planned FMS profile was not feasible under the current assumptions. The downstream intervention layer is expected to resolve these cases by issuing additional commands and replacing or adjusting the affected trajectories.

### Artifact Manifest

`manifest.json` summarizes:

- source paths,
- generated and skipped counts,
- skipped-arrival reasons,
- simulation success/failure counts,
- simulation failure messages,
- wait-point success rates and cluster counts,
- aggregate raw/compressed point counts,
- generation parameters,
- per-flight artifact status.

The current manifest format is useful for quickly identifying how many base trajectories are infeasible before any intervention layer is applied.

## Runtime Manager

`ScenarioManager.__init__()` loads all resources eagerly:

- events,
- fix sequences,
- SIMAP arrival artifact JSONL,
- ADS-B compressed flight JSONL,
- optional SIMAP artifact manifest,
- optional ADS-B compressed metadata.

It also initializes:

```python
self.diff: list[dict[str, Any]] = []
```

and builds a fix-sequence index by `flight_id`.

### Arrivals

`arrival_schedule()`:

1. Iterates over catalog arrivals.
2. Looks up the matching fix sequence.
3. Looks up the matching SIMAP arrival artifact.
4. Skips the arrival if either lookup is missing.
5. Copies the artifact payload.
6. Calls `_apply_diff(payload)`.
7. Normalizes runtime fields.
8. Sorts arrivals by `time_at_first_fix`, then `flight_id`.

Runtime fields added or normalized include:

- `time_at_first_fix`
- `time_at_first_fix_utc`
- `time_at_last_event`
- `time_at_last_event_utc`
- `runway`
- `route_type`
- `fix_sequence`
- `fix_count`
- `final_fix`
- `baseline_final_fix`
- `base_route`
- `atc_wait_point`
- `wait_atc_point`

The manager prefers artifact fields over catalog fallback fields. For example, `time_at_first_fix` is derived from artifact `first_time` when present, otherwise from the fix sequence catalog.

For route metadata, the manager prefers:

1. `base_route.fix_sequence`,
2. payload `fix_sequence`,
3. `base_route.lateral_path`,
4. catalog `fix_sequence`.

For `fix_count`, it prefers:

1. `base_route.fix_count`,
2. payload `fix_count`,
3. count derived from the runtime route sequence,
4. catalog `fix_count`.

### Departures

`departure_schedule()`:

1. Sorts catalog departures by `event_time`, then `flight_id`.
2. Looks up the matching compressed ADS-B trajectory.
3. Skips the departure if no trajectory is found.
4. Copies the compressed trajectory payload.
5. Adds catalog metadata:

- `flight_id`
- `callsign`
- `icao24`
- `departure_time`
- `departure_time_utc`
- `runway`

Departures currently use ADS-B compressed tracks directly, not SIMAP base-route artifacts.

### Health

`health()` reports:

- loaded event, arrival, departure, fix-sequence, and trajectory counts,
- missing arrival fix sequences,
- missing arrival SIMAP artifacts,
- missing departure ADS-B trajectories,
- skipped departure count from the SIMAP artifact manifest when available,
- active resource paths.

The health response keeps some legacy-compatible field names, such as both `artifact_flights_count` and `simap_arrival_flights_count`.

## API

Run the API with:

```sh
scenario-manager-api
```

The server binds to:

```text
http://127.0.0.1:8000
```

### GET `/health`

Returns the `health()` payload. Use this first to verify that the configured files loaded and to check missing-resource counts.

### GET `/departures`

Returns the departure schedule assembled from the events catalog and ADS-B compressed trajectories.

Flights without compressed ADS-B trajectories are omitted.

### GET `/arrivals`

Returns the arrival schedule assembled from the events catalog, fix sequence catalog, and SIMAP arrival artifact JSONL.

Flights without a fix sequence or SIMAP arrival artifact are omitted.

The returned trajectory is the base artifact today because intervention patching is not yet implemented.

### GET `/diff`

Returns `ScenarioManager.intervention_diff()`, currently a shallow copy of `self.diff`.

Today this is normally:

```json
[]
```

There is no route that mutates `diff` yet.

## Intervention Diff

The intended intervention model is:

```text
served arrival = base artifact + intervention diff
```

The base artifact should remain a reproducible precomputed baseline. Agent commands should be captured as intervention records, and selected records should override or replace fields for specific flights when `/arrivals` is assembled.

What exists today:

- `ScenarioManager.diff` is an in-memory list of dictionaries.
- `/diff` exposes that list.
- `arrival_schedule()` calls `_apply_diff()` before adding runtime arrival fields.
- `_apply_diff()` is the intended patching hook.

What does not exist yet:

- no intervention schema,
- no endpoint to add, update, or remove interventions,
- no persistence for interventions,
- no logic to select intervention records by `flight_id`,
- no code that overrides `points`, `fix_sequence`, `base_route`, timing, or feasibility metadata.

### Recommended Future Diff Shape

No schema is enforced by current code. A practical future intervention record should include enough information to audit why a base trajectory changed and to replace the affected trajectory deterministically.

Recommended fields:

```json
{
  "id": "intervention-0001",
  "flight_id": "NKS220M1a91e6e",
  "created_at_utc": "2026-05-13T00:00:00Z",
  "source": "agent",
  "reason": "resolve vertical infeasibility and restore orderly flow",
  "command": {
    "type": "vector_or_route_adjustment"
  },
  "overrides": {
    "route_type": "intervention-route",
    "fix_sequence": "MUZZY>BEREE>...>RW18R",
    "fix_count": 8,
    "points": [],
    "lateral_breakpoint_times": [],
    "altitude_breakpoint_times": [],
    "first_time": 0,
    "last_time": 0,
    "simulation": {
      "success": true,
      "message": "intervention trajectory feasible"
    }
  },
  "base": {
    "route_type": "base-route",
    "baseline_final_fix": {}
  }
}
```

The eventual `_apply_diff()` implementation should be explicit about patch semantics:

- match interventions by `flight_id`;
- choose a deterministic record when multiple interventions exist for one flight;
- preserve `base_route` and `baseline_final_fix` for auditability;
- override trajectory-bearing fields atomically, especially `points`, `columns`, breakpoints, timing, route metadata, and simulation metadata;
- reject or ignore malformed records rather than serving partial corrupted trajectories.

For safety, the diff should probably be append-only at first, with later records superseding earlier records for the same flight.

## Expected Agent Workflow

The intended agent workflow is:

1. Read `/arrivals` and identify base trajectories with `simulation.success == false`, flow conflicts, spacing issues, or route-ordering problems.
2. Use the base artifact as the initial condition and audit baseline.
3. Generate additional commands for selected flights.
4. Replan those selected flights into feasible, orderly, safe trajectories.
5. Store each intervention as a diff record.
6. Serve `/arrivals` as base artifacts plus matching intervention overrides.
7. Serve `/diff` as the audit trail of active interventions.

Until `_apply_diff()` and mutation endpoints are implemented, steps 5 through 7 are only partially wired.

## Regenerating Artifacts

Basic command:

```sh
scenario-manager-precompute-artifact
```

Useful options:

```sh
scenario-manager-precompute-artifact \
  --events-path data/adsb/catalogs/2026-04-01_landings_and_departures.csv \
  --fix-sequences-path data/adsb/catalogs/2026-04-01_fix_sequences.csv \
  --raw-adsb-dir data/adsb/raw \
  --fixes-csv data/kdfw_procs/airport_related_fixes.csv \
  --output-dir data/artifacts \
  --processes 8
```

For a small smoke test:

```sh
scenario-manager-precompute-artifact --limit 10 --processes 1
```

After regenerating, check:

- `data/artifacts/simap_arrival_flights.jsonl`
- `data/artifacts/manifest.json`
- `/health`
- `/arrivals`

## Common Failure Modes

`missing raw ADS-B flight`

The events catalog contains an arrival flight ID that is not present in the loaded raw ADS-B tracks.

`missing route fixes before runway`

The catalog fix sequence cannot produce a usable route before the runway.

`missing ATC decision point`

The route could not be associated with a KDFW cluster capture polygon.

`no final fix for runway within tolerance`

No non-runway fix was found near the extended runway centerline within the configured cross-track tolerance.

`simulation.success == false`

The base trajectory exists, but SIMAP could not build a feasible vertical profile under the current route, speed, altitude, and distance assumptions. This is the primary case the intervention layer is expected to address.

## Current Limitations

- Arrival artifacts are generated only for arrivals.
- Departures are served from compressed ADS-B tracks.
- Interventions are not applied yet.
- Interventions are not persisted yet.
- The API has no write endpoints.
- The diff list is process-local memory.
- Runtime resource loading is eager, so missing or malformed configured files fail startup.
- Infeasible base-route simulations are still served unless the consumer filters or intervenes.

## Testing Pointers

The scenario manager behavior is covered by:

```sh
pytest tests/test_scenario_manager.py
pytest tests/test_precompute_artifact.py
pytest tests/test_wait_atc_point.py
```

The tests verify:

- default manifest loading,
- departure schedule assembly,
- arrival schedule assembly from SIMAP artifacts,
- health missing-resource counts,
- API route exposure,
- artifact payload and manifest creation,
- final fix selection behavior,
- wait-point detection behavior.
