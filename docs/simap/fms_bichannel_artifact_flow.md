# FMS Bichannel Arrival Artifact Flow: A Complete Guide to the FMS Module

Arrival artifact here refers to the whole pipeline from the flight plan (described as fix sequence) with initial (or boundary) state, to the prescription of target speed, descent with minimum thrust, decelerate at around 10,000ft to 250kts, and slow down in full configuration to land at runway threshold. 

There are modes: 290kt cruise, 290kt descent, 250kt decel, then slow down to reference landing speed (just above stall speed a bit).

The "base" (or pre-intervened) flight path will follow the fix sequence extracted by the ADS-B data, but the ATC will take over when the aircraft enters one of the "ATC takeover recognition areas", described by 4 polygons for 4 different arrival directions. This is the Wait for ATC or ATC takeover point. Then from this takeover point, it will only add one Final fix (which is whatever fix lined up with the selected runway).

There is absolutely no guarantee that the base flight path is even feasible. 

This document explains how the scenario-manager precompute pipeline uses the
SIMAP FMS bichannel module to generate base-route arrival artifacts.

Primary codepaths:

- `src/mcp_tools/scenario_manager/precompute_artifact.py`
- `src/simap/fms_bichannel/core.py`
- `src/simap/fms/core.py`
- `src/simap/lateral_dynamics.py`
- `src/simap/path_geometry.py`
- `src/simap/nlp_colloc/tactical/builder.py`

The short version:

```text
catalog rows + raw ADS-B + fix catalog
  -> base-route construction
  -> route tokens resolved to waypoints
  -> ReferencePath
  -> raw ADS-B seed state
  -> tactical request
  -> FMSRequest
  -> FMSBiChannelRequest
  -> plan_fms_bichannel()
  -> full simulated trajectory
  -> compressed JSONL arrival artifact
  -> manifest summary
```

## Responsibilities

The pipeline is split into three layers.

### Scenario precompute layer

`src/mcp_tools/scenario_manager/precompute_artifact.py` is a batch artifact
generator. It does not implement aircraft dynamics directly. Its job is to:

- read arrival catalog rows and route fix sequences;
- read raw ADS-B tracks;
- choose a base route for each arrival;
- seed the simulation from observed ADS-B state;
- build the SIMAP request objects;
- run `plan_fms_bichannel()`;
- compress the simulated trajectory;
- write `simap_arrival_flights.jsonl` and `manifest.json`.

### Tactical request layer

`src/simap/nlp_colloc/tactical/builder.py` converts a high-level tactical
command into a `CoupledDescentPlanRequest`. This resolves route tokens into
waypoints, builds the continuous `ReferencePath`, initializes aircraft
configuration and performance models, and creates upstream and threshold
boundary conditions.

The precompute pipeline only uses this builder as a request factory. It does not
run the collocation solver.

### Bichannel FMS layer

`src/simap/fms_bichannel/core.py` combines:

- a longitudinal FMS profile from `src/simap/fms/core.py` or
  `src/simap/fms/holds.py`;
- a lateral path-following replay from `src/simap/lateral_dynamics.py`.

The longitudinal channel owns elapsed time, distance-to-go, altitude, and
airspeed. The lateral channel owns map position, heading, bank, cross-track
error, and track error.

The two channels share the same reference path and the same time grid.

The reference path is built before either channel can do useful work. It is the
geometric spine of the entire replay:

```text
fix sequence / base route
  -> resolved waypoint lat/lon sequence
  -> ReferencePath
  -> FMSRequest.reference_path
  -> longitudinal stationing and lateral map guidance
```

The longitudinal FMS does not compute map position, heading, or bank. It only
computes a schedule along this already-built path. The lateral channel then
turns that schedule into actual map motion.

## Input Resources

`precompute_artifacts()` receives these paths:

- `events_path`: landings/departures catalog CSV.
- `fix_sequences_path`: arrival route fix sequence CSV.
- `raw_adsb_dir`: raw ADS-B CSV directory.
- `fixes_csv`: airport-related fix catalog.
- `output_dir`: destination for generated artifacts.

Default paths are defined near the top of
`src/mcp_tools/scenario_manager/precompute_artifact.py`:

```text
data/adsb/catalogs/2026-04-01_landings_and_departures.csv
data/adsb/catalogs/2026-04-01_fix_sequences.csv
data/adsb/raw
data/kdfw_procs/airport_related_fixes.csv
data/artifacts
```

The relevant event columns are:

- `flight_id`
- `callsign`
- `icao24`
- `operation`
- `runway`

The relevant fix sequence columns are:

- `flight_id`
- `first_time`
- `last_time`
- `fix_sequence`
- `fix_count`

The raw ADS-B track must provide at least:

- `time`
- `lat`
- `lon`
- `geoaltitude`
- optional `heading`

## Batch-Level Flow

`precompute_artifacts()` is the public batch function.

It first calls `_arrival_rows()`:

```text
events CSV
  -> keep rows where operation == arrival
  -> normalize flight_id

fix sequence CSV
  -> normalize flight_id

arrival rows + fix sequence rows
  -> merge on flight_id
```

It then calls `_flight_raw_tracks()`:

```text
load_raw_adsb(raw_adsb_dir)
  -> split_tracks_by_gap(...)
  -> group by flight_id
```

Each arrival row becomes an `ArtifactTask`. The task contains the catalog row,
the matching raw ADS-B DataFrame if available, and the precompute parameters:

- compression tolerances;
- final-fix target distance and centerline tolerance;
- FMS time step;
- top-of-descent search tolerance and iteration limit.

Tasks are processed by `_run_artifact_tasks()`. With one process it runs in
process; with more than one process it uses `multiprocessing.Pool` and
`imap_unordered()`.

## Per-Arrival Flow

`_process_arrival_task()` is the per-flight control flow.

The high-level steps are:

```text
ArtifactTask
  -> validate raw ADS-B exists
  -> normalize original route tokens
  -> load fix catalog
  -> detect ATC decision point
  -> build base route
  -> resolve base-route tokens into a ReferencePath
  -> pick raw ADS-B seed near first base-route fix
  -> build FMSRequest + FMSBiChannelState
  -> call plan_fms_bichannel()
  -> build compressed payload
```

Any exception in this flow becomes a skipped `ArtifactResult` with the exception
type and message as the reason. That keeps the batch run from failing because
one arrival has unusable data.

## Route Normalization

The route begins as the catalog `fix_sequence`, usually formatted like:

```text
FIXA>FIXB>FIXC
```

`_route_tokens()`:

1. splits on `>`;
2. strips whitespace;
3. uppercases tokens;
4. drops `NAN`;
5. normalizes the runway to `RW##` plus optional `L`, `C`, or `R`;
6. appends the runway if it is not already present.

For example:

```text
fix_sequence = "fixa>fixb"
runway = "35c"

route = ["FIXA", "FIXB", "RW35C"]
```

Coordinates are also supported later in the tactical path machinery as
`(lat_deg, lon_deg)` tuples or parseable coordinate tokens, but the current
catalog route normalization path returns string tokens.

## ATC Decision Point Detection

`detect_wait_atc_point()` identifies the point where the base route should
stop following the observed arrival route and switch to a direct-to-final-fix
route.

The detector:

1. resolves route tokens into waypoints;
2. finds the runway waypoint;
3. finds the route crossing of a default 50 NM ring around the runway;
4. classifies that crossing into an arrival quadrant: `NE`, `NW`, `SE`, or
   `SW`;
5. selects the last named route fix inside the corresponding capture polygon.

The return payload includes:

- selected fix identifier;
- lat/lon;
- `route_index`;
- distance to runway;
- arrival cluster;
- gate-crossing diagnostics.

If no acceptable point is found, the arrival is skipped with reason
`missing ATC decision point`.

## Base Route Construction

`_build_base_route()` constructs the route SIMAP will simulate.

It starts with the original normalized route and the ATC decision point:

```text
original route:
  [FIXA, FIXB, FIXC, FIXD, RW35C]

ATC route_index:
  2

route prefix:
  [FIXA, FIXB, FIXC]
```

Then it selects a runway-aligned final fix and appends the runway:

```text
base route:
  [FIXA, FIXB, FIXC, FINAL_FIX, RW35C]
```

The output type is `BaseRoute`, which carries:

- `lateral_path`: route tokens used by SIMAP;
- `upstream_identifier`: first token, or `COORD01` for coordinate starts;
- `runway_identifier`;
- `final_fix`;
- `atc_point`;
- final-fix selection parameters.

Consecutive duplicate route tokens are removed after appending final fix and
runway.

## Final Fix Selection

`_select_final_fix()` searches the fix catalog for a non-runway fix near the
extended runway centerline.

The algorithm:

1. normalize and look up the runway waypoint;
2. derive runway true heading:
   - use the reciprocal runway if available;
   - otherwise use runway number times 10 degrees;
3. define the outbound direction away from the runway threshold;
4. project every non-runway fix into the runway-local east/north frame;
5. compute along-track and cross-track distance against the extended centerline;
6. keep only fixes that are in front of the runway and within cross-track
   tolerance;
7. choose the candidate with:
   - smallest error from target along-track distance;
   - then smallest absolute cross-track distance;
   - then smallest direct distance.

The defaults are:

```text
target final-fix distance: 7.0 NM
centerline tolerance:      0.15 NM
```

If no candidate exists, the flight is skipped through the exception handling in
`_process_arrival_task()`.

## Worked Example: Fix Sequence To Reference Path

Suppose the catalog gives this arrival:

```text
flight_id     = ARR123
runway        = 35C
fix_sequence  = JEN>BOOVE>ALIAN>DFW
```

The exact fix names are not important. What matters is the transformation:

```text
catalog fix sequence
  -> normalized route tokens
  -> ATC decision point
  -> base route
  -> resolved waypoints
  -> ReferencePath
```

After route normalization, the runway token is guaranteed to be present:

```text
[JEN, BOOVE, ALIAN, DFW, RW35C]
```

Assume `detect_wait_atc_point()` selects `BOOVE` as the ATC decision point. The
base-route builder keeps the prefix through that point:

```text
[JEN, BOOVE]
```

Then `_select_final_fix()` finds a runway-aligned final fix near the configured
target distance from the threshold. If that selected fix is `FINAL35C`, the
base route becomes:

```text
[JEN, BOOVE, FINAL35C, RW35C]
```

That base route is the lateral path passed into `TacticalCommand`:

```python
TacticalCommand(
    lateral_path=["JEN", "BOOVE", "FINAL35C", "RW35C"],
    upstream=TacticalCondition(
        fix_identifier="JEN",
        cas_kts=...,
        altitude_ft=...,
    ),
    altitude_constraints=(),
)
```

`build_tactical_plan_request()` then resolves the route tokens through the fix
catalog:

```text
JEN      -> PathWaypoint(identifier="JEN", lat_deg=..., lon_deg=...)
BOOVE    -> PathWaypoint(identifier="BOOVE", lat_deg=..., lon_deg=...)
FINAL35C -> PathWaypoint(identifier="FINAL35C", lat_deg=..., lon_deg=...)
RW35C    -> PathWaypoint(identifier="RW35C", lat_deg=..., lon_deg=...)
```

Those waypoint coordinates become a continuous `ReferencePath`:

```text
PathWaypoint sequence
  -> build_reference_path()
  -> ReferencePath.from_geographic()
  -> local east/north samples, track angles, curvature, and s_m stationing
```

The final waypoint, here `RW35C`, becomes the local map origin. The upstream
fix, here `JEN`, is near `s_m = total_length_m`. The runway threshold is
`s_m = 0`.

This means both channels are already tied to route geometry before simulation:

- the longitudinal FMS moves along `ReferencePath.s_m`;
- the lateral FMS projects actual east/north position back onto the same
  `ReferencePath`;
- output lat/lon is converted from lateral east/north using the same path
  origin.

## ADS-B Seed State

The simulation is anchored to observed ADS-B through `_seed_for_flight_at_fix()`.

The seed finder:

1. keeps rows with non-null `time`, `lat`, `lon`, and `geoaltitude`;
2. requires at least two valid rows;
3. finds the ADS-B row closest to the first base-route fix;
4. chooses the next row as the speed neighbor, or previous row if the closest
   point is the last point;
5. computes ground speed from point-to-point distance divided by time delta;
6. rejects non-finite or very low ground speed;
7. carries `heading` when available.

The resulting `SeedState` contains:

```text
time_s
lat_deg
lon_deg
geoaltitude_m
heading_deg | None
ground_speed_mps
```

The seed time becomes the absolute timestamp offset for the generated artifact.
The seed altitude and speed become the upstream condition for SIMAP.

## Request Assembly

`_build_request_bundle()` converts the base route and seed into the SIMAP
objects used by `plan_fms_bichannel()`.

First it converts observed state:

```text
h_m = max(seed.geoaltitude_m, 1.0)
cas_mps = tas2cas(max(seed.ground_speed_mps, 1.0), h_m)
```

The code treats ADS-B ground speed as the best available TAS proxy for the
upstream condition. The converted CAS is clamped to at least 80 kt when building
the tactical command.

Then it builds:

```python
TacticalCommand(
    lateral_path=base_route.lateral_path,
    upstream=TacticalCondition(
        fix_identifier=base_route.upstream_identifier,
        cas_kts=max(80.0, mps_to_kts(cas_mps)),
        altitude_ft=m_to_ft(h_m),
    ),
    altitude_constraints=(),
)
```

`build_tactical_plan_request()` resolves that command into a
`TacticalPlanBundle`:

- fix tokens become `PathWaypoint` objects;
- `ReferencePath` is built from those waypoints;
- aircraft config and performance backend are initialized with OpenAP data;
- threshold altitude and landing speed are set from runway/catalog data;
- upstream altitude and CAS are set from the seed;
- the tactical constraint envelope is built.

At this point the `ReferencePath` already exists inside `bundle.request`. The
rest of the FMS request assembly does not create route geometry; it attaches
FMS-specific start altitude, speed, target altitude, and time-step settings to
that existing path.

The precompute pipeline then creates:

```text
FMSRequest.from_coupled_request(
    bundle.request,
    start_s_m=bundle.request.reference_path.total_length_m,
    dt_s=fms_dt_s,
)
```

That starts the FMS at the upstream end of the reference path.

Finally, it builds the explicit `FMSBiChannelState`:

```text
t_s        = 0.0
s_m        = fms_request.start_s_m
h_m        = fms_request.start_h_m
v_tas_mps  = cas2tas(fms_request.start_cas_mps, fms_request.start_h_m)
east/north = reference_path.position_ne(start_s_m)
psi_rad    = ADS-B heading converted to math angle, or path track angle
phi_rad    = 0.0
```

These first lateral values do not come from the longitudinal planner:

- `east_m` and `north_m` come from `reference_path.position_ne(start_s_m)`;
- `psi_rad` comes from ADS-B heading when available, otherwise from
  `reference_path.track_angle_rad(start_s_m)`;
- `phi_rad` starts at `0.0`.

ADS-B heading is aviation heading: degrees clockwise from north. The lateral
dynamics use math angle in the east/north coordinate plane. The conversion is:

```text
psi_rad = wrap(radians(90 - heading_deg))
```

## Reference Path Convention

`ReferencePath` is the shared geometry object used by both the longitudinal and
lateral channels.

It is built from the resolved base-route waypoint sequence, with the final
waypoint as the local origin. For arrivals, the final waypoint is the runway
threshold. It is not just a container for named fixes; it is a sampled geometric
path derived from those fixes.

The most important convention:

```text
s_m = remaining distance to threshold
```

So:

- `s_m = total_length_m` is the upstream end of the route;
- `s_m = 0` is the runway threshold;
- as the aircraft flies toward the runway, `s_m` decreases.

`ReferencePath.from_geographic()` also builds fly-by geometry around eligible
interior waypoints. That means the map path can curve before fixes instead of
being a sharp polyline. Its stationing is scaled so `total_length_m` remains the
original route chord length used by the longitudinal FMS.

Note that like real FMS, the aircraft RNAV might produce early-turn behavior. It lives here. During construction, sharp interior corners
can become:

```text
line to turn-start -> circular fly-by arc -> line from turn-end
```

The resulting samples carry track angle and `curvature_inv_m`. Later,
`compute_lateral_command()` previews that curvature through
`ReferencePath.curvature_many()` and uses it as lateral feed-forward.

The reference path provides:

- `position_ne(s_m)`;
- `position_ne_many(s_m)`;
- `project_s_m(east_m, north_m)`;
- `latlon_from_ne_many(east_m, north_m)`;
- `track_angle_rad(s_m)`;
- `tangent_hat(s_m)`;
- `normal_hat(s_m)`;
- `curvature_many(s_m)`.

## FMS Bichannel Request

`FMSBiChannelRequest` wraps:

```text
base_request: FMSRequest | HoldAwareFMSRequest
guidance: LateralGuidanceConfig
initial_state: FMSBiChannelState | None
```

`precompute_artifact.py` uses a plain `FMSRequest`, default
`LateralGuidanceConfig`, and an explicit initial state from ADS-B.

The default lateral guidance parameters are:

```text
lookahead_m                 = 1500.0
cross_track_gain            = 1.0
track_error_gain            = 2.0
curvature_feedforward_gain  = 0.25
min_lookahead_m             = 150.0
max_los_angle_rad           = 89 degrees
integration_step_s          = 0.5
```

## Longitudinal Channel

`plan_fms_bichannel()` first runs the longitudinal planner:

```text
FMSRequest
  -> plan_fms_descent()
  -> FMSResult
```

or, for hold-aware requests:

```text
HoldAwareFMSRequest
  -> plan_hold_aware_fms_descent()
  -> FMSResult
```

The plain FMS state is:

```text
x_long = [s_m, h_m, v_tas_mps]
```

The managed descent simulator computes:

```text
mode = mode_for_s(cfg, s_m)
CAS = cas_from_tas(v_tas_mps, h_m, weather)
target_CAS = speed_targets.for_mode(mode, h_m)
speed_error = CAS - target_CAS
pitch = nominal_pitch + Kp * speed_error + Ki * integral(speed_error)
vertical_speed = clip(v_tas * sin(pitch), min_vs, max_vs)
thrust = idle_thrust(...)
drag = drag(...)
ground_speed = along-track ground speed from TAS and wind
```

Then it advances:

```text
s_m       <- s_m - ground_speed * dt
h_m       <- h_m + vertical_speed * dt
v_tas_mps <- v_tas_mps + ((thrust - drag) / mass - g * sin(gamma)) * dt
```

`plan_fms_descent()` wraps this simulator in a top-of-descent search. It finds
the shortest descent distance that can reach the target altitude by threshold,
then stitches a level segment ahead of the descent segment.

If the route does not have enough along-track distance, the planner returns an
infeasible `FMSResult` with a truncated descent-to-threshold profile. The
bichannel layer still replays lateral motion over whatever longitudinal result
it receives.

The longitudinal planner also applies the FMS speed target rule for the
altitude speed cap, so target CAS is limited to the configured cap below the
configured altitude, typically 250 kt below 10,000 ft.

## Lateral Channel

After the longitudinal result is available, `_lateral_response()` computes the
lateral replay.

The first lateral state is already available before this loop starts. It is
either:

- the explicit `FMSBiChannelState` supplied by the caller, which is what the
  artifact precompute path uses; or
- a default state from `FMSBiChannelState.on_reference_path()`, which places the
  aircraft on `base.reference_path` at the first longitudinal `s_m`.

So the lateral loop is not asking the longitudinal solver for map state. It is
combining a pre-existing lateral state with the longitudinal schedule.

For each longitudinal sample index:

1. copy scheduled `t_s`, `s_m`, `h_m`, and `v_tas_mps` into the lateral state;
2. keep the current lateral `east_m`, `north_m`, `psi_rad`, and `phi_rad`;
3. find the active mode for the scheduled `s_m`;
4. call `compute_lateral_command()`;
5. append all lateral diagnostics;
6. integrate to the next longitudinal sample with `_advance_lateral_state()`.

At index 0, those lateral values are the initialized values described above.
At later indexes, they are the values produced by the previous
`_advance_lateral_state()` call.

The output length always matches the longitudinal result length.

### Guidance Command

`compute_lateral_command()` works in ground track.

It first computes wind-aware ground velocity:

```text
v_air = v_tas * [cos(psi), sin(psi)]
wind  = [wind_east, wind_north]
v_gnd = v_air + wind

ground_speed = norm(v_gnd)
ground_track = atan2(v_gnd_north, v_gnd_east)
```

Then it projects the actual aircraft position onto the reference path:

```text
ref_s = reference_path.project_s_m(east, north)
ref_position = reference_path.position_ne(ref_s)
tangent = reference_path.tangent_hat(ref_s)
normal = reference_path.normal_hat(ref_s)
ref_track = reference_path.track_angle_rad(ref_s)
```

This projection is based on actual map position, not merely the scheduled
longitudinal `s_m`. That makes capture behavior robust when the aircraft has
cross-track or along-track error.

It computes:

```text
error_vector = actual_position - ref_position
cross_track_m = dot(error_vector, normal)
track_error_rad = wrap(ground_track - ref_track)
alongtrack_speed_mps = max(0, dot(v_gnd, tangent))
along_path_offset_m = dot(error_vector, tangent)
```

The controller chooses a lookahead target downstream on the route:

```text
target_s = max(0, ref_s - lookahead_m)
target_position = reference_path.position_ne(target_s)
```

Near the threshold, if the aircraft has already moved beyond the closest
projected point, the target is extrapolated forward along the tangent so the
controller does not command a turn back toward the threshold sample.

Line-of-sight error gives the feedback curvature:

```text
los_track = atan2(target_north - north, target_east - east)
los_error = clamp(wrap(los_track - ground_track), +/- max_los_angle)

l1_gain = track_error_gain * sqrt(cross_track_gain)
curvature_feedback = l1_gain * sin(los_error) / lookahead_m
```

The controller also previews reference-path curvature between the projected
point and the lookahead point. The largest magnitude curvature in that preview
becomes feed-forward:

```text
curvature_cmd =
    curvature_feedforward_gain * curvature_feedforward
    + curvature_feedback
```

Finally it converts curvature to bank:

```text
phi_req = atan(ground_speed^2 * curvature_cmd / g)
phi_max = bank_limit_rad(cfg, mode, CAS)
phi_req = clamp(phi_req, -phi_max, +phi_max)
```

The resulting `LateralCommand` contains:

- east/north ground velocity;
- ground speed and ground track;
- along-track speed;
- cross-track error;
- track error;
- commanded curvature;
- requested bank;
- bank limit.

### State Integration

`_advance_lateral_state()` advances map position, heading, and bank between two
longitudinal samples.

The longitudinal output interval may be larger than the lateral integration
step. The function substeps using:

```text
dt = min(guidance.integration_step_s, remaining_interval)
```

For each substep it linearly interpolates the scheduled longitudinal values:

```text
scheduled_t
scheduled_s
scheduled_h
scheduled_v_tas
```

Then it recomputes the lateral command at the current integrated map state.

Bank and heading are updated with `lateral_rates()`:

```text
phi_dot = clip((phi_req - phi) / tau_phi, -p_max, +p_max)
psi_dot = g * tan(phi) / max(v_tas, 1.0)
```

Notice that heading rate uses current bank, not requested bank. This creates a
simple roll lag: the aircraft must roll toward the request before the heading
rate fully responds.

Map position advances with commanded ground velocity:

```text
east  <- east  + east_dot  * velocity_scale * dt
north <- north + north_dot * velocity_scale * dt
psi   <- wrap(psi + psi_dot * dt)
phi   <- phi + phi_dot * dt
```

`velocity_scale` is a correction that keeps the integrated map motion consistent
with the scheduled path-distance progress:

```text
scheduled_path_speed =
    distance(reference_path.position_ne(next_s),
             reference_path.position_ne(current_s)) / dt

velocity_scale =
    clip(scheduled_path_speed / command.ground_speed, 0.0, 2.0)
```

This matters because the fly-by map geometry and the FMS stationing are not
identical physical distances.

## FMSBiChannelResult

`FMSBiChannelResult` contains the full longitudinal `FMSResult` plus lateral
arrays:

```text
east_m
north_m
lat_deg
lon_deg
psi_rad
phi_rad
ground_track_rad
ground_speed_mps
alongtrack_speed_mps
cross_track_m
track_error_rad
curvature_cmd_inv_m
phi_req_rad
phi_max_rad
```

It also computes summary diagnostics:

- `max_abs_cross_track_m`;
- `max_abs_track_error_rad`;
- `max_bank_command_ratio`;
- `final_threshold_error_m`.

`success` and `message` are copied from the longitudinal result. A generated
artifact can therefore have `simulation.success = false` when the base route is
vertically infeasible, even though the bichannel module still produced a
trajectory over the truncated longitudinal profile.

## Artifact Payload Conversion

`_payload_from_result()` converts the bichannel result into the JSONL artifact
payload.

Absolute timestamps are:

```text
times = round(seed.time_s + result.t_s)
```

Latitude and longitude normally come from `FMSBiChannelResult.lat_deg` and
`FMSBiChannelResult.lon_deg`. The fallback path can derive lat/lon from
`result.s_m` and a `ReferencePath`, but the bichannel result already has map
coordinates.

Altitude comes from:

```text
geoaltitudes = result.h_m
```

Then `compress_breakpoints()` selects the minimal points needed to preserve the
trajectory within the configured lateral and altitude tolerances.

The stored point format is:

```json
["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"]
```

Breakpoint mask bits are:

```json
{
  "lateral": 1,
  "altitude": 2
}
```

So:

- `0`: retained point but not a lateral or altitude breakpoint;
- `1`: lateral breakpoint;
- `2`: altitude breakpoint;
- `3`: both lateral and altitude breakpoint.

Each generated arrival payload includes:

- flight identity fields;
- normalized runway;
- base-route fix sequence;
- compressed points;
- lateral and altitude breakpoint times;
- ATC decision point payload;
- selected final fix payload;
- full base-route payload;
- compression tolerances;
- simulation diagnostics;
- lateral guidance parameters.

## Manifest

The batch function also writes `manifest.json`.

The manifest summarizes:

- creation time;
- artifact type;
- input and output paths;
- number of arrivals processed;
- generated and skipped counts;
- skipped departure count;
- skipped-arrival reason counts;
- simulation success/failure counts;
- simulation failure message counts;
- wait-ATC-point success and cluster counts;
- raw and compressed point totals;
- compression and FMS settings;
- lateral guidance settings;
- per-flight status records.

The manifest is useful for deciding whether the generated base artifacts are
ready to serve. In particular, `simulation_failure_count` does not mean the
JSONL is missing those flights. It means those flights have base trajectories
whose vertical FMS profile was infeasible under the current assumptions.

## Important Invariants

The main invariants to keep in mind:

- `s_m` is distance remaining to threshold, so it decreases during flight.
- `ReferencePath` coordinates are local east/north meters with the runway
  threshold as origin.
- `psi_rad` is a math angle in east/north coordinates, not aviation heading.
- `FMSBiChannelResult` length matches its `longitudinal` result length.
- Bichannel success is longitudinal success; lateral diagnostics are separate.
- The precompute pipeline catches per-flight failures and records them as
  skipped artifacts.
- Generated arrivals are base-route artifacts, not intervention-adjusted
  trajectories.

## Common Failure Modes

`_process_arrival_task()` can skip an arrival for expected data-quality reasons:

- missing raw ADS-B flight;
- missing route fixes before runway;
- missing ATC decision point;
- missing raw seed near the first base-route fix;
- unknown lateral-path fix;
- missing runway fix;
- no runway-aligned final fix inside tolerance;
- invalid FMS time step;
- request construction errors from the tactical builder.

Generated artifacts can still have `simulation.success = false`. That usually
means SIMAP could not complete the vertical profile before the threshold with
the available route distance, seed altitude, speed targets, and aircraft
performance assumptions.

## End-to-End Information Flow

The complete information flow is:

```text
events CSV
  -> arrival row: flight_id, callsign, icao24, runway

fix sequence CSV
  -> fix_sequence for flight_id

raw ADS-B
  -> track for flight_id
  -> SeedState near first base-route fix

fix catalog
  -> waypoint resolution
  -> runway waypoint
  -> ATC point detection
  -> runway-aligned final fix selection

base route tokens
  -> PathWaypoint sequence from fix catalog
  -> ReferencePath construction

BaseRoute
  -> TacticalCommand
  -> TacticalPlanBundle
  -> FMSRequest
  -> FMSBiChannelRequest

FMSRequest
  -> plan_fms_descent()
  -> FMSResult: t_s, s_m, h_m, speeds, forces, modes, phase, success

FMSBiChannelRequest + FMSResult
  -> _lateral_response()
  -> FMSBiChannelResult: lat/lon, east/north, heading, bank, lateral diagnostics

FMSBiChannelResult + SeedState
  -> absolute times
  -> trajectory compression
  -> JSONL payload

all task results
  -> manifest summary
```
