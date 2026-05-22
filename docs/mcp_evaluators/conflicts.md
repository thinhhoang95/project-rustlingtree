# Conflict Evaluator

`ConflictEvaluator` detects arrival-pair conflicts in the current
scenario-manager trajectory set. It lives in
`src/mcp_tools/evaluators/conflict.py` and is exposed by the scenario-manager API
at:

```text
GET /tools/evals/conflicts
```

The evaluator uses `ScenarioManager.arrival_schedule()`, not the raw artifact
files directly. That means it evaluates the same served arrival trajectories
that clients receive from `/arrivals`, including any scenario-manager diff that
has already been applied.

## What It Answers

The evaluator answers:

> Do any two served arrival trajectories occupy overlapping protection
> envelopes at the same time?

Each aircraft is treated as the center of a protection envelope. Two arrivals
are in conflict when their center-to-center separation is within both thresholds
at the same time:

- lateral threshold: `10.0 NM`
- vertical threshold: `1000.0 ft`

The lateral threshold is `2 * CONFLICT_ENVELOPE_RADIUS_NM`, where
`CONFLICT_ENVELOPE_RADIUS_NM = 5.0`. The vertical threshold is
`CONFLICT_ENVELOPE_HEIGHT_FL * 10 ft`, where
`CONFLICT_ENVELOPE_HEIGHT_FL = 100.0`.

## Input Flow

The input is the list returned by:

```python
manager.arrival_schedule()
```

Every evaluated arrival must have:

- `columns`: a list of column names.
- `points`: a list of trajectory rows.

The `columns` list must include:

- `time`: epoch seconds.
- `lat`: latitude in degrees.
- `lon`: longitude in degrees.
- `geoaltitude_m`: geometric altitude in meters.

The evaluator finds the column indexes by name, so the required columns do not
need to be adjacent. Each point must contain finite numeric values for those
columns.

These identity fields are copied into each conflict event:

- `callsign` -> `flight_number`
- `icao24`
- `flight_id`
- `runway`

Optional tolerance fields may also be present on an arrival:

- `lateral_tolerance_m`
- `altitude_tolerance_m`

They must be finite, nonnegative numbers when present. These tolerances are used
to report `possible` conflicts when trajectory compression uncertainty can
bridge a separation gap.

## Example Input

This example is used throughout the walkthrough. `ARR_A` flies east across
`ARR_B`, which remains stationary. Both are at `1000 m` altitude.

```json
[
  {
    "flight_id": "ARR_A",
    "callsign": "A100",
    "icao24": "a10001",
    "runway": "RW35C",
    "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
    "points": [
      [1775021200, 32.0, -97.3, 1000.0, 3],
      [1775021260, 32.0, -96.7, 1000.0, 3]
    ]
  },
  {
    "flight_id": "ARR_B",
    "callsign": "B200",
    "icao24": "b20002",
    "runway": "RW35C",
    "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
    "points": [
      [1775021200, 32.0, -97.0, 1000.0, 3],
      [1775021260, 32.0, -97.0, 1000.0, 3]
    ]
  }
]
```

## Algorithm Walkthrough

The evaluator works on continuous line segments, not fixed time samples. That
lets it detect a conflict even when the closest approach happens between two
stored trajectory points.

1. Load the served arrivals.

   ```python
   arrivals = manager.arrival_schedule()
   ```

   If there are fewer than two arrivals, the evaluator returns `[]`.

2. Build a local projection.

   The evaluator scans every trajectory point, validates `lat` and `lon`, and
   computes the average latitude and longitude. That average becomes the origin
   of a local equirectangular projection.

   Each `(lat, lon)` point is converted into `(x_m, y_m)` so the rest of the
   algorithm can use meters:

   ```text
   x_m = earth_radius_m * (lon_rad - lon0_rad) * cos(lat0_rad)
   y_m = earth_radius_m * (lat_rad - lat0_rad)
   ```

   In the example, all points have latitude `32.0`, so the movement is mostly in
   local `x_m`.

3. Convert arrivals into trajectories.

   For each arrival, the evaluator extracts:

   - `time`
   - projected `x_m`
   - projected `y_m`
   - `geoaltitude_m` as `z_m`

   Then it handles duplicate timestamp groups. For adjacent rows with the same
   timestamp, the last row is kept as the representative point. The maximum
   lateral and altitude difference between the duplicate rows and the
   representative row is added to that trajectory's tolerance. This keeps the
   timeline strictly increasing while preserving uncertainty introduced by
   compression.

   After duplicate collapse, each trajectory must still have at least two unique
   times, and times must be strictly increasing.

4. Split trajectories into segments.

   Each adjacent pair of trajectory points becomes one `_Segment`. A segment
   stores:

   - start and end time: `t0`, `t1`
   - start and end position: `(x0, y0, z0)`, `(x1, y1, z1)`
   - velocity components: `vx`, `vy`, `vz`
   - min and max bounds for `x`, `y`, and `z`

   In the example:

   - `ARR_A` has one segment from `1775021200` to `1775021260`.
   - `ARR_B` has one segment over the same time interval.

5. Generate candidate segment pairs.

   The evaluator first reduces the number of expensive checks. It sorts segments
   by start time and only compares pairs whose time ranges overlap. It also
   rejects pairs from the same flight.

   For each remaining pair, it checks whether the segment bounding boxes could
   overlap after expanding them by the conflict thresholds and any tolerances:

   ```text
   lateral_threshold_m = 10.0 * 1852.0 = 18520.0
   vertical_threshold_m = 1000.0 * 0.3048 = 304.8
   ```

   The example pair overlaps in time and its bounding boxes can overlap within
   the lateral and vertical thresholds, so it becomes a candidate.

6. Solve the conflict interval analytically.

   For each candidate pair, the evaluator computes the relative position and
   relative velocity over the common time interval.

   Lateral separation is solved as a quadratic inequality:

   ```text
   (dx0 + dvx * t)^2 + (dy0 + dvy * t)^2 <= lateral_threshold_m^2
   ```

   Vertical separation is solved as a linear absolute-value inequality:

   ```text
   abs(dz0 + dvz * t) <= vertical_threshold_m
   ```

   The segment pair is a conflict only if the lateral-valid interval and
   vertical-valid interval overlap.

   In the example, both aircraft are at the same altitude, so the vertical
   interval is the full `60 s`. `ARR_A` crosses the `10 NM` lateral boundary
   about `10 s` after the segment starts and leaves it about `50 s` after the
   segment starts. The conflict interval is therefore:

   ```text
   start_time = 1775021210
   end_time   = 1775021250
   ```

7. Find closest approach.

   Inside the conflict interval, the evaluator chooses the time that minimizes
   normalized three-dimensional separation. Lateral distance is normalized by
   `18520 m`; vertical separation is normalized by `304.8 m`.

   In the example, closest approach occurs at the crossing point:

   ```text
   closest_time = 1775021230
   latitude     = 32.0
   longitude    = -97.0
   ```

8. Create a conflict hit.

   The evaluator records the two flights ordered by `(flight_id, flight_number,
   icao24)`, the conflict interval, closest-approach position, lateral distance,
   vertical separation, severity, and confidence.

   Severity is a normalized penetration score:

   ```text
   lateral_penetration =
       max(0, (lateral_threshold_m - lateral_distance_m) / lateral_threshold_m)

   vertical_penetration =
       max(0, (vertical_threshold_m - vertical_separation_m) / vertical_threshold_m)

   severity = min(lateral_penetration, vertical_penetration)
   ```

   In the example, the aircraft are colocated laterally at closest approach and
   have zero vertical separation, so severity is effectively `1.0`.

9. Add confidence.

   A hit is `confirmed` when it satisfies the standard `10 NM` and `1000 ft`
   thresholds using the reconstructed trajectory.

   A hit is `possible` when it does not satisfy the standard thresholds but does
   satisfy thresholds expanded by `lateral_tolerance_m`,
   `altitude_tolerance_m`, or duplicate-timestamp tolerance. Possible conflicts
   are useful for compressed trajectories: they mean uncertainty could bridge the
   apparent gap.

10. Merge adjacent hits.

    Different segment pairs for the same flight pair can produce adjacent or
    overlapping hits. The evaluator merges hits for the same pair when the gap is
    no more than `5 s`.

    During a merge, it keeps the widest event interval and uses the best
    closest-approach hit. Confirmed hits beat possible hits; otherwise higher
    severity, lower lateral distance, and lower vertical separation win.

11. Sort final events.

    Final `ConflictEvent` objects are sorted by:

    1. confirmed before possible
    2. descending severity
    3. ascending lateral distance
    4. ascending vertical separation
    5. ascending start time
    6. flight identifiers

## Example Output

The Python evaluator returns `ConflictEvent` dataclasses. The API serializes the
same fields as JSON:

```json
[
  {
    "flight_a": {
      "flight_number": "A100",
      "icao24": "a10001",
      "flight_id": "ARR_A",
      "runway": "RW35C"
    },
    "flight_b": {
      "flight_number": "B200",
      "icao24": "b20002",
      "flight_id": "ARR_B",
      "runway": "RW35C"
    },
    "start_time": 1775021210,
    "end_time": 1775021250,
    "closest_time": 1775021230,
    "closest_time_utc": "2026-04-01T05:27:10Z",
    "latitude": 32.0,
    "longitude": -97.0,
    "lateral_distance_nmi": 0.0,
    "vertical_separation_ft": 0.0,
    "lateral_threshold_nmi": 10.0,
    "vertical_threshold_ft": 1000.0,
    "severity": 1.0,
    "confidence": "confirmed"
  }
]
```

Small floating-point values near zero may appear in direct Python output, for
example `3.2e-13 NM` instead of exactly `0.0`.

An empty array means no arrival-pair conflicts were detected among the served
arrivals.

## Output Fields

- `flight_a`, `flight_b`: the conflicting arrivals, ordered deterministically by
  identity.
- `start_time`: rounded epoch second when the merged conflict begins.
- `end_time`: rounded epoch second when the merged conflict ends.
- `closest_time`: rounded epoch second of closest approach inside the conflict.
- `closest_time_utc`: UTC rendering of `closest_time`.
- `latitude`, `longitude`: midpoint between the two aircraft at closest
  approach.
- `lateral_distance_nmi`: center-to-center lateral distance at closest approach.
- `vertical_separation_ft`: center-to-center vertical separation at closest
  approach.
- `lateral_threshold_nmi`: lateral threshold used for confirmed conflicts.
- `vertical_threshold_ft`: vertical threshold used for confirmed conflicts.
- `severity`: normalized conflict penetration, where `0.0` is at the threshold
  boundary and values approach `1.0` as both separations approach zero.
- `confidence`: `confirmed` or `possible`.

## Validation Rules

The evaluator raises `ValueError` for malformed trajectory data. Important
validation rules:

- `columns` must be a list of strings.
- `columns` must include `time`, `lat`, `lon`, and `geoaltitude_m`.
- `points` must be a list.
- each point must be a list or tuple with enough values for the required column
  indexes.
- required point values must be finite numbers, not booleans.
- each trajectory must contain at least two points.
- after duplicate timestamp collapse, each trajectory must contain at least two
  unique times.
- times must be strictly increasing after duplicate collapse.
- optional tolerances must be finite, nonnegative numbers.

The error message includes the best available flight label, such as:

```text
flight_id=ARR_A callsign=A100 trajectory columns must include geoaltitude_m
```

## Usage

Direct Python usage:

```python
from dataclasses import asdict
from mcp_tools.evaluators import ConflictEvaluator

events = ConflictEvaluator(manager).evaluate()
payload = [asdict(event) for event in events]
```

API usage:

```bash
curl http://127.0.0.1:8000/tools/evals/conflicts
```

## Practical Interpretation

A confirmed event means the reconstructed served trajectories violate both the
lateral and vertical thresholds at the same time. A possible event means the
stored trajectory points do not prove a threshold violation by themselves, but
the evaluator's tolerance model says the compressed or duplicate-timestamp data
could hide one.

The usual remediation loop is to adjust one or both affected arrivals, re-serve
the scenario through `arrival_schedule()`, and rerun this evaluator until the
conflict list is empty or only acceptable residual risks remain.
