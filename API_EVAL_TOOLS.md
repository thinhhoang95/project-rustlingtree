# API Eval Tools

The scenario-manager API exposes evaluator endpoints under `/tools/evals/*`.

## Feasibility

`GET /tools/evals/feasibility`

Returns arrivals whose current scenario-manager trajectory is not feasible. The
result uses the served arrival schedule, so it reflects the precomputed artifact
after any scenario-manager diff has been applied.

Flights are ranked from worst to least bad by missing runway-threshold distance.

### Example Request

```bash
curl http://127.0.0.1:8000/tools/evals/feasibility
```

### Example Response

```json
[
  {
    "flight_number": "AAL123",
    "icao24": "a1b2c3",
    "flight_id": "AAL123M1a2b3c",
    "runway": "RW35C",
    "missing_distance_nmi": 2.14,
    "missing_distance_m": 3963.28,
    "simulation_message": "infeasible: not enough along-track distance to complete FMS profile before threshold"
  }
]
```

### Fields

- `flight_number`: arrival callsign from the scenario manager.
- `icao24`: aircraft ICAO24 code.
- `flight_id`: scenario-manager flight identifier.
- `runway`: normalized runway identifier.
- `missing_distance_nmi`: missing threshold distance in nautical miles.
- `missing_distance_m`: missing threshold distance in meters.
- `simulation_message`: feasibility message from the current arrival trajectory.

An empty array means no served arrivals are currently marked infeasible.

## Conflicts

`GET /tools/evals/conflicts`

Returns arrival-pair conflicts in the current scenario-manager trajectory set.
The result uses the served arrival schedule, so it reflects the precomputed
artifact after any scenario-manager diff has been applied.

The evaluator treats each aircraft as the center of a cylinder:

- cylinder radius: `CONFLICT_ENVELOPE_RADIUS_NM = 5`
- cylinder total height: `CONFLICT_ENVELOPE_HEIGHT_FL = 100`, interpreted here as `1000 ft`

Because the rule is cylinder intersection, two aircraft are in conflict when
their center-to-center separation is within both:

- lateral threshold: `10 NM`
- vertical threshold: `1000 ft`

Trajectory points are treated as piecewise-linear in time. The evaluator first
uses time buckets and segment bounding boxes to reduce candidate checks, then
solves each overlapping segment pair analytically rather than scanning fixed
timesteps.

### Example Request

```bash
curl http://127.0.0.1:8000/tools/evals/conflicts
```

### Example Response

```json
[
  {
    "flight_a": {
      "flight_number": "AAL123",
      "icao24": "a1b2c3",
      "flight_id": "AAL123M1a1b2c3",
      "runway": "RW35C"
    },
    "flight_b": {
      "flight_number": "DAL456",
      "icao24": "d4e5f6",
      "flight_id": "DAL456M1d4e5f6",
      "runway": "RW36L"
    },
    "start_time": 1775021200,
    "end_time": 1775021264,
    "closest_time": 1775021231,
    "closest_time_utc": "2026-04-01T07:27:11Z",
    "latitude": 32.9123,
    "longitude": -97.0412,
    "lateral_distance_nmi": 3.42,
    "vertical_separation_ft": 420.0,
    "lateral_threshold_nmi": 10.0,
    "vertical_threshold_ft": 1000.0,
    "severity": 0.58,
    "confidence": "confirmed"
  }
]
```

### Fields

- `flight_a`, `flight_b`: the conflicting arrivals, ordered by `flight_id`.
- `start_time`, `end_time`: approximate conflict interval in epoch seconds.
- `closest_time`: timestamp of the closest approach within the conflict interval.
- `closest_time_utc`: UTC rendering of `closest_time`.
- `latitude`, `longitude`: midpoint between the two aircraft at closest approach.
- `lateral_distance_nmi`: center-to-center lateral distance at closest approach.
- `vertical_separation_ft`: center-to-center vertical separation at closest approach.
- `lateral_threshold_nmi`: threshold used for lateral cylinder intersection.
- `vertical_threshold_ft`: threshold used for vertical cylinder intersection.
- `severity`: normalized penetration score from `0.0` at the boundary toward `1.0` at perfect overlap.
- `confidence`: `confirmed` for conflicts in the reconstructed trajectory, or `possible` when only compression tolerance expansion creates an overlap.

An empty array means no conflicts are detected among the served arrivals.

## Runway Overlapping Uses

`GET /tools/evals/runway-overlaps`

Returns overlapping runway-use intervals across the current scenario-manager
arrival and departure schedules. The result uses the served schedules, so it
reflects the precomputed artifact after any scenario-manager diff has been
applied.

The evaluator applies fixed runway occupancy assumptions:

- each departure occupies the runway for `90 s` from `departure_time`
- each arrival occupies the runway for `60 s` from `time_at_last_event`, treated as the runway-threshold arrival time

Runway identifiers are normalized for matching. `RW35C`, `RWY35C`, and `35C`
match the same runway end, and reciprocal ends are grouped as one physical
runway surface. For example, `RW35C` and `17C` both report under `17C/35C`.

### Example Request

```bash
curl http://127.0.0.1:8000/tools/evals/runway-overlaps
```

### Example Response

```json
[
  {
    "runway": "17C/35C",
    "use_a": {
      "flight_number": "AAL123",
      "icao24": "a1b2c3",
      "flight_id": "AAL123M1a1b2c3",
      "operation": "arrival",
      "runway": "RW35C"
    },
    "use_b": {
      "flight_number": "DAL456",
      "icao24": "d4e5f6",
      "flight_id": "DAL456M1d4e5f6",
      "operation": "departure",
      "runway": "17C"
    },
    "start_time": 1775021230,
    "end_time": 1775021260,
    "overlapping_time": 1775021230,
    "overlapping_time_utc": "2026-04-01T07:27:10Z",
    "overlapping_duration": 30
  }
]
```

### Fields

- `runway`: normalized physical runway key used for overlap grouping.
- `use_a`, `use_b`: the overlapping runway uses, ordered by interval start time.
- `use_a.operation`, `use_b.operation`: either `arrival` or `departure`.
- `use_a.runway`, `use_b.runway`: served runway identifier for each operation before physical-runway grouping.
- `start_time`, `end_time`: overlap interval in epoch seconds.
- `overlapping_time`: timestamp when the overlap begins; equivalent to `start_time`.
- `overlapping_time_utc`: UTC rendering of `overlapping_time`.
- `overlapping_duration`: overlap duration in seconds.

An empty array means no overlapping runway uses are detected among the served
arrivals and departures.
