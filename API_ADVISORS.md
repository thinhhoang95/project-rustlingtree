# API Advisor Tools

The scenario-manager API exposes advisory endpoints under `/tools/advisors/*`.
Advisors are read-only what-if tools. They do not mutate the scenario manager or
write intervention diffs.

Profile-based advisors recompute SIMAP longitudinal profiles from the currently
served arrival payload. Schedule-based advisors such as AMAN read the active
served arrival/departure schedules directly. In both cases, advisory results are
based on the scenario-manager view after any saved diffs have been applied.

## Common Concepts

### Units

- `miles_to_gain_nmi`: nautical miles.
- `minutes_to_gain`: minutes.
- `seconds_to_gain`: seconds.
- `*_m`: meters.
- `*_s`: seconds.
- `cas_kts`: calibrated airspeed in knots.
- `s_m`: SIMAP along-track station in meters remaining to runway threshold.

### `s_m` Convention

`s_m` follows SIMAP's distance-to-threshold convention:

- `s_m = 0` is the runway threshold.
- Larger `s_m` values are farther upstream.
- A speed-control station must be within the arrival reference path:
  `0 <= s_m <= request.start_s_m`.

### Baseline And What-If Profiles

SIMAP profile advisor responses contain baseline and what-if planner status
fields:

- `baseline_*`: result from the served arrival rebuilt as a longitudinal profile.
- `what_if_*`: result after applying the advisory what-if.

For an already feasible baseline, feasibility advice returns zero distance and
uses the baseline profile as the what-if profile.

### Shared Response Fields

All SIMAP profile advisor responses include:

| Field | Type | Description |
| --- | --- | --- |
| `flight_number` | string | Arrival callsign from the served schedule. |
| `icao24` | string | Aircraft ICAO24 identifier. |
| `flight_id` | string | Scenario-manager flight identifier. |
| `runway` | string | Served runway identifier. |
| `miles_to_gain_nmi` | number | Actual extra route distance in nautical miles. Positive means a longer path. |
| `minutes_to_gain` | number | What-if total profile time minus baseline profile time, in minutes. Positive means longer flight time. |
| `baseline_success` | boolean | Whether the rebuilt baseline longitudinal planner succeeded. |
| `baseline_message` | string | Planner message for the baseline profile. |
| `what_if_success` | boolean | Whether the what-if longitudinal planner succeeded. |
| `what_if_message` | string | Planner message for the what-if profile. |
| `baseline_time_s` | number | Baseline profile total time in seconds. |
| `what_if_time_s` | number | What-if profile total time in seconds. |

## AMAN Advisor

```text
GET /tools/advisors/aman
```

The AMAN advisor computes arrival delays that would clear runway-use overlaps
involving arrivals. Departures remain fixed. Arrivals are processed first-come,
first-served per physical runway using the active served schedules, so saved
scenario-manager diffs are reflected.

It uses the same runway occupancy assumptions and physical-runway grouping as
`/tools/evals/runway-overlaps`.

### Example Request

```bash
curl "http://127.0.0.1:8000/tools/advisors/aman" | python -m json.tool
```

### Example Response

Only arrivals with positive advised delay are returned.

```json
[
  {
    "flight_number": "AAL123",
    "icao24": "a1b2c3",
    "flight_id": "AAL123M1a1b2c3",
    "runway": "RW35C",
    "physical_runway": "17C/35C",
    "original_time_at_last_event": 1775021200,
    "original_time_at_last_event_utc": "2026-04-01T07:26:40Z",
    "advised_time_at_last_event": 1775021260,
    "advised_time_at_last_event_utc": "2026-04-01T07:27:40Z",
    "seconds_to_gain": 60,
    "minutes_to_gain": 1.0
  }
]
```

### AMAN-Specific Fields

| Field | Type | Description |
| --- | --- | --- |
| `physical_runway` | string | Normalized physical runway surface used for overlap grouping. |
| `original_time_at_last_event` | integer | Current served arrival threshold time in epoch seconds. |
| `original_time_at_last_event_utc` | string | UTC rendering of the original threshold time. |
| `advised_time_at_last_event` | integer | Delayed threshold time that avoids arrival-involving overlaps on the runway. |
| `advised_time_at_last_event_utc` | string | UTC rendering of the advised threshold time. |
| `seconds_to_gain` | integer | Delay to apply to the arrival threshold time. |
| `minutes_to_gain` | number | `seconds_to_gain / 60.0`. |

### Client Interpretation

- Empty list means no arrival needs runway-overlap delay advice.
- AMAN does not attempt to resolve departure-departure overlaps.
- If each returned arrival gained the advised delay, the runway-overlap
  evaluator should report no remaining overlap where either use is an arrival.

## Feasibility Advisor

```text
GET /tools/advisors/feasibility
GET /tools/advisors/feasibility?flight_id=<flight_id>
```

The feasibility advisor estimates how much extra along-track distance is needed
to make the rebuilt longitudinal profile feasible.

It does not use `simulation.final_threshold_error_m` as the required distance.
Instead, it extends the upstream end of the reference path, reruns
`plan_fms_descent()`, and searches for the smallest extension that succeeds.

### Query Parameters

| Parameter | Required | Type | Description |
| --- | --- | --- | --- |
| `flight_id` | no | string | If provided, return advice only for this arrival. If omitted, return advice for all served arrivals. |

### Example Request

```bash
curl "http://127.0.0.1:8000/tools/advisors/feasibility?flight_id=NKS220M1a91e6e" \
  | python -m json.tool
```

### Example Response

The endpoint returns a list, even when `flight_id` is supplied.

```json
[
  {
    "flight_number": "NKS220M1",
    "icao24": "a91e6e",
    "flight_id": "NKS220M1a91e6e",
    "runway": "RW18R",
    "miles_to_gain_nmi": 36.09,
    "miles_to_gain_m": 66831.16,
    "minutes_to_gain": 8.93,
    "baseline_success": false,
    "baseline_message": "infeasible: not enough along-track distance to complete FMS profile before threshold; showing FMS response truncated at threshold",
    "what_if_success": true,
    "what_if_message": "stitched level cruise and descent reached target altitude at threshold",
    "baseline_time_s": 1029.27,
    "what_if_time_s": 1565.19,
    "search_converged": true
  }
]
```

### Feasibility-Specific Fields

| Field | Type | Description |
| --- | --- | --- |
| `miles_to_gain_m` | number | Same required extension as `miles_to_gain_nmi`, in meters. |
| `search_converged` | boolean | `true` when bracketing and bisection found a successful extension within the search bounds. `false` means the reported distance is the max attempted extension and the profile still did not succeed. |

### Client Interpretation

- Empty list means no matching arrivals, or no served arrivals when no
  `flight_id` is supplied.
- If `baseline_success` is `true`, expect `miles_to_gain_nmi = 0.0` and
  `minutes_to_gain = 0.0`.
- If `search_converged` is `false`, treat the result as "at least this much
  distance was insufficient or the configured search limit was reached."

## Vectoring Advisor

```text
GET /tools/advisors/vectoring?flight_id=<flight_id>&extra_distance_nmi=<nmi>
```

The vectoring advisor evaluates the effect of adding a requested amount of
along-track distance.

It extends the upstream end of the reference path by `extra_distance_nmi`,
recomputes the longitudinal profile, and reports the time delta.

### Query Parameters

| Parameter | Required | Type | Description |
| --- | --- | --- | --- |
| `flight_id` | yes | string | Arrival to advise. |
| `extra_distance_nmi` | yes | number | Extra along-track distance to test, in nautical miles. Must be nonnegative. |

### Example Request

```bash
curl "http://127.0.0.1:8000/tools/advisors/vectoring?flight_id=NKS220M1a91e6e&extra_distance_nmi=5" \
  | python -m json.tool
```

### Example Response

```json
{
  "flight_number": "NKS220M1",
  "icao24": "a91e6e",
  "flight_id": "NKS220M1a91e6e",
  "runway": "RW18R",
  "miles_to_gain_nmi": 5.0,
  "minutes_to_gain": 1.19,
  "baseline_success": false,
  "baseline_message": "infeasible: not enough along-track distance to complete FMS profile before threshold; showing FMS response truncated at threshold",
  "what_if_success": false,
  "what_if_message": "infeasible: not enough along-track distance to complete FMS profile before threshold; showing FMS response truncated at threshold",
  "baseline_time_s": 1029.27,
  "what_if_time_s": 1100.40,
  "requested_extension_nmi": 5.0,
  "requested_extension_m": 9260.0,
  "required_feasibility_miles_nmi": 36.09,
  "required_feasibility_m": 66831.16,
  "feasibility_search_converged": true
}
```

### Vectoring-Specific Fields

| Field | Type | Description |
| --- | --- | --- |
| `requested_extension_nmi` | number | Requested path extension, in nautical miles. Same value as `miles_to_gain_nmi`. |
| `requested_extension_m` | number | Requested path extension, in meters. |
| `required_feasibility_miles_nmi` | number | Feasibility advisor's estimated required extension for the same baseline arrival. |
| `required_feasibility_m` | number | Feasibility required extension, in meters. |
| `feasibility_search_converged` | boolean | Whether the feasibility search found a successful extension. |

### Client Interpretation

- `what_if_success = false` means the requested extension still did not make the
  profile feasible.
- Compare `requested_extension_nmi` to `required_feasibility_miles_nmi` to decide
  whether the tested vectoring distance is likely enough for vertical
  feasibility.
- `minutes_to_gain` is recomputed by SIMAP; do not assume it equals distance
  divided by a fixed cruise speed.

## Speed-Control Advisor

```text
GET /tools/advisors/speed-control?flight_id=<flight_id>&s_m=<meters>&cas_kts=<knots>
```

The speed-control advisor evaluates a lower-CAS instruction accepted at a given
distance-to-threshold station.

V1 supports speed reductions only. The requested `cas_kts` must be lower than
the base managed target at the acceptance station and within SIMAP's planned CAS
bounds for the relevant mode. Invalid speed requests return an API error.

### Query Parameters

| Parameter | Required | Type | Description |
| --- | --- | --- | --- |
| `flight_id` | yes | string | Arrival to advise. |
| `s_m` | yes | number | Acceptance station in meters remaining to threshold. Must be within the route. |
| `cas_kts` | yes | number | New prescribed calibrated airspeed in knots. Must be positive and lower than the base managed profile at `s_m`. |

### Example Request

```bash
curl "http://127.0.0.1:8000/tools/advisors/speed-control?flight_id=NKS220M1a91e6e&s_m=100000&cas_kts=180" \
  | python -m json.tool
```

### Example Response

```json
{
  "flight_number": "NKS220M1",
  "icao24": "a91e6e",
  "flight_id": "NKS220M1a91e6e",
  "runway": "RW18R",
  "miles_to_gain_nmi": 0.0,
  "minutes_to_gain": 3.3,
  "baseline_success": false,
  "baseline_message": "infeasible: not enough along-track distance to complete FMS profile before threshold; showing FMS response truncated at threshold",
  "what_if_success": false,
  "what_if_message": "infeasible: not enough along-track distance to complete FMS profile before threshold; showing FMS response truncated at threshold",
  "baseline_time_s": 1029.27,
  "what_if_time_s": 1227.13,
  "s_m": 100000.0,
  "cas_kts": 180.0,
  "equivalent_vectoring_miles_nmi": 23.15,
  "equivalent_vectoring_m": 42882.3
}
```

### Speed-Control-Specific Fields

| Field | Type | Description |
| --- | --- | --- |
| `s_m` | number | Accepted speed-control station, in meters remaining to threshold. |
| `cas_kts` | number | Requested calibrated airspeed in knots. |
| `equivalent_vectoring_miles_nmi` | number | Time gain converted to equivalent extra path distance using baseline pre-TOD/level-segment groundspeed. |
| `equivalent_vectoring_m` | number | Equivalent vectoring distance in meters. |

### Client Interpretation

- `miles_to_gain_nmi` is always `0.0` for speed-only advice because the route is
  not lengthened.
- Use `minutes_to_gain` for the actual time effect.
- Use `equivalent_vectoring_miles_nmi` only as a comparison against vectoring.
  It is not an actual route delta.
- A speed instruction can add time without making an infeasible profile feasible;
  check `what_if_success`.

## Error Handling

The API currently lets validation and advisor errors surface as HTTP error
responses. Common causes:

- Unknown `flight_id`.
- Missing or malformed served arrival fields such as `base_route`, `points`,
  `columns`, or route fixes.
- Missing or malformed served runway-overlap fields such as `runway`,
  `time_at_last_event`, or `departure_time`.
- `extra_distance_nmi < 0`.
- `s_m < 0` or `s_m` beyond the arrival reference path.
- `cas_kts <= 0`.
- Speed-control CAS is not lower than the base managed profile at `s_m`.
- Speed-control CAS is outside planned mode bounds.

Client code should treat non-2xx responses as advisory calculation failures and
show the returned error body when available.

## Suggested Client Workflow

1. Read `/arrivals` to display the active served schedule.
2. Use `/tools/evals/feasibility` to rank currently infeasible arrivals.
3. Use `/tools/evals/runway-overlaps` and `/tools/advisors/aman` to identify
   arrival delays needed to clear runway-use overlaps.
4. For a selected arrival, call `/tools/advisors/feasibility?flight_id=...` to
   estimate the required extra path distance.
5. Call `/tools/advisors/vectoring` with candidate vector lengths and compare
   `what_if_success`, `minutes_to_gain`, and `required_feasibility_miles_nmi`.
6. Call `/tools/advisors/speed-control` for candidate lower-speed instructions
   and compare `minutes_to_gain` against vectoring alternatives.
