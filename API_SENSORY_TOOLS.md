# API Sensory Tools

Sensory tools are read-only helpers for LLM-based agents that need map-aware
context before using edit tools. They inspect the currently served
ScenarioManager arrival schedule, including active diffs, but they do not save
diffs or create edit drafts.

## Vector Assist

```text
POST /tools/sensory/vector-assist
```

Vector Assist recommends dogleg route edits that can gain a requested amount of
arrival elapsed time. It evaluates identified fixes and free-coordinate fixes
inside an operational vectoring mask for the arrival direction, then returns
ready-to-send `/tools/path-stretch/simulate` request payloads.

### Request

```json
{
  "flight_id": "NKS220M1a91e6e",
  "target_time_gain_s": 90,
  "grid_spacing_nm": 1.0,
  "identified_threshold_s": 10.0,
  "include_map": false,
  "max_exact_candidates": 40
}
```

| Field | Required | Description |
| --- | --- | --- |
| `flight_id` | yes | Arrival `flight_id` from `/arrivals`. |
| `target_time_gain_s` | yes | Desired additional elapsed time, in seconds. |
| `grid_spacing_nm` | no | Free-coordinate grid spacing in nautical miles. Default `1.0`. |
| `identified_threshold_s` | no | Prefer identified fixes when their score is within this many seconds of the best free fix. Default `10.0`. |
| `include_map` | no | Include estimated free-grid time-gain cells. Default `false`. |
| `max_exact_candidates` | no | Maximum shortlisted candidates to run through exact SIMAP scoring. Default `40`. |

### Response Shape

```json
{
  "flight_number": "NKS220M1",
  "icao24": "a91e6e",
  "flight_id": "NKS220M1a91e6e",
  "runway": "RW18R",
  "arrival_cluster": "SE",
  "operational_mask": "south",
  "target_time_gain_s": 90.0,
  "attempt_status": {
    "previous_attempt_count": 0,
    "remaining_attempts": 2,
    "replaced_dogleg_used": false,
    "replaced_dogleg_available": true
  },
  "best_identified_candidate": null,
  "best_free_candidate": null,
  "recommendation": null,
  "rejected_counts": {},
  "evaluated_candidate_count": 0,
  "map_cells": null
}
```

Candidate fields include:

- `candidate_kind`: `identified` or `free`.
- `variant`: `sandwiched_dogleg` or `replaced_dogleg`.
- `lat`, `lon`, and optional `fix_identifier`.
- `projected_segment_index`, `f_a`, and `f_b` for the route segment used.
- `actual_time_gain_s`, `estimated_time_gain_s`, and error fields.
- `metrics` from the exact path-stretch simulation.
- `path_stretch_request`, a complete request body for
  `/tools/path-stretch/simulate`.

### Workflow

1. Call AMAN or evaluator tools to determine the time that should be gained.
2. Call `POST /tools/sensory/vector-assist`.
3. Send `recommendation.path_stretch_request` to
   `POST /tools/path-stretch/simulate`.
4. Re-run feasibility, conflict, and runway-overlap evaluators.
5. Save the path-stretch draft with `PUT /diff/path-stretch/{flight_id}` if the
   evaluated result is acceptable.

### Operational Rules

- `NE` and `NW` arrivals use the north operational mask, defined by `TTT`,
  `WLLTR`, and `PRX`.
- `SE` and `SW` arrivals use the south operational mask, defined by `TTT`,
  `BGTOE`, and `WAITT`.
- A vector-assist-tagged path stretch can be saved at most twice per flight.
- Only one vector-assist-tagged `replaced_dogleg` is allowed per flight.
- Generic path-stretch edits without `vector_assist` metadata keep existing
  behavior.

### Errors

Validation failures return HTTP `400`, for example:

- unknown `flight_id`;
- negative or non-finite target time;
- missing served `base_route`, `cas_profile`, or trajectory points;
- missing or unsupported `arrival_cluster`;
- missing named operational-mask fixes;
- exhausted vector-assist attempt limits.
