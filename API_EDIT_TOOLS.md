# API Edit Tools

The scenario-manager API exposes trajectory edit endpoints for simulation
arrivals. Edit endpoints are write-capable workflows: a `simulate` request
creates an in-memory draft, and a later `save` request promotes that draft into
the active scenario-manager diff.

Current edit tools:

- Path Stretch: edit an arrival's lateral route and recompute the trajectory.
- Speed Intervention: add along-track speed advisories and recompute the trajectory.

## Common Concepts

### Base URL

Examples assume the Scenario Manager API is running on:

```text
http://127.0.0.1:8000
```

Start it with:

```bash
uvicorn mcp_tools.scenario_manager.api:app --host 127.0.0.1 --port 8000
```

### Arrivals Only

Both edit workflows operate on served simulation arrivals from:

```text
GET /arrivals
```

Use the `flight_id` field from an arrival item when calling edit endpoints.
Departures are not supported by these tools.

### Draft Then Save

Edit tools use a two-step mutation model:

1. `POST /tools/.../simulate` validates the request, runs SIMAP, and returns a
   draft trajectory. This does not mutate `/arrivals`.
2. `PUT /diff/.../{flight_id}` saves a returned `draft_id` as the active diff
   for that arrival. After save, `/arrivals` serves the edited trajectory.

Drafts are stored in Scenario Manager memory. Restarting the API clears drafts
and active diffs.

### One Active Trajectory Edit Per Flight

Saving a Path Stretch or Speed Intervention removes any existing active
`path-stretch` or `speed-intervention` diff for the same `flight_id`, then adds
the newly saved diff. This keeps exactly one trajectory override active per
arrival.

### Units And Coordinates

| Field | Unit / convention |
| --- | --- |
| `lat`, `lon` | Decimal degrees in API request and response payloads. |
| Route coordinate token | `[lat, lon]` when stored in JSON payloads. |
| SIMAP coordinate route token | `(lat, lon)` internally. |
| Client map coordinate | `[lon, lat]` in MapLibre UI code, not the API. |
| `s_m` | Meters remaining to runway threshold. `0` is the runway threshold; larger values are farther upstream. |
| `station_nm_to_runway` | `s_m / 1852`, in nautical miles remaining to runway threshold. |
| `cas_kts` | Calibrated airspeed in knots. |
| `*_nm` | Nautical miles. |
| `*_min` | Minutes. |

### Error Shape

Validation failures are returned as HTTP `400` responses:

```json
{
  "detail": "path-stretch route requires at least two points"
}
```

FastAPI request-shape validation errors may use the standard HTTP `422`
validation response when required JSON fields have the wrong type.

## Inspect Active Diffs

`GET /diff`

Returns the active in-memory intervention diff records.

### Example Request

```bash
curl http://127.0.0.1:8000/diff | python -m json.tool
```

### Example Response

```json
[
  {
    "id": "path-stretch-ARR1-1a2b3c4d5e6f",
    "flight_id": "ARR1",
    "created_at_utc": "2026-05-21T00:00:00Z",
    "source": "path-stretching",
    "type": "path-stretch",
    "command": {
      "type": "path_stretch",
      "handles": [],
      "route": [
        {
          "token_type": "fix",
          "lat": 32.95,
          "lon": -97.04,
          "fix_identifier": "TTT"
        },
        {
          "token_type": "coordinate",
          "lat": 32.92,
          "lon": -97.08,
          "fix_identifier": null
        },
        {
          "token_type": "fix",
          "lat": 32.899,
          "lon": -97.037,
          "fix_identifier": "RW35C"
        }
      ],
      "old_route": ["TTT", "RW35C"],
      "new_route": ["TTT", [32.92, -97.08], "RW35C"]
    },
    "overrides": {
      "route_type": "path-stretch",
      "points": []
    },
    "base": {
      "route_type": "base-route",
      "base_route": {}
    }
  }
]
```

### Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Saved draft id. Same value as the `draft_id` returned by the corresponding simulate response. |
| `flight_id` | string | Arrival flight id this diff applies to. |
| `created_at_utc` | string | UTC creation timestamp for the draft/diff. |
| `source` | string | Workflow source. Current values are `path-stretching` and `speed-intervention`. |
| `type` | string | Active edit type. Current values are `path-stretch` and `speed-intervention`. |
| `command` | object | User command payload retained for audit/debugging. Shape depends on edit type. |
| `overrides` | object | Trajectory-bearing payload fields served over the base arrival after save. |
| `base` | object | Selected base-arrival metadata captured before the edit was applied. |

`overrides` is the full recomputed trajectory payload. It can include:

- `route_type`
- `fix_sequence`
- `fix_count`
- `columns`
- `breakpoint_mask_bits`
- `points`
- `cas_profile`
- `lateral_breakpoint_times`
- `altitude_breakpoint_times`
- `first_time`
- `last_time`
- `raw_point_count`
- `compressed_point_count`
- `lateral_tolerance_m`
- `altitude_tolerance_m`
- `simulation`
- `final_fix`
- `baseline_final_fix`
- `base_route`
- `atc_wait_point`
- `wait_atc_point`
- `path_stretch`
- `speed_intervention`

Only these trajectory override fields are applied back onto `/arrivals`.

`base` captures the selected pre-edit metadata:

| Field | Type | Description |
| --- | --- | --- |
| `route_type` | string or null | Served route type before the edit. |
| `fix_sequence` | string or null | Served route sequence before the edit. |
| `fix_count` | integer or null | Served route fix count before the edit. |
| `base_route` | object or null | Served base route metadata before the edit. |
| `final_fix` | object or null | Served final-fix metadata before the edit. |
| `baseline_final_fix` | object or null | Baseline final-fix metadata before the edit. |
| `simulation` | object or null | Served SIMAP diagnostic payload before the edit. |

## Shared Simulate Response Fields

Both simulate endpoints return a common draft envelope.

| Field | Type | Description |
| --- | --- | --- |
| `draft_id` | string | In-memory draft identifier. Use this value in the matching save endpoint. |
| `flight_id` | string | Arrival flight id from the request. |
| `created_at_utc` | string | Draft creation time in UTC. |
| `trajectory` | object | Recomputed arrival payload. It has the same general shape as an `/arrivals` item and includes trajectory points, timing, route metadata, `simulation`, and edit-specific metadata. |
| `metrics` | object | Distance and elapsed-time comparison between the served/base trajectory and the simulated edit. |
| `simulation` | object | SIMAP diagnostic payload for the edited trajectory. |

Common `metrics` fields:

| Field | Type | Description |
| --- | --- | --- |
| `old_distance_nm` | number | Distance along the currently served arrival trajectory before the edit, in nautical miles. |
| `new_distance_nm` | number | Distance along the simulated edited trajectory, in nautical miles. |
| `delta_distance_nm` | number | `new_distance_nm - old_distance_nm`. Positive means the edited trajectory is longer. |
| `old_elapsed_min` | number | Elapsed time before the edit, in minutes. |
| `new_elapsed_min` | number | Elapsed time after the simulated edit, in minutes. |
| `delta_elapsed_min` | number | `new_elapsed_min - old_elapsed_min`. Positive means the edited trajectory takes longer. |

Common `simulation` fields:

| Field | Type | Description |
| --- | --- | --- |
| `success` | boolean | Whether SIMAP/FMS planning succeeded. |
| `message` | string | SIMAP/FMS status message. |
| `max_abs_cross_track_m` | number | Maximum absolute lateral cross-track error in meters. |
| `max_abs_track_error_rad` | number | Maximum absolute track-angle error in radians. |
| `final_threshold_error_m` | number | Final distance/position error at runway threshold in meters. |

The exact `trajectory` payload is intentionally compatible with `/arrivals`.
Common fields include `columns`, `points`, `first_time`, `last_time`,
`base_route`, `cas_profile`, `route_type`, `final_fix`, `baseline_final_fix`,
`path_stretch`, and `speed_intervention` when applicable.

## Path Stretch

Path Stretch lets a client submit an edited full lateral route for an arrival.
Scenario Manager normalizes the route, runs FMSBiChannel SIMAP, and returns a
draft trajectory.

### Simulate Path Stretch

`POST /tools/path-stretch/simulate`

Preferred request body:

```json
{
  "flight_id": "ARR1",
  "route": [
    {
      "token_type": "fix",
      "fix_identifier": "TTT",
      "lat": 32.869,
      "lon": -97.041
    },
    {
      "token_type": "coordinate",
      "lat": 32.92,
      "lon": -97.08
    },
    {
      "token_type": "fix",
      "fix_identifier": "RW35C",
      "lat": 32.899,
      "lon": -97.037
    }
  ]
}
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/tools/path-stretch/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "flight_id": "ARR1",
    "route": [
      {"token_type": "fix", "fix_identifier": "TTT", "lat": 32.869, "lon": -97.041},
      {"token_type": "coordinate", "lat": 32.92, "lon": -97.08},
      {"token_type": "fix", "fix_identifier": "RW35C", "lat": 32.899, "lon": -97.037}
    ]
  }' | python -m json.tool
```

Example response:

```json
{
  "draft_id": "path-stretch-ARR1-1a2b3c4d5e6f",
  "flight_id": "ARR1",
  "created_at_utc": "2026-05-21T00:00:00Z",
  "trajectory": {
    "flight_id": "ARR1",
    "route_type": "path-stretch",
    "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
    "points": [
      [1775021000, 32.869, -97.041, 3000.0, 3],
      [1775021900, 32.899, -97.037, 185.0, 3]
    ],
    "base_route": {
      "type": "path-stretch",
      "selection_method": "interactive_path_stretch",
      "lateral_path": ["TTT", [32.92, -97.08], "RW35C"],
      "route_points": [
        {"token_type": "fix", "lat": 32.869, "lon": -97.041, "fix_identifier": "TTT"},
        {"token_type": "coordinate", "lat": 32.92, "lon": -97.08, "fix_identifier": null},
        {"token_type": "fix", "lat": 32.899, "lon": -97.037, "fix_identifier": "RW35C"}
      ]
    },
    "path_stretch": {
      "handles": [],
      "handle_count": 0,
      "route_points": [
        {"token_type": "fix", "lat": 32.869, "lon": -97.041, "fix_identifier": "TTT"},
        {"token_type": "coordinate", "lat": 32.92, "lon": -97.08, "fix_identifier": null},
        {"token_type": "fix", "lat": 32.899, "lon": -97.037, "fix_identifier": "RW35C"}
      ],
      "route_point_count": 3
    },
    "simulation": {
      "success": true,
      "message": "stitched level cruise and descent reached target altitude at threshold",
      "max_abs_cross_track_m": 8.2,
      "max_abs_track_error_rad": 0.01,
      "final_threshold_error_m": 12.4
    }
  },
  "metrics": {
    "old_distance_nm": 42.1,
    "new_distance_nm": 47.8,
    "delta_distance_nm": 5.7,
    "old_elapsed_min": 12.4,
    "new_elapsed_min": 13.8,
    "delta_elapsed_min": 1.4
  },
  "old_route_tokens": ["TTT", "RW35C"],
  "new_route_tokens": ["TTT", "32.920000,-97.080000", "RW35C"],
  "simulation": {
    "success": true,
    "message": "stitched level cruise and descent reached target altitude at threshold",
    "max_abs_cross_track_m": 8.2,
    "max_abs_track_error_rad": 0.01,
    "final_threshold_error_m": 12.4
  }
}
```

#### Request Fields

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `flight_id` | yes | string | Arrival flight id from `/arrivals`. Leading/trailing whitespace is stripped. |
| `route` | preferred | array | Full edited lateral route. If supplied, it replaces the legacy `handles` workflow. Requires at least two points. |
| `handles` | legacy | array | Insert-only legacy shape. Supported for compatibility when `route` is omitted. Requires at least one handle. |

#### `route[]` Fields

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `token_type` | yes | string | Either `fix` or `coordinate`. |
| `lat` | yes | number | Route point latitude in decimal degrees. Must be finite and valid latitude. |
| `lon` | yes | number | Route point longitude in decimal degrees. Must be finite and valid longitude. |
| `fix_identifier` | yes for `fix` | string or null | Named fix/runway token. It is uppercased and must exist in the configured fix catalog. Ignored for `coordinate` points. |

For `fix` route points, SIMAP receives the uppercase `fix_identifier` string. For
`coordinate` route points, SIMAP receives a `(lat, lon)` tuple and the saved
JSON route token is `[lat, lon]`.

#### Legacy `handles[]` Fields

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `insert_after_index` | yes | integer | Zero-based base-route segment start index. The handle is inserted after this base route token. Must target an existing segment. |
| `token_type` | yes | string | Either `fix` or `coordinate`. |
| `lat` | yes | number | Inserted point latitude in decimal degrees. |
| `lon` | yes | number | Inserted point longitude in decimal degrees. |
| `fix_identifier` | yes for `fix` | string or null | Named fix token. Required and fix-catalog validated when `token_type` is `fix`. |

#### Path Stretch Response Fields

Path Stretch adds these fields to the shared simulate response:

| Field | Type | Description |
| --- | --- | --- |
| `old_route_tokens` | array of strings | Display tokens for the served base route before editing. Coordinate tokens are formatted as `"lat,lon"`. |
| `new_route_tokens` | array of strings | Display tokens for the edited route sent to SIMAP after normalization and consecutive duplicate removal. |
| `trajectory.path_stretch` | object | Edit metadata stored on the simulated trajectory payload. |
| `trajectory.base_route.route_points` | array or omitted | Full normalized route-point list when the preferred `route` request shape was used. |
| `trajectory.base_route.handles` | array | Normalized legacy handles inserted into the base route. Empty for full-route edits. |

Path Stretch saved `diff.command` fields:

| Field | Type | Description |
| --- | --- | --- |
| `type` | string | Always `path_stretch`. |
| `handles` | array | Normalized legacy handles used to create the edit. Empty for full-route edits. |
| `route` | array or null | Normalized full route points from the request, or `null` for legacy handle requests. |
| `old_route` | array | Base lateral route tokens before editing. Fix tokens are strings; coordinate tokens are `[lat, lon]`. |
| `new_route` | array | Edited lateral route tokens after normalization and duplicate removal. Fix tokens are strings; coordinate tokens are `[lat, lon]`. |

`trajectory.path_stretch` fields:

| Field | Type | Description |
| --- | --- | --- |
| `handles` | array | Normalized legacy handles. Empty when full-route editing is used. |
| `handle_count` | integer | Number of normalized legacy handles. |
| `route_points` | array or null | Normalized full route points from the request, or `null` for legacy handle requests. |
| `route_point_count` | integer or null | Number of full route points from the request, or `null` for legacy handle requests. |

#### Path Stretch Validation

The endpoint returns HTTP `400` when:

- `flight_id` is empty or unknown.
- the arrival lacks `base_route.lateral_path`.
- neither `route` nor `handles` is provided.
- `route` contains fewer than two points.
- the normalized edited route has fewer than two distinct points.
- the edited route is identical to the served base route.
- a `fix` route point or handle omits `fix_identifier`.
- a `fix_identifier` is not in the configured fix catalog.
- a coordinate is not finite or is outside valid latitude/longitude bounds.
- a legacy `insert_after_index` does not target an existing base-route segment.
- the arrival does not have enough trajectory seed data to run SIMAP.

### Save Path Stretch

`PUT /diff/path-stretch/{flight_id}`

Body:

```json
{
  "draft_id": "path-stretch-ARR1-1a2b3c4d5e6f"
}
```

Example request:

```bash
curl -X PUT http://127.0.0.1:8000/diff/path-stretch/ARR1 \
  -H "Content-Type: application/json" \
  -d '{"draft_id": "path-stretch-ARR1-1a2b3c4d5e6f"}' \
  | python -m json.tool
```

Example response:

```json
{
  "diff": {
    "id": "path-stretch-ARR1-1a2b3c4d5e6f",
    "flight_id": "ARR1",
    "created_at_utc": "2026-05-21T00:00:00Z",
    "source": "path-stretching",
    "type": "path-stretch",
    "command": {
      "type": "path_stretch",
      "handles": [],
      "route": [
        {"token_type": "fix", "lat": 32.869, "lon": -97.041, "fix_identifier": "TTT"},
        {"token_type": "coordinate", "lat": 32.92, "lon": -97.08, "fix_identifier": null},
        {"token_type": "fix", "lat": 32.899, "lon": -97.037, "fix_identifier": "RW35C"}
      ],
      "old_route": ["TTT", "RW35C"],
      "new_route": ["TTT", [32.92, -97.08], "RW35C"]
    },
    "overrides": {
      "route_type": "path-stretch",
      "points": []
    },
    "base": {
      "route_type": "base-route",
      "base_route": {}
    }
  },
  "arrival": {
    "flight_id": "ARR1",
    "route_type": "path-stretch",
    "path_stretch": {
      "handle_count": 0,
      "route_point_count": 3
    }
  }
}
```

#### Save Request Fields

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `draft_id` | yes | string | Draft id returned by `POST /tools/path-stretch/simulate`. |

#### Save Response Fields

| Field | Type | Description |
| --- | --- | --- |
| `diff` | object | Saved active diff record. |
| `arrival` | object | Updated served arrival after the diff is applied. This is the same arrival shape returned by `/arrivals`. |

Save returns HTTP `400` when the draft id is unknown or when the draft belongs
to a different `flight_id` than the path parameter.

## Speed Intervention

Speed Intervention lets a client add one or more ATC speed advisories at SIMAP
remaining along-track stations. Scenario Manager runs both a baseline
FMSBiChannel plan and a speed-intervened FMSBiChannel plan, then returns the
intervened draft trajectory plus before/after metrics.

### Simulate Speed Intervention

`POST /tools/speed-intervention/simulate`

Request body:

```json
{
  "flight_id": "ARR1",
  "advisories": [
    {
      "s_m": 42000.0,
      "cas_kts": 180.0,
      "lat": 32.91,
      "lon": -97.04
    }
  ]
}
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/tools/speed-intervention/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "flight_id": "ARR1",
    "advisories": [
      {"s_m": 42000.0, "cas_kts": 180.0, "lat": 32.91, "lon": -97.04}
    ]
  }' | python -m json.tool
```

Example response:

```json
{
  "draft_id": "speed-intervention-ARR1-a1b2c3d4e5f6",
  "flight_id": "ARR1",
  "created_at_utc": "2026-05-21T00:00:00Z",
  "trajectory": {
    "flight_id": "ARR1",
    "route_type": "speed-intervention",
    "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
    "points": [
      [1775021000, 32.91, -97.04, 3000.0, 3],
      [1775022050, 32.899, -97.037, 185.0, 3]
    ],
    "base_route": {
      "type": "speed-intervention",
      "selection_method": "interactive_speed_intervention",
      "speed_advisories": [
        {
          "s_m": 42000.0,
          "station_nm_to_runway": 22.67818574514039,
          "cas_kts": 180.0,
          "lat": 32.91,
          "lon": -97.04
        }
      ]
    },
    "speed_intervention": {
      "advisories": [
        {
          "s_m": 42000.0,
          "station_nm_to_runway": 22.67818574514039,
          "cas_kts": 180.0,
          "lat": 32.91,
          "lon": -97.04
        }
      ],
      "advisory_count": 1
    },
    "simulation": {
      "success": true,
      "message": "stitched level cruise and descent reached target altitude at threshold",
      "max_abs_cross_track_m": 7.5,
      "max_abs_track_error_rad": 0.01,
      "final_threshold_error_m": 10.1
    }
  },
  "metrics": {
    "old_distance_nm": 42.1,
    "new_distance_nm": 42.1,
    "delta_distance_nm": 0.0,
    "old_elapsed_min": 12.4,
    "new_elapsed_min": 14.2,
    "delta_elapsed_min": 1.8,
    "equivalent_distance_nm": 13.3,
    "baseline_distance_nm": 42.1,
    "speed_intervention_distance_nm": 42.1
  },
  "advisories": [
    {
      "s_m": 42000.0,
      "station_nm_to_runway": 22.67818574514039,
      "cas_kts": 180.0,
      "lat": 32.91,
      "lon": -97.04
    }
  ],
  "simulation": {
    "success": true,
    "message": "stitched level cruise and descent reached target altitude at threshold",
    "max_abs_cross_track_m": 7.5,
    "max_abs_track_error_rad": 0.01,
    "final_threshold_error_m": 10.1
  },
  "baseline_simulation": {
    "success": true,
    "message": "baseline profile succeeded",
    "max_abs_cross_track_m": 6.9,
    "max_abs_track_error_rad": 0.01,
    "final_threshold_error_m": 9.8
  }
}
```

#### Request Fields

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `flight_id` | yes | string | Arrival flight id from `/arrivals`. Leading/trailing whitespace is stripped. |
| `advisories` | yes | array | One or more speed advisories. The server sorts them descending by `s_m` before building SIMAP ATC speed segments. |

#### `advisories[]` Fields

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `s_m` | yes | number | Remaining along-track station in meters to runway threshold. Must be finite and within the arrival reference path: `0 <= s_m <= start_s_m`. |
| `cas_kts` | yes | number | Issued calibrated airspeed in knots. Must be finite and positive. SIMAP/FMS may reject values that are not lower than the managed profile or that violate mode CAS bounds. |
| `lat` | no | number or null | Advisory display latitude in decimal degrees. Validated if supplied. Used for payload/UI metadata, not for station computation. |
| `lon` | no | number or null | Advisory display longitude in decimal degrees. Validated if supplied. Used for payload/UI metadata, not for station computation. |

#### Speed Intervention Response Fields

Speed Intervention adds these fields to the shared simulate response:

| Field | Type | Description |
| --- | --- | --- |
| `advisories` | array | Normalized advisory payloads, sorted descending by `s_m`. |
| `baseline_simulation` | object | SIMAP diagnostic payload for the baseline run without ATC speed advisories. |
| `trajectory.speed_intervention` | object | Edit metadata stored on the simulated trajectory payload. |
| `trajectory.base_route.speed_advisories` | array | Same normalized advisory list attached to route metadata. |

Speed Intervention saved `diff.command` fields:

| Field | Type | Description |
| --- | --- | --- |
| `type` | string | Always `speed_intervention`. |
| `advisories` | array | Normalized advisory payloads, sorted descending by `s_m`. |
| `base_route` | array or null | Lateral route tokens used for the speed-intervened SIMAP request. Fix tokens are strings; coordinate tokens are `[lat, lon]`. |

`advisories[]` response fields:

| Field | Type | Description |
| --- | --- | --- |
| `s_m` | number | Remaining station in meters to runway threshold. |
| `station_nm_to_runway` | number | Remaining station in nautical miles to runway threshold. |
| `cas_kts` | number | Issued calibrated airspeed in knots. |
| `lat` | number or null | Advisory display latitude, if supplied in the request. |
| `lon` | number or null | Advisory display longitude, if supplied in the request. |

Speed-specific `metrics` fields:

| Field | Type | Description |
| --- | --- | --- |
| `equivalent_distance_nm` | number | Time gain converted to vectoring-equivalent distance using baseline pre-TOD or initial ground speed. |
| `baseline_distance_nm` | number | Distance along the served/base trajectory. |
| `speed_intervention_distance_nm` | number | Distance along the speed-intervened trajectory. |

For Speed Intervention, `old_elapsed_min` is based on the baseline SIMAP run and
`new_elapsed_min` is based on the speed-intervened SIMAP run. The lateral route
usually does not change, so `delta_distance_nm` may be near zero while
`delta_elapsed_min` and `equivalent_distance_nm` are positive.

`baseline_simulation` uses the same diagnostic fields as `simulation`.

#### Speed Intervention Validation

The endpoint returns HTTP `400` when:

- `flight_id` is empty or unknown.
- the arrival lacks `base_route.lateral_path`.
- no advisories are provided.
- `s_m` is not finite.
- `s_m` is outside the arrival reference path.
- `cas_kts` is not finite or is less than or equal to zero.
- optional `lat` or `lon` is supplied but invalid.
- SIMAP/FMS rejects the ATC speed segment, for example because the requested CAS
  is not lower than the managed base profile at the station or violates mode
  CAS bounds.

### Save Speed Intervention

`PUT /diff/speed-intervention/{flight_id}`

Body:

```json
{
  "draft_id": "speed-intervention-ARR1-a1b2c3d4e5f6"
}
```

Example request:

```bash
curl -X PUT http://127.0.0.1:8000/diff/speed-intervention/ARR1 \
  -H "Content-Type: application/json" \
  -d '{"draft_id": "speed-intervention-ARR1-a1b2c3d4e5f6"}' \
  | python -m json.tool
```

Example response:

```json
{
  "diff": {
    "id": "speed-intervention-ARR1-a1b2c3d4e5f6",
    "flight_id": "ARR1",
    "created_at_utc": "2026-05-21T00:00:00Z",
    "source": "speed-intervention",
    "type": "speed-intervention",
    "command": {
      "type": "speed_intervention",
      "advisories": [
        {
          "s_m": 42000.0,
          "station_nm_to_runway": 22.67818574514039,
          "cas_kts": 180.0,
          "lat": 32.91,
          "lon": -97.04
        }
      ],
      "base_route": ["TTT", "RW35C"]
    },
    "overrides": {
      "route_type": "speed-intervention",
      "points": []
    },
    "base": {
      "route_type": "base-route",
      "base_route": {}
    }
  },
  "arrival": {
    "flight_id": "ARR1",
    "route_type": "speed-intervention",
    "speed_intervention": {
      "advisory_count": 1
    }
  }
}
```

#### Save Request Fields

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `draft_id` | yes | string | Draft id returned by `POST /tools/speed-intervention/simulate`. |

#### Save Response Fields

| Field | Type | Description |
| --- | --- | --- |
| `diff` | object | Saved active diff record. |
| `arrival` | object | Updated served arrival after the diff is applied. This is the same arrival shape returned by `/arrivals`. |

Save returns HTTP `400` when the draft id is unknown or when the draft belongs
to a different `flight_id` than the path parameter.

## End-To-End Usage

### Path Stretch Workflow

1. Pick an editable arrival:

   ```bash
   curl http://127.0.0.1:8000/arrivals | python -m json.tool
   ```

2. Submit an edited full route to simulate:

   ```bash
   curl -X POST http://127.0.0.1:8000/tools/path-stretch/simulate \
     -H "Content-Type: application/json" \
     -d '{"flight_id":"ARR1","route":[{"token_type":"fix","fix_identifier":"TTT","lat":32.869,"lon":-97.041},{"token_type":"coordinate","lat":32.92,"lon":-97.08},{"token_type":"fix","fix_identifier":"RW35C","lat":32.899,"lon":-97.037}]}'
   ```

3. Save the returned `draft_id`:

   ```bash
   curl -X PUT http://127.0.0.1:8000/diff/path-stretch/ARR1 \
     -H "Content-Type: application/json" \
     -d '{"draft_id":"path-stretch-ARR1-1a2b3c4d5e6f"}'
   ```

4. Reload `/arrivals`; the saved arrival now has `route_type:
   "path-stretch"` and includes `path_stretch` metadata.

### Speed Intervention Workflow

1. Pick an editable arrival from `/arrivals`.

2. Submit one or more speed advisories:

   ```bash
   curl -X POST http://127.0.0.1:8000/tools/speed-intervention/simulate \
     -H "Content-Type: application/json" \
     -d '{"flight_id":"ARR1","advisories":[{"s_m":42000,"cas_kts":180,"lat":32.91,"lon":-97.04}]}'
   ```

3. Save the returned `draft_id`:

   ```bash
   curl -X PUT http://127.0.0.1:8000/diff/speed-intervention/ARR1 \
     -H "Content-Type: application/json" \
     -d '{"draft_id":"speed-intervention-ARR1-a1b2c3d4e5f6"}'
   ```

4. Reload `/arrivals`; the saved arrival now has `route_type:
   "speed-intervention"` and includes `speed_intervention` metadata.

## Implementation Notes

- Path Stretch is implemented in
  `src/mcp_tools/scenario_manager/path_stretching.py`.
- Speed Intervention is implemented in
  `src/mcp_tools/scenario_manager/speed_intervention.py`.
- HTTP routes are defined in `src/mcp_tools/scenario_manager/api.py`.
- Saved edit diffs are applied when serving arrivals through
  `apply_path_stretch_diff()`.
- The client-side implementation described by `project-rustlingleaves/docs`
  converts MapLibre `[lon, lat]` coordinates into API `lat` and `lon` before
  calling these endpoints.
