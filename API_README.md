# Scenario Manager API

This API serves the KDFW scenario resources from memory so the catalog files are loaded once at startup instead of on every request.

It exposes two read-only schedule endpoints:

- departure schedule: ADS-B compressed trajectory payload plus departure metadata
- arrival schedule: SIMAP compressed trajectory payload plus arrival metadata

Related tool documentation:

- [Advisors](API_ADVISORS.md)
- [Evaluators](API_EVAL_TOOLS.md)
- [Edit tools](API_EDIT_TOOLS.md)
- [Sensory tools](API_SENSORY_TOOLS.md)

## Run The API

Start the server with either command:

```bash
scenario-manager-api
```

or:

```bash
uvicorn mcp_tools.scenario_manager.api:app --host 127.0.0.1 --port 8000
```

By default the server reads the entry marked `"default": true` in `data_manifest.json`.
The current default entry is:

- `data/adsb/catalogs/2026-04-01_landings_and_departures.csv`
- `data/adsb/catalogs/2026-04-01_fix_sequences.csv`
- `data/artifacts/simap_arrival_flights.jsonl`
- `data/adsb/compressed/adsb_compressed_flights.jsonl`

## Endpoints

### `GET /health`

Returns a summary of the loaded resources.

Example:

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "events_count": 2046,
  "arrivals_count": 1026,
  "departures_count": 1020,
  "fix_sequences_count": 12048,
  "compressed_flights_count": 2017,
  "artifact_flights_count": 787,
  "simap_arrival_flights_count": 787,
  "adsb_compressed_flights_count": 2017,
  "arrivals_missing_fix_sequences_count": 0,
  "arrivals_missing_trajectories_count": 239,
  "arrivals_missing_artifacts_count": 239,
  "arrivals_missing_simap_trajectories_count": 239,
  "departures_missing_trajectories_count": 15,
  "departures_missing_adsb_trajectories_count": 15,
  "skipped_departures_count": 1020,
  "resource_paths": {
    "data_manifest": "data_manifest.json",
    "data_date": "2026-04-01",
    "events": "data/adsb/catalogs/2026-04-01_landings_and_departures.csv",
    "landings_and_departures": "data/adsb/catalogs/2026-04-01_landings_and_departures.csv",
    "fix_sequences": "data/adsb/catalogs/2026-04-01_fix_sequences.csv",
    "simap_arrival_trajectories": "data/artifacts/simap_arrival_flights.jsonl",
    "simap_arrival_artifact_manifest": "data/artifacts/manifest.json",
    "adsb_compressed_trajectories": "data/adsb/compressed/adsb_compressed_flights.jsonl",
    "adsb_compressed_metadata": "data/adsb/compressed/metadata.json"
  }
}
```

### `GET /departures`

Returns the departure schedule sorted by `departure_time`. Departures use compressed ADS-B trajectories directly because SIMAP does not modify departures.

Each item keeps the ADS-B compressed trajectory payload shape and adds:

- `departure_time`
- `departure_time_utc`
- `runway`

The trajectory payload includes:

- `flight_id`
- `callsign`
- `icao24`
- `columns`
- `points`
- `breakpoint_mask_bits`
- `lateral_breakpoint_times`
- `altitude_breakpoint_times`
- `first_time`
- `last_time`
- `raw_point_count`
- `compressed_point_count`
- tolerance metadata

Example:

```bash
curl http://127.0.0.1:8000/departures | python -m json.tool
```

Example response shape:

```json
[
  {
    "flight_id": "AAL2658M1a08a1a",
    "callsign": "AAL2658M1",
    "icao24": "a08a1a",
    "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
    "points": [
      [1775024459, 32.86746267545021, -97.05159712810904, 662.94, 3],
      [1775026139, 35.419891357421875, -99.24519872188568, 10408.92, 3]
    ],
    "breakpoint_mask_bits": {
      "lateral": 1,
      "altitude": 2
    },
    "lateral_breakpoint_times": [1775024459, 1775026139],
    "altitude_breakpoint_times": [1775024459, 1775026139],
    "first_time": 1775024459,
    "last_time": 1775026139,
    "raw_point_count": 18,
    "compressed_point_count": 2,
    "lateral_tolerance_m": 100.0,
    "altitude_tolerance_m": 50.0,
    "departure_time": 1775024459,
    "departure_time_utc": "2026-04-01T06:20:59Z",
    "runway": "36L"
  }
]
```

### `GET /arrivals`

Returns the arrival schedule sorted by the first-fix handoff time. Arrivals use SIMAP-generated compressed trajectories for the base route from the first route fix through the ATC wait point, selected final fix, and runway threshold.

Each item keeps the original compressed trajectory payload shape and adds:

- `time_at_first_fix`
- `time_at_first_fix_utc`
- `time_at_last_event`
- `time_at_last_event_utc`
- `runway`
- `route_type`: currently `"base-route"` for generated arrival artifacts
- `fix_sequence`: the base-route sequence from the first fix through runway threshold
- `fix_count`
- `base_route`
- `atc_wait_point`
- `wait_atc_point`: compatibility alias for `atc_wait_point`
- `final_fix`
- `baseline_final_fix`: compatibility alias for `final_fix`
- `cas_profile`: full-resolution SIMAP calibrated airspeed profile in knots

The trajectory payload remains compatible with the ADS-B compressed trajectory format:

- `columns`
- `points`
- `breakpoint_mask_bits`
- `lateral_breakpoint_times`
- `altitude_breakpoint_times`
- `first_time`
- `last_time`
- `raw_point_count`
- `compressed_point_count`
- tolerance metadata

The `fix_sequence` is not the original catalog sequence. It is the unique base-route sequence SIMAP used: original fixes up to and including the ATC wait point, then the runway-aligned final fix, then the runway threshold.

The `cas_profile` is separate from trajectory `points` so existing geometry consumers do not need to reinterpret the compressed point schema. It is sampled at the full SIMAP FMS result timestep, not only at lateral or altitude breakpoints. Older artifacts generated before CAS support may omit this field until `simap_arrival_flights.jsonl` is regenerated.

`cas_profile` fields:

- `columns`: `["time", "cas_kts"]`
- `units`: currently `{"cas_kts": "kt"}`
- `source`: currently `"simap_fms_bichannel"`
- `points`: `[time, cas_kts]` samples, where `time` is the epoch timestamp matching the simulated arrival timeline

`base_route` fields:

- `type`: `"base-route"`
- `selection_method`
- `fix_sequence`
- `fix_count`
- `lateral_path`: route tokens in order
- `upstream_identifier`: first route fix used as the SIMAP upstream condition
- `runway`
- `atc_point`
- `final_fix`
- `runway_true_heading_deg`

Arrival artifacts include `atc_wait_point`, a metadata object identifying the fix where the aircraft waits for ATC instruction before being routed direct to the selected final fix. `wait_atc_point` is returned with the same object for backward compatibility.

`atc_wait_point` fields:

- `source`: currently `"fix"` for generated base-route artifacts
- `identifier`: selected fix identifier
- `lat`
- `lon`
- `lateral_path_token`: selected fix identifier
- `route_index`: zero-based index in the original route used to build the base-route prefix
- `distance_nm`
- `ring_inner_nm`
- `ring_outer_nm`
- `selection_method`
- `arrival_cluster`
- `gate_cluster`
- `gate_radius_nm`
- `gate_lat`
- `gate_lon`
- `gate_classification_fallback`
- `capture_margin_nm`

`final_fix` fields:

- `identifier`
- `lat`
- `lon`
- `distance_nm`
- `along_track_nm`
- `cross_track_nm`
- `target_distance_nm`
- `cross_track_tolerance_nm`

### RNAV/PBN Fly-By Turn Example

Arrival lateral paths are treated as RNAV-style fly-by paths, not as a sequence of fly-over corner points. Public FAA RNAV/PBN guidance distinguishes fly-by waypoints from fly-over waypoints: at a fly-by waypoint, the aircraft anticipates the turn and begins banking before reaching the fix so it can roll out on the next course. The turn anticipation distance is driven mainly by course change, groundspeed, and bank capability.

For example, an arrival may expose this base route:

```json
{
  "fix_sequence": "KIILO>SHMPP>ZROBA>CURLE>TANNO>DELMO>SILER>ZINGG>RW17C",
  "base_route": {
    "lateral_path": ["KIILO", "SHMPP", "ZROBA", "CURLE", "TANNO", "DELMO", "SILER", "ZINGG", "RW17C"],
    "atc_point": {
      "identifier": "SILER"
    },
    "final_fix": {
      "identifier": "ZINGG"
    },
    "runway": "RW17C"
  }
}
```

SIMAP constructs a continuous line-LNAV path through that route:

1. It follows the inbound line toward each interior fix.
2. For eligible course changes, it computes a tangent circular turn before the fix.
3. It exits the turn on the outbound line toward the next fix.

In the `SILER > ZINGG > RW17C` segment, this means the simulated aircraft should begin banking before `ZINGG`, then roll out aligned with the runway course toward `RW17C`. The compressed trajectory `points` therefore may not pass exactly through every intermediate fix coordinate. That is expected for fly-by RNAV behavior; use `base_route.lateral_path` for the route intent and the trajectory `points` for the flown path.

FAA references:

- [FAA AIP ENR 1.16, RNAV routes and waypoints](https://www.faa.gov/air_traffic/publications/atpubs/aip_html/part2_enr_section_1.16.html)
- [FAA ATBARC RNAV flight behavior and turn anticipation](https://www.faa.gov/air_traffic/publications/atpubs/atbarc/03-5.htm)

Example:

```bash
curl http://127.0.0.1:8000/arrivals | python -m json.tool
```

Example response shape:

```json
[
  {
    "flight_id": "NKS220M1a91e6e",
    "callsign": "NKS220M1",
    "icao24": "a91e6e",
    "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
    "points": [
      [1775020679, 31.909439086914062, -95.5360107421875, 10972.800000000001, 3],
      [1775021708, 32.89953642491388, -97.0298451398072, 159.7152, 3]
    ],
    "breakpoint_mask_bits": {
      "lateral": 1,
      "altitude": 2
    },
    "cas_profile": {
      "columns": ["time", "cas_kts"],
      "units": {
        "cas_kts": "kt"
      },
      "source": "simap_fms_bichannel",
      "points": [
        [1775020679, 238.5],
        [1775020681, 238.1],
        [1775020683, 237.8]
      ]
    },
    "lateral_breakpoint_times": [1775020679, 1775021708],
    "altitude_breakpoint_times": [1775020679, 1775021708],
    "route_type": "base-route",
    "fix_sequence": "MUZZY>BEREE>TACKE>PAXTN>FIVIS>STONZ>LEGRE>RW18R",
    "fix_count": 8,
    "atc_wait_point": {
      "source": "fix",
      "identifier": "STONZ",
      "lat": 32.97284722222222,
      "lon": -96.90284444444445,
      "lateral_path_token": "STONZ",
      "route_index": 5,
      "distance_nm": 8.380134836797653,
      "selection_method": "cluster_capture_polygon",
      "arrival_cluster": "SE",
      "gate_cluster": "SE"
    },
    "final_fix": {
      "identifier": "LEGRE",
      "lat": 33.04105833333333,
      "lon": -97.05399444444444,
      "distance_nm": 7.519805065979901,
      "along_track_nm": 7.519805056162219,
      "cross_track_nm": 0.000384257868011151,
      "target_distance_nm": 7.0,
      "cross_track_tolerance_nm": 0.15
    },
    "base_route": {
      "type": "base-route",
      "selection_method": "atc_direct_to_runway_aligned_final_fix",
      "fix_sequence": "MUZZY>BEREE>TACKE>PAXTN>FIVIS>STONZ>LEGRE>RW18R",
      "fix_count": 8,
      "lateral_path": ["MUZZY", "BEREE", "TACKE", "PAXTN", "FIVIS", "STONZ", "LEGRE", "RW18R"],
      "upstream_identifier": "MUZZY",
      "runway": "RW18R"
    },
    "wait_atc_point": {
      "source": "fix",
      "identifier": "STONZ",
      "lat": 32.97284722222222,
      "lon": -96.90284444444445,
      "lateral_path_token": "STONZ",
      "route_index": 5,
      "distance_nm": 8.380134836797653,
      "selection_method": "cluster_capture_polygon",
      "arrival_cluster": "SE",
      "gate_cluster": "SE"
    },
    "baseline_final_fix": {
      "identifier": "LEGRE",
      "lat": 33.04105833333333,
      "lon": -97.05399444444444,
      "distance_nm": 7.519805065979901,
      "along_track_nm": 7.519805056162219,
      "cross_track_nm": 0.000384257868011151,
      "target_distance_nm": 7.0,
      "cross_track_tolerance_nm": 0.15
    },
    "first_time": 1775020679,
    "last_time": 1775021708,
    "raw_point_count": 516,
    "compressed_point_count": 45,
    "lateral_tolerance_m": 100.0,
    "altitude_tolerance_m": 50.0,
    "time_at_first_fix": 1775020679,
    "time_at_first_fix_utc": "2026-04-01T05:17:59Z",
    "time_at_last_event": 1775021708,
    "time_at_last_event_utc": "2026-04-01T05:35:08Z",
    "runway": "RW18R"
  }
]
```

## Python Client Example

```python
from __future__ import annotations

import json
from urllib.request import urlopen

base_url = "http://127.0.0.1:8000"

with urlopen(f"{base_url}/health") as response:
    health = json.load(response)
with urlopen(f"{base_url}/departures") as response:
    departures = json.load(response)
with urlopen(f"{base_url}/arrivals") as response:
    arrivals = json.load(response)

print(health["status"])
print(departures[0]["flight_id"], departures[0]["departure_time_utc"])
print(arrivals[0]["flight_id"], arrivals[0]["time_at_first_fix_utc"], arrivals[0]["time_at_last_event_utc"])
```

## Notes

- The API is read-only.
- Resources are loaded once at startup and reused across requests.
- Arrival schedules use trajectory times from simulation artifacts (`first_time`/`last_time`) and fall back to fix-sequence values when needed.
