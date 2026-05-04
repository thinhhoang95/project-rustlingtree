# Scenario Manager API

This API serves the KDFW scenario resources from memory so the catalog files are loaded once at startup instead of on every request.

It exposes two read-only schedule endpoints:

- departure schedule: ADS-B compressed trajectory payload plus departure metadata
- arrival schedule: SIMAP compressed trajectory payload plus arrival metadata

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
  "events_count": 506,
  "arrivals_count": 364,
  "departures_count": 142,
  "fix_sequences_count": 11829,
  "compressed_flights_count": 2046,
  "artifact_flights_count": 988,
  "simap_arrival_flights_count": 988,
  "adsb_compressed_flights_count": 2046,
  "arrivals_missing_fix_sequences_count": 0,
  "arrivals_missing_trajectories_count": 2,
  "arrivals_missing_artifacts_count": 2,
  "arrivals_missing_simap_trajectories_count": 2,
  "departures_missing_trajectories_count": 2,
  "departures_missing_adsb_trajectories_count": 2,
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

Returns the arrival schedule sorted by the first-fix handoff time. Arrivals use SIMAP-generated compressed trajectories.

Each item keeps the original compressed trajectory payload shape and adds:

- `time_at_first_fix`
- `time_at_first_fix_utc`
- `time_at_last_event`
- `time_at_last_event_utc`
- `runway`
- `original_fix_sequence`
- `original_fix_count`

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

Example:

```bash
curl http://127.0.0.1:8000/arrivals | python -m json.tool
```

Example response shape:

```json
[
  {
    "flight_id": "AAL1111M1ab2c04",
    "callsign": "AAL1111M1",
    "icao24": "ab2c04",
    "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
    "points": [
      [1743465659, 34.45838928222656, -95.25550063775512, 11170.920000000002, 3],
      [1743465839, 34.20744323730469, -95.51782724808676, 11148.06, 2]
    ],
    "time_at_first_fix": 1743465659,
    "time_at_first_fix_utc": "2025-04-01T00:00:59Z",
    "time_at_last_event": 1743465839,
    "time_at_last_event_utc": "2025-04-01T00:02:19Z",
    "runway": "35C",
    "original_fix_sequence": "PLEZE>BIRLE>CHMLI>...",
    "original_fix_count": 14
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
