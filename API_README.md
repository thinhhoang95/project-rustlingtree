# Scenario Manager API

This API serves the KDFW scenario resources from memory so the catalog files are loaded once at startup instead of on every request.

It exposes two read-only schedule endpoints:

- departure schedule: time and runway for each departure
- arrival schedule: the original `flights.jsonl`-style trajectory payload plus arrival metadata

## Run The API

Start the server with either command:

```bash
scenario-manager-api
```

or:

```bash
uvicorn mcp_tools.scenario_manager.api:app --host 127.0.0.1 --port 8000
```

By default the server reads:

- `data/adsb/catalogs/2025-04-01_landings_and_departures.csv`
- `data/adsb/catalogs/2025-04-01_fix_sequences.csv`
- `data/adsb/compressed/flights.jsonl`

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
  "compressed_flights_count": 502,
  "arrivals_missing_fix_sequences_count": 0,
  "arrivals_missing_trajectories_count": 2,
  "departures_missing_trajectories_count": 2,
  "resource_paths": {
    "events": "data/adsb/catalogs/2025-04-01_landings_and_departures.csv",
    "fix_sequences": "data/adsb/catalogs/2025-04-01_fix_sequences.csv",
    "compressed_flights": "data/adsb/compressed/flights.jsonl"
  }
}
```

### `GET /departures`

Returns the departure schedule sorted by `departure_time`.

Each item contains:

- `flight_id`
- `callsign`
- `icao24`
- `departure_time`
- `departure_time_utc`
- `runway`

Example:

```bash
curl http://127.0.0.1:8000/departures | python -m json.tool
```

### `GET /arrivals`

Returns the arrival schedule sorted by the first-fix handoff time.

Each item keeps the original compressed trajectory payload shape and adds:

- `arrival_time`
- `arrival_time_utc`
- `runway`
- `original_fix_sequence`
- `original_fix_count`

The trajectory payload remains compatible with the existing `data/adsb/compressed/flights.jsonl` format:

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
    "arrival_time": 1743465659,
    "arrival_time_utc": "2025-04-01T00:00:59Z",
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
print(arrivals[0]["flight_id"], arrivals[0]["arrival_time_utc"])
```

## Notes

- The API is read-only.
- Resources are loaded once at startup and reused across requests.
- Arrival schedules intentionally use the first-fix time from the fix-sequence catalog, not the runway threshold time, so the schedule can be updated later when runway or fix edits are introduced.
