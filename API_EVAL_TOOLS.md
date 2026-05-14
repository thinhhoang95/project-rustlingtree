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
