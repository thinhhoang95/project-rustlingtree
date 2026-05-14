# Ground Distance Estimation Tool

`GroundDistanceEstimationTool` estimates the straight-line ground distance required
for an aircraft to descend and land from an initial longitudinal state.

The tool lives in `mcp_tools.utils` and reuses the SIMAP FMS descent model.
Inputs are altitude in feet and true airspeed in knots. TAS is converted to CAS
internally before the FMS request is built.

## Usage

```python
from mcp_tools.utils import GroundDistanceEstimationTool

tool = GroundDistanceEstimationTool()

estimate = tool.estimate(altitude_ft=26_000.0, tas_kts=430.0)
print(estimate.required_ground_distance_nmi)
print(estimate.initial_cas_kts)
print(estimate.success, estimate.message)

# Concise form: returns only required distance in nautical miles.
distance_nmi = tool(altitude_ft=26_000.0, tas_kts=430.0)
```

For the default A320 setup, `26_000 ft` and `430 kt TAS` currently estimates
about `110 nmi` of required descent distance.

## API

```python
GroundDistanceEstimationTool(
    aircraft_type="A320",
    payload_kg=12_000.0,
    runway_altitude_ft=620.0,
    dt_s=1.0,
)
```

`estimate()` returns a `GroundDistanceEstimate` with:

- `required_ground_distance_m`
- `required_ground_distance_nmi`
- `initial_cas_kts`
- `descent_time_s`
- `tod_distance_nmi`
- `success`
- `message`

## Notes

- The estimate is route-independent and uses a synthetic straight reference path.
- No wind is applied by default.
- The returned distance is the required descent segment distance, not the full
  synthetic path length.
- `altitude_ft` must be above `runway_altitude_ft`, and `tas_kts` must be positive.
