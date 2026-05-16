# Feasibility Evaluator

`FeasibleEvaluator` reports arrivals whose current scenario-manager trajectory is
not vertically feasible. It lives in
`src/mcp_tools/evaluators/feasible.py` and is exposed by the scenario-manager API
at:

```text
GET /tools/evals/feasibility
```

The evaluator uses `ScenarioManager.arrival_schedule()`, not the raw artifact
files directly. That is important because `arrival_schedule()` is the served
view of the scenario: base arrival artifacts plus any scenario-manager diff that
has been applied before runtime fields are returned.

## What It Answers

The evaluator answers:

> Which served arrival trajectories failed SIMAP vertical-profile simulation, and
> how far short of the runway threshold did each one end?

It does not recompute the FMS profile. It reads the simulation result already
attached to each arrival and turns failures into a compact, ranked list.

## Input Flow

The input is the list returned by:

```python
manager.arrival_schedule()
```

Each arrival must contain a `simulation` dictionary with:

- `success`: a boolean. `true` means the vertical profile is feasible.
- `final_threshold_error_m`: a finite, nonnegative number in meters.
- `message`: an explanatory simulation message. It is converted to a string in
  the output.

These arrival identity fields are copied into the result:

- `callsign` -> `flight_number`
- `icao24`
- `flight_id`
- `runway`

The identity fields are converted to strings, so missing values become empty
strings. The simulation metadata is stricter: missing or malformed simulation
fields raise `ValueError`.

## Example Input

This example has three arrivals. `OK100` is feasible, while `SHORT100` and
`SHORT200` are infeasible.

```json
[
  {
    "flight_id": "ARR_OK",
    "callsign": "OK100",
    "icao24": "ok100",
    "runway": "RW35C",
    "simulation": {
      "success": true,
      "message": "ok",
      "final_threshold_error_m": 500.0
    }
  },
  {
    "flight_id": "ARR_SHORT_1",
    "callsign": "SHORT100",
    "icao24": "s10001",
    "runway": "RW36L",
    "simulation": {
      "success": false,
      "message": "short by one nautical mile",
      "final_threshold_error_m": 1852.0
    }
  },
  {
    "flight_id": "ARR_SHORT_2",
    "callsign": "SHORT200",
    "icao24": "s20002",
    "runway": "RW35C",
    "simulation": {
      "success": false,
      "message": "short by two nautical miles",
      "final_threshold_error_m": 3704.0
    }
  }
]
```

## Algorithm Walkthrough

The evaluator performs one pass over the served arrival schedule.

1. Load the served arrivals.

   ```python
   arrivals = manager.arrival_schedule()
   ```

   In the example, this returns the three arrivals shown above.

2. Read and validate `simulation`.

   Each arrival must have a dictionary at `arrival["simulation"]`. The evaluator
   then requires `simulation.success` to be a boolean and
   `simulation.final_threshold_error_m` to be a finite, nonnegative number.

   `OK100` has:

   ```json
   {
     "success": true,
     "final_threshold_error_m": 500.0
   }
   ```

   This is valid, even though the missing-distance field is nonzero. The
   `success` flag is authoritative for whether the arrival is reported.

3. Skip feasible arrivals.

   If `simulation.success` is `true`, the arrival is ignored.

   In the example, `OK100` is skipped.

4. Convert failed arrivals into `FeasibleFlight` records.

   For each arrival with `simulation.success == false`, the evaluator copies the
   identity fields and computes nautical miles from meters:

   ```text
   missing_distance_nmi = final_threshold_error_m / 1852.0
   ```

   For `SHORT100`:

   ```text
   1852.0 m / 1852.0 = 1.0 nmi
   ```

   For `SHORT200`:

   ```text
   3704.0 m / 1852.0 = 2.0 nmi
   ```

5. Sort the failures.

   The result is sorted by:

   1. descending `missing_distance_m`
   2. ascending `flight_number`
   3. ascending `icao24`

   That puts the most severe vertical infeasibility first. In the example,
   `SHORT200` appears before `SHORT100`.

## Example Output

The Python evaluator returns `FeasibleFlight` dataclasses. The API serializes the
same fields as JSON:

```json
[
  {
    "flight_number": "SHORT200",
    "icao24": "s20002",
    "flight_id": "ARR_SHORT_2",
    "runway": "RW35C",
    "missing_distance_nmi": 2.0,
    "missing_distance_m": 3704.0,
    "simulation_message": "short by two nautical miles"
  },
  {
    "flight_number": "SHORT100",
    "icao24": "s10001",
    "flight_id": "ARR_SHORT_1",
    "runway": "RW36L",
    "missing_distance_nmi": 1.0,
    "missing_distance_m": 1852.0,
    "simulation_message": "short by one nautical mile"
  }
]
```

An empty array means every served arrival has `simulation.success == true`.

## Output Fields

- `flight_number`: the arrival callsign.
- `icao24`: the aircraft ICAO24 identifier.
- `flight_id`: the scenario-manager flight identifier.
- `runway`: the normalized runway identifier.
- `missing_distance_nmi`: runway-threshold shortfall in nautical miles.
- `missing_distance_m`: runway-threshold shortfall in meters.
- `simulation_message`: the message from the arrival's simulation metadata.

## Validation Rules

The evaluator raises `ValueError` when an arrival has malformed simulation
metadata. This is intentional because the evaluator should not silently hide an
invalid served trajectory.

Required validation:

- `simulation` must exist and must be a dictionary.
- `simulation.success` must exist and must be a boolean.
- `simulation.final_threshold_error_m` must be an integer or float, but not a
  boolean.
- `simulation.final_threshold_error_m` must be finite and `>= 0.0`.

The error message includes the best available flight label, such as:

```text
flight_id=ARR_SHORT_1 callsign=SHORT100 has missing or malformed simulation.success
```

## Usage

Direct Python usage:

```python
from dataclasses import asdict
from mcp_tools.evaluators import FeasibleEvaluator

items = FeasibleEvaluator(manager).evaluate()
payload = [asdict(item) for item in items]
```

API usage:

```bash
curl http://127.0.0.1:8000/tools/evals/feasibility
```

## Practical Interpretation

Each returned item marks an arrival that exists in the served scenario but whose
vertical profile did not have enough distance to complete before the runway
threshold under the current assumptions. The usual next step is to intervene on
the affected flight by changing route, timing, speed, altitude, or another
trajectory-bearing field, then re-read `/arrivals` and this evaluator to confirm
that the replacement trajectory is feasible.
