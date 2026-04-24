# SIMAP Implementation Walkthrough

This document walks through the current longitudinal RNAV implementation starting from [`src/simap/examples/run_a320.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/examples/run_a320.py).

The goal is to explain how the example assembles the planner inputs, how the solver works, and what each supporting module is responsible for so the code can be maintained or extended safely.

## End-to-End Flow

At a high level, the example follows this sequence:

1. Load OpenAP aircraft and engine data.
2. Build an aircraft configuration and performance backend.
3. Define the threshold boundary, upstream boundary, and constraint envelope.
4. Build a longitudinal plan request.
5. Solve the distance-domain planning problem.
6. Plot the resulting plan and the envelope.
7. Print solver diagnostics and sampled output.

The important point is that the old “simulate forward and clip things until they look feasible” behavior is gone. The current implementation solves a constrained planning problem over distance, then replays the result for verification.

## 1. Example Entry Point

[`run_a320.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/examples/run_a320.py) is intentionally small. It is the best place to understand the intended usage pattern.

### Step 1: Load OpenAP

```python
openap = load_openap("A320")
aircraft_data = extract_aircraft_data(openap)
```

This gives the example:

- aircraft geometry and performance metadata
- engine metadata
- WRAP speed anchors like `landing_speed`, `finalapp_vcas`, and `descent_const_vcas`

### Step 2: Pick a representative mass

```python
mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=12_000.0)
```

The mass is fixed for the solve in v1. The helper chooses a practical approach mass by clipping `OEW + payload` against `MLW`.

### Step 3: Build the config and performance backend

```python
cfg, openap = build_default_aircraft_config("A320", mass_kg=mass_kg, openap_objects=openap)
perf = EffectivePolarBackend(cfg=cfg, openap=openap)
```

`cfg` carries the mode-specific limits, drag polars, and gating distances. `perf` is the force model used by the planner.

### Step 4: Define the terminal and upstream boundaries

```python
threshold = ThresholdBoundary(...)
upstream = UpstreamBoundary(...)
```

The threshold boundary anchors the stabilized approach side of the problem. The upstream boundary supplies the free-TOD target region and makes the problem unique.

### Step 5: Build the constraint envelope

```python
envelope = ConstraintEnvelope.from_profiles(...)
```

This is where the example specifies the altitude, CAS, and flight-path-angle bounds over distance. The solver uses this envelope together with mode-based limits from `cfg`.

### Step 6: Create the request and solve

```python
request = LongitudinalPlanRequest(...)
plan = plan_longitudinal_descent(request)
```

`plan_longitudinal_descent()` is the main API. It solves a direct-collocation problem with `trust-constr`, then replays the result with `solve_ivp` and records residuals.

### Step 7: Visualize and inspect

```python
plot_constraint_envelope(envelope)
plot_longitudinal_plan(plan, envelope=envelope)
print(plan.to_pandas().head())
print(plan.to_pandas().tail())
```

The plan result is also tabular, which makes it easy to inspect in pandas or export to CSV.

## 2. Core Data Types

### `ThresholdBoundary`

Defined in [`longitudinal_planner.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_planner.py), this holds the stabilized threshold-side anchor:

- `h_m`
- `cas_mps`
- `gamma_rad`

Use it when the downstream end of the approach is known and should be enforced exactly.

### `UpstreamBoundary`

Also in [`longitudinal_planner.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_planner.py). It defines the free upstream endpoint:

- `h_m`
- `cas_window_mps`
- `gamma_rad`

This is what turns TOD into a unique optimization target instead of an underconstrained geometric curve.

### `ConstraintEnvelope`

Defined in [`longitudinal_profiles.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_profiles.py). It stores nodewise lower and upper bounds for:

- altitude
- CAS
- optional `gamma`
- optional thrust
- optional `CL_max`

The important behavior is that it is distance-indexed and interpolated. It is not a single fixed speed schedule.

### `LongitudinalPlanRequest` and `LongitudinalPlanResult`

These are the solver input and output contracts in [`longitudinal_planner.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_planner.py).

The request bundles:

- aircraft config
- performance backend
- boundaries
- envelope
- weather
- optimizer settings

The result carries:

- solved `s_m`, `h_m`, `v_tas_mps`, `v_cas_mps`
- `t_s`, `gamma_rad`, `thrust_n`
- mode labels
- solver status/message
- collocation and replay residuals

## 3. Module Walkthrough

### `openap_adapter.py`

[`openap_adapter.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/openap_adapter.py) is the OpenAP integration layer.

It does three jobs:

1. Load OpenAP aircraft, engine, WRAP, drag, and thrust objects.
2. Extract the aircraft data needed to build `AircraftConfig`.
3. Provide tiny helpers like `wrap_default()` and `wrap_sample()`.

Maintenance rule: keep OpenAP calls here. If you need a new OpenAP field, add it here first so the rest of the code stays insulated from package-specific APIs.

### `calibration.py`

[`calibration.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/calibration.py) translates OpenAP metadata into the simulator’s configuration.

It:

- suggests a practical approach mass
- fits effective drag polars from OpenAP non-clean drag samples
- builds the default `AircraftConfig`

This module is where mode-specific coefficients come from. If the drag model changes, this is the first place to look.

### `config.py`

[`config.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/config.py) defines the aircraft and mode configuration dataclasses.

Important responsibilities:

- `ModeConfig` stores per-mode drag, speed, and motion limits
- `AircraftConfig` stores aircraft-wide parameters and gate distances
- `mode_for_s()` chooses clean / approach / final from distance to threshold
- `planned_cas_bounds_mps()` combines mode CAS limits with stall-margin logic
- `bank_limit_rad()` computes the allowable bank limit for lateral code

Example:

```python
mode = mode_for_s(cfg, 10_000.0)
cas_min, cas_max = planned_cas_bounds_mps(cfg, 10_000.0)
```

### `weather.py`

[`weather.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/weather.py) defines the weather interface.

It is intentionally small:

- `WeatherProvider` is a protocol with `wind_ne_mps()` and `delta_isa_K()`
- `ConstantWeather` is the default still-air / ISA-offset implementation
- `alongtrack_wind_mps()` projects northeast wind into the procedure track direction

This keeps the planner independent from any specific weather source.

### `backends.py`

[`backends.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/backends.py) is the performance abstraction.

`PerformanceBackend` defines the methods the solver needs:

- `drag_newtons()`
- `idle_thrust_newtons()`
- `thrust_bounds_newtons()`

`EffectivePolarBackend` implements those methods by combining:

- the mode drag polar from `cfg`
- the OpenAP idle thrust model
- OpenAP engine metadata for thrust bounding

Practical takeaway: if the solver needs a different aerodynamic or thrust source, swap the backend implementation rather than touching the planner.

### `longitudinal_profiles.py`

[`longitudinal_profiles.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_profiles.py) provides interpolation and envelope utilities.

`ScalarProfile` is the core helper:

- strictly increasing distance nodes
- linear interpolation by `value()`
- finite-difference slope by `slope()`

`ConstraintEnvelope.from_profiles()` merges multiple profiles onto a common distance grid.

The module also contains `build_speed_schedule_from_wrap()`, which pulls WRAP defaults into a simple CAS schedule. In the current workflow, this schedule is used as a starting policy and as a source for the CAS envelope, not as the final answer.

### `longitudinal_dynamics.py`

[`longitudinal_dynamics.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_dynamics.py) contains the distance-domain physics helpers.

The two main helpers are:

- `quasi_steady_cl()`
- `distance_state_derivatives()`

`quasi_steady_cl()` computes the lift coefficient assuming quasi-steady flight with `L ≈ m g cos(gamma)`.

`distance_state_derivatives()` turns a candidate state and control into derivatives for:

- altitude
- TAS
- accumulated time

This is the model the collocation solver enforces at the nodes.

### `longitudinal_planner.py`

[`longitudinal_planner.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_planner.py) is the center of the rewrite.

The solver workflow is:

1. Build a normalized collocation mesh.
2. Construct a physically plausible initial guess.
3. Set nodewise equality constraints for the trapezoidal defects and boundary conditions.
4. Set inequality constraints for altitude, CAS, thrust, `gamma`, and `CL`.
5. Optimize with `scipy.optimize.minimize(method="trust-constr")`.
6. Replay the solution with `solve_ivp`.
7. Record collocation and replay residuals in the result.

The module is easiest to maintain if you think of it in three layers:

- request/result types
- mesh transcription and objective
- replay/diagnostics

Example usage:

```python
request = LongitudinalPlanRequest(...)
plan = plan_longitudinal_descent(request)
```

The returned `plan` is the canonical output of the implementation.

### `simap_plot.py`

[`simap_plot.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/simap_plot.py) provides the post-solve visualization helpers.

It focuses on distance-domain views:

- altitude vs distance
- CAS vs distance
- flight-path angle vs distance
- thrust vs distance
- envelope overlays

The main entry point for the example is `plot_longitudinal_plan()`.

### `path_geometry.py`

[`path_geometry.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/path_geometry.py) is the geographic reference-path utility.

It is not part of the longitudinal planner itself, but it remains important for the broader simulator stack and future RNAV extensions.

Responsibilities:

- build a centerline from waypoints
- interpolate east/north, lat/lon, track, and curvature
- expose path tangents and normals

If you later add a lateral-aware RNAV planner, this is the module you will build on.

### `__init__.py`

[`__init__.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/__init__.py) is the public facade.

It intentionally re-exports:

- planner types
- config and profile helpers
- OpenAP adapters
- the performance backend
- plotting helpers

If a new module is added to the supported public API, update this file deliberately. It is the easiest place for downstream import breakage to start.

## 4. Data Flow Through the Example

The example’s data flow is worth keeping in mind when debugging:

1. OpenAP metadata becomes `AircraftConfig`.
2. `AircraftConfig` plus `EffectivePolarBackend` define the aircraft physics.
3. `ScalarProfile` objects define distance-indexed constraints.
4. `ConstraintEnvelope` merges those profiles into a solver-ready grid.
5. `LongitudinalPlanRequest` packages the whole problem.
6. `plan_longitudinal_descent()` solves and verifies the trajectory.
7. `LongitudinalPlanResult` becomes both a table and a plot source.

If something looks wrong in the result, inspect those layers in that order.

## 5. Practical Maintenance Notes

- Keep OpenAP-specific logic in [`openap_adapter.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/openap_adapter.py) and [`backends.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/backends.py).
- Keep profile shape logic in [`longitudinal_profiles.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_profiles.py), not in the planner.
- Keep the planner focused on transcription and optimization. If you find physics logic creeping into the objective or constraints, move it to `longitudinal_dynamics.py` or `backends.py`.
- Use the result diagnostics when comparing changes. `solver_message` and replay residuals are often more useful than the raw plotted curve.
- If you add a new output field, update:
  - `LongitudinalPlanResult`
  - `to_pandas()`
  - the plotting helpers
  - the example
  - tests that assert schema

## 6. Minimal Example Snippets

Build a schedule from WRAP defaults:

```python
speed_schedule = build_speed_schedule_from_wrap(openap.wrap)
```

Create a bounded envelope from profiles:

```python
envelope = ConstraintEnvelope.from_profiles(
    altitude_lower=alt_lower,
    altitude_upper=alt_upper,
    cas_lower=cas_lower,
    cas_upper=cas_upper,
)
```

Solve and inspect:

```python
plan = plan_longitudinal_descent(request)
print(plan.solver_success)
print(plan.to_pandas().head())
```

Plot the result:

```python
plot_longitudinal_plan(plan, envelope=envelope)
```

## 7. What To Read Next

1. [`src/simap/examples/run_a320.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/examples/run_a320.py)
2. [`src/simap/longitudinal_planner.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_planner.py)
3. [`src/simap/backends.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/backends.py)
4. [`src/simap/longitudinal_profiles.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/longitudinal_profiles.py)
5. [`src/simap/simap_plot.py`](/Volumes/CrucialX/project-rustlingtree/src/simap/simap_plot.py)

