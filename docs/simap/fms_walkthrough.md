# FMS Walkthrough

This document explains the FMS heuristic in `src/simap/fms/core.py` and
`src/simap/fms/holds.py`.

It is intentionally separate from `planner_walkthrough.md`, which covers the NLP
collocation planner.

## What The Module Does

The FMS module is the distance-aware, flight-management-style descent heuristic.
It is not a nonlinear optimizer. The core path is a forward simulation with a
small control law wrapped around it.

The module has three layers:

1. `simulate_fms_descent()` is the basic time-stepped descent simulation.
2. `plan_fms_descent()` searches for a top-of-descent location, then stitches a
   level segment to the descent segment.
3. `plan_hold_aware_fms_descent()` adds explicit hold instructions before doing
   the same stitched profile construction.

The common input shape is:

```text
CoupledDescentPlanRequest
  -> FMSRequest.from_coupled_request()
  -> plan_fms_descent() or plan_hold_aware_fms_descent()
  -> FMSResult
```

The example script in `src/simap/examples/run_a320_kdfw_fms.py` shows the same
flow end to end.

## 1. Model

The basic state in the FMS simulator is:

```text
x(t) = [s(t), h(t), V_tas(t)]
```

where:

- `s` is distance from the runway threshold measured upstream.
- `h` is altitude.
- `V_tas` is true airspeed.

The simulation also carries a PI integral term for speed tracking, but that term
is a controller state rather than an aircraft state.

The main inputs are:

- `AircraftConfig` for mass, gates, and CAS limits.
- `PerformanceBackend` for thrust and drag.
- `ReferencePath` for the along-track geometry and track angle.
- `WeatherProvider` for wind and ISA deviation.
- `FMSSpeedTargets` for the clean, approach, and final CAS targets.
- `FMSPIConfig` for the pitch controller.

The current mode comes from `mode_for_s(cfg, s)`, so the target speed is a
function of along-track position. That means the same aircraft can be in clean,
approach, or final behavior at different points in one descent.

`FMSSpeedTargets.for_mode()` applies one more rule: below 10,000 ft it clamps
the target to the configured altitude speed cap, typically 250 kt, if that is
lower than the mode target.

## 2. Constraints

The request dataclasses enforce the basic preconditions:

- `dt_s` must be positive.
- `max_time_s` must be positive.
- `start_s_m` must be on the reference path.
- `start_cas_mps` must be positive.
- `FMSPIConfig` requires a sensible vertical-speed interval and positive
  integral limit.

The simulation itself respects these dynamic constraints:

- vertical speed is clipped between `min_vertical_speed_mps` and
  `max_vertical_speed_mps`
- the pitch command is bounded indirectly by that vertical-speed clip
- throttle is held at idle in the managed descent
- the integration step is shortened so the state does not overshoot the target
  altitude or the end of the reference path
- the loop stops when the target altitude is reached, the path ends, or the time
  limit is hit

The hold-aware path adds its own constraints:

- hold altitudes are sorted from high to low
- each hold must be below the start altitude and above the final target altitude
- each hold can either capture the current speed or use an explicit hold speed
- the hold controller has its own altitude and speed PI limits

## 3. Objective

There is no outer scalar optimizer inside `simulate_fms_descent()`. The control
law itself is the objective.

In plain terms, the managed descent tries to:

- keep CAS near the target for the current mode
- stay within the vertical-speed envelope
- use idle thrust
- remain smooth enough to be physically plausible

The pitch command is computed as a PI controller on CAS error:

```text
pitch = nominal_pitch + Kp * CAS_error + Ki * integral(CAS_error)
```

The outer `plan_fms_descent()` wrapper does introduce a small search problem:
it varies top-of-descent distance until the simulated descent is just feasible
from that splice point to the threshold.

So the two layers are different:

- `simulate_fms_descent()` is an initial-value simulation.
- `plan_fms_descent()` is a one-dimensional root-finding wrapper around that
  simulation.

The hold-aware planner uses the same idea, but each function evaluation includes
the hold switching logic as well.

## 4. The ODE Problem

The simulator integrates the point-mass dynamics forward in time with an
explicit stepper. The underlying continuous model is:

```text
ds/dt = -V_ground
dh/dt = v_vertical
dV_tas/dt = (T - D) / m - g * sin(gamma)
```

The control law defines the algebraic quantities at each step:

```text
CAS = cas_from_tas(V_tas, h, weather)
error = CAS - target_CAS(mode, h)
pitch = PI(error)
v_vertical = clip(V_tas * sin(pitch), min_vs, max_vs)
gamma = asin(clamp(v_vertical / V_tas))
V_ground = max(1, V_tas + alongtrack_wind)
```

The sign convention can be confusing at first:

- `s` measures distance upstream from the threshold.
- As the aircraft moves toward the runway, `s` decreases.

That is why the code updates `s` with a subtraction step.

The simulator uses weather-dependent CAS/TAS conversion from OpenAP and a
performance backend for thrust/drag. The state update is a forward Euler step,
not a generic ODE solver.

## 5. Mental Model

The simplest way to picture the module is as three layers stacked on top of
each other:

```text
inputs
  -> request assembly
  -> closed-loop forward simulation
  -> optional TOD search and stitching
  -> result arrays
  -> plotting / replay / reporting
```

For the normal FMS path, the flow is:

```text
FMSRequest
  -> simulate_fms_descent()
  -> FMSResult
```

For the stitched path, the flow is:

```text
FMSRequest
  -> plan_fms_descent()
  -> find TOD
  -> simulate level segment
  -> simulate descent segment
  -> stitch results
  -> FMSResult
```

For the hold-aware path, the flow is:

```text
FMSRequest + HoldInstruction[]
  -> HoldAwareFMSRequest
  -> plan_hold_aware_fms_descent()
  -> repeated hold-aware simulation
  -> stitch level + descent
  -> FMSResult
```

The example script then turns that result into plots and a pandas table.

## 6. Examples

### Example A: Basic Managed Descent

```python
fms_request = FMSRequest.from_coupled_request(bundle.request, dt_s=0.5)
result = plan_fms_descent(fms_request)
```

This is the cleanest version of the algorithm:

- start at the upstream end of the route
- infer the speed targets from the coupled request
- integrate the descent
- stop at the target altitude

The resulting `FMSResult.phase` will mostly show the descent phase, and the
plots in `run_a320_kdfw_fms.py` will show altitude, CAS, pitch, vertical speed,
and thrust as continuous histories.

### Example B: Hold Without An Explicit Hold Speed

```python
hold_request = HoldAwareFMSRequest(
    base_request=fms_request,
    holds=(
        HoldInstruction(
            holding_altitude_ft=12_000.0,
            holding_speed_kts=None,
            holding_time_s=120.0,
        ),
    ),
)
result = plan_hold_aware_fms_descent(hold_request)
```

When `holding_speed_kts` is `None`, the hold captures the CAS already present at
the hold altitude. That makes the hold a two-stage event:

1. capture the current speed
2. stay at the hold altitude for the requested duration

This is the part of the code that is easiest to misread if you only look at the
result arrays. The `phase` field tells the story:

- `hold_decelerate` means the controller is still trying to catch the target
  speed
- `hold` means the speed has been captured and the hold timer is running

### Example C: Why The Planner Searches TOD

The top-of-descent search is a one-variable feasibility problem:
distance from the threshold where the level segment should end.

If TOD is too close to the threshold, the managed descent runs out of distance
before it reaches the target altitude.
If TOD is moved farther upstream, the descent becomes easier, so the planner can
keep reducing the level segment until it reaches the feasibility boundary.
`plan_fms_descent()` samples candidate TOD values, brackets that transition, and
then refines the root.

This is why the planner needs a search wrapper even though the inner simulation
is just a forward time integration.

### Example D: Interpreting The Plot

The example plot in `run_a320_kdfw_fms.py` is useful because it shows four
different ideas at once:

- altitude and CAS tell you whether the managed descent is hitting the target
- pitch and vertical speed show how hard the controller is working
- thrust shows whether the profile is still in the idle-thrust regime
- the `phase` span shows where holds or level flight are happening

If the CAS line tracks the mode target and the vertical-speed line stays within
its bounds, the controller is behaving as intended even if the altitude curve is
not perfectly smooth.

## 7. Reading Order

If you want to trace the implementation in code, read it in this order:

1. `src/simap/fms/core.py`
2. `src/simap/fms/holds.py`
3. `src/simap/examples/run_a320_kdfw_fms.py`
4. `src/simap/__init__.py`

The most useful functions are:

- `FMSRequest.from_coupled_request()`
- `simulate_fms_descent()`
- `plan_fms_descent()`
- `simulate_hold_aware_fms_descent()`
- `plan_hold_aware_fms_descent()`
