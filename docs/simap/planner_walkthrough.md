# Coupled Planner Walkthrough

This document explains how SIMAP now plans the longitudinal and lateral
trajectory in one distance-domain nonlinear program.

The important change is that cross-track error, heading error, bank, roll rate,
wind, and along-track speed are part of the optimizer. The simulator can still
replay a result, but the planner is now the source of truth for the coupled
trajectory.

## 1. Full-stack flow

The main flow is:

1. Build `AircraftConfig`, `PerformanceBackend`, `WeatherProvider`, and
   `ReferencePath`.
2. Build the longitudinal `ConstraintEnvelope`.
3. Build `CoupledDescentPlanRequest`, including the reference path and optional
   lateral boundary conditions.
4. Call `plan_coupled_descent()`.
5. Inspect `CoupledDescentPlanResult`, which now contains altitude, speed,
   thrust, cross-track, heading, bank, ground speed, and map position.
6. Optionally call `simulate_plan()` as a validation replay.

## 2. Decision variables

The optimizer uses a fixed node grid from threshold to top of descent:

- `s = 0`: runway threshold
- `s = TOD`: upstream end of the planned segment

At each node it optimizes:

- `h`: altitude
- `V`: true airspeed
- `t`: time from threshold
- `y`: cross-track error
- `chi`: heading error relative to reference-path tangent
- `phi`: bank angle
- `gamma`: flight-path angle
- `T`: thrust
- `p`: roll rate

It also optimizes:

- `TOD`
- `constraint_slack`

## 3. Coupled Dynamics

Let the reference path provide:

```text
theta(s) = reference track angle
kappa(s) = reference curvature
tau(s) = [cos(theta), sin(theta)]
n(s) = [-sin(theta), cos(theta)]
```

The aircraft heading is:

```text
psi = theta + chi
```

The air and ground velocity are:

```text
v_air = V [cos(psi), sin(psi)]
v_ground = v_air + wind(s, h, t)
```

Project ground velocity into path coordinates:

```text
q = v_ground dot tau
r = v_ground dot n
```

Where:

- `q` is along-track speed toward the threshold.
- `r` is cross-track speed.

The distance-domain state is:

```text
x(s) = [h, V, t, y, chi, phi]
```

The controls are:

```text
u(s) = [gamma, T, p]
```

The coupled ODE is:

```text
dh/ds   = -tan(gamma)
dV/ds   = -(((T - D) / m) - g sin(gamma)) / (V cos(gamma))
dt/ds   = 1 / q
dy/ds   = -r / q
dchi/ds = -g tan(phi) / (V q) + kappa(s)
dphi/ds = -p / q
```

The signs come from the coordinate convention: the optimizer grid increases
from threshold upstream, while the aircraft flies toward decreasing `s`.

## 4. Constraints

The planner keeps the existing longitudinal constraints:

- threshold altitude, CAS, and flight-path angle
- upstream altitude, flight-path angle, and CAS window
- altitude envelope
- CAS envelope combined with aircraft-mode CAS limits
- thrust bounds from the performance backend and optional envelope
- optional flight-path-angle bounds
- optional maximum lift coefficient

It adds lateral constraints:

- threshold and upstream cross-track boundary conditions
- threshold and upstream heading-error boundary conditions
- threshold and upstream bank boundary conditions
- bank angle within `bank_limit_rad(cfg, mode, CAS)`
- roll rate within `mode.p_max_rps`
- along-track speed above `optimizer.min_alongtrack_speed_mps`

Banked lift coefficient is checked with the load-factor correction:

```text
CL = m g cos(gamma) / (0.5 rho V^2 S cos(phi))
```

## 5. Objective

The objective keeps the existing longitudinal terms:

```text
- top-of-descent reward
+ thrust-above-idle penalty
+ gamma smoothness penalty
+ slack penalty
```

It adds lateral regularization:

```text
+ cross-track penalty
+ heading-error penalty
+ bank penalty
+ roll-rate penalty
```

Those terms make the optimizer prefer a centered, smooth path, but endpoint
constraints still force recovery when the upstream lateral boundary starts
off-path.

## 6. Mental Model

The old design was:

```text
solve longitudinal profile first
then replay lateral response afterward
```

The new design is:

```text
solve vertical motion, speed, cross-track recovery, heading, bank, and roll in
one NLP
```

So a cross-track offset can change the optimized solution. If bank or roll
limits make recovery difficult, the optimizer may need a different speed,
longer `TOD`, different timing, or slack.

## 7. Example: 150 m Upstream Offset

Set:

```python
CoupledDescentPlanRequest(
    ...,
    upstream_lateral=LateralBoundary(cross_track_m=150.0),
)
```

The optimized result should satisfy:

- `cross_track_m[-1] ~= 150`
- `cross_track_m[0] ~= 0`
- `phi_rad` becomes nonzero during recovery
- `abs(phi_rad) <= phi_max_rad`
- `alongtrack_speed_mps > 0`

That behavior is now planned before replay, not discovered afterward by
`simulate_plan()`.

## 8. Reading Order

Start with:

1. `CoupledDescentPlanRequest` and `CoupledDescentPlanResult` in
   `src/simap/coupled_descent_planner.py`
2. `_TrajectoryEvaluation.state_derivatives`
3. `_equality_constraints`
4. `_inequality_constraints`
5. `_objective`
6. `simulate_plan()` in `src/simap/simulator.py` for validation replay
