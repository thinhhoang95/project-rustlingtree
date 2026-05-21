# SIMAP FMS Longitudinal Dynamics Blueprint

This document describes the managed-descent model used by the SIMAP FMS
bichannel implementation in `src/simap/fms_bichannel/core.py`, including the
longitudinal FMS in `src/simap/fms/core.py`, hold-aware extension in
`src/simap/fms/holds.py`, lateral replay in `src/simap/lateral_dynamics.py`,
and the shared route geometry in `src/simap/path_geometry.py`.

The model is a reduced-order point-mass FMS replay. It is intended to produce a
flight-management-style arrival profile: level flight until top of descent,
idle-thrust managed descent with target CAS tracking, mode-dependent slowdown,
optional altitude holds, and lateral path following against the same reference
route.

## State And Coordinates

The longitudinal FMS state is:

```text
x_long(t) = [s(t), h(t), V(t)]
```

where:

- `s` is remaining along-track distance to the runway threshold in meters.
- `h` is altitude in meters.
- `V` is true airspeed, `V_tas_mps`, in meters per second.

The controller also carries integral state. In managed descent this is:

```text
I_v(t) = integral(CAS - CAS_target) dt
```

The hold-aware path adds altitude and speed integrals while a hold is active:

```text
I_h(t) = integral(h_target - h) dt
I_hold_v(t) = integral(CAS_hold_target - CAS) dt
```

The bichannel replay adds the lateral state:

```text
x_lat(t) = [E(t), N(t), psi(t), phi(t)]
```

where `E` and `N` are local east/north coordinates, `psi` is aircraft heading,
and `phi` is bank angle. The bichannel state is therefore:

```text
x(t) = [s, h, V, E, N, psi, phi]
```

The longitudinal channel owns `s`, `h`, `V`, elapsed time, target speed, thrust,
drag, and phase. The lateral channel replays `E`, `N`, `psi`, and `phi` on the
same time grid.

The sign convention is important: `s = 0` is the runway threshold, larger `s`
is farther upstream, and `s` decreases as the aircraft flies toward the runway.

## Reference Path And Weather

`ReferencePath` is the geometric spine shared by both channels. It provides:

```text
p_ref(s) = [E_ref(s), N_ref(s)]
chi_ref(s) = reference track angle
tau(s) = [cos(chi_ref), sin(chi_ref)]
n(s) = [-sin(chi_ref), cos(chi_ref)]
kappa_ref(s) = reference curvature
```

The path is built from geographic waypoints with straight legs and fly-by arc
transitions. Its stationing is expressed as remaining distance `s`. The physical
fly-by path can be shorter than the original route chord length, but stationing
is stretched to preserve the longitudinal distance budget.

Weather enters through:

```text
w(E,N) represented as [w_east(s,h,t), w_north(s,h,t)]
Delta ISA = delta_isa_K(s,h,t)
```

The longitudinal channel uses only the wind component along the reference track:

```text
w_s = w_east cos(chi_ref(s)) + w_north sin(chi_ref(s))
V_g_long = max(1, V + w_s)
```

CAS/TAS conversion uses OpenAP with the local `Delta ISA`.

## Aircraft Modes

Mode is selected only from remaining distance:

```text
mode(s) =
  final     if s <= cfg.final_gate_m
  approach  if s <= cfg.approach_gate_m
  clean     otherwise
```

The default gate semantics are terminal phase gates, not explicit flap dynamics.
The mode affects:

- target CAS through `FMSSpeedTargets.for_mode()`;
- aerodynamic polar coefficients `cd0` and `k`;
- thrust upper bounds in the performance backend;
- bank limits and roll response in the lateral channel;
- minimum/maximum planned CAS limits where configured.

On a mode transition, the managed speed integral is reset to zero. This prevents
integral windup from the previous target-speed regime from carrying into the
new regime.

The speed target is:

```text
CAS_target_raw =
  clean_cas_mps     in clean mode
  approach_cas_mps  in approach mode
  final_cas_mps     in final mode

CAS_target = min(CAS_target_raw, below_altitude_limit_cas_mps)
             when h <= below_altitude_limit_h_m
CAS_target = CAS_target_raw otherwise
```

The default altitude speed cap is 250 kt below 10,000 ft.

## Managed Descent Equations

At each time step the simulator computes CAS from TAS:

```text
CAS = tas2cas(V, h, Delta ISA)
e_v = CAS - CAS_target(mode(s), h)
```

The managed pitch command is a PI law on CAS error:

```text
theta_cmd = theta_nominal + Kp e_v + Ki I_v
```

The implementation records this command as `pitch_rad`, but there is no
separate pitch-attitude state. After vertical-speed limiting, the recorded
`pitch_rad` and `gamma_rad` are both the flight-path angle.

Vertical speed and flight-path angle are:

```text
v_z_raw = max(1, V) sin(theta_cmd)
v_z = clip(v_z_raw, VS_min, VS_max)
gamma = asin(clip(v_z / max(1, V), -0.95, 0.95))
```

Managed descent uses idle thrust:

```text
T = T_idle(mode, V, h, Delta ISA)
```

In the current implementation `T_idle` is the lower value returned by
`PerformanceBackend.thrust_bounds_newtons()`. Drag comes from the performance
backend:

```text
D = D(mode, mass, wing_area, V, h, gamma, Delta ISA)
```

For the default `EffectivePolarBackend`:

```text
q = 0.5 rho V^2
L = mass g cos(gamma) / cos(phi_drag)
CL = L / (q S)
CD = cd0(mode) + k(mode) CL^2
D = q S CD
```

The longitudinal managed-descent dynamics are:

```text
ds/dt = -V_g_long
dh/dt = v_z
dV/dt = (T - D) / mass - g sin(gamma)
```

The speed integral has anti-windup logic. Let the vertical-speed command be
saturated at the upper level limit or lower descent limit. Then:

```text
dI_v/dt = 0    if saturated level and e_v > 0
dI_v/dt = 0    if saturated descent and e_v < 0
dI_v/dt = e_v  otherwise
I_v = clip(I_v, -I_limit, I_limit)
```

The saturation cases match situations where integrating the error would push
the already-clipped vertical-speed command farther into saturation.

## Level Segment Equations

`plan_fms_descent()` searches for a top-of-descent point. Before that point,
the stitched profile contains a level segment.

During the level segment:

```text
h(t) = h_start
CAS(t) = CAS_start
V(t) = V_start = cas2tas(CAS_start, h_start, Delta ISA at level start)
gamma = 0
v_z = 0
T = D(mode, V, h, gamma=0)
ds/dt = -V_g_long
```

The level segment is not an idle-thrust segment. Thrust is set equal to drag so
the speed is held constant in level flight. The current implementation keeps
the initial level-segment TAS value fixed rather than re-solving TAS from CAS
as weather or station changes during the level segment. The result phase is
`level`.

If the selected top of descent is at or upstream of the start, the level segment
degenerates to a single sample at the initial state.

## Hold-Aware Equations

`HoldAwareFMSRequest` wraps the base FMS request with altitude holds. Holds are
sorted from high altitude to low altitude and must be strictly below the start
altitude and above the final target altitude.

A hold activates when the descending aircraft reaches the hold altitude:

```text
h <= h_hold
```

At activation, the implementation snaps `h` to `h_hold`. If no hold speed is
provided, the current CAS is captured as the hold target and the hold is treated
as speed-captured immediately. If a hold speed is provided, the phase starts as
`hold_decelerate` until the speed error is within tolerance, then switches to
`hold`.

The hold altitude controller is:

```text
e_h = h_hold - h
v_z = clip(Kp_h e_h + Ki_h I_h, VS_hold_min, VS_hold_max)
gamma = asin(clip(v_z / max(1, V), -0.95, 0.95))
```

The hold speed controller commands acceleration:

```text
e_hold_v = CAS_hold_target - CAS
a_cmd = clip(Kp_v e_hold_v + Ki_v I_hold_v,
             -a_hold_limit, +a_hold_limit)
```

Thrust is selected to realize that acceleration, then clipped to backend thrust
bounds:

```text
T_raw = D + mass (a_cmd + g sin(gamma))
T = clip(T_raw, T_min, T_max)
```

The same point-mass speed equation is then used:

```text
dV/dt = (T - D) / mass - g sin(gamma)
```

Hold integrals are clipped independently. When speed capture is complete, the
hold timer counts down; when it reaches zero, the active hold is cleared and
managed descent resumes with a reset managed speed integral.

## Lateral Equations In The Bichannel Replay

The bichannel simulation first computes a complete longitudinal `FMSResult`.
The lateral channel then walks through the longitudinal rows and computes one
map state per row.

For the current lateral state:

```text
v_air = V [cos(psi), sin(psi)]
v_ground = v_air + [w_east, w_north]
V_g_lat = ||v_ground||
chi_g = atan2(v_ground_north, v_ground_east)
```

The aircraft is projected onto the reference path:

```text
s_ref = project_s_m(E, N)
p_ref = p_ref(s_ref)
e_xtk = ([E, N] - p_ref) dot n(s_ref)
e_chi = wrap(chi_g - chi_ref(s_ref))
```

A lookahead target is selected ahead on the path:

```text
L = clamp(s_ref, min_lookahead_m, lookahead_m)
s_target = max(0, s_ref - L)
p_target = p_ref(s_target)
```

Near the threshold, if the aircraft is already beyond the endpoint, the target
is placed on a tangent extension instead of clamping back to `s = 0`.

The line-of-sight error and feedback curvature are:

```text
chi_los = atan2(N_target - N, E_target - E)
eta = clip(wrap(chi_los - chi_g), -eta_max, +eta_max)
K_l1 = track_error_gain sqrt(cross_track_gain)
kappa_feedback = K_l1 sin(eta) / L
```

The controller previews reference curvature between `s_ref` and `s_target` and
uses the largest-magnitude value as feed-forward:

```text
kappa_cmd = curvature_feedforward_gain kappa_preview + kappa_feedback
```

Curvature becomes a bank request:

```text
phi_req_raw = atan(V_g_lat^2 kappa_cmd / g)
phi_max = min(phi_comfort_max,
              phi_procedure_max,
              phi_stall_margin(CAS, mode))
phi_req = clip(phi_req_raw, -phi_max, +phi_max)
```

The roll and heading dynamics are:

```text
dphi/dt = clip((phi_req - phi) / tau_phi(mode),
               -p_max(mode), +p_max(mode))
dpsi/dt = g tan(phi) / max(V, 1)
```

The heading rate uses the current bank angle, not the requested bank angle, so
the response includes roll lag.

The lateral position derivative before replay scaling is:

```text
dE/dt = V cos(psi) + w_east
dN/dt = V sin(psi) + w_north
```

During bichannel replay, this velocity is multiplied by a scale factor so map
motion follows the scheduled physical distance between the two longitudinal
stations:

```text
scheduled_path_speed =
  ||p_ref(s_next) - p_ref(s_current)|| / dt

velocity_scale =
  clip(scheduled_path_speed / V_g_lat, 0, 2)
  when V_g_lat > 0 and scheduled_path_speed > 0
velocity_scale = 1 otherwise

dE/dt = velocity_scale (V cos(psi) + w_east)
dN/dt = velocity_scale (V sin(psi) + w_north)
```

This is a bichannel consistency device: the longitudinal stationing remains the
authority for time and distance while lateral geometry can include fly-by arcs.

## Numerical Integration

The longitudinal FMS uses explicit forward Euler integration on a time grid set
by `FMSRequest.dt_s`.

For a step of length `dt`:

```text
t_next = t + dt
s_next = s - V_g_long dt
h_next = h + v_z dt
V_next = max(1, V + V_dot dt)
```

The step size is shortened to avoid overshooting terminal events:

- `max_time_s`;
- reference-path end when `stop_at_reference_path_end=True`;
- target altitude in managed descent;
- next hold altitude in hold-aware descent;
- remaining hold time while a captured hold is active.

Samples are appended before the state update. The final sample is therefore the
first sampled state that satisfies the stop condition.

The lateral channel uses the longitudinal output rows as its public output grid.
Between row `i` and row `i + 1`, `_advance_lateral_state()` substeps with:

```text
dt_lat <= LateralGuidanceConfig.integration_step_s
```

Within each lateral substep, scheduled `t`, `s`, `h`, and `V` are linearly
interpolated between the two longitudinal rows. The lateral rates are evaluated
at the beginning of the substep and then advanced with forward Euler:

```text
phi_next = phi + phi_dot dt_lat
psi_next = wrap(psi + psi_dot dt_lat)
E_next = E + E_dot_scaled dt_lat
N_next = N + N_dot_scaled dt_lat
```

If the Euler roll update would cross `phi_req`, the bank angle is snapped to
`phi_req` to avoid overshoot from the first-order roll loop.

## Top-Of-Descent Search

`simulate_fms_descent()` is an initial-value simulation. `plan_fms_descent()` is
a wrapper that chooses where the descent segment should begin.

The planner evaluates a candidate top of descent by simulating from
`start_s_m = candidate_tod_s_m` with `stop_at_reference_path_end=True`.

The feasibility metric is:

```text
metric = s_final                         if target altitude is reached
metric = -(h_final - h_target clipped >= 0) otherwise
```

A positive metric means the descent reached the target altitude before the
threshold and had distance left over. A negative metric means the aircraft
reached the threshold before reaching target altitude.

The algorithm:

1. Simulates from the full available distance. If that is infeasible, returns a
   failed, threshold-truncated result.
2. Bisection-searches between `s = 1e-3` and the available distance.
3. Stops when the metric magnitude or bracket width is within tolerance.
4. Re-simulates the selected descent and performs up to eight small adjustments
   so `tod_s_m` matches the descent distance flown.
5. Simulates the level segment from the original start to `tod_s_m`.
6. Stitches level and descent results, dropping the duplicate descent start row.

The hold-aware planner uses the same top-of-descent search, but each candidate
simulation includes the hold logic.

## Result Fields

`FMSResult` contains the longitudinal history:

```text
t_s
s_m
distance_flown_m
h_m
v_tas_mps
v_cas_mps
target_cas_mps
pitch_rad
gamma_rad
vertical_speed_mps
thrust_n
drag_n
ground_speed_mps
mode
speed_error_mps
phase
```

`FMSBiChannelResult` wraps that result and adds:

```text
east_m, north_m, lat_deg, lon_deg
psi_rad, phi_rad
ground_track_rad, ground_speed_mps
alongtrack_speed_mps
cross_track_m, track_error_rad
curvature_cmd_inv_m
phi_req_rad, phi_max_rad
```

The bichannel `ground_speed_mps` is the lateral wind-aware ground speed from
heading and wind. The longitudinal result also has `ground_speed_mps`, computed
from TAS plus along-track wind along the reference path.

## Assumptions And Limitations

- The longitudinal model is a 3-state point-mass model plus controller
  integrals. There are no pitch-attitude, angle-of-attack, engine spool, flap,
  gear, or mass-burn states.
- `pitch_rad` in the result is effectively flight-path angle after vertical
  speed clipping; it is not an independent aircraft pitch attitude.
- Managed descent holds thrust at idle. Only level segments and holds command
  thrust above idle.
- Aircraft mass is constant.
- Mode changes are instantaneous distance-gate changes. They do not model flap
  or landing-configuration transition dynamics.
- The longitudinal channel does not include bank in drag. The default drag
  backend accepts `bank_rad`, but managed descent calls it with the default
  zero-bank value.
- Lateral bank limits affect lateral tracking only; they do not feed back into
  the longitudinal descent rate or drag.
- Longitudinal ground speed is computed from the reference-path track, not from
  the actual lateral heading produced by the bichannel replay.
- The lateral channel is slaved to the longitudinal schedule. It can report
  cross-track error and bank saturation, but it does not slow, re-time, or
  reshape the longitudinal profile when lateral tracking is poor.
- CAS/TAS conversion and atmospheric density come from OpenAP with the current
  `Delta ISA`; wind and temperature are supplied by `WeatherProvider`.
- The model enforces a minimum TAS of `1 m/s` and a minimum longitudinal ground
  speed of `1 m/s` to avoid singular updates.
- The implementation uses explicit Euler integration. Accuracy is therefore
  tied directly to `dt_s` and lateral `integration_step_s`.
- Top-of-descent planning assumes the feasibility boundary is monotonic with
  available descent distance. That is true for the intended idle managed
  descent use case, but future constraints should preserve or revisit this
  assumption.

## Extension Guidance

Future features should decide which channel owns the new behavior:

- Features that change target speed, thrust, vertical speed, or altitude belong
  in the longitudinal FMS or hold-aware controller.
- Features that change path capture, bank limits, turn prediction, or map
  position belong in lateral guidance.
- Features that couple bank, drag, speed, and descent feasibility require a
  model change: the current bichannel replay is intentionally decoupled after
  the longitudinal profile has been generated.

When adding new modes, keep the mode effects explicit: speed target, CAS bounds,
aero polar, thrust limits, bank/roll limits, and integral reset behavior should
all be reviewed together.
