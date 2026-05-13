# Lateral FMS Walkthrough

This document explains the lateral FMS guidance used by the bi-channel FMS
replay in:

- `src/simap/path_geometry.py`
- `src/simap/lateral_dynamics.py`
- `src/simap/fms_bichannel/core.py`

The lateral channel is separate from the longitudinal descent channel. The
longitudinal FMS supplies elapsed time, distance-to-go, altitude, and speed. The
lateral channel turns that schedule into map position, heading, bank, cross-track
error, and track error against a fly-by RNAV reference path.

The current implementation is a reduced-order line-LNAV/RNAV controller:

1. Build a route path from straight line legs and fly-by circular transitions.
2. Project the aircraft onto that curved reference path.
3. Aim at a lookahead target downstream on the path.
4. Add preview curvature feed-forward from the upcoming fly-by arc.
5. Convert the resulting curvature command into a bank request.
6. Apply bank and roll-rate limits before replaying the aircraft response.

It is not a proprietary Boeing LNAV implementation. It uses public RNAV/FMS
concepts: fly-by waypoints, turn anticipation, tangent path geometry,
cross-track path capture, and bank-limited coordinated turns.

## 1. Coordinate System

The reference path is represented by `ReferencePath` in
`src/simap/path_geometry.py`.

The important convention is:

```text
s = distance remaining to the runway threshold
```

So:

- `s = 0` is the runway threshold.
- Larger `s` values are farther upstream.
- As the aircraft flies toward the runway, `s` decreases.

At any path station, the reference path provides:

```text
p_ref(s) = [east_ref, north_ref]
theta(s) = reference track angle
tau(s) = [cos(theta), sin(theta)]
n(s) = [-sin(theta), cos(theta)]
kappa_ref(s) = reference curvature
```

Where:

- `tau` is the local unit tangent pointing along the path toward decreasing
  `s`.
- `n` is the local left-normal associated with that tangent.
- `kappa_ref` is nonzero primarily inside fly-by arc transitions.

The aircraft lateral state is:

```text
x_lat(t) = [east, north, psi, phi]
```

Where:

- `east`, `north` are local map coordinates in meters.
- `psi` is aircraft heading angle in radians.
- `phi` is bank angle in radians.

The longitudinal channel supplies:

```text
x_long(t) = [s, h, V_tas]
```

Where:

- `h` is altitude.
- `V_tas` is true airspeed.

## 2. Fly-By RNAV Reference Path

`ReferencePath.from_geographic()` now builds a fly-by path instead of a simple
polyline.

For each eligible interior waypoint:

```text
inbound course -> waypoint -> outbound course
```

the path builder computes:

```text
turn_angle = angle between inbound and outbound legs
R_nominal = V_nominal^2 / (g tan(phi_nominal))
lead = R_nominal * tan(turn_angle / 2)
```

Then it caps the lead distance by:

- a maximum fraction of the inbound leg
- a maximum fraction of the outbound leg
- a hard maximum lead distance

The default constants live near the top of `src/simap/path_geometry.py`:

```python
_DEFAULT_FLYBY_BANK_RAD = np.deg2rad(12.0)
_DEFAULT_FLYBY_SPEED_MPS = 230.0 * 0.514444
_MAX_FLYBY_LEG_FRACTION = 0.45
_MAX_FLYBY_LEAD_M = 7.0 * 1852.0
```

The generated path is:

```text
line segment to turn-start
circular fly-by arc
line segment from turn-end
```

This is the main Boeing-style/RNAV behavior change: the reference path bends
before the fix, so the bank command can begin before the aircraft reaches the
fix.

### Distance Convention

The physical fly-by path is shorter than the original corner-to-corner polyline.
However, `ReferencePath.total_length_m` remains the original route chord length.
The sampled fly-by path stationing is stretched to that total length.

That keeps the longitudinal planner's distance budget stable while the map
geometry still contains fly-by turn anticipation.

## 3. Full Flow

The normal bi-channel flow is:

```text
FMSBiChannelRequest
  -> plan_fms_descent() or simulate_fms_descent()
  -> FMSResult longitudinal profile
  -> _lateral_response()
  -> FMSBiChannelResult
```

At each longitudinal output node, `_lateral_response()`:

1. Copies scheduled `t_s`, `s_m`, `h_m`, and `v_tas_mps` into the lateral state.
2. Calls `compute_lateral_command()`.
3. Stores position, heading, bank, ground track, cross-track error, track error,
   commanded curvature, and bank request.
4. Advances the lateral state to the next longitudinal node with
   `_advance_lateral_state()`.

`_advance_lateral_state()` substeps inside one longitudinal time step. The
default substep is `0.5 s`, controlled by
`LateralGuidanceConfig.integration_step_s`.

During substeps, map motion is scaled to the physical map distance between the
current scheduled path station and the next scheduled path station. This matters
because fly-by map geometry is physically shorter than the stationing used by
the longitudinal FMS.

## 4. Guidance Inputs And Outputs

`compute_lateral_command()` receives:

```text
s_m, east_m, north_m, h_m, t_s, psi_rad, v_tas_mps
AircraftConfig
ModeConfig
ReferencePath
WeatherProvider
LateralGuidanceConfig
```

It returns `LateralCommand`:

```text
east_dot_mps
north_dot_mps
ground_speed_mps
alongtrack_speed_mps
ground_track_rad
cross_track_m
track_error_rad
curvature_cmd_inv_m
phi_req_rad
phi_max_rad
```

`phi_req_rad` is the actual lateral control request. The other fields are
diagnostics or kinematic quantities used by replay and plotting.

## 5. Wind-Aware Ground Track

The guidance law works in ground track, not just aircraft heading.

```text
v_air = V_tas [cos(psi), sin(psi)]
w = [wind_east, wind_north]
v_ground = v_air + w
ground_speed = norm(v_ground)
ground_track = atan2(v_ground_north, v_ground_east)
```

The controller uses `ground_track` for line-of-sight error and reported
`track_error_rad`.

Ground speed is also used when converting path curvature to bank:

```text
phi_req = atan(ground_speed^2 * kappa_cmd / g)
```

A tailwind therefore increases the bank needed for the same ground-path
curvature, while a headwind reduces it.

## 6. Active Path Projection

The controller does not blindly use the scheduled longitudinal `s_m` to measure
lateral error. Instead it projects the current map position to the closest point
on the reference path:

```text
s_ref = project_s_m(east, north)
p_ref = position_ne(s_ref)
```

Then it gets the local path basis:

```text
tau = tangent_hat(s_ref)
n = normal_hat(s_ref)
theta = track_angle_rad(s_ref)
```

The signed cross-track error is:

```text
cross_track = ([east, north] - p_ref) dot n
```

The reported ground-track error is:

```text
track_error = wrap(ground_track - theta)
```

This closest-point projection keeps capture robust when the aircraft is not at
the same path station as the longitudinal schedule.

## 7. Lookahead Target And Curvature Command

The lookahead distance starts from:

```text
L_base = guidance.lookahead_m
L_min = guidance.min_lookahead_m
```

Defaults:

```text
lookahead_m = 1500.0
min_lookahead_m = 150.0
```

Inside the finite path bounds:

```text
L = clamp(s_ref, L_min, L_base)
s_target = max(0, s_ref - L)
p_target = position_ne(s_target)
```

Because `s` decreases toward the threshold, subtracting `L` moves the target
ahead of the aircraft.

The line-of-sight track is:

```text
chi_los = atan2(target_north - north, target_east - east)
eta = wrap(chi_los - ground_track)
```

`eta` is clipped by `guidance.max_los_angle_rad`.

The feedback curvature is:

```text
K_l1 = track_error_gain * sqrt(cross_track_gain)
kappa_feedback = K_l1 * sin(eta) / L
```

The controller also previews curvature between `s_ref` and `s_target`:

```text
kappa_preview = max_abs_curvature(reference_path.curvature_many(s_ref ... s_target))
```

Then:

```text
kappa_cmd = curvature_feedforward_gain * kappa_preview + kappa_feedback
```

This feed-forward is what makes the controller predictive in turns. The
lookahead target detects the upcoming curved path, and the curvature preview
adds explicit bank demand before line-of-sight error grows.

### Endpoint Extension

Near or beyond the threshold, the finite reference path clamps to `s = 0`. To
avoid commanding a turn back toward the endpoint, when:

```text
s_ref <= L_min
along_path_offset > 0
```

the target is placed on a tangent extension:

```text
p_target = p_ref + (along_path_offset + L_base) tau
```

## 8. Curvature To Bank

The commanded curvature is converted into a coordinated-turn bank request:

```text
phi_req = atan(ground_speed^2 * kappa_cmd / g)
```

The raw bank request is clipped to the active aircraft/mode envelope:

```text
phi_max = bank_limit_rad(cfg, mode, CAS)
phi_req = clip(phi_req, -phi_max, +phi_max)
```

CAS is computed from TAS, altitude, and weather:

```text
CAS = tas2cas(V_tas, h, delta_isa)
```

The bank limit combines:

- comfort bank limit
- procedure bank limit
- stall-margin bank limit

## 9. Roll And Heading Dynamics

The bank request is not applied as an instantaneous heading change.

The roll loop is first order:

```text
phi_dot = (phi_req - phi) / tau_phi
```

and then clipped by the mode roll-rate limit:

```text
phi_dot = clip(phi_dot, -p_max, +p_max)
```

The heading rate is:

```text
psi_dot = g tan(phi) / V_tas
```

`psi_dot` uses current bank, not requested bank, so heading response still lags
until the roll loop catches up. The fly-by path and curvature preview compensate
for this by starting the bank request upstream of the fix.

## 10. Replay Integration

`_lateral_response()` stores one output row per longitudinal FMS row, so
`FMSBiChannelResult` remains aligned with `FMSResult`.

Between two longitudinal rows, `_advance_lateral_state()` performs smaller
lateral substeps:

```text
while elapsed < longitudinal_dt:
    interpolate scheduled t, s, h, V_tas
    compute lateral command at current map state
    compute roll and heading rates
    scale map velocity to scheduled physical path distance
    advance east, north, psi, phi
```

The map position update is:

```text
velocity_scale = scheduled_path_speed / command.ground_speed
east_next  = east  + east_dot  * velocity_scale * dt
north_next = north + north_dot * velocity_scale * dt
```

The attitude update is:

```text
phi_next = phi + phi_dot * dt
psi_next = wrap(psi + psi_dot * dt)
```

If the roll update would step past `phi_req`, the code snaps to `phi_req`.

## 11. Tuning Parameters

`LateralGuidanceConfig` currently contains:

```python
LateralGuidanceConfig(
    lookahead_m=1500.0,
    cross_track_gain=1.0,
    track_error_gain=2.0,
    curvature_feedforward_gain=0.25,
    min_lookahead_m=150.0,
    max_los_angle_rad=np.deg2rad(89.0),
    integration_step_s=0.5,
)
```

### lookahead_m

Primary downstream target distance.

Smaller values:

- capture the path more aggressively
- command higher bank
- can oscillate or saturate in tight terminal geometry

Larger values:

- produce smoother commands
- may start responding to upcoming path shape earlier
- can cut corners if feedback is too weak

### cross_track_gain

Scales the L1 gain through:

```text
K_l1 = track_error_gain * sqrt(cross_track_gain)
```

Increasing it makes capture stronger.

### track_error_gain

Directly scales the line-of-sight feedback curvature.

The default `2.0` corresponds to the standard pure-pursuit/L1 coefficient:

```text
kappa = 2 sin(eta) / L
```

### curvature_feedforward_gain

Scales the curvature preview term from the upcoming fly-by path.

Smaller values:

- rely more on line-of-sight feedback
- reduce early bank and saturation
- can become more reactive through tight turns

Larger values:

- start banking more predictively
- can reduce peak cross-track in some turns
- can over-lead and saturate bank in tight or slow segments

The current default `0.25` was chosen from the KDFW ADS-B cross-check because it
kept cross-track low without over-driving the bank request.

### min_lookahead_m

Prevents near-threshold or very-short-path singular behavior.

### max_los_angle_rad

Bounds the line-of-sight error before converting it to curvature. This is not a
bank limit; it only prevents the nonlinear guidance law from trying to turn
toward a target that is effectively behind the aircraft.

### integration_step_s

Controls lateral replay substepping inside each longitudinal FMS time step.

Smaller values:

- improve numerical tracking quality
- increase runtime

Larger values:

- run faster
- can degrade tracking on tight turns or at high ground speed

## 12. Diagnostics

The most useful result fields are:

```text
cross_track_m
track_error_rad
curvature_cmd_inv_m
phi_req_rad
phi_rad
phi_max_rad
max_abs_cross_track_m
max_abs_track_error_rad
max_bank_command_ratio
final_threshold_error_m
```

Interpretation:

- `cross_track_m` should stay near zero after capture.
- `track_error_rad` can be nonzero in turns and during intercepts.
- `curvature_cmd_inv_m` shows the combined feed-forward and feedback command.
- `phi_req_rad` shows what the guidance law wants.
- `phi_rad` shows what the roll dynamics actually achieved.
- `phi_max_rad` shows the active bank envelope.
- `max_bank_command_ratio` near `1.0` means the requested bank touched the
  current bank limit.
- `final_threshold_error_m` is a scalar check of how close the replay ended to
  the runway threshold position.

For the ADS-B cross-check workflow, run:

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/x_check_simap_adsb.py AAL2802M2,ab30f0
```

## 13. Example: AAL2802M2 / ab30f0

The `AAL2802M2,ab30f0` KDFW arrival is a useful validation case because the
route includes several terminal turns:

```text
KIILO > SHMPP > ZROBA > CURLE > TANNO > DELMO > SILER > ZINGG > RW17C
```

The current RNAV fly-by controller generates bank requests before fixes instead
of after them. In the fresh cross-check run:

```text
max |cross-track|:      106.787 m
RMS cross-track:         26.112 m
p95 |cross-track|:       57.368 m
final threshold miss:   184.899 m
max |actual bank|:       24.648 deg
max |requested bank|:    25.000 deg
```

Representative turn-initiation leads from the same run:

```text
SHMPP: bank request begins about 2.4 km before the fix
SILER: bank request begins about 5.4 km before the fix
ZINGG: bank request begins about 12.0 km before the fix
```

The script still reports the separate longitudinal/VNAV status:

```text
infeasible: not enough along-track distance to complete FMS profile before threshold
```

That message is not a lateral tracking failure. It means the longitudinal FMS
profile could not complete the descent to threshold altitude within the route
distance. The lateral replay still reached low cross-track error against the
fly-by RNAV path.

## 14. Why The Algorithm Works Better

The previous controller had lookahead, but the path itself was effectively a
corner-to-corner route. The aircraft could only start banking when the lookahead
target or projection geometry began to reveal the turn.

The current controller improves this in two places:

1. The reference path now bends before the waypoint using line-arc-line fly-by
   geometry.
2. The bank command includes preview curvature from the upcoming path, so the
   aircraft can start rolling before cross-track error grows.

The feedback still matters. If the aircraft is offset from the path, the
lookahead line-of-sight term pulls it back toward the reference while the
feed-forward term supplies the nominal turn curvature.

## 15. Known Limitations

This is still a reduced-order model.

Important simplifications:

- The fly-by radius uses fixed nominal speed and bank constants, not per-waypoint
  predicted groundspeed and active bank limit.
- There is no ARINC 424 leg-type parser or path terminator model.
- There is no RF-leg, heading-to-intercept, direct-to, or fly-over waypoint mode.
- There is no lateral acceleration or jerk envelope beyond bank and roll-rate
  limits.
- There is no separate localizer capture mode.
- The longitudinal path stationing remains based on original route chord length
  to preserve existing descent-distance behavior.

The design goal is pragmatic simulation quality: predictive turn initiation,
low cross-track error, and plausible bank/roll response.

## 16. Reading Order

Start with:

1. Fly-by geometry helpers in `src/simap/path_geometry.py`
2. `ReferencePath.from_geographic()` in `src/simap/path_geometry.py`
3. `LateralGuidanceConfig` in `src/simap/lateral_dynamics.py`
4. `compute_lateral_command()` in `src/simap/lateral_dynamics.py`
5. `lateral_rates()` in `src/simap/lateral_dynamics.py`
6. `_advance_lateral_state()` in `src/simap/fms_bichannel/core.py`
7. `_lateral_response()` in `src/simap/fms_bichannel/core.py`

## 17. Public Reference Context

The implemented controller is not copied from any OEM source. Public Boeing and
Airbus flight-management implementations are proprietary. The design is based on
public RNAV/LNAV concepts:

- fly-by waypoints begin the turn before the fix
- turn anticipation depends on course change, speed, and bank/turn radius
- path tracking uses cross-track and track-angle geometry
- bank follows coordinated-turn curvature demand
- roll response and bank limits constrain the realized aircraft response

Useful public references:

- FAA AIP ENR 1.16, RNAV route and waypoint terminology
- FAA ATBARC RNAV flight behavior guidance on fly-by turns and turn anticipation
