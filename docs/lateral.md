# Lateral FMS Walkthrough

This document explains the lateral FMS guidance used by the bi-channel FMS
replay in:

- `src/simap/lateral_dynamics.py`
- `src/simap/fms_bichannel/core.py`

The lateral FMS is intentionally separate from the longitudinal FMS descent
heuristic. The longitudinal channel supplies a time history of distance,
altitude, and speed. The lateral channel replays aircraft map position, heading,
bank, cross-track error, and track error against the same reference path.

The current lateral guidance law is a nonlinear lookahead controller. It is
similar in spirit to public transport-aircraft LNAV path-capture logic and
L1/pure-pursuit guidance: find the active path tangent, aim at a target point
ahead on the path, then command the curvature needed to rotate the current
ground track toward that line of sight.

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
```

Where:

- `tau` is the local unit tangent pointing along the path toward decreasing
  `s`.
- `n` is the local left-normal associated with that tangent.

The aircraft state used by the lateral channel is:

```text
x_lat(t) = [east, north, psi, phi]
```

Where:

- `east`, `north` are local map coordinates in meters.
- `psi` is aircraft heading angle in radians.
- `phi` is bank angle in radians.

The longitudinal channel supplies the schedule:

```text
x_long(t) = [s, h, V_tas]
```

Where:

- `h` is altitude.
- `V_tas` is true airspeed.

## 2. Full Flow

The normal bi-channel flow is:

```text
FMSBiChannelRequest
  -> plan_fms_descent() or simulate_fms_descent()
  -> FMSResult longitudinal profile
  -> _lateral_response()
  -> FMSBiChannelResult
```

The lateral replay loop performs this sequence at each longitudinal output
node:

1. Copy the scheduled `t_s`, `s_m`, `h_m`, and `v_tas_mps` into the lateral
   state.
2. Call `compute_lateral_command()`.
3. Store position, heading, bank, ground track, cross-track error, track error,
   commanded curvature, and bank request.
4. Advance the lateral state to the next longitudinal node with
   `_advance_lateral_state()`.

`_advance_lateral_state()` can take several substeps inside one longitudinal
time step. By default the lateral substep is `0.5 s`, controlled by
`LateralGuidanceConfig.integration_step_s`.

This is important because the precompute pipeline often runs the longitudinal
FMS at `2.0 s`. Holding one lateral command for two seconds can noticeably
degrade path tracking in tight terminal turns.

## 3. Guidance Inputs And Outputs

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

Only `phi_req_rad` is the actual lateral control request. The other fields are
diagnostics or kinematic quantities used by the replay and plots.

## 4. Wind-Aware Ground Track

The guidance law works in ground track, not just aircraft heading. This matters
because crosswind can make the aircraft heading look aligned while the ground
track is drifting across the path.

The air-relative velocity is:

```text
v_air = V_tas [cos(psi), sin(psi)]
```

The weather provider gives:

```text
w = [wind_east, wind_north]
```

The ground velocity is:

```text
v_ground = v_air + w
```

Then:

```text
ground_speed = norm(v_ground)
ground_track = atan2(v_ground_north, v_ground_east)
```

The lateral command uses `ground_track` for the line-of-sight error and for the
reported `track_error_rad`.

## 5. Active Path Projection

The controller does not blindly use the scheduled longitudinal `s_m` to measure
lateral error. Instead it projects the current map position to the closest point
on the reference path:

```text
s_ref = project_s_m(east, north)
p_ref = position_ne(s_ref)
```

Then it gets the local tangent and normal at `s_ref`:

```text
tau = tangent_hat(s_ref)
n = normal_hat(s_ref)
theta = track_angle_rad(s_ref)
```

The position error vector is:

```text
e = [east, north] - p_ref
```

The signed cross-track error is:

```text
cross_track = e dot n
```

The along-path offset from the projected point is:

```text
along_path_offset = e dot tau
```

The reported ground-track error is:

```text
track_error = wrap(ground_track - theta)
```

This closest-point projection is what keeps the controller responsive when the
aircraft has drifted away from the scheduled path station. The longitudinal
profile can still say "we should be at `s = 15 km`", but the lateral controller
uses the nearest path geometry to decide which way the aircraft should turn.

## 6. Lookahead Target Selection

The lookahead distance starts from:

```text
L_base = guidance.lookahead_m
L_min = guidance.min_lookahead_m
```

The default values are:

```text
lookahead_m = 1500.0
min_lookahead_m = 150.0
```

Inside the path bounds, the effective lookahead is:

```text
L = clamp(s_ref, L_min, L_base)
```

Then the target point is downstream along the reference path:

```text
s_target = max(0, s_ref - L)
p_target = position_ne(s_target)
```

Because `s` decreases toward the threshold, subtracting `L` moves the target
ahead of the aircraft in the direction of flight.

### Endpoint Extension

There is one special case near the threshold. Some simulations intentionally
continue beyond `s = 0`. Once the aircraft passes the end of the finite
reference path, projecting to the closest point clamps `s_ref` to the endpoint.
If the controller kept aiming at that endpoint, it would command an unrealistic
turn back to the runway threshold.

To avoid that, when:

```text
s_ref <= L_min
along_path_offset > 0
```

the target is placed on the tangent extension beyond the endpoint:

```text
p_target = p_ref + (along_path_offset + L_base) tau
```

This lets straight-path test cases and threshold-crossing replays continue along
the path extension instead of reversing back toward the endpoint.

## 7. Line-Of-Sight Error

The line-of-sight track from aircraft position to target point is:

```text
chi_los = atan2(target_north - north, target_east - east)
```

The controller compares this target direction to the current ground track:

```text
eta = wrap(chi_los - ground_track)
```

`eta` is clipped by `guidance.max_los_angle_rad`. The default is `89 deg`, which
allows aggressive capture but prevents singular behavior near a 180 degree
look-back condition.

```text
eta = clip(eta, -max_los_angle, +max_los_angle)
```

The sign convention is:

- Positive `eta` means the target lies left of the current ground track.
- Negative `eta` means the target lies right of the current ground track.

The commanded curvature follows the L1/pure-pursuit shape:

```text
kappa_cmd = K_l1 sin(eta) / L
```

Where:

```text
K_l1 = track_error_gain * sqrt(cross_track_gain)
```

The defaults are:

```text
cross_track_gain = 1.0
track_error_gain = 2.0
K_l1 = 2.0
```

The previous lateral law used this approximate form:

```text
kappa_cmd = kappa_ref
            - cross_track_gain * cross_track / L^2
            - track_error_gain * track_error / L
```

That linear feedback can under-command when a large cross-track offset and an
intercepting track error cancel each other. The lookahead law avoids that
cancellation because the cross-track error changes the target line-of-sight
geometry directly.

## 8. Curvature To Bank

The controller converts commanded ground-path curvature into a coordinated-turn
bank request:

```text
phi_req = atan(ground_speed^2 * kappa_cmd / g)
```

`ground_speed` is used here, not TAS. That is intentional: the map-path
curvature is a ground-track curvature. A tailwind increases the bank needed for
the same ground-path turn radius, and a headwind reduces it.

The raw bank request is clipped to the current aircraft and mode limit:

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

Those limits come from `AircraftConfig`, `ModeConfig`, and `bank_limit_rad()`.

## 9. Roll And Heading Dynamics

The lateral command is not applied as an instantaneous heading change. The
roll/heading response is modeled by `lateral_rates()`.

The roll loop is first order:

```text
phi_dot = (phi_req - phi) / tau_phi
```

Then it is clipped by the mode roll-rate limit:

```text
phi_dot = clip(phi_dot, -p_max, +p_max)
```

The heading rate is the coordinated-turn relation:

```text
psi_dot = g tan(phi) / V_tas
```

Note that `psi_dot` uses the current bank angle, not the requested bank angle.
That means heading response naturally lags until the bank response catches up.

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
    advance east, north, psi, phi
```

The map position update is:

```text
east_next  = east  + east_dot  * dt
north_next = north + north_dot * dt
```

The attitude update is:

```text
phi_next = phi + phi_dot * dt
psi_next = wrap(psi + psi_dot * dt)
```

If the roll update would step past `phi_req`, the code snaps to `phi_req`
instead of overshooting it.

The scheduled longitudinal quantities are linearly interpolated during
substeps:

```text
t, s, h, V_tas = lerp(current_longitudinal_row, next_longitudinal_row)
```

That lets the lateral controller see the evolving speed, altitude, and mode
inside a large longitudinal time step.

## 11. Tuning Parameters

`LateralGuidanceConfig` currently contains:

```python
LateralGuidanceConfig(
    lookahead_m=1500.0,
    cross_track_gain=1.0,
    track_error_gain=2.0,
    min_lookahead_m=150.0,
    max_los_angle_rad=np.deg2rad(89.0),
    integration_step_s=0.5,
)
```

### lookahead_m

Primary tuning knob.

Smaller values:

- capture the path more aggressively
- command higher bank
- can oscillate or saturate in tight terminal geometry

Larger values:

- produce smoother commands
- reduce peak bank
- can cut corners and leave larger cross-track error

The current default of `1500 m` was chosen because it gives tight terminal
tracking for the KDFW ADS-B cross-check case without requiring continuous bank
saturation.

### cross_track_gain

Scales the L1 gain through:

```text
K_l1 = track_error_gain * sqrt(cross_track_gain)
```

Increasing it makes capture stronger. Because it is inside a square root, it is
less sensitive than a direct proportional cross-track gain.

### track_error_gain

Directly scales the curvature command. Increasing it makes the aircraft rotate
toward the lookahead target faster.

The default of `2.0` corresponds to the standard pure-pursuit/L1 coefficient:

```text
kappa = 2 sin(eta) / L
```

### min_lookahead_m

Prevents near-threshold or very-short-path singular behavior.

If this is too small, the target point can become too close to the aircraft and
produce noisy curvature near the threshold. If it is too large, the final
approach can become less precise.

### max_los_angle_rad

Bounds the line-of-sight error before converting it to curvature.

This is not a bank limit. It only prevents the nonlinear guidance law from
trying to turn toward a target that is effectively behind the aircraft.

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
- `phi_req_rad` shows what the guidance law wants.
- `phi_rad` shows what the roll dynamics actually achieved.
- `phi_max_rad` shows the active bank envelope.
- `max_bank_command_ratio` near `1.0` means the requested bank touched the
  current bank limit.
- `final_threshold_error_m` is a quick scalar check of how close the replay
  ended to the runway threshold position.

For the ADS-B cross-check workflow, run:

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/x_check_simap_adsb.py AAL860M1,a35b39
```

The companion diagnostics script can also be used to compute scalar tracking
metrics from `FMSBiChannelResult`.

## 13. Example Behavior

On the `AAL860M1,a35b39` KDFW arrival, the previous linear curvature-feedback
law showed poor lateral tracking:

```text
max |cross-track|:     about 1489.6 m
RMS cross-track:       about 506.9 m
p95 |cross-track|:     about 1281.9 m
final threshold miss:  about 990.7 m
```

The current lookahead law with lateral substepping gives:

```text
max |cross-track|:     about 165.6 m
RMS cross-track:       about 15.5 m
p95 |cross-track|:     about 5.2 m
final threshold miss:  about 16.9 m
```

The peak cross-track error occurs around a tight terminal capture segment. Most
of the route is much tighter than the peak, which is why the RMS and p95 numbers
are far lower than the maximum.

## 14. Why The Algorithm Works Better

The key difference is how cross-track error influences the command.

The old law computed two separate linear terms:

```text
cross-track correction
track-error correction
```

Those terms could oppose each other. For example, if the aircraft was far off
path but already pointed somewhat back toward the path, the negative track error
term could cancel the positive cross-track term. The aircraft then under-turned,
lagged the path, and accumulated large cross-track error through the turn.

The new law first turns the path error into a geometric target point:

```text
aircraft position -> lookahead target
```

Then it asks one question:

```text
How much curvature is needed to rotate the current ground track toward that
target?
```

That makes large offsets naturally produce larger line-of-sight angles, while
small errors near the centerline naturally produce small commands.

## 15. Known Limitations

This is still a reduced-order model, not a proprietary Airbus or Boeing LNAV
implementation.

Important simplifications:

- There is no explicit fly-by turn anticipation based on waypoint turn radius.
- The reference path itself is already sampled and smoothed by `ReferencePath`;
  the controller follows that path rather than constructing ARINC leg geometry.
- There is no lateral acceleration or jerk envelope beyond bank and roll-rate
  limits.
- There is no separate localizer capture mode.
- There is no explicit heading-select or direct-to transition logic.

The design goal is pragmatic simulation quality: tight path tracking against
the reference path with plausible bank and roll response.

## 16. Reading Order

Start with:

1. `LateralGuidanceConfig` in `src/simap/lateral_dynamics.py`
2. `compute_lateral_command()` in `src/simap/lateral_dynamics.py`
3. `lateral_rates()` in `src/simap/lateral_dynamics.py`
4. `_lateral_response()` in `src/simap/fms_bichannel/core.py`
5. `_advance_lateral_state()` in `src/simap/fms_bichannel/core.py`
6. `ReferencePath.project_s_m()` in `src/simap/path_geometry.py`

## 17. Public Reference Context

The implemented controller is not copied from any OEM source. Public Boeing and
Airbus flight-management implementations are proprietary. The design is instead
based on public guidance concepts that are common across transport LNAV,
UAV-style L1 guidance, and pure-pursuit path following:

- measure cross-track error against a tangent point
- compute a desired target direction ahead on the path
- command curvature from line-of-sight geometry
- convert curvature to bank with coordinated-turn dynamics
- enforce bank and roll-rate limits before integrating aircraft response

Publicly available material that motivated the design includes NASA B-737 FMS
lateral guidance reports and public patents describing tangent-point,
cross-track, track-error, and LNAV path-following concepts.
