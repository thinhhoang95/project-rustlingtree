# Advisory Tools

Advisory tools provide on-demand context for planning interventions. Compared
to evaluators, they answer what-if questions: how many more along-track miles
are needed to make an arrival feasible, how much time a vectoring extension
adds, or how much time a lower speed instruction gains.

## Important Approach Revision

Do not use `simulation.final_threshold_error_m` as the advisory
`miles_to_gain` source. That field is useful diagnostic metadata for the served
trajectory, but it is not reliable as the additional along-track distance needed
to make a longitudinal profile feasible. In current bichannel artifacts it is a
final threshold-position error, while infeasible longitudinal profiles can also
be threshold-truncated with substantial altitude remaining.

Advisors should instead rebuild a longitudinal `FMSRequest` from the served
arrival payload, then recompute baseline and what-if profiles. This keeps the
answer aligned with the active ScenarioManager view, including future diff
application, and avoids mixing base-artifact metadata with advisory semantics.

## Advisory Tool Output

Core output fields:

- `miles_to_gain_nmi`: extra along-track route miles. Positive means more path
  distance is flown.
- `minutes_to_gain`: extra flight time minutes. Positive means the flight time
  is longer.
- Baseline and what-if planner success/message fields, so the caller can see
  whether the advisory is based on a feasible profile.

For speed-only advisories, the actual route distance does not change, so
`miles_to_gain_nmi` should be `0.0`. Return an additional
`equivalent_vectoring_miles_nmi` field that converts the time gain into an
equivalent route-extension distance using the baseline pre-TOD/level-segment
groundspeed, falling back to initial groundspeed when no level segment exists.

## Three Advisory Tools

1. Feasibility Advisor
   - Rebuild the baseline longitudinal profile from the served arrival.
   - If the baseline is feasible, return zero miles and zero minutes.
   - If the baseline is infeasible, prepend virtual along-track distance to the
     reference path and search for the smallest extension that makes
     `plan_fms_descent()` succeed.
   - Use bracket doubling followed by bisection to the configured TOD tolerance.

2. Vectoring Advisor
   - Given an extra along-track distance `x`, extend the reference path by `x`
     and recompute the profile.
   - Return `miles_to_gain_nmi = x`.
   - Compute `minutes_to_gain` from the recomputed profile time delta, not from
     a fixed cruise-speed shortcut.
   - Include the feasibility-advisor required distance for context.

3. Speed-Control Advisor
   - Given an along-track station `s_m` and lower prescribed CAS, add an
     `ATCSpeedSegment` at that station and recompute the profile.
   - V1 supports reductions only. The requested CAS must be lower than the base
     managed profile at the acceptance station and within existing planned CAS
     bounds.
   - Return actual route miles as `0.0`, plus `minutes_to_gain` and
     `equivalent_vectoring_miles_nmi`.
