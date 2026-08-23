# Scope of Traffic Scaling

One `TrafficScenario` contains one one-hour demand window. A `TrafficScenarioBatch` contains many such scenarios, but they are simulated independently.

## Full day or individual windows?

Hailmary does not run one continuous full-day simulation. **It only concerns the flights whose terminal-entry times are within the intervened window (say 09:00–10:00)**. The first event in the event queue could start at or later than 09:00, and the last event will correspond to the last flight whose original terminal-arrival time is within the intervend window.

```text
Window 1 → fresh simulator → finish scenario
Window 2 → new fresh simulator → finish scenario
Window 3 → new fresh simulator → finish scenario
...
```

The learning population persists between windows, but traffic and simulator state do not.

Within one scenario:

- It includes every selected flight and runway in that one-hour terminal-entry window.
- The simulator starts from a fresh event queue.
- Flights may continue flying after the nominal window end until their trajectories complete. The window controls which flights are included, not a hard simulation stop time.
- At decision points, temporary rollouts run only to a local outcome horizon—usually until the relevant pair crosses its shared resource—not through the whole day.

This loop is visible in [`Phase0ExperimentRunner._train_one()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/learning/phase0.py:947).

One important current limitation: `Phase0ExperimentRunner` accepts only scale `1.0`. Scaled scenarios can be generated, but I found no separate end-to-end `Phase1ExperimentRunner`. Therefore, scaled traffic is not automatically passed into the current training runner.

In order to test the demand scaling, run the script:
```bash
./.venv/bin/python -m hailmary.cli.visualize_demand_scaling \
  --scale 2.0 \
  --timezone America/Chicago \
  --window-start "2026-04-01 09:00"
```

---

# Hail Mary Proceedings: What Happens Inside One Window?

The simplified sequence is:

```text
Traffic scenario (One intervened window, and only flights pertaining to this window are considered to build an event queue)
    ↓
Create event-driven simulator
    ↓
Advance to next event
    ↓
Find adjacent leader/follower pairs
    ↓
Generate feasible actions and features
    ↓
Match/create candidate rules
    ↓
Run selected/rival/no-op branches
    ↓
Assign causal credit
    ↓
Update/evolve rules
    ↓
Apply only selected action to real scenario
    ↓
Continue to next event
```

## Step 1 — Create the Event Queue

The route graph is already built offline. For the current scenario, Hailmary creates an event-driven simulator containing events such as:

- flight released at the reconstructed 50-NM terminal-entry time;
- action station crossed;
- route resource crossed;
- runway threshold crossed;
- flight completed;
- exogenous disturbance, if configured.

This is an event heap rather than a newly learned “event graph.” The simulator processes simultaneous events together in [`Simulator.advance_next()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/simulator/engine.py:207).

### Why an Event-Driven Simulator?

This is because the simulator does not try to advance the timeline by second or milisecond. Instead, it will look at the key moments in the timeline (similar to *keyframes* in movie-making), and the timeline will just *jump* to these timeline moments. So that's a *leapfrogging* behavior instead.

Generally, an event is **consumed**, which mostly means updating some bookkeeping system like refining the event queues (removing past events), creating new decision epoch, change the lifecycle flag to `active`... It does not change the takeoff time, or the flight path just because an event is consumed (technically there is an exception: the *exogeneous disturbance* event will also mutate the flight state—takeoff time for instance).

Think of each trajectory as a prerecorded film, and events as bookmarks in that film.

The simulator does not “push” an aircraft forward every second. Instead, it knows:

```text
trajectory(elapsed time) → position, altitude, speed
```

For flight `F`:

```text
physical_state(t) = F.current_trajectory(t - F.trajectory_origin_time)
```

The event queue says when to stop fast-forwarding the films because something relevant happens. An action can replace the unplayed remainder of one film.

### A Concrete Two-Aircraft Example

Suppose two arrivals share runway `RWY`:

```text
Leader L predicted runway time:   t=200
Follower F predicted runway time: t=250
Required runway spacing:             90 seconds
```

The predicted spacing is:

```text
250 - 200 = 50 seconds
```

So the follower is predicted to arrive 40 seconds too soon:

```text
required delay = 90 - 50 = 40 seconds
```

Flight `F` has a speed-action station that it will reach at `t=130`.

At initialization, part of the queue might look like:

```text
t=100  RELEASED(L)
t=110  RELEASED(F)
t=130  ACTION_STATION_CROSSED(F, station=3)
t=200  RESOURCE_CROSSED(L, RWY)
t=200  COMPLETED(L)
t=250  RESOURCE_CROSSED(F, RWY)
t=250  COMPLETED(F)
```

#### 1. At `t=100`: Leader Release

The engine jumps directly to `t=100` and consumes `RELEASED(L)`.

State changes:

```text
simulation time: 100
L.lifecycle: scheduled → active
release event: removed from queue
```

What does not change:

```text
L.current_trajectory remains BASE_L
trajectory arrays remain immutable
```

Nevertheless, asking “where is L at `t=105`?” works by sampling `BASE_L` five seconds after its trajectory origin.

#### 2. At `t=110`: Follower Release

The same thing happens:

```text
simulation time: 110
F.lifecycle: scheduled → active
```

No trajectory changes.

#### 3. At `t=130`: Follower Reaches an Action Station

The event time was calculated in advance from the trajectory:

```text
BASE_F reaches station 3 at elapsed time 20 s
F trajectory origin = 110
event time = 110 + 20 = 130
```

When the event is processed:

```text
simulation time: 130
station 3 added to F.crossed_action_stations
event removed from queue
decision epoch created
```

Again, the event itself does not change the trajectory. It records:

> According to the current trajectory and clock, F is exactly at station 3 now.

The controller then examines the post-event state:

```text
leader runway ETA:   200
follower runway ETA: 250
spacing deficit:      40 seconds
```

The action catalog might offer:

```text
no-op
speed/light
speed/medium
speed/heavy
path-stretch
```

These are possibilities, not commands.

#### 4. Counterfactual Comparison

Suppose the controller temporarily forks the world.

##### No-op branch

```text
F keeps BASE_F
F runway event remains t=250
spacing remains 50 seconds
```

##### Slowdown branch

A slower trajectory variant `SLOW_F` is created. At `t=130`, it is spliced onto the aircraft’s current location:

```text
Past:
    t=110–130 uses BASE_F and remains unchanged

Future:
    t=130 onward uses SLOW_F
```

Suppose this variant adds 40 seconds before the runway:

```text
old F runway time: 250
new F runway time: 290

new spacing: 290 - 200 = 90 seconds
```

The slowdown branch therefore receives a better spacing score.

The temporary branches are discarded. If the policy selects slowdown, the same action is applied to the real simulator.

#### 5. Applying the Selected Action

Applying slowdown changes:

```text
F.current_variant_id:
    BASE_F → SLOW_F

F.action_history:
    [] → [speed-action-id]

F.speed_action_count:
    0 → 1

predicted RWY crossing:
    250 → 290
```

The queue changes to:

```text
t=200  RESOURCE_CROSSED(L, RWY)
t=200  COMPLETED(L)
t=290  RESOURCE_CROSSED(F, RWY)
t=290  COMPLETED(F)
```

Crucially:

```text
F's position at t=130 before action
=
F's position at t=130 after action
```

There is no teleportation. Only its future changes. This is the purpose of the splice logic in [engine.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/simulator/engine.py:607).

### Where Is the Conflict Event?

There isn’t one.

The spacing deficit at `t=130` is derived from predicted runway times. It motivates the action, but it is not a queued event.

Similarly, suppose the two aircraft violate airborne separation between `t=157.3` and `t=164.8`. The conflict detector can analytically discover that interval even if the event queue contains only:

```text
t=150  some station crossing
t=170  some resource crossing
```

The engine does not need `CONFLICT_STARTED` and `CONFLICT_ENDED` events. Conflict is an evaluated property of the trajectories over an interval; events are the timestamps where the simulator must update discrete state or consult the controller.

---

## Step 2 — Detect Useful Aircraft Pairs

At every decision-triggering event, Hailmary examines the current route-segment flows.

It finds aircraft that are adjacent on a shared directed segment:

```text
leader → follower
```

This is recomputed dynamically because previous speed or path actions may change the predicted merge order. See [`build_current_segment_anchors()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/features/anchors.py:311).

If a window has no actionable shared-segment pair, it remains in traffic auditing but contributes no learning experiment.

## Step 3 — Enumerate Feasible Actions

For each leader/follower pair, the action catalog generates actions bound to the follower, including:

- no-op;
- slowdown bands;
- path-stretch alternatives.

Infeasible actions are omitted. This happens through [`ActionCatalog`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/actions/catalog.py:284).

> **Notice:** Even though the simulator may stop at every action station for every flight, the actions that can be considered are strictly reserved for the pertaining flight only. In other words, if flight `A` hits the action station, then flight `B` does not have the chance to take action.

## Step 4 — Compute the Feature Vector

Hailmary converts the current pair and surrounding traffic into features, including concepts such as:

- predicted spacing error;
- required separation;
- distance and time to the shared resource;
- slowdown and path-stretch capacity;
- traffic pressure;
- remaining intervention freedom;
- downstream-trailer margins;
- exact airport, runway, segment and cluster identities.

This is done by [`simulator_state_vector()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/features/state_vector.py:409).

## Step 5 — Match or Create Rules

Existing mutable rules are tested against the feature vector.

Conceptually, a rule looks like:

```text
IF
    spacing error is between −90 and 0 seconds
    AND speed capacity is sufficient
    AND airport/runway/segment/cluster identities match
THEN
    apply medium slowdown to the follower
```

If the current state lacks rules advocating some feasible actions, the covering mechanism creates initial rules around the current feature point. See [`_freeze_match_sets()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/learning/trainer.py:660).

Thus, candidate actions come from the physical action catalog; candidate rules are created and evolved by the learner.

## Step 6 — Run Three Temporary Rollout Arms

For one selected pair, Hailmary forks the exact same simulator state into:

```text
Arm A: selected action
Arm B: strongest rival action
Arm C: no-op
```

Each branch runs to the same local outcome horizon under the same frozen continuation policy. It scores spacing, safety, intervention cost and related outcome terms.

The three branches are temporary and discarded afterward. This occurs in [`CausalTrainer.process_epoch()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/learning/trainer.py:1253).

## Step 7 — Update the Rules

Hailmary compares the scores:

```text
selected − rival
selected − no-op
rival − no-op
```

Those differences update the evidence attached to matching rules. Over repeated windows and decision epochs, the learner:

- accumulates means, variances and confidence bounds;
- mutates and crosses rules within action niches;
- specializes overly broad rules;
- deletes weak rules when the population is full;
- periodically certifies sufficiently supported rules into a frozen deployment rulebook.

Only Arm A’s initial action is committed to the real scenario. The rival and no-op branches never alter it. The simulator then advances to the next real event.

---

## Rule Credit Assignment: A Worked Example

This section elaborates on how the scoring and credit assignment in Steps 6 and 7 actually work end to end.

The key mental model is: the route graph defines a one-dimensional queue on each shared directed segment. An action is graded by how it repairs one adjacent gap without damaging the following gaps too much.

Also, rules do not receive the branch’s absolute score. They receive score differences between selected, rival, and no-op arms.

### Concrete Route-Graph Example

Suppose five aircraft share segment `S`, whose exit resource requires 90-second spacing:

```text
upstream                                      segment exit
L → F → T1 → T2 → T3  ────────────────────────────►
    ^
    action is applied to F
```

The ordering comes from physical progress for current occupants, followed by entry ETA for future entrants. Adjacent pairs become anchors ([anchors.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/features/anchors.py:311)).

For the decision anchored on `L → F`, the generic outcome cohort is frozen as:

```text
bound pair:  L → F
trailers:        F → T1 → T2 → T3
```

Assume the original predicted exit times are:

| Aircraft | Exit time |
|---|---:|
| L | 1000 |
| F | 1050 |
| T1 | 1140 |
| T2 | 1230 |
| T3 | 1320 |

Thus the original gaps are:

```text
L→F   = 50 s    deficient
F→T1  = 90 s    ideal
T1→T2 = 90 s    ideal
T2→T3 = 90 s    ideal
```

### Converting One Gap into a Score

With required interval \(R=90\), each adjacent gap is converted using the piecewise function in [spacing.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/evaluation/spacing.py:87):

```text
gap ≤ 0                  → −1
0 < gap < 90             → −1 + 2 × gap/90
90 ≤ gap ≤ 112.5         → +1
112.5 < gap < 180        → declines linearly
gap ≥ 180                → −0.5
```

Therefore:

```text
50-second gap:
−1 + 2 × 50/90 = +0.111

90-second gap:
+1.000

30-second gap:
−1 + 2 × 30/90 = −0.333
```

Notice that a 50-second spacing violation still has a slightly positive score. Zero on this scale occurs at 45 seconds; it is not the boundary between legal and illegal spacing.

### Three Rollout Arms

Suppose the controller compares:

```text
Arm A: medium slowdown, adding 40 s to F
Arm B: heavy slowdown, adding 60 s to F
Arm C: no-op
```

Assume all trajectories remain dynamically feasible and no later interventions occur.

#### Arm C: No-op

The gaps remain:

```text
50, 90, 90, 90
```

Therefore:

```text
pair score        = score(L→F) = 0.111
propagation score = mean(1, 1, 1) = 1.000
intervention term = 0
throughput term   = 0
```

Total:

```text
Y_noop = 0.111 + 1.000 = 1.111
```

#### Arm A: Medium Slowdown

F moves from `1050` to `1090`:

```text
L→F   = 90 s
F→T1  = 50 s
T1→T2 = 90 s
T2→T3 = 90 s
```

The action perfectly repairs the bound pair, but transfers some compression to the first trailer:

```text
pair score        = 1.000
propagation score = mean(0.111, 1, 1)
                  = 0.704
```

For one 15-knot speed action, the normalized intervention cost is approximately `0.361`; with weight `0.3`, its contribution is `−0.108` ([outcome.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/evaluation/outcome.py:176)).

No gap exceeds 90 seconds, so the excess-gap throughput term is zero:

```text
Y_medium = 1.000 + 0.704 − 0.108
         = 1.595
```

#### Arm B: Heavy Slowdown

F moves to `1110`:

```text
L→F   = 110 s
F→T1  = 30 s
T1→T2 = 90 s
T2→T3 = 90 s
```

The 110-second bound gap still receives the maximum `+1`, but the first trailer is now heavily compressed:

```text
pair score        = 1.000
propagation score = mean(−0.333, 1, 1)
                  = 0.556
intervention term = −0.133
throughput term   = −0.006
```

The throughput penalty appears because the 110-second gap contains 20 seconds of excess spacing. It is averaged over the four cohort edges and given only weight `0.1`.

```text
Y_heavy = 1.000 + 0.556 − 0.133 − 0.006
        = 1.417
```

The full composition is implemented directly in [outcome.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/evaluation/outcome.py:339).

#### Intuitive Mental Model of the Score

```
pair:
    Did we repair the target gap?

propagation:
    Did repairing it damage the next three gaps?

intervention:
    How complicated/aggressive was the solution?

throughput:
    Did the solution create unnecessarily large gaps?
```

### How the Rules Are Actually Credited

The arm ordering is:

```text
medium slowdown: 1.595
heavy slowdown:  1.417
no-op:           1.111
```

The learner derives:

```text
medium − heavy = +0.179
medium − no-op = +0.484
heavy  − no-op = +0.306
```

These differences—not the raw scores—grade the matching rules ([credit.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/learning/credit.py:57)):

- Medium-slowdown rules receive `+0.179` in their rival/evolution ledger and `+0.484` in their no-op/deployment ledger.
- Heavy-slowdown rules receive `−0.179` in their rival ledger, but still receive `+0.306` versus no-op.
- No-op rules receive veto evidence of `1.111 − 1.595 = −0.484`.

That distinction is important: heavy slowdown is learned as beneficial relative to doing nothing, but inferior to medium slowdown.

### Nuances: Pair Identity Freezing During Rollouts

When flows merge together, the pair identities will adapt correctly; but in a three-arm rollout, the pair identities stay frozen. This is to prevent unstable learning of causality.

Yes, with one crucial qualification: pair identities are refreshed between real decision epochs, but frozen during one three-arm rollout.

#### Are Pairs Adapted Over Time?

At every real decision epoch, Hailmary rebuilds the flow for every active route segment:

- Aircraft already occupying the segment are ordered by physical progress.
- Future entrants are ordered by predicted entry time.
- Adjacent aircraft become leader–follower pairs.
- Aircraft that have crossed the segment exit are removed.

This happens in [anchors.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/features/anchors.py:311).

Suppose two inbound flows merge:

```text
Before merge:

Flow A: A1 → A2
Flow B: B1 → B2

After predicted merge ordering:

A1 → B1 → A2 → B2
```

At the next real decision epoch, the resulting anchors could be:

```text
A1→B1
B1→A2
A2→B2
```

So yes, the learner adapts to the current physical/predicted sequence rather than permanently treating `A1→A2` as a pair.

However, once a three-arm experiment begins, its cohort is deliberately frozen:

```text
real decision epoch:
    freeze L, F, T1, T2, T3

selected rollout:
    score those same identities

rival rollout:
    score those same identities

no-op rollout:
    score those same identities
```

This is enforced by `freeze_outcome_cohort()` ([outcome.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/evaluation/outcome.py:56)).

If an action makes the frozen follower overtake its leader in one branch, Hailmary does not silently redefine the pair to make the branch look better. Their exit-time difference becomes zero or negative and receives a bad score. Freezing is necessary so all three arms answer the same causal question.

After the chosen action is committed and the real simulator advances to the next decision event, pairs are rebuilt again.

So the lifecycle is:

```text
Real epoch 1:
    rebuild current pairs
    freeze selected pair for three-arm comparison
    commit one action

Real epoch 2:
    rebuild pairs from the changed real state
    freeze the new selected pair
    ...
```

#### Where Is a Long Segment Evaluated?

Not at the segment beginning. The segment has three conceptually different locations:

```text
segment entry         action station(s)          segment exit
     │                       │                         │
     ▼                       ▼                         ▼
determine membership   trigger a decision       evaluate spacing
and merge ordering     and apply an action       at this resource
```

The scoring resource is the segment’s **exit gate**. When an anchor is created:

```python
anchor.resource_id = segment.exit_resource_id
```

([anchors.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/features/anchors.py:383))

The entry gate is used to distinguish:

- aircraft already physically occupying the segment;
- aircraft committed to enter it later.

The exit gate is where Hailmary asks:

```text
When will the leader cross the segment exit?
When will the follower cross it?
Is the difference close to the required interval?
```

Actions are considered when the follower crosses an eligible action station—not necessarily at the segment entry ([catalog.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/actions/catalog.py:335)).

For example:

```text
Segment length: 60 NM
Required exit interval: 90 s

t=100: F crosses an upstream action station
       current predicted exit times:
           L = 900
           F = 950
       predicted exit spacing = 50 s

       → three-arm rollout begins at t=100
       → slowdown/path-stretch/no-op are compared
       → each arm is graded at the segment exit resource
```

The branches run until a common horizon:

```text
last frozen trailer’s baseline segment-exit time
+ one required spacing interval
```

If there are no trailers, it is the follower’s baseline exit time plus one interval ([outcome.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/evaluation/outcome.py:86)).

Thus “evaluation at the exit” does not mean the real controller waits until the aircraft physically reaches the exit before choosing. It means the decision is made upstream using temporary rollouts, and the consequence is measured through predicted exit-crossing times after those rollouts.

On a long segment with several action stations, Hailmary can reconsider the traffic repeatedly:

```text
station 1 → rebuild pairs → rollout → commit
station 2 → rebuild pairs → rollout → commit
station 3 → rebuild pairs → rollout → commit
segment exit
```

That repeated re-anchoring is precisely what lets the system adapt when merge order or leader–follower identity changes over time.

---

## The Key Distinction

```text
Within a window:
    state and actions evolve sequentially.

Between windows:
    simulator state resets completely,
    but the learned rule population persists.
```

So Hailmary learns across a day’s collection of windows, but it does not model the day as one continuous interaction. Each window is an independent traffic experiment contributing evidence to the same evolving rule population.
