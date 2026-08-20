# Scope of Traffic Scaling

One `TrafficScenario` contains one one-hour demand window. A `TrafficScenarioBatch` contains many such scenarios, but they are simulated independently.

## a. Full day or individual windows?

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


# Hail Mary proceedings: What happens inside one window?

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

### 1. Create the event queue

The route graph is already built offline. For the current scenario, Hailmary creates an event-driven simulator containing events such as:

- flight released at the reconstructed 50-NM terminal-entry time;
- action station crossed;
- route resource crossed;
- runway threshold crossed;
- flight completed;
- exogenous disturbance, if configured.

This is an event heap rather than a newly learned “event graph.” The simulator processes simultaneous events together in [`Simulator.advance_next()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/simulator/engine.py:207).

#### Why is there an Event-driven simulator and why does the event queue matter at all?

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

### A concrete two-aircraft example

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

### 1. At `t=100`: leader release

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

### 2. At `t=110`: follower release

The same thing happens:

```text
simulation time: 110
F.lifecycle: scheduled → active
```

No trajectory changes.

### 3. At `t=130`: follower reaches an action station

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

### 4. Counterfactual comparison

Suppose the controller temporarily forks the world.

#### No-op branch

```text
F keeps BASE_F
F runway event remains t=250
spacing remains 50 seconds
```

#### Slowdown branch

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

### 5. Applying the selected action

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

### Where is the conflict event?

There isn’t one.

The spacing deficit at `t=130` is derived from predicted runway times. It motivates the action, but it is not a queued event.

Similarly, suppose the two aircraft violate airborne separation between `t=157.3` and `t=164.8`. The conflict detector can analytically discover that interval even if the event queue contains only:

```text
t=150  some station crossing
t=170  some resource crossing
```

The engine does not need `CONFLICT_STARTED` and `CONFLICT_ENDED` events. Conflict is an evaluated property of the trajectories over an interval; events are the timestamps where the simulator must update discrete state or consult the controller.

---

### 2. Detect useful aircraft pairs

At every decision-triggering event, Hailmary examines the current route-segment flows.

It finds aircraft that are adjacent on a shared directed segment:

```text
leader → follower
```

This is recomputed dynamically because previous speed or path actions may change the predicted merge order. See [`build_current_segment_anchors()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/features/anchors.py:311).

If a window has no actionable shared-segment pair, it remains in traffic auditing but contributes no learning experiment.

### 3. Enumerate feasible actions

For each leader/follower pair, the action catalog generates actions bound to the follower, including:

- no-op;
- slowdown bands;
- path-stretch alternatives.

Infeasible actions are omitted. This happens through [`ActionCatalog`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/actions/catalog.py:284).

> **Notice:** Even though the simulator may stop at every action station for every flight, the actions that can be considered are strictly reserved for the pertaining flight only. In other words, if flight `A` hits the action station, then flight `B` does not have the chance to take action.

### Shared Resources—to correctly identify the leader–follower pairs, and provide the necessary delay baseline to compute the pair's features such as: `required_delay_s`, `required_delay_over_speed_capacity`, `required_delay_over_path_capacity`

Resources represent shared sequencing gates, including merge boundaries. They are used to establish flows, leader–follower ordering, ETAs, and required spacing. Shared resources produce events (entering and exiting shared resources such as runway threshold or segment begin/end, which defines the keyframes where the event-driven simulator will hop to).

In the current implementation, the concrete resources are:

| Resource | Example identity | Purpose |
|---|---|---|
| Runway threshold | `KATL:RW18R:threshold` | Final runway crossing and throughput/spacing reference |
| Route-segment entry | `segment_<hash>:entry` | Marks entry into a shared corridor; often corresponds to a merge/intercept boundary |
| Route-segment exit | `segment_<hash>:exit` | The downstream gate where that segment’s sequence spacing is evaluated |

Every route segment automatically receives entry and exit resources ([graph.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/topology/graph.py:207)). The underlying boundary node is classified as `merge`, `corridor`, or `runway_endpoint`, depending on how route-cluster membership changes there ([graph.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/topology/graph.py:522)). So the “merge resource” is normally represented as a segment entry/exit gate rather than a separately named `MERGE_POINT` resource.

A resource also carries a required crossing interval—90 seconds by default ([models.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/scenario/models.py:261)).

Concretely, suppose A and B approach a shared final segment from different branches:

1. The shared segment’s entry resource represents the merge boundary.
2. Before entry, aircraft are ordered by predicted entry ETA.
3. After crossing the entry resource, they become physical occupants, ordered by actual progress.
4. Their predicted times at the segment’s exit resource determine the spacing:
   `B exit ETA − A exit ETA`.
5. If that interval is 55 seconds against a required 90 seconds, Hailmary derives a 35-second required delay. This delay value will be used to compute the pair's features such as `required_delay_s`, `required_delay_over_speed_capacity`, `required_delay_over_path_capacity`. 

> **Notice:**  this delay value is not enforced by the simulator, it's sole purpose is to compute the feature values. 

Remark: the (global) conflict detector's use is to be employed by the low-medium-high path stretch to measure which variant of the path stretching is optimal. 

### 4. Compute the feature vector

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

### 5. Match or create rules

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

### 6. Run three temporary rollout arms

For one selected pair, Hailmary forks the exact same simulator state into:

```text
Arm A: selected action
Arm B: strongest rival action
Arm C: no-op
```

Each branch runs to the same local outcome horizon under the same frozen continuation policy. It scores spacing, safety, intervention cost and related outcome terms.

The three branches are temporary and discarded afterward. This occurs in [`CausalTrainer.process_epoch()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/learning/trainer.py:1253).

### 7. Update the rules

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

## The key distinction

```text
Within a window:
    state and actions evolve sequentially.

Between windows:
    simulator state resets completely,
    but the learned rule population persists.
```

So Hailmary learns across a day’s collection of windows, but it does not model the day as one continuous interaction. Each window is an independent traffic experiment contributing evidence to the same evolving rule population.
