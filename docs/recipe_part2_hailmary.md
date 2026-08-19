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


## b. What happens inside one window?

The simplified sequence is:

```text
Traffic scenario
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