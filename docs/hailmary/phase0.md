# Phase 0: Mental Model and End-to-End Walkthrough

Phase 0 is a controlled scientific experiment around the Hailmary learner. It is
not merely a training script, and it is not a phase in the simulator timeline.

Its central question is:

> Can the system discover that a delay should be handled according to the
> available physical capacity—speed when speed can absorb it, path stretch when
> speed cannot—rather than learning a convenient confound such as distance to
> final?

The executable orchestration is implemented in
[`phase0.py`](../../src/hailmary/learning/phase0.py), and its scientific
acceptance gates are implemented in
[`validation.py`](../../src/hailmary/learning/validation.py).

## The shortest useful mental model

- The **mutable population** is the laboratory notebook: it contains tentative
  `condition => action` hypotheses and their evolving evidence.
- A **three-arm rollout** is one controlled experiment on those hypotheses.
- The **evolution ledger** records whether an action beat its strongest rival.
- The **deployment ledger** records whether an action was actually better than
  doing nothing.
- The **frozen rulebook** is the published result: only independently certified
  rules, detached from the mutable population.
- The **Phase-0 runner** is the laboratory protocol around all of these pieces.
  It repeats the experiment across controlled scenarios, random seeds, and
  publication intervals, then evaluates the resulting rulebook on held-out
  scenarios.

```mermaid
flowchart LR
    A["Balanced two-aircraft scenarios"] --> B["Mutable rule population"]
    B --> C["Choose one root action"]
    C --> D["Three common-root rollout arms"]
    D --> E["Update causal ledgers"]
    E --> F["GA, specialization, and deletion"]
    F --> G["Slow certification tick"]
    G --> H["Immutable deployed rulebook"]
    H --> I["Held-out and robustness tests"]
    I --> J["Phase-0 acceptance report"]
```

## 1. The experimental hypothesis

Phase 0 focuses on one operational concept:

```text
small required delay relative to speed capacity
    -> one of the speed actions should survive

required delay beyond speed capacity but within path capacity
    -> path_stretch/oracle_short_medium_long should survive
```

The important phrase is **relative to capacity**.

A correlational learner might discover that speed tends to work when an
aircraft is far from final. That may predict the training data but still be the
wrong explanation. The intended causal rule is closer to:

```text
required_delay / speed_capacity <= 1
```

Distance to final affects capacity, but it should not replace capacity as the
rule's explanation. Phase 0 therefore generates cases that separate these
variables—for example, large errors detected late—so that a distance-based
confound cannot survive merely because it was usually correct in naturalistic
traffic.

## 2. Experimental inputs

The runner receives two already-generated `FactorialScenarioBatch` objects:

- a training batch;
- a materially disjoint held-out batch.

The runner validates the batches; it does not silently manufacture successful
evidence.

### Two-aircraft scope

Every Phase-0 scenario contains exactly:

```text
leader -> follower
```

The leader must release before the follower, and the follower must expose both
speed and path-stretch action opportunities.

This is intentionally smaller than the general rollout machinery. The generic
outcome can follow downstream trailers, but Phase 0 isolates the basic
speed-versus-path concept using one leader–follower pair.

### Balanced four-factor design

Each batch must be a balanced full factorial over four binary factors:

- commitment;
- error magnitude;
- pressure;
- time to final.

That gives at least `2^4 = 16` cells per replicate.

The factors must be realized in simulator-visible state:

- **error magnitude** changes the timing disturbance;
- **time to final** changes when the disturbance occurs;
- **pressure** changes schedule compression;
- **commitment** changes the follower's remaining operational freedom.

They are not accepted as metadata-only labels.

### Correlation audit

The generated batch audits registered factor pairs, including:

- error magnitude versus time to final;
- error magnitude versus pressure;
- commitment versus pressure;
- commitment versus error magnitude.

The default absolute-correlation limit is 0.30. This audit is part of the
scientific design: the learner cannot demonstrate removal of a confound if the
generator never separates the confounded variables.

### Training/held-out separation

Training and held-out scenarios are fingerprinted from their material physical
content after normalizing cosmetic identifiers, ordering, and labels. A
physically equivalent realization cannot occur in both sets merely under a new
scenario ID, flight ID, or runway label.

## 3. Fixed action vocabulary

Phase 0 retains the existing Hailmary macro-actions:

```text
no_op/no_op
speed/light       = 10 kt reduction
speed/medium      = 15 kt reduction
speed/heavy       = 20 kt reduction
path_stretch/oracle_short_medium_long
```

A speed band is a physical command reduction. It does not promise to absorb a
fixed amount of time; realized delay depends on the aircraft's route, speed
envelope, and remaining distance.

Path stretch is one learner-visible macro-action. Its realization layer may
evaluate short, medium, and long candidate geometries, but the learner sees one
stable identity. The geometry-only selector is retained as an explicit
ablation.

No swap, hold, meter-upstream, or delay-target action is introduced.

## 4. Experiment matrix

The default `Phase0ExperimentConfig` requests:

- seeds `17` and `29`;
- rulebook refresh intervals `100` and `500`;
- one pass over the training batch;
- a vanilla accuracy-based control.

The causal learner runs for every:

```text
seed x refresh interval
```

With the defaults, that is four causal runs.

The vanilla learner runs once per seed at the primary, largest refresh
interval. It shares the scenario design, rule representation, population
mechanics, covering, exploration, and action vocabulary. Its credit signal is
the important difference.

The selected primary causal result is currently the first configured seed at
the largest configured refresh interval. The other runs are retained as
robustness evidence rather than discarded.

## 5. One training scenario

Every scenario starts with a fresh simulator constructed by the fixed
`ActionRuntime`.

The runner advances simulator events until a decision epoch appears. At that
point, the trainer:

1. constructs the current leader–follower anchor;
2. enumerates feasible action candidates for the follower;
3. computes the canonical feature vector;
4. builds a frozen match set for that decision.

Important features include:

- spacing deviation;
- time to final;
- pressure;
- commitment;
- required delay divided by speed capacity;
- required delay divided by path capacity.

The last two ratios are the intended concept vocabulary. They turn the
diagonal question "is the error small relative to what this aircraft can
absorb?" into an axis-aligned interval that one XCSR rule can express.

## 6. Rules, matching, and covering

A mutable action rule looks approximately like:

```text
IF role = leader_follower
AND required_delay_over_speed_capacity in [0.0, 1.0]
AND spacing_deviation_s in [...]
THEN speed/light
```

At a decision epoch:

1. Rules with the correct role and feature-schema hash are considered.
2. Rules whose interval conditions contain the current vector enter the match
   set.
3. If a currently feasible action lacks an advocate, covering creates a new
   matching rule for that action.
4. The match set is frozen for the remainder of the decision.

Covering never invents an infeasible action. A newly covered rule may be chosen
as the root experiment, but it is uncertified and therefore cannot act in
deployment or inside rollout continuation.

Freezing the match set is important: later mutations, deletion, or covering
cannot retroactively change which rules receive evidence from this rollout.

## 7. Root action selection

Exploration is scheduled by coarse regions rather than only by global training
time. The default region axes are:

```text
commitment band x pressure band x absolute-error band
```

The scheduler tracks visits and completed experiments separately for each
feasible action. Undercovered actions receive experimental opportunities.
After coverage is sufficient, the learner normally exploits the current
published rulebook, with an optional residual exploration rate.

In causal mode, no-op is withheld from root coverage while no certified
physical rival exists. Selecting no-op at that point would produce three
identical no-op arms and no causal evidence, potentially starving all physical
actions. Once a certified physical contender exists, no-op becomes a valid
root experiment and can accumulate veto evidence.

## 8. The three-arm experiment

Suppose the selected action is `speed/medium`. The frozen evaluation snapshot
selects the strongest certified alternative at the same anchor.

The experiment becomes:

```text
Arm A: selected action
Arm B: strongest certified rival
Arm C: root no-op
```

For example:

```text
A = speed/medium
B = path_stretch/oracle_short_medium_long
C = no_op/no_op
```

If there is no certified ordinary rival, Arm B falls back to no-op.

### Frozen experimental boundary

Before any arm runs, the trainer freezes:

- the simulator parent hash;
- the outcome cohort;
- the common horizon;
- baseline crossing predictions;
- the cumulative intervention baseline;
- the published continuation rulebook;
- the action-runtime configuration.

Each arm forks the same dynamic simulator content, including pending events and
RNG state.

Historical interventions remain part of the physical starting state but do not
count toward the temporary rollout's intervention penalty. Only the root
action and later actions caused within the rollout window are scored as
interventions.

### Shared continuation policy

Each arm:

1. applies its own root action;
2. continues under the same immutable certified rulebook;
3. runs to the same horizon;
4. receives an outcome score.

Training Arm C is:

```text
root no-op + the same frozen continuation policy
```

It is not permanent no-op. Sharing continuation is what makes the root action
the controlled difference among A, B, and C.

## 9. Outcome score

The generic semi-local outcome combines:

- bound-pair separation quality;
- downstream propagation quality;
- rollout-only intervention parsimony;
- throughput.

Because Phase 0 contains exactly two aircraft, the bound-pair, throughput, and
intervention terms do most of the work. Downstream propagation becomes a
central signal in Phase 1.

The intervention term excludes all actions already present at the root. This
prevents historical actions from saturating or diluting the cost of the
current experiment.

## 10. Four causal signals

Let the three arm scores be:

```text
y_A = selected outcome
y_B = contender outcome
y_C = no-op-root outcome
```

The trainer derives:

```text
delta_rival          = y_A - y_B
delta_selected_noop  = y_A - y_C
delta_contender_noop = y_B - y_C
delta_veto           = y_C - max(y_A, y_B)
```

These differences answer different questions:

- **`delta_rival`**: should the selected rule survive and reproduce relative
  to its strongest competitor?
- **`delta_selected_noop`**: is the selected action genuinely better than
  doing nothing and therefore eligible for deployment?
- **`delta_contender_noop`**: does the contender also deserve deployment
  consideration, even if it lost to Arm A?
- **`delta_veto`**: did doing nothing beat the best distinct physical
  proposal in this region?

This is why one ordinary reward or one two-arm comparison is insufficient.

## 11. Credit assignment

Only co-advocates in the frozen root match set receive evidence.

For causal credit:

- selected-action advocates receive:
  - `delta_rival` in their evolution ledger;
  - `delta_selected_noop` in their deployment ledger;
- contender advocates receive:
  - `-delta_rival` in their evolution ledger;
  - `delta_contender_noop` in their deployment ledger;
- matching no-op rules receive `delta_veto` as rival-grounded veto evidence.

Rules matching other anchors, rules that happen to act later during
continuation, and rules created after the root match set receive no root
credit.

The vanilla control instead learns the selected arm's outcome prediction and
prediction accuracy. It shares the rest of the population mechanics but does
not use the causal differences as its fitness signal.

## 12. Commit only Arm A

All three temporary rollout branches are discarded.

Only the initially selected root action—Arm A—is applied to the real training
simulator. Arm B and Arm C exist only to produce counterfactual evidence.

The trainer records a content-hashed epoch trace containing:

- parent and committed-state hashes;
- selected anchor and all three root actions;
- match-set and co-advocate rule IDs;
- exploration and contender-selection reasons;
- frozen rulebook and outcome-plan hashes;
- arm initial and final hashes;
- all three scores and four deltas;
- ledger recipients;
- GA and certification events.

## 13. Evolution

At scheduled intervals, the genetic algorithm operates inside an exact niche,
such as:

```text
leader_follower x speed/light
leader_follower x path_stretch/oracle_short_medium_long
```

Action identity never changes through mutation or crossover.

The evolutionary machinery can:

- select parents by rival-ledger lower confidence bound;
- widen, narrow, shift, add, or remove interval predicates;
- cross conditions between same-niche parents;
- delete weak or overrepresented rules population-wide;
- conservatively subsume narrower same-action rules.

Offspring may inherit discounted statistics, but inherited evidence is tracked
as a birth baseline. It cannot be misrepresented as independent post-birth
evidence for certification.

## 14. Slow certification and publication

At a certification tick, the mutable population is evaluated.

The intended causal gates are:

```text
Action rule:
    enough no-op-grounded samples
    no-op-grounded LCB > 0

No-op/veto rule:
    enough rival-grounded samples
    rival-grounded LCB > 0
```

The standard starting thresholds are 30 independent no-op-grounded samples for
an action and 60 independent rival-grounded samples for a veto, both with a
positive lower confidence bound.

Passing rules are copied into a new immutable `FrozenRulebookPolicy`. The
published rulebook contains no live references to mutable population rules.

Publication also creates an authenticated `EvaluationSnapshot` containing:

- the detached rulebook;
- publication epoch;
- certified source-rule IDs;
- frozen contender-ranking fields;
- exact action/veto certification evidence;
- feature, action, and configuration hashes.

Between publication ticks:

- the mutable population continues to learn;
- deployment behavior remains unchanged;
- rollout continuation remains unchanged;
- exploration-disabled training exploit reads only the published snapshot.

## 15. Training artifacts

Every `Phase0TrainingRun` retains:

- seed and refresh interval;
- causal or vanilla credit mode;
- trainer epoch and committed-experiment count;
- publication generations;
- one trace hash per epoch;
- full mutable population JSON;
- immutable rulebook;
- complete evaluation snapshot;
- publication-time certification evidence;
- joint rule-region hyperrectangles.

This separation lets an auditor answer two different questions:

1. What did the mutable population know at the end?
2. What evidence and rules were actually published for deployment?

## 16. Held-out evaluation

The selected primary causal run is evaluated on every held-out factorial
scenario.

### Training/deployment agreement

The runner compares:

```text
exploration-disabled trainer decision
versus
direct deployed-rulebook decision
```

They must agree exactly. Live population updates between publication ticks
cannot affect this comparison.

### Learned policy versus permanent no-op

Held-out evaluation uses a different control from training Arm C:

```text
learned arm:
    deployed root action + deployed continuation policy

permanent-no-op control:
    root no-op + no-op at every later decision
```

This asks whether the complete deployed controller is useful, rather than only
whether one root action helps under a shared continuation policy.

Each comparison includes authenticated provenance for:

- root and final simulator hashes;
- the complete frozen outcome plan;
- runtime manifest and hash;
- learned and permanent-no-op policy fingerprints;
- component outcome scores for both arms.

### Path-stretch ablation

If the rulebook publishes a path-stretch action, the runner evaluates the same
learner-visible action identity under:

- the normal semi-local-outcome oracle;
- the geometry-clearance-only selector.

Both runtimes must provide authenticated configuration manifests.

### Vanilla control

When enabled, the vanilla rulebook is evaluated on the exact same held-out
scenario order and against permanent-no-op controls with identical auditable
provenance.

## 17. Acceptance gates

The final `Phase0AcceptanceReport` checks:

1. **Exact vocabulary** — the rulebook configuration contains exactly the five
   supported action identities.
2. **Capacity-ratio predicates** — every published action rule contains a
   speed- or path-capacity ratio.
3. **No distance confound** — no published action rule directly conditions on
   distance to final or distance to the resource.
4. **Positive no-op-grounded certification** — every published action rule has
   matching publication-time evidence, enough samples, and a positive LCB.
5. **Exploit/deployment agreement** — exploration-disabled training behavior
   matches direct deployment.
6. **Held-out improvement** — mean learned-policy improvement over permanent
   no-op is positive.
7. **Paired vanilla control** — when vanilla is enabled, causal and vanilla
   evidence uses the exact same held-out controls.
8. **Path ablation** — a published path-stretch rule reports both oracle and
   geometry-only results with authenticated runtime provenance.
9. **Refresh robustness** — tested refresh intervals produce real publications
   and the same qualitative held-out action map.
10. **Seed robustness** — independently seeded runs publish comparable joint
    rule hyperrectangles, with a default minimum mean IoU of 0.5.

The runner does not force these checks to pass. It can complete normally and
return a failed report, which is the scientifically correct behavior.

## 18. What Phase 0 establishes

If the acceptance report passes on a sufficiently powered registered run,
Phase 0 supports the following limited claim:

> Under the controlled two-aircraft factorial distribution, the causal learner
> can discover compact capacity-relative speed/path rules, publish them through
> independent no-op-grounded certification, reproduce them across seeds and
> publication intervals, and outperform permanent no-op on held-out scenarios.

It does not establish arbitrary operational rule discovery, network-wide
traffic control, or broad transfer across airports and traffic distributions.

## 19. What is still missing from the scientific Phase 0

The implementation machinery is complete, but passing the software tests is
not the same as completing the scientific claim.

### Full-scale registered experimental run

A final Phase-0 study still needs:

- fixed generator ranges and replicate counts;
- thresholds committed before examining results;
- enough training epochs to reach the real 30/60-sample certification gates;
- more random seeds;
- archived artifacts and environment manifests;
- reporting of failed as well as successful rules.

### Stronger statistical acceptance

The current held-out gate requires mean improvement greater than zero. It does
not yet require:

- a paired confidence interval excluding zero;
- a minimum scientifically meaningful effect;
- seed-level uncertainty;
- bootstrap or permutation analysis;
- an explicit power calculation.

The path-stretch gate requires both ablations to be present, but it does not
require a registered difference or direction between them.

### Stronger causal-versus-vanilla gate

The vanilla check currently verifies correct pairing and reports comparative
metrics. It does not require causal performance to exceed vanilla performance,
and it does not formally require the predicted structural contrast:

```text
causal rules retain capacity-ratio predicates
vanilla rules retain the distance-to-final confound
```

That contrast is the intended Phase-0 headline and should become an explicit
acceptance criterion.

### Interventional predicate ablation

The tutorial proposes a stronger load-bearing-predicate audit:

1. take each surviving rule;
2. flip one predicate while holding the other predicates matched;
3. execute the rule's action;
4. verify that flipping a causal predicate erases or reverses its advantage.

The current acceptance gate checks rule syntax—capacity ratio present and
distance confound absent—but does not yet run this predicate-level
intervention.

### Distillation baseline

The vanilla accuracy-based XCSR control is implemented. The second proposed
baseline—a trained policy followed by decision-tree distillation—is not.

### External validity

Phase 0 intentionally does not establish transfer across:

- airports and runway geometries;
- aircraft mixes;
- larger traffic streams;
- weather regimes;
- broad operational distributions.

Those questions belong primarily to later phases.

## 20. Phase 1

Phase 1 is explicitly planned. It retains the same five action identities but
changes the scientific target.

### Structural change

Phase 0:

```text
leader -> follower
```

Phase 1:

```text
leader -> follower -> trailer
```

### Scientific question

Phase 1 asks:

> Can the learner recognize that an action which helps the bound pair may harm
> the downstream sequence, and learn when not to intervene?

Its intended targets are:

- downstream propagation credit;
- high-commitment "freeze the sequence" behavior;
- explicit no-op/veto rules;
- region-scheduled exploration of rare late-action cases;
- veto over-generality auditing.

For example, a heavy slowdown might improve the leader–follower interval while
compressing the follower–trailer interval. The multi-aircraft outcome should
make that downstream damage visible.

If no-op repeatedly beats distinct physical alternatives in a high-commitment
region, a no-op rule can certify as a scoped veto. At deployment, that veto
suppresses otherwise positive action scores at the matching anchor.

Phase 1 should report:

- how often each veto fires;
- the mean score of the actions it suppresses;
- whether a veto is overly broad;
- high-commitment sampling density with and without region scheduling;
- held-out downstream propagation benefit.

The core foundations already exist:

- multi-aircraft outcome scoring;
- three-arm rollouts;
- causal veto credit and certification;
- immutable rulebooks;
- region-scheduled exploration.

What does not yet exist is a dedicated `Phase1ExperimentRunner`, a Phase-1
scenario contract, and a Phase-1 acceptance report.

## 21. Phase 2 and later extensions

The tutorial's Phase 2 scales to broader distributions while retaining the
same action vocabulary:

- more scenarios and larger populations;
- broader persistence and replay campaigns;
- cross-region and cross-geometry generalization;
- stronger concept-correspondence evaluation.

New operational levers, multi-anchor simultaneous interventions, flow/resource
learning vectors, and oblique predicates remain separately scoped extensions.

## Summary

Phase 0 is the smallest experiment that can test whether causal rollout credit
discovers the intended speed-capacity versus path-capacity concept and produces
a genuinely deployable frozen rulebook.

Its full loop is:

```text
balanced generator
    -> mutable hypotheses
    -> root action selection
    -> common-root A/B/C experiment
    -> rival and no-op grounded evidence
    -> GA and specialization
    -> slow independent certification
    -> immutable rulebook
    -> held-out, seed, refresh, and ablation checks
    -> acceptance report
```

Phase 1 then adds the first downstream trailer and tests whether the same
machinery can discover the equally important heuristic: sometimes the correct
action is to freeze the sequence and do nothing.
