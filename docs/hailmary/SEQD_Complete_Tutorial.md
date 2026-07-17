# SEQD, From First Principles to Working System

*A complete, self-contained tutorial of the revised Sequencing Design framework. It assumes no memory of the earlier notes: everything needed to understand, implement, and evaluate the method is rebuilt here in order, with the revisions from the stress-test integrated as the default design rather than patched on. Read it linearly; later sections lean on earlier ones.*

---

## Part I — What we are building and why

### 1. The problem in plain language

Terminal-area arrival sequencing is the job of taking a stream of aircraft converging on an airport from several directions and turning it into a single, safely spaced, efficiently packed line onto the runway. Human approach controllers do this with a repertoire of *heuristics* — compact, reusable rules of thumb like "if the spacing error is small, fix it with speed; if it is large, stretch the path," or "once an aircraft is established near final, stop touching the sequence." These heuristics are the accumulated operational wisdom of the profession, but they exist mostly in controllers' heads and in training folklore. They are not written down as formal, testable objects, and nobody knows which parts of each heuristic are genuinely *causal* (the world really works that way) versus merely *habitual* (it correlates with success because of how traffic usually arrives).

SEQD (the Sequencing Design problem) is the problem of making an artificial agent **rediscover these heuristics from scratch**, purely by interacting with a traffic simulator — and, crucially, rediscover them in a form where every clause of every rule has *earned its place by experiment*. The deliverable is not a black-box policy that sequences traffic well, nor is it the mutable training population. The deliverable is a **certified rulebook**: a frozen subset of human-readable rules, each of the form

```
IF   <condition on the traffic situation>
THEN <apply this kind of intervention, roughly this hard, here>
```

where every predicate in every condition has survived an interventional test: keeping it demonstrably changes outcomes, removing it demonstrably would not have.

The one-sentence thesis, in its revised form:

> SEQD learns an XCS-style population of `condition ⇒ (anchor, lever, band)` rules for terminal-area sequencing using three-arm interventional rollouts. A rival-grounded ledger drives evolution, a no-op-grounded ledger certifies and scores deployed action rules, and a stricter rival-grounded gate certifies no-op rules as scoped vetoes. The shipped artifact is the certified rulebook `π_deploy`, not the mutable population.

Three distinctions organize the revised design. First, the strongest rival is an instrument for evolution, while no-op is the permanent grounding baseline for deployment. Second, the interventional estimate comes from three same-seed simulator forks rather than matched episodes or occasional re-grounding. Third, the population that learns is not the rulebook that acts: deployment and rollout continuation both use the same slowly refreshed, certified `π_deploy`. These distinctions are developed in Parts III and IV.

### 2. Why not the obvious alternatives

**Why not train a deep RL policy and distil rules from it?** Distillation fits a symbolic model (a decision tree, typically) to the input–output behavior of a trained network. Whatever it recovers is *correlational*: a predicate can appear in the distilled tree because it predicts the policy's behavior, even if it is causally inert — a confound riding on the true cause. The existing interpretable arrival-management literature (simulation-optimization plus tree distillation) has exactly this shape and exactly this limitation. SEQD drops the distillation route entirely and differentiates against it as a baseline.

**Why not attention-based explanation?** Attention maps over traffic situations are post-hoc and correlational for the same reason. A foil, not a competitor.

**Why a classifier system at all?** Because the target object — a population of `condition ⇒ action` rules with thresholded conditions, maintained and refined by experience — *is* a Learning Classifier System. XCS gives us, off the shelf: rule matching, rule discovery (covering plus a genetic algorithm), niching so that different situations maintain different rules, generalization pressure, and subsumption of redundant rules. We are not inventing a representation. We are replacing exactly one component — the credit signal that decides which rules live and reproduce — and keeping the rest.

**The closest philosophical relative** is CURLS-style causal rule learning: rules that describe subgroups by *treatment effect* rather than by correlation. SEQD is that idea moved from static, observational, tabular data with a fixed treatment into an online semi-Markov decision process where the agent generates its own interventions. The four-way intersection — classifier-system rules × interventional credit × temporal event graph × terminal-area sequencing — is, on current searching, open. One residual search in the causal-ML/online-RL literature ("interventional fitness," "heterogeneous treatment effect online RL") remains advisable before final commitment.

---

## Part II — Background primers

You need working knowledge of three things: the operational quantities of sequencing, the mechanics of XCS, and just enough causal inference to see why accuracy-based fitness fails. Each primer is self-contained.

### 3. Sequencing primitives

Every arriving aircraft has, at any moment, a **minimum-time map**: the earliest time it could feasibly reach each downstream point (merge fix, final approach fix, runway threshold) given its position, speed envelope, and route options. From minimum-time maps, three quantities that the whole framework leans on can be defined *even for aircraft on different routes or being vectored*:

**Spacing deviation** for an ordered pair (leader L, follower F) toward a shared resource: the difference between the *predicted* interval at that resource and the *required* interval (wake-turbulence and radar separation make the requirement pair-specific). Negative deviation means predicted compression — a future conflict or forced go-around. Positive deviation means a gap — wasted runway capacity.

**Slack** for a single aircraft: the difference between its assigned or desired time at a resource and its earliest feasible time. Slack is delay the aircraft can absorb "for free" in the time dimension; it comes in flavors depending on the lever that would realize it — *speed slack* (how much delay speed reduction alone can absorb given remaining track-miles) and *path slack* (how much a geometric stretch can absorb given available airspace).

**Pressure** on a resource: how many aircraft want it per unit time in an upcoming window, against its service rate. High pressure with low aggregate slack is the classic recipe for late compression.

Two operational facts motivate everything else. First, **actions propagate**: delaying one aircraft changes the spacing situation of everything behind it in the sequence; costs and benefits land on aircraft other than the one you touched, seconds to minutes later. Second, **controllability decays**: the closer an aircraft gets to final approach, the fewer options remain for changing its time or its place in the order, and the more expensive any change becomes. The profession's name for the accumulated cost-of-change is *commitment*. Point Merge — the published route structure with a merge point fed by equidistant sequencing legs — is the one systematized abstraction the field already has: it turns "path stretch" into a clean, bounded delay reservoir. It is evidence that the abstractions we want to discover are real enough to have been engineered around, and it is deliberately *not* hard-coded into our state.

### 4. XCS in one page

A Learning Classifier System maintains a **population [P]** of classifiers. Each classifier is:

```
classifier = {
  condition:   a conjunction of predicates over the input
               (in XCSR, real-valued inputs: one interval [l_i, u_i] per feature,
                with "don't care" = the full range),
  action:      one action from the action set,
  bookkeeping: prediction p, prediction error ε, fitness F,
               experience count, numerosity, average niche size
}
```

The execution cycle at each decision point: form the **match set [M]** of all classifiers whose conditions hold in the current input; if some action has no advocate, **covering** creates a new classifier matching the current input (with randomly widened intervals) advocating it; compute a fitness-weighted **prediction array** (expected payoff per action); select an action (explore/exploit); the classifiers in [M] advocating the selected action form the **action set [A]**; the environment returns reward; every classifier in [A] updates its prediction `p ← p + β(R − p)` and error `ε ← ε + β(|R − p| − ε)`.

Rule discovery: fitness in standard XCS is **accuracy-based** — a classifier is accurate if ε is below a threshold, and fitness is its accuracy *relative to the other classifiers in its niche*. A steady-state **genetic algorithm** runs inside action sets: it selects parents proportional to fitness, crosses over and mutates their condition intervals, and inserts offspring, deleting weak classifiers population-wide. **Subsumption** lets an accurate, more general classifier absorb an accurate, more specific one, which is the pressure toward maximally general rules. The niched GA is why one population can maintain *different* rules for different regions of the input space simultaneously — exactly the "repertoire of heuristics" structure we want.

SEQD keeps these population-learning mechanics, but splits XCS's prediction and fitness roles across interventional ledgers and separates the trained population from the certified controller (§§9.5–11).

### 5. Why accuracy is not enough, in ninety seconds

Pearl's ladder distinguishes *seeing* (associations, rung 1) from *doing* (interventions, rung 2). Accuracy-based fitness is a rung-1 criterion: a classifier is rewarded for **predicting** payoff well in the states it matches. But a predicate can predict payoff perfectly while being causally inert. The canonical sequencing example: suppose that, in naturalistic traffic, aircraft that need delay are almost always upstream of waypoint X (because that is where demand bunches). Then the predicate `upstream of X` predicts "delay action pays off here" just as accurately as the true cause, `enough absorbable track-miles remain`. Vanilla XCS has no way to prefer the second; both rules are accurate. A distilled tree has the same blindness.

The rung-2 fix is an **intervention**: apply the action in matched situations, *withhold or replace it* in otherwise-identical situations, and credit the rule with the outcome gap. Withholding an action in a system of cascading, delayed effects is a **blocking intervention**, and there is theory to lean on: in chain-reaction systems with overlapping delayed effects — which follower chains in a terminal area are — causal structure is identifiable from blocking interventions where purely observational criteria provably fail. That result does real work in this design (Part V, the "freeze" heuristic), not decorative work.

One more piece of vocabulary: the outcome gap we estimate, conditional on a rule's predicates holding, is a **conditional average treatment effect** (CATE). A rule in SEQD is, in causal-inference terms, a *description of a subgroup with a significant treatment effect* — that is the CURLS connection — except the agent chooses its own treatments online.

---

## Part III — The world, the features, and the interface

### 6. The state: a typed temporal event graph

The simulator state at any moment is exposed to the learner as a heterogeneous graph.

**Node types:** `aircraft`, `waypoint` (fixes, merge points, runway thresholds), `runway`, and `intervention` (each applied action becomes a node, so the graph remembers what has been done and rules can, in principle, condition on recent interventions).

**Edge types:** `proximity` (spatial-temporal closeness), `assigned-to-waypoint` (routing), `leader–follower` (a *derived*, directed edge: A's predicted occupancy of a shared resource constrains B — computed from predicted resource order plus the pair-specific separation requirement), `flow-membership` (a derived grouping: aircraft sharing waypoint histories, downstream fixes, and similar minimum-time maps), and `intervention-applied-to`.

Nodes and edges carry features (positions, speeds, envelopes, minimum-time maps, predicted resource times, separation requirements). The graph is **event-driven**: a decision epoch occurs when something discrete happens — an aircraft crosses a fix, a proximity or leader–follower edge forms or breaks, a runway slot frees, an intervention completes. Between epochs, the simulator just flies the aircraft. Formally this makes the learning problem a **semi-Markov decision process**: actions span wall-clock time, and epochs are irregularly spaced.

The derived edges matter philosophically: *flow* and *leader–follower* are not hand-labeled by scenario designers; they are computed by fixed, simple detectors over raw trajectories (recurring waypoint-chain motifs; predicted-order-plus-constraint). The learner then decides, via survival, whether predicates built on them are worth anything.

### 7. The feature bank, including the ratio library

Rules do not condition on the raw graph directly; they condition on a **feature bank**: a fixed library of scalar features computed from the graph, each attached to a node, an edge, or a (node, resource) pair. The bank has three tiers.

**Tier 1 — raw local features.** Per aircraft: speed, altitude, distance-to-final, remaining track-miles, time-to-final, heading relative to intercept. Per leader–follower edge: predicted spacing deviation at the shared resource, required separation. Per resource: demand count in the next N minutes, service rate.

**Tier 2 — derived single quantities.** Per aircraft: speed-absorption capacity (delay absorbable by speed alone, from the envelope and remaining distance), path-absorption capacity (delay absorbable by available stretch geometry), slack (earliest-feasible minus assigned), path freedom (a scalar summarizing remaining route options), intercept-established flag. Per resource: pressure (demand over capacity in a window). Per flow: aggregate slack, size.

**Tier 3 — derived comparisons and ratios.** This tier is the important revision. It includes, as directly matchable scalar features: `required_delay / speed_capacity`, `required_delay / path_capacity`, `slack / required_delay`, `time_to_final / time_needed_to_reorder`, pairwise spacing deviation normalized by requirement (the gap/compression scale), and `pressure × (1 − mean slack)` style compound stress terms.

Why Tier 3 exists: the *causally correct* condition of many heuristics is a comparison between two derived quantities — "the error is small **relative to what speed can absorb**" — which is a *diagonal* boundary in raw-feature space. XCSR conditions are axis-aligned boxes; they can only tile a diagonal with many small rules. Without Tier 3, the learner would discover the right causal structure but express it as a dozen fragmented boxes, and the concept-correspondence evaluation would fail for representational reasons, not causal ones. With Tier 3, a single interval on a single ratio feature expresses the heuristic, and the discovery claim becomes precise and honest: **the agent selects which derived quantities are causally load-bearing** (most of the bank will turn out inert or confounded; survival is the filter). Composing new ratios from scratch, via oblique linear predicates, is explicitly a phase-2 extension, not the core claim.

The Tier 2/3 library is also, deliberately, the operationalization of the controller-concept vocabulary — flow, slack, commitment, pressure, gap, compression risk, reservoir. This makes the eventual concept-correspondence measurement (Part VI) a direct readout instead of an interpretive exercise.

### 8. The action interface: (anchor, lever, band)

Phase 0 learns the physical macro-actions already exposed by Hailmary. It emits a bound action identity and hands that identity to the shared realization layer:

```
tuple = (anchor, lever, band)

anchor:  a node or edge in the graph, provisionally bound when a rule matches
         and finally bound only if that anchor–action candidate is selected
         for execution (see §9) —
         currently "the follower of this leader–follower edge"
lever:   one of { speed, path_stretch, no_op }
band:    one of { light, medium, heavy, oracle_short_medium_long, no_op }
```

The three speed bands are physical command reductions of 10, 15, and 20 kt. They are not promises to absorb fixed amounts of time: the realized delay depends on the remaining route and speed envelope. Path stretch is one learner-visible macro-action. Its realization layer evaluates the existing short, medium, and long geometries and logs the winning geometry and all candidate scores. The geometry-only selector remains an explicit ablation.

All simulators—real, forked rollout arms, resumed checkpoints, and deployment—use the same frozen action runtime. Thus an action identity has one physical meaning everywhere. Unsupported operational actions such as swap, hold, and meter-upstream are outside the current scientific claim rather than silently represented by approximate substitutes.

`no_op` is a first-class identity during learning, but it has different deployment semantics from an action. A certified no-op rule becomes a **scoped veto**: wherever its condition matches, action at that anchor is suppressed. This makes prohibitions ("don't touch the sequence near final") explicit, storable, reportable, and enforceable (§9.6).

The exact vocabulary is therefore `no_op/no_op`, `speed/light`, `speed/medium`, `speed/heavy`, and `path_stretch/oracle_short_medium_long`. It is serialized and content-hashed from the action catalog configuration. Every distinct identity remains a separate action for credit purposes.

### 9. Rules and role-centric matching

Here is the second structural revision. A controller heuristic like "the follower absorbs the spacing error" refers to a **role**, not to a particular aircraft. Classic XCS matches a condition against *the* global state vector; that cannot express roles. SEQD instead makes matching **relational**:

A rule's condition is a conjunction of interval predicates over the feature bank **evaluated on an ego-graph rooted at a candidate anchor**. At each epoch, the system enumerates candidate anchors by role type — every leader–follower edge, every flow node, every (aircraft, resource) pair currently live — computes the ego-graph features for each, and builds a **match set per candidate anchor**. A match means only that a rule is *eligible* at that anchor. If the same rule matches three aircraft pairs, it creates three distinct candidate applications of its one lever–band action, each with a different provisional anchor binding. It does **not** execute the action three times. The anchor slot is provisionally bound by matching; global arbitration later selects at most one anchor–action candidate for physical execution, and only that winner receives a final binding.

For example, a follower-anchored rule `R1 ⇒ speed` might match both edges `X→Y` and `Y→Z`. This creates `(R1, X→Y, speed-on-Y)` and `(R1, Y→Z, speed-on-Z)` as separate candidates. If the first candidate wins arbitration, speed is applied to Y only. The second is an unexecuted opportunity, not an automatic second speed command; at the next epoch it must match and win again in the recomputed state before it can be applied.

Concretely, a rule looks like:

```
RULE r_17   (role: leader–follower edge, anchored on follower)
  IF    spacing_deviation(L,F)@merge   ∈ [-300 s, -90 s]        # large compression
    AND required_delay / path_capacity ∈ [0.0, 0.8]             # stretch can absorb it
    AND time_to_final(F)               ∈ [8 min, ∞)             # not yet committed
  THEN  (anchor = F, lever = path_stretch, band = oracle_short_medium_long)
  EVOLUTION  Δ̄_rival = +0.30, σ²_rival = 0.15, n_rival = 71
  DEPLOYMENT Δ̄_noop  = +0.42, σ²_noop  = 0.11, n_noop  = 63
  TRAINING   numerosity = 4
```

Everything in XCS that was defined per global state is redefined per **role context**: covering triggers when a candidate anchor's match set lacks an advocate for some action; GA niches are (role type × action) within similar contexts; subsumption compares rules of the same role type. This is a modest extension — it is how relational RL and graph-pattern classifiers already work — but it must be explicit in the implementation because it changes the unit of matching, the niche definition, and what "the state" means everywhere below.

### 9.5. The deployed policy `π_deploy`

The artifact SEQD produces is not the population [P]. It is a **rulebook**: the subset of [P] that has passed certification, with frozen deployment fields. The rulebook performs no exploration, covering, GA, subsumption, statistics updates, rollouts, or confidence-bound calculations. It is the controller evaluated in Part VII and, critically, the continuation policy inside every rollout (§10.5).

Certification is an offline operation performed when a deployment snapshot is refreshed:

```
action rules (lever != no-op):
    certified iff n_noop >= n_min and LCB(Delta_mean_noop) > 0     [z = 1.96]
    ship with  w_r = Delta_mean_noop,r
               F_r = n_noop,r / sigma_squared_noop,r

prohibition rules (lever = no-op):
    certified iff n_rival >= n_min_veto and LCB(Delta_mean_rival) > 0
                  [z = 1.96; n_min_veto > n_min]
    ship with a veto flag; no w and no F
```

At each deployed epoch:

```
1. enumerate anchors and compute ego-graph features
2. build match sets from certified rules only; never cover
3. veto every anchor matched by a certified no-op rule
4. for each surviving anchor alpha and advocated action a:
       Q(alpha,a) = sum_r(w_r * F_r) / sum_r(F_r)
5. choose (alpha*,a*) = argmax Q with deterministic tie-breaking
6. if no candidate survives, or Q(alpha*,a*) <= 0, choose no-op
7. otherwise apply a* at alpha* and advance to the next event
```

Thus inference is two frozen numbers per action rule, one precision-weighted average, and one global argmax. Only one action may fire per epoch. The high event density makes deferral cheap; simultaneous multi-fire is reserved for Phase 2 (§20).

### 9.6. Why prohibitions are vetoes, not advocates

A no-op rule's advocated action is the grounding baseline, so its no-op-grounded advantage is identically zero: `Delta_mean_noop = 0`. It can never win a positive-`Q` argmax. Its evidence instead lives in the rival ledger: positive `Delta_mean_rival` means that, in this condition region, doing nothing beat the best proposed action. The executable meaning is therefore a **scoped veto** at the matching anchor, not a scored no-op advocate.

Because a veto is a strong operator, it uses the stricter certification bar above and receives an over-generality audit. For every certified veto, report its firing fraction and the mean `Q` of the actions it suppressed. A rule that fires broadly while suppressing high-`Q` actions is an over-general rule the GA failed to specialize, not a heuristic.

Encoding A (§15) is consequently executable as well as reportable. Comparing its veto region with the survival frontier of the existing speed and path-stretch rules checks whether active and passive prohibition evidence agree.

---

## Part IV — The learning algorithm

This part is the heart of the method: the full decision-epoch loop, then the fitness mathematics.

### 10. The decision-epoch loop, step by step

At each event-driven epoch *t*:

**Step 1 — Enumerate candidate anchors.** Walk the graph; collect all live role instances (leader–follower edges, flows, aircraft–resource pairs). Compute each anchor's ego-graph feature vector from the bank. These are locations where a rule *could* apply, not interventions that have already been chosen.

**Step 2 — Build match sets and candidate applications.** For each candidate anchor α, form [M]ₐ = all rules of that role type whose interval conditions hold on α's features. Each match produces a candidate `(rule r, anchor α, action advocated by r)`; the same rule may therefore produce several candidates at different anchors. Matching neither selects nor executes any of them. If some lever–band action has no advocate in a context where the covering policy says it should (see the exploration note below), create a covering rule: intervals centered on the current feature values, widened by a random spread, advocating the missing action, initialized with zero effect statistics and low experience.

**Step 3 — Select at most one intervention for this epoch.** The root behavior policy is the current frozen `π_dep,t = π_deploy(certify(P_snapshot))` (§9.5), augmented by exploration and covering. On exploit steps it computes the deployed precision-pooled `Q` scores and applies the same veto and global arbitration rules as deployment; on explore steps the region scheduler may choose a different advocated action, including one from an uncertified covering rule. Every nonwinning candidate remains unexecuted. The three-arm rollout evaluates the selected pair and supplies learning samples; it does not retroactively replace the selected action when another arm scores better. Simultaneous multi-anchor interventions are a Phase 2 extension because they entangle credit.

**Exploration is region-scheduled, not merely time-annealed.** Maintain visitation counts over a coarse discretization of the concept axes that matter for the target heuristics — commitment band × pressure band × error-magnitude band is enough to start. Boost exploration probability in undersampled cells, and do not anneal exploration in a cell before it has minimum coverage. The reason is concrete: the agent can only learn "do not intervene near final" by *executing late speed or path-stretch actions and observing their cost*, and a naive time-annealed schedule typically stops exploring before the rare high-commitment region has been sampled. Dangerous exploration is fine — this is a simulator — but it must actually happen.

This root off-policy-ness is intended. Exploration and covering change which states are sampled and therefore how a rule's effect is averaged over its condition region. For heuristic discovery that breadth is desirable: a rule should hold across its full region, not only where the current controller visits. This differs from off-policy continuation in Step 5, which would change the quantity being estimated and is therefore a bug.

**Step 4 — Choose the counterfactual contender.** Let *a* be the selected action at α*. The contender *c* is the strongest rival action `c != a` among **certified action advocates** in α*'s match set, ranked by evolution-ledger `LCB(Delta_mean_rival)`; if none exists, use `no-op`. This comparison is local to α*. The contender is an experimental instrument, not a policy: it supplies Arm B's first move and never appears in the deployed rulebook merely by being selected here.

**Step 5 — Use the frozen deployed policy and run three arms (the interventional estimate).** The current **certified rulebook**, `π_dep,t`, is the read-only continuation policy in every arm. It was produced from a population snapshot at the most recent certification tick. Uncertified rules — new covering rules, partially tested GA offspring, and anything below its certification threshold — cannot act inside rollouts because they cannot act at deployment. Rolling out with the raw population would estimate the advantage of a policy that will never run.

Refresh `π_dep,t` only on a slow timescale: every *N* real epochs, with `N ≈ 500` as the starting value. Within an evaluation phase the baseline is constant, so effect samples remain comparable across rules and time. The resulting cycle is generalized policy iteration: evaluate against a frozen target, improve the population, then re-certify and re-freeze.

Fork the current simulator state three ways on a shared exogenous seed and pending event stream:

```
Arm A: initial action = a       (selected action)
Arm B: initial action = c       (strongest rival)
Arm C: initial action = no-op   (deployment ground)
all three then continue under the same pi_dep,t to horizon H
```

The no-op arm is not an occasional diagnostic. Its effect statistic is the deployment score, so all three arms run every epoch. This costs 1.5 times as much simulation as two arms and supplies both main ledgers from every epoch.

"Same policy" means the same certified rulebook, not the same later action sequence. The initial interventions put the arms into different states, so different certified rules may match and later actions may differ. Those downstream differences descend from the initial choice and belong in the outcome. For example, speed in Arm A may leave a clean sequence, while stretch in Arm B creates geometry that leads `π_dep,t` to slow a trailing aircraft. That extra slowdown is legitimately charged to Arm B.

No learning or population change occurs in any temporary arm: no statistics updates, GA, covering, subsumption, exploration, or LCB calculation. Continuation is deterministic deployed arbitration with deterministic fallbacks. Thus the only systematic differences are the root actions and their consequences. Shared-seed pairing removes the dominant scenario-level variance. Cross-episode matching is only a robustness check or fallback when simulator forking is impossible.

**Step 6 — Compute outcomes and update the two ledgers plus veto evidence.** Compute the same semi-local outcome *y* (§12) in each arm, including continuation actions, then distribute:

```
Delta_rival = y_A - y_B                         -> EVOLUTION ledger
    +Delta_rival to every rule in [M]_alpha* advocating a
    -Delta_rival to every rule in [M]_alpha* advocating c

Delta_noop(a) = y_A - y_C                       -> DEPLOYMENT ledger
    +Delta_noop(a) to every rule in [M]_alpha* advocating a

Delta_noop(c) = y_B - y_C                       -> DEPLOYMENT ledger
    +Delta_noop(c) to every rule in [M]_alpha* advocating c

Delta_veto = y_C - max(y_A, y_B)                -> VETO evidence
    +Delta_veto to every no-op rule in [M]_alpha*
```

Veto evidence is stored in a no-op rule's rival/evolution ledger because it measures no-op against the best proposed action. The rival ledger drives evolution; the no-op ledger drives deployment. Rules that acted only later inside an arm still receive nothing. A rule matching another, nonselected anchor receives no extra sample. Update recipients in the real population with Welford's online mean and variance update.

**Step 7 — Commit only the first action and continue the world.** Discard all three temporary branches, return to the untouched real state at epoch *t*, and apply only *a*. None of Arm A's later simulated actions becomes real. Advance to the next event and solve a fresh three-arm problem. In short: **long closed-loop rollout for evaluation, one-step commitment in reality**.

**Step 8 — Rule discovery and slow certification.** Roughly every 50 epochs, run the steady-state GA inside each `(role type × action)` niche. Select parents and survivors using rival-ledger `LCB(Delta_mean_rival)`, mutate and cross condition intervals, and insert offspring with discounted statistics. Offspring are uncertified and cannot act inside deployment or rollout continuation until they earn enough no-op-grounded evidence. Subsumption likewise uses the evolution ledger. Roughly every 500 real epochs, run certification over the population and atomically replace `π_dep,t`; only then does the continuation target move.

### 11. The fitness mathematics, precisely

Each action rule maintains two independent running ledgers:

```
evolution:   (Delta_mean_rival, sigma_squared_rival, n_rival)
deployment:  (Delta_mean_noop,  sigma_squared_noop,  n_noop)
```

No-op rules use rival/evolution statistics as veto evidence; their no-op-grounded effect is zero by construction.

As in XCS, prediction and fitness answer different questions:

```
prediction: w_r = Delta_mean_noop,r
fitness:    F_r = n_noop,r / sigma_squared_noop,r
```

Prediction says what effect to expect relative to doing nothing. Fitness is precision — the inverse squared standard error — and says how strongly to trust that estimate. Deployment pools overlapping certified advocates as

```
Q(alpha,a) = sum_r(w_r * F_r) / sum_r(F_r).
```

This is inverse-variance fixed-effect pooling: the minimum-variance combined estimate at that `(anchor, action)`. Fitness must not be numerosity or a count of matching rules. GA offspring and subsumption remnants create correlated near-clones; vote-counting would let a family shout louder merely because it bred well. A rule whose condition straddles effect and no-effect regions receives scattered samples, inflating `sigma_squared_noop` and collapsing `F`. Precision weighting therefore penalizes over-generality by the same logic as XCS's accuracy-based fitness while remaining interventional.

The lower confidence bound remains

```
LCB(r) = Delta_mean_r - z * sigma_r / sqrt(n_r).
```

It is used at the offline certification gate (`z = 1.96`) and for GA selection and survival on the rival ledger (`z ≈ 1.0–1.96`). It is never computed at inference. Young, noisy rules therefore must prove themselves before breeding or shipping, while deployed arbitration uses the frozen mean and precision only.

Conditional on a rule's predicates, `Delta_mean_rival` estimates the policy-mediated advantage of its root action over the contender mix, while `Delta_mean_noop` estimates its policy-mediated advantage over no-op. All three arms continue under the same frozen `π_dep,t`; the estimate includes downstream corrections caused by the root choice rather than pretending the first maneuver acts in isolation.

There is **no value function anywhere in SEQD, deliberately**. Triple-arm same-seed differencing cancels the common state-value term rather than estimating it. `w_r` is a local, condition-averaged advantage; `F_r` is estimator precision; neither is a critic or `V(s)`. State enters `Q` only through matching, which selects the constant-valued boxes to pool.

Exploration at the root changes the sampling distribution, so uniform choice among advocated actions on explore steps helps keep comparisons interpretable. Exploration is never used inside rollout continuation. Rival-grounded effects will shrink as the population improves and contenders become stronger; that is expected and benign. The slow snapshot interval *N* bounds target-policy drift, removing the need for periodic no-op re-grounding because Arm C supplies that ground every epoch.

**Sample sharing.** All co-advocates in the selected match set share the applicable ledger samples. This lets general rules accumulate evidence quickly. If the generator visits the whole condition region, an over-general rule mixes effect and no-effect samples: its mean falls, variance rises, deployment precision and GA LCB collapse, and specialization carves out the region where the effect holds.

### 12. The outcome function: windowed and semi-local

The naive outcome — episode-level throughput — makes the method fail quietly: one 60-second speed adjustment changes episode throughput by far less than the episode-to-episode variance of procedurally generated traffic, so per-rule ATEs would need astronomical sample counts. Pairing (Step 5) removes cross-scenario variance; the windowed outcome removes irrelevant-traffic variance. Define, for an intervention anchored on pair (L, F) at epoch t:

```
y = w₁ · pair term:        spacing outcome of (L,F) at the shared resource —
                           graded margin score, violation flag, go-around flag
  + w₂ · propagation term: same graded scores summed over the k trailing
                           aircraft in the affected sequence, measured up to
                           horizon H (past the k-th trailer's resource crossing)
  + w₃ · parsimony term:   −(count and magnitude of interventions issued in
                           the temporary rollout window, including the initial
                           action and all later actions selected by frozen pi_dep,t)
  + w₄ · global residual:  small-weight throughput term over the window,
                           insurance against effects escaping the k-window
```

Defaults to start: k = 3, H = the k-th trailer's threshold crossing plus one slot, w₁ : w₂ : w₃ : w₄ ≈ 1 : 1 : 0.3 : 0.1.

**The separation penalty must be graded, not catastrophic.** Score margin erosion piecewise (full margin → partial erosion → violation → go-around) rather than as a single huge negative constant. With a cliff penalty, Δ samples in risky regions are all-or-nothing and their variance swamps the estimator; with a graded score, "this late intervention ate 40% of the trailing pair's margin" is a smooth, learnable signal that points in the same direction as the rare disaster. (Fuel, workload-as-entropy, and monitorability terms remain deliberately dropped from the outcome — they dilute the credit signal and are second-paper material. Parsimony stays because it is the pressure toward sparse, controller-like intervention patterns.)

---

## Part V — The scenario generator as part of the method

### 13. Why generation is not the backdrop

The interventional test can only separate two predicates if they **vary independently somewhere in the data**. If every situation that needs delay is also upstream of waypoint X, no method — interventional or not — can attribute the effect between the two predicates; the do-operator distinguishes causes from confounds only where the confound is broken. Naturalistic traffic is full of exactly such entanglements, because the entanglements are *why controllers formed the habits in the first place*. Therefore the generator must **actively decorrelate the features that controllers conflate**, and this is a first-class design requirement with a named failure mode, not a data-engineering afterthought.

The revision from the stress-test sharpens this from a generic principle into a protocol:

**Pre-register per-heuristic confound pairs.** For each heuristic family you intend to rediscover, write down before training which predicate pairs the generator must break, then verify it broke them. For the two worked heuristics of Part VI: (error magnitude ⟂ distance-to-final) and (error magnitude ⟂ pressure) for the speed-versus-path rule; (commitment ⟂ pressure) and (commitment ⟂ error magnitude) for the freeze rule.

**Generate factorially over those axes.** The generator's controllable parameters (arrival rates per entry fix, entry-time jitter, where and when spacing errors are injected, geometry variants) are sampled so the registered pairs get all four quadrants: late-detected *large* errors, early-detected *small* errors, high-commitment/low-pressure streams, low-commitment/high-pressure streams. These quadrants are rare in naturalistic traffic; that rarity is exactly why habitual and causal predicates are observationally indistinguishable to humans.

**Audit.** After generation, compute the empirical correlation matrix over all Tier 2/3 features across the sampled situation distribution; flag any registered pair with |ρ| above ~0.3 as unresolvable and fix the generator before training. Publish the audit with the results — it is the reader's warrant that surviving predicates were genuinely tested.

The situation *distribution* also carries a second duty from the original framing, unchanged: the learning objective is expected effect over the distribution, and that is what makes a surviving regularity a *heuristic* (a generalizable rule) rather than an overfitted reaction to one scenario.

---

## Part VI — Two heuristics, traced end to end

These traces are the proof-of-concept in prose: one positive action rule and one prohibition, the case that requires separate rival/no-op evidence and executable veto semantics. Numbers are illustrative but of realistic magnitude.

### 14. Trace 1 — "Small error → speed, large error → path"

**Target.** The delay-absorption heuristic decomposes into exact-identity rules over a leader–follower pair, anchored on the follower:

```
For each b ∈ {light, medium, heavy}:
R_speed,b: spacing_deviation ∈ [−90 s, 0)  ∧  required_delay/speed_capacity ≤ 1
           ⇒ (F, speed, b)
R_path:   spacing_deviation ∈ [−300 s, −90 s)  ∧  required_delay/path_capacity ≤ 1
          ⇒ (F, path_stretch, oracle_short_medium_long)
```

Each condition is a single interval region on Tier 3 ratios plus one on the raw deviation — one macro-rule per exact action identity because learned rules never contain union bands. Without Tier 3, the same causal content shatters into a tiling of boxes over (deviation × track-miles × speed-margin) space.

**A learning event, concretely.** Epoch t: pair (L, F), predicted deviation −60 s at the merge, F has speed capacity ≈ 90 s. The match set contains `R_speed`, a stretch rival, and a no-op rule. Exploration picks speed; the strongest certified rival is stretch. Under the same frozen `π_dep,t`, Arm A (speed) yields `y_A ≈ +0.8`, Arm B (stretch) yields `y_B ≈ +0.4`, and Arm C (no-op) leaves compression and yields `y_C ≈ −0.4`. Thus `Delta_rival = +0.4` drives evolution, while `Delta_noop(speed) = +1.2` supplies `R_speed`'s deployment ledger and eventual shipped score. The stretch advocates also receive the free no-op-grounded sample `+0.8`. The downstream correction's advocate receives no root update. All arms are discarded and only the initial speed action is applied in reality. Over many events, rival-ledger LCBs guide specialization while no-op-ledger means and precision determine which action rules certify and how they arbitrate at deployment.

**The confound test.** The natural confounder of "error is small" is "close to final" (small errors are usually *detected* late under naturalistic traffic, so the two co-occur). A confounded rule `distance_to_final < θ ⇒ speed` predicts success well observationally. The generator's registered decorrelation produces late-detected *large* errors; the confounded rule is eligible there and, when its speed candidate is selected, speed fails to absorb the error, the pair term of y_A goes negative, and the rule inherits negative Δ samples it cannot escape. It dies not because it predicted badly — it predicted fine on the naturalistic majority — but because its advocated action performed badly where its predicate and the true cause came apart. That sentence is the entire epistemic difference from accuracy-based fitness and from distillation, exhibited on one rule.

**Expected artifact.** A surviving exact speed-band rule family and path-stretch macro rule whose load-bearing predicates are the slack-ratio features — i.e., the *slack* concept, emerged and certified. The vanilla-XCS baseline run on identical experience is predicted to retain the distance-to-final confound; that contrast is the Phase 0 headline figure.

### 15. Trace 2 — "Freeze the sequence near final"

**Target.** A prohibition: when commitment is high, avoid unnecessary speed or path interventions. Two encodings exist, and the framework should produce **both** and show they coincide:

```
Encoding A (explicit rule):
  time_to_final < θ  ∧  intercept_established  ∧  path_freedom low
  ⇒ (—, no_op, no_op)
Encoding B (population boundary):
  action rules fail to survive in the high-commitment region;
  the heuristic is the survival frontier of the physical-action population.
```

**Why deployment needs veto semantics.** A no-op rule has zero advantage over the no-op grounding baseline, so it cannot act as a scored advocate. Its evidence instead asks whether no-op beat the best proposed action. When that rival-grounded estimate passes the stricter veto gate, the rule ships as a scoped veto. The framework can therefore represent the heuristic both actively (Encoding A suppresses action) and passively (Encoding B is the missing physical-action survival region).

**Why the effect is detectable.** A late slowdown or stretch can move the bound aircraft while compressing a trailing edge, so the cost lands on the third and fourth aircraft in trail seconds to minutes later. The *windowed multi-aircraft outcome* (k ≥ 3) sees that damage, the *shared-seed three-arm rollout* compares the physical action with its strongest rival and a root no-op under the same frozen continuation policy, and *graded margin scoring* turns partial margin erosion into a smooth signal aligned with the rare catastrophe.

**A learning event.** High-commitment epoch: F is established on intercept, 4 min to threshold, and a heavy slowdown is still technically feasible. Region-scheduled exploration selects `speed/heavy`. With no certified non-no-op rival, Arm B is no-op and therefore coincides with Arm C. Under frozen `π_dep,t`, the slowdown produces `y_A ≈ −0.9`; no-op produces `y_B = y_C ≈ +0.1`. Heavy-speed advocates receive `Delta_rival = −1.0` in evolution and `Delta_noop = −1.0` in deployment. By the fixed `Delta_veto = y_C - max(y_A, y_B)` definition, matching no-op rules receive `0.0` here because Arm B already is the no-op baseline. Veto evidence is produced when a distinct physical rival is present, for example when no-op is selected and speed is Arm B. Repeated evidence specializes action rules away from this region; repeated distinct-rival evidence can certify the explicit no-op rule as a veto.

**The confound test.** High commitment co-occurs naturally with high pressure. A correlational learner therefore tends to learn `pressure high ⇒ no intervention`, which is wrong when a low-commitment aircraft still has useful speed or path capacity. Registered high-commitment/low-pressure and low-commitment/high-pressure cases let the action deltas remove pressure from the prohibition while retaining the commitment signature.

---

## Part VII — Evaluation, staging, and diagnostics

### 16. What counts as success

The north star is "the learned concepts are real and the shipped controller is the evaluated artifact," so evaluation targets both the population's discoveries and the certified rulebook.

**Deployment consistency (first-class).** Run `π_deploy(certify(P_final))` on held-out scenarios with no exploration, covering, LCB calculations, or rollouts, and compare it with the training-time behavior policy. Material divergence indicates that Step 5's frozen-snapshot discipline was violated. This experiment certifies that the deliverable is the controller whose continuation behavior defined the learned effects.

**Compilation check (optional).** Compile the certified population into a conflict-free decision list ordered by specificity, with fixed role priority across anchors and first match firing. Report the performance delta. A statement such as "compiling the population into a flat rule list costs X% of its advantage" is useful in its own right, and the list may be more defensible as a human-readable artifact than precision-weighted voting.

**Causal load-bearing (primary).** For each surviving rule and each of its predicates: use the *generator* to manufacture matched situations where that predicate is flipped while the rest of the condition holds, run the rule's action, and measure the effect. A load-bearing predicate's flip should erase or reverse the advantage; an inert predicate's flip should not change it — and inert predicates should already have been stripped by subsumption, so finding one is a bug report. This is predicate ablation done interventionally, closing the loop on the method's own claim.

**Generalization.** Train on one region of generator-parameter space; test surviving rules' effects on held-out regions (different demand levels, different geometry variants). A heuristic, by definition, transfers.

**Concept correspondence.** Because Tier 2/3 features are the operationalized concept vocabulary, this is a direct readout: report which concepts' features appear as load-bearing predicates in surviving rules, against the target table (flow, leader–follower, pressure, slack, commitment, intervention region, reservoir, gap, compression risk). Include the frontier readouts (Encoding B objects) alongside explicit rules.

**Baseline beat.** Two concrete baselines on *identical experience streams*: (a) vanilla accuracy-based XCSR — isolates the contribution of the credit signal alone, everything else held fixed; (b) a trained policy (GNN or tabular) plus decision-tree distillation — the existing interpretable-AMAN recipe. The predicted, falsifiable contrasts: both baselines retain the registered confounds (distance-to-final in H1, pressure in H2); the interventional population does not; and interventional rules transfer better across generator regions.

### 17. The staged plan

**Phase 0 — one pair, the five Hailmary action identities (weeks, not months).** A single leader–follower pair on a simple merge geometry; actions {`no_op/no_op`, three physical speed bands, one path-stretch macro}; the H1 decorrelation registered and audited. Deliver both the learned population and the certified rulebook, and pass the deployment-consistency check. Success: H1 is recovered in compact rules whose load-bearing predicates are capacity ratios, while vanilla XCS on the same runs keeps the confound. Every later phase inherits the three-arm rollout, two ledgers, frozen continuation, windowed outcome, and audit.

**Phase 1 — three-aircraft chain, same action vocabulary.** Exercise the executable veto mechanism and its over-generality audit without introducing a new realization contract. Validate propagation credit and region-scheduled exploration by measuring high-commitment sampling density with and without it.

**Phase 2 — full distribution, same action vocabulary.** Scale scenarios, population size, persistence, and generalization evaluation while retaining the five audited identities. New levers, multi-anchor simultaneous interventions, flow/resource learning vectors, and oblique predicates remain separately scoped extensions.

### 18. Failure modes and their designed answers

| Failure mode | Symptom | Designed answer |
|---|---|---|
| Effect drowned in noise | LCBs never separate from zero | Shared-seed three-arm rollouts (§10.5) + windowed outcome (§12); check w₄ is small |
| Concept fragmentation | Many small same-action rules tiling a diagonal | Tier 3 ratio features (§7); phase-2 oblique predicates |
| Confound survives | Registered pair predicate persists in rules | Generator audit failed — fix decorrelation, retrain (§13) |
| Prohibitions unlearnable | No certified vetoes where freezing is right | Verify `Delta_veto` credit and the stricter veto gate (§9.6, §10.6) |
| Exploration dies early | High-commitment cells unsampled; no late-action data | Region-scheduled exploration with per-cell coverage floors (§10.3) |
| Lucky junk reproduces | Volatile population, young rules breeding | Select on LCB, not mean; raise z; raise GA experience threshold (§11) |
| Over-general credit harvesting | Broad rule with bimodal Δ samples | Generator coverage of full condition region + specialization pressure (§11) |
| Cliff-penalty variance | Δ variance explodes near separation limits | Graded margin scoring, not catastrophic constants (§12) |
| Band data starvation | Per-(lever,band) n too small | Keep the fixed three speed bands and one path-stretch macro; do not add bands (§8, §17) |
| Fitness drift | Rival-grounded effects shrink as the population improves | Expected and benign; bound target-policy drift with snapshot interval N (§10.5) |
| Train/deploy mismatch | Rulebook underperforms the training behavior policy | Rollout continuation = frozen `π_deploy(certify(P_t))` (§10.5) |
| Junk acting inside rollouts | High Δ variance; young rules win continuation arbitration | Apply certification to the rollout snapshot, not only export (§10.5) |
| Veto over-generality | A certified no-op rule fires broadly and suppresses high-Q actions | Audit firing rate × mean suppressed Q; use stricter `n_min_veto` (§9.6) |

### 19. Starting parameter sheet

| Parameter | Starting value |
|---|---:|
| Population size | 1,000 in Phase 0; increase by phase |
| Online statistics | Exact Welford updates; no learning-rate β |
| GA-selection LCB z | 1.0 exploratory; up to 1.96 for reproduction eligibility |
| GA experience threshold | 20 rival-grounded samples |
| Action-rule certification | `n_min = 30` no-op-grounded samples and LCB > 0 |
| Veto certification | `n_min_veto = 60` rival-grounded samples and LCB > 0 |
| Certification z | 1.96 |
| Deployment snapshot refresh N | 500 real epochs |
| Trailers k | 3 |
| Horizon H | k-th trailer's threshold crossing + one slot |
| Outcome weights | 1 : 1 : 0.3 : 0.1 |
| Action identities | no-op, three speed bands, one path-stretch macro |
| Decorrelation audit | `abs(rho) <= 0.3` |
| Exploration coverage floor | 200 visits per registered concept-axis cell |
| No-op re-grounding fraction | obsolete; Arm C runs every epoch |

### 20. What was deliberately dropped, and the open questions that remain

Dropped as decisions, unchanged from the consolidated note: autonomous procedure design (second paper); fuel/workload/monitorability reward terms (dilute the signal); concepts-as-embedding-clusters (replaced by the interventional-survival definition — one definition, not three); distillation as a method (kept only as the baseline it is meant to beat); neuro-symbolic-from-scratch (one mechanism, not a menu).

One action fires per epoch in training and deployment. Dense events make deferral inexpensive (the complete workflow below shows about 90 seconds). Simultaneous multi-fire remains a Phase 2 extension; if implemented, its counterfactual must become leave-one-out ablation from the fired set, `Arm B = S_t \ {r's action}`, rather than action replacement, because individually good rules can jointly overcorrect. Contender drift is bounded by snapshot interval *N*, which is now the tunable.

Genuinely open after the revisions, in rough order of risk: the constants of the windowed outcome (`k`, `H`, weights) and their sensitivity; whether sample sharing and generator coverage suffice against over-general harvesting or require per-rule heterogeneity tests; whether the veto's stricter certification bar starves prohibition rules of samples, given that vetoed anchors produce no deployment rollouts even though training exploration can override vetoes; the best value of *N*; and the remaining causal-ML/online-RL literature sweep.

---

## Appendix A — Glossary

**Anchor** — a graph node or edge where a rule may be eligible; physical action occurs only if its candidate wins arbitration.

**Band** — a coarse magnitude slot of an intervention tuple.

**Blocking intervention** — withholding or replacing an action in a cascade system; the theoretical basis for identifiability here.

**Candidate application** — a rule match paired with one anchor and advocated action; eligible, not yet executed.

**CATE** — conditional average treatment effect; here, a policy-mediated root-action advantage conditional on a rule's predicates.

**Certification** — the offline snapshot-time gate `n >= n_min and LCB > 0`; the only deployment-related place a confidence bound is computed.

**Commitment** — accumulated cost of changing an aircraft's role or order; a target concept.

**Contender** — the strongest certified rival action at the selected anchor by rival-ledger LCB; an experimental instrument, never a deployed policy.

**Covering** — the training-only XCS mechanism that creates a rule when a context lacks an advocate.

**Deployment ledger** — no-op-grounded `(Delta_mean, sigma_squared, n)`; supplies a shipped action rule's prediction `w` and precision `F`.

**Ego-graph** — the local subgraph rooted at a candidate anchor over which rule predicates are evaluated.

**Encoding B / frontier** — a prohibition read passively from the survival boundary of an action-rule population.

**Epoch** — an event-driven decision point of the semi-MDP.

**Evolution ledger** — rival-grounded `(Delta_mean, sigma_squared, n)`; drives GA selection, survival, and veto evidence.

**Frozen policy** — the certified `π_dep,t`, refreshed every *N* real epochs and used read-only in all rollout continuations.

**LCB** — lower confidence bound used for certification and GA selection; never an inference-time score.

**Lever** — in the current implementation, speed, path stretch, or no-op.

**Match set** — rules whose conditions hold at one candidate anchor; membership means eligibility, not execution.

**Minimum-time map** — per-aircraft earliest-feasible times at downstream points; basis of spacing deviation, slack, and pressure.

**Niche** — the `(role type × action)` context in which the GA competes rules.

**Precision fitness** — `F = n / sigma_squared`; an estimator weight, not a value or numerosity vote.

**Receding-horizon execution** — use a long simulated future to evaluate one move, commit only that first action, then recompute.

**Region-scheduled exploration** — per-cell coverage floors over concept axes, preventing premature annealing.

**Role-centric matching** — evaluating conditions per candidate anchor; one rule can create several candidates but at most one action fires per epoch.

**Rulebook** — the certified subset of [P], containing conditions plus frozen `w` and `F` for action rules or a veto flag for prohibitions; the deliverable, deployed policy, and rollout continuation policy.

**SEQP** — the lower-level realization layer that turns tuples into legal geometry.

**Slack** — absorbable delay, including speed-slack and path-slack variants.

**Subsumption** — absorption of a specific rule by a confidently at-least-as-good, more general rule.

**Three-arm rollout** — same-seed temporary forks for the selected action, strongest rival, and no-op, all continued by the same frozen rulebook.

**Tier 3 features** — derived ratios and comparisons forming the operationalized concept vocabulary.

**Veto** — a certified no-op rule that suppresses action at each anchor where it matches.

**Windowed outcome** — semi-local outcome combining pair, k-trailer propagation, parsimony, and a small global residual.

---
## Appendix B — Complete end-to-end workflow

This section works deployment and training end to end on the same event. It makes explicit which artifact acts, when each ledger moves, and when the frozen policy may change.

**The scene.** Four arrivals converging on merge point M for runway 25, in predicted order: AAL12 (a heavy), then UAL88 (a 737), then DAL34 (an A320), then JBU77 well behind. The graph holds three leader–follower edges — (AAL12→UAL88), (UAL88→DAL34), (DAL34→JBU77) — one flow node grouping all four, and the aircraft–runway pairs.

**The trigger.** At sim time 12:02:10, UAL88 crosses entry fix KODIA. That's a discrete event, so an epoch fires. Its minimum-time map is recomputed, and the prediction now shows trouble ahead.

---

**Step 1 — Enumerate anchors, compute features.** The system walks the graph and computes each anchor's ego-graph features. The two that matter this epoch:

*Anchor 1, edge (AAL12→UAL88), anchored on follower UAL88.* Required separation at M for a heavy→medium pair: 120 s. Predicted interval: 55 s. So spacing deviation = 55 − 120 = **−65 s** — compression; if nothing is done, UAL88 arrives 65 s too close behind the heavy. Required delay: 65 s. UAL88's speed-absorption capacity (from its envelope and remaining track-miles): 80 s. Path-absorption capacity: 240 s. So the Tier-3 ratios: `required_delay/speed_capacity` = 65/80 = **0.81** (speed alone can just barely absorb it), `required_delay/path_capacity` = **0.27**. Time-to-final: 14 min, intercept not established, path freedom high — low commitment.

*Anchor 2, edge (UAL88→DAL34).* Predicted deviation **+45 s** — a modest gap, wasted capacity. Some rules see an opportunity here.

Before this epoch, the latest slow-timescale certification pass produced this relevant rulebook:

```
r_A: deviation [-90 s, 0), speed ratio [0, 1.0]
     -> speed/light; ships with w = 0.62, F = 160

r_B: deviation [-300 s, -60 s), path ratio [0, 1.0]
     -> path_stretch/oracle_short_medium_long; ships with w = 0.55, F = 56

r_C: time_to_final < 5 min and intercept_established
     -> VETO
```

### Deployment path for this event

**Step 2D — Match certified rules only.** On Anchor 1, `r_A` matches because −65 is in `[−90, 0)` and `0.81 <= 1`; `r_B` matches because −65 is in `[−300, −60)` and `0.27 <= 1`. `r_C` does not match because UAL88 is 14 minutes from final and not established. There is no veto. Deployment never considers uncertified rules and never covers.

**Step 3D — Score and act.** Each action has one advocate here:

```
Q(anchor 1, speed)   = (0.62 * 160) / 160 = +0.62
Q(anchor 1, path_stretch/oracle_short_medium_long) = (0.55 * 56) / 56 = +0.55
```

The global argmax selects speed at Anchor 1 because `+0.62 > 0`. The runtime receives `(UAL88, speed, light)`; its realized delay is audited rather than assumed from the band name. Only one action fires. A positive candidate on Anchor 2 would be deferred and recomputed at the next event. Deployment has used no exploration, covering, contender, rollout, LCB, or update.

For contrast, suppose the next event is DAL34 at 4 minutes to final with intercept established. If `r_A` and `r_C` both match there, `r_C` vetoes the anchor before global arbitration. The `+0.62` speed score is suppressed and the controller chooses no-op. The veto is operational, not decorative.

### Training path for the original event

Training begins from the same original state and includes the deployment path, then adds experimentation and learning.

**Step 2T — Match the whole population and cover.** Suppose Anchor 2 lacks a `speed/medium` advocate. Covering mints `r_cov112` with widened intervals and zeroed ledgers. It can accumulate evidence at the root, but with `n = 0` it is uncertified and cannot act in deployment or inside rollout continuation.

**Step 3T — Explore or exploit.** The region cell `(low commitment, moderate pressure, medium error)` has 640 visits, above its floor of 200, so this epoch exploits and chooses speed as deployment did. In an undersampled cell, exploration could choose any feasible speed band, the path-stretch macro, no-op, or an uncertified covering rule. That root experiment is intentional; none of those privileges carries into continuation.

**Step 4T — Choose the contender.** Among certified non-speed advocates at Anchor 1, `r_B` has the strongest frozen rival-ledger LCB, so `c = path_stretch/oracle_short_medium_long`. The contender is local to Anchor 1 and is used only as Arm B's first action.

**Step 5T — Run three same-seed arms.** Use the already frozen `π_dep,t`, not the raw population. Fork from 12:02:10 with identical pending events and roll to JBU77's threshold crossing plus one slot, about 12 simulated minutes:

```
Arm A  speed   -> clean pair, no later correction                    y_A = +1.49
Arm B  path_stretch -> geometry disturbs DAL34; pi_dep,t corrects    y_B = +0.82
Arm C  no-op   -> compression persists; late large correction        y_C = +0.11
```

All continuation decisions use certified rules, deterministic precision pooling, active vetoes, and no learning. Different later actions across arms are consequences of different root states and properly affect each arm's outcome.

**Step 6T — Update both ledgers and veto evidence.** The three outcomes yield:

```
Delta_rival    = y_A - y_B           = +0.67
Delta_noop(a)  = y_A - y_C           = +1.38
Delta_noop(c)  = y_B - y_C           = +0.71
Delta_veto     = y_C - max(y_A,y_B)  = -1.38
```

Consequently, every selected speed advocate at Anchor 1 receives `+0.67` in its evolution ledger and `+1.38` in its deployment ledger. Every path-stretch macro advocate receives `−0.67` in evolution and `+0.71` in deployment. Both facts can be true: path stretch lost to speed but still beat doing nothing. Any matching no-op rule receives `−1.38` in its rival/veto ledger, evidence that a prohibition does not belong in this region. Rules that acted only during continuation receive no update.

A representative online update is:

```
r_A evolution:   n 52->53, mean 0.34 -> 0.34 + (0.67 - 0.34)/53 = 0.346
r_A deployment:  n 40->41, mean 0.62 -> 0.62 + (1.38 - 0.62)/41 = 0.639

r_B evolution:   n 44->45, mean 0.28 -> 0.28 + (-0.67 - 0.28)/45 = 0.259
r_B deployment:  n 36->37, mean 0.55 -> 0.55 + (0.71 - 0.55)/37 = 0.554
```

`r_cov112` receives nothing in this exploit epoch because it was neither selected nor the contender. It remains in the population, not the rulebook.

**Step 7T — Commit one action; reality continues.** Delete all three temporary arms, return to the untouched real state at 12:02:10, and issue only Arm A's first command: UAL88 flies 210 kt. No simulated continuation action is queued. When DAL34 crosses a fix at 12:03:40, a new epoch recomputes every anchor. The deferred opportunity on Anchor 2 waited about 90 seconds, illustrating why one-action arbitration is practical when events are dense.

**Step 8T — Evolve and, later, re-certify.** At roughly epoch 50, the niche GA selects parents by rival-ledger LCB, mutates conditions, and inserts discounted-stat offspring. Those offspring remain unable to act in continuation. At roughly epoch 500, certification recomputes action-rule `w` and `F`, applies the stricter veto gate, freezes the next rulebook, and atomically replaces `π_dep,t`. Only this slow tick changes the deployment and continuation target.

---

The workflow exposes the division of labor. Rival-grounded evidence says which rule wins competition and reproduction. No-op-grounded evidence says whether an action deserves to ship and how it should be pooled. Veto evidence says where even a positive action score must be suppressed. The same frozen rulebook performs deployment and every rollout continuation; only root exploration, ledger updates, GA, and slow re-certification distinguish training.

---
### What changes when

**Every epoch:** the selected rules' ledgers update globally in the mutable population. Their frozen deployment scores do **not** change immediately; the current `π_dep,t` remains read-only until the next certification tick. New world state from the single committed action changes which rules match at the next event.

**Every roughly 50 epochs:** GA and subsumption change conditions and population membership using the evolution ledger. Covering can add a rule at any root epoch. None of these changes enters the frozen rulebook immediately.

**Every roughly 500 epochs:** certification reads the current ledgers, admits or removes rules, freezes `w` and `F` or veto flags, and publishes a new `π_dep`. This is the policy-improvement boundary and the only moment continuation semantics change.

The one-sentence version: evidence moves fast, rule structure moves at the GA timescale, the deployed target moves slowly, and reality advances from one committed first action while all three longer futures remain temporary evidence.
