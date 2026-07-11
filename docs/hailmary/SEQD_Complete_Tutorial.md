# SEQD, From First Principles to Working System

*A complete, self-contained tutorial of the revised Sequencing Design framework. It assumes no memory of the earlier notes: everything needed to understand, implement, and evaluate the method is rebuilt here in order, with the revisions from the stress-test integrated as the default design rather than patched on. Read it linearly; later sections lean on earlier ones.*

---

## Part I — What we are building and why

### 1. The problem in plain language

Terminal-area arrival sequencing is the job of taking a stream of aircraft converging on an airport from several directions and turning it into a single, safely spaced, efficiently packed line onto the runway. Human approach controllers do this with a repertoire of *heuristics* — compact, reusable rules of thumb like "if the spacing error is small, fix it with speed; if it is large, stretch the path," or "once an aircraft is established near final, stop touching the sequence." These heuristics are the accumulated operational wisdom of the profession, but they exist mostly in controllers' heads and in training folklore. They are not written down as formal, testable objects, and nobody knows which parts of each heuristic are genuinely *causal* (the world really works that way) versus merely *habitual* (it correlates with success because of how traffic usually arrives).

SEQD (the Sequencing Design problem) is the problem of making an artificial agent **rediscover these heuristics from scratch**, purely by interacting with a traffic simulator — and, crucially, rediscover them in a form where every clause of every rule has *earned its place by experiment*. The deliverable is not a black-box policy that sequences traffic well. The deliverable is a **population of human-readable rules**, each of the form

```
IF   <condition on the traffic situation>
THEN <apply this kind of intervention, roughly this hard, here>
```

where every predicate in every condition has survived an interventional test: keeping it demonstrably changes outcomes, removing it demonstrably would not have.

The one-sentence thesis, in its revised form:

> SEQD is the problem of learning an XCS-style population of `condition ⇒ (anchor, lever, band)` rules for terminal-area sequencing, in which a rule's fitness is **interventional** rather than accuracy-based: a rule earns credit for the paired-rollout **advantage of its advocated action over its strongest counterfactual contender**, so that the recovered heuristics are causally load-bearing rather than merely predictive — and the abstract concepts controllers reason with (flow, slack, commitment, pressure) emerge as the predicates that survive this test.

Two things changed in that sentence relative to the original framing, and both came out of tracing real heuristics end to end. First, the counterfactual is no longer always "withhold the lever" (`no-op`); it is the strongest rival action, with `no-op` as the fallback. This is what makes *prohibition* heuristics — arguably the most characteristic controller heuristics of all — learnable as first-class rules rather than readable only as absences. Second, "paired-rollout" is now part of the definition: the interventional estimate comes from forking the simulator, not from matching across episodes. Both changes are motivated in detail in Parts III and V.

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

Everything in this page we keep, except the meaning of fitness.

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

SEQD does not steer aircraft. It emits an **abstract intervention tuple** and hands it to a lower-level realization layer (SEQP) that owns the geometry:

```
tuple = (anchor, lever, band)

anchor:  a node or edge in the graph, provisionally bound when a rule matches
         and finally bound only if that anchor–action candidate is selected
         for execution (see §9) —
         e.g. "the follower of this leader–follower edge",
         "this flow", "this region near merge point M"
lever:   one of { speed, stretch, hold, swap, meter-upstream, no-op }
band:    a coarse magnitude — e.g. "absorb ~0–1.5 min", "absorb ~2–4 min",
         "stretch within ~4 nm, headings 220–230"
```

SEQP consumes the tuple plus concrete geometry, realizes a legal maneuver (or reports infeasibility), and rolls the simulator to the next epoch. The division of labor is the whole point of the interface: heuristics live at the tuple level; geometry lives below it.

`no-op` is a first-class lever, and under the revised fitness (Part IV) it is not merely a control arm — rules can *advocate* it and earn positive credit for it. That is how prohibitions ("don't touch the sequence near final") become explicit, storable, reportable rules.

A practical warning about bands that shapes the staging plan: every distinct `(lever, band)` pair is a separate action for credit purposes, so bands **split the effect-estimation data**. Start with two coarse bands per lever, and treat band refinement as a specialization operator applied only to rules whose effect estimate is already confidently positive (§13).

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
  THEN  (anchor = F, lever = stretch, band = absorb 2–4 min)
  STATS Δ̄ = +0.42, σ² = 0.11, n = 63, LCB = +0.34, numerosity = 4
```

Everything in XCS that was defined per global state is redefined per **role context**: covering triggers when a candidate anchor's match set lacks an advocate for some action; GA niches are (role type × action) within similar contexts; subsumption compares rules of the same role type. This is a modest extension — it is how relational RL and graph-pattern classifiers already work — but it must be explicit in the implementation because it changes the unit of matching, the niche definition, and what "the state" means everywhere below.

---

## Part IV — The learning algorithm

This part is the heart of the method: the full decision-epoch loop, then the fitness mathematics.

### 10. The decision-epoch loop, step by step

At each event-driven epoch *t*:

**Step 1 — Enumerate candidate anchors.** Walk the graph; collect all live role instances (leader–follower edges, flows, aircraft–resource pairs). Compute each anchor's ego-graph feature vector from the bank. These are locations where a rule *could* apply, not interventions that have already been chosen.

**Step 2 — Build match sets and candidate applications.** For each candidate anchor α, form [M]ₐ = all rules of that role type whose interval conditions hold on α's features. Each match produces a candidate `(rule r, anchor α, action advocated by r)`; the same rule may therefore produce several candidates at different anchors. Matching neither selects nor executes any of them. If some lever–band action has no advocate in a context where the covering policy says it should (see the exploration note below), create a covering rule: intervals centered on the current feature values, widened by a random spread, advocating the missing action, initialized with zero effect statistics and low experience.

**Step 3 — Select at most one intervention for this epoch.** Aggregate candidate advocates into a fitness-weighted prediction per `(anchor, action)` pair (as in XCS, but the "prediction" is the LCB of the effect estimate, §11). Global arbitration compares those pairs across all anchors and chooses one winner `(α*, a*)` to serve as Arm A's initial action and later be applied in the real simulator — greedy on exploit steps, exploratory otherwise. Every nonwinning candidate, including another match of the same rule at a different anchor, remains unexecuted this epoch. The twin rollout evaluates the selected pair and supplies a learning sample; it does not retroactively replace the selected action with Arm B when an exploratory Arm A scores poorly. Simultaneous multi-anchor interventions are a phase-2 extension because they entangle credit.

**Exploration is region-scheduled, not merely time-annealed.** Maintain visitation counts over a coarse discretization of the concept axes that matter for the target heuristics — commitment band × pressure band × error-magnitude band is enough to start. Boost exploration probability in undersampled cells, and do not anneal exploration in a cell before it has minimum coverage. The reason is concrete: the agent can only learn "don't swap near final" by *executing late swaps and getting burned*, and a naive time-annealed schedule typically stops exploring before the rare high-commitment region has been sampled. Dangerous exploration is fine — this is a simulator — but it must actually happen.

**Step 4 — Choose the counterfactual contender.** This is the revised core. Let *a* be the selected action at selected anchor α*. The contender *c* is the **strongest rival**: the action ≠ *a* with the highest LCB among advocates in α*'s match set; if no rival has positive support, *c* = `no-op`. This comparison is local to the winning anchor: a candidate at a different anchor is not a contender merely because it was considered during global arbitration. Selecting the strongest rival (rather than always `no-op`) is what makes the fitness an *advantage* and what makes prohibitions learnable; the do-versus-don't test survives as the special case where the contender is `no-op`.

**Step 5 — Freeze the policy and run the twin rollout (the interventional estimate).** After covering and contender selection are complete, take a snapshot `P_t` of the rule population. The snapshot fixes the rules, their conditions, their scores, the arbitration method, and deterministic tie-breaking. Fork the current simulator state into two temporary arms that share the exogenous random seed and pending event stream. Arm A realizes the selected initial action *a* via SEQP; arm B realizes the initial contender *c*. Roll both arms forward to the outcome horizon *H* (§12), using the same frozen policy `π(P_t)` at every later decision epoch in both arms.

"Same policy" means the same rule-to-action mapping, not the same later action sequence. The initial interventions put the arms into different states, so different rules may subsequently match and their anchor–action candidates may win later arbitrations. Those downstream differences are descendants of the initial choice and therefore belong in the measured outcome. For example, speed in arm A may leave a clean sequence requiring no correction, while stretch in arm B may create geometry that causes the same frozen policy to select a slowdown for a trailing aircraft. The extra slowdown is legitimately charged to arm B even though another rule selected it.

No learning or population change is allowed inside either temporary branch: no statistics updates, GA activity, covering, subsumption, or exploration draws that differ arbitrarily between arms. If stochastic exploration is retained, its random stream must be deliberately coupled; the simpler default is deterministic exploit-mode arbitration with deterministic fallbacks (such as `no-op` when no frozen rule advocates a feasible action). Thus the only systematic difference is the initial action and the state/action consequences it causes. Pairing removes scenario-level variance, which is the dominant noise source. (Cross-episode matching, the observational alternative, is demoted to a robustness check and to any future setting where simulator forking is forbidden.)

**Step 6 — Compute the windowed outcome and the effect sample.** For each arm, compute the semi-local outcome *y* (§12), including all later interventions issued by the frozen policy. The effect sample is Δ = y_A − y_B. Distribute it only to the rules involved in the initial choice at α*: every rule in α*'s original match set that advocated *a* receives +Δ; every rule there that advocated *c* receives −Δ; rules advocating other initial actions and rules that acted only later inside a rollout receive nothing from this comparison. A rule that also matched a nonselected anchor receives at most this one update because it advocated at α*; its nonselected candidate creates neither an extra sample nor an extra physical action. Update each recipient's running statistics (mean, variance, count) once, in the real population, with the standard online (Welford) update. This assigns the initial advocates credit for choosing a state transition that led the whole fixed policy into a better or worse future; it does not claim that they personally chose every downstream action.

**Step 7 — Commit only the first action and continue the world.** Discard both temporary rollout branches and return to the untouched real state at epoch *t*. Apply only the selected initial action *a* in the real simulator; none of arm A's later simulated actions is automatically made real. Advance the real simulator until its next event-driven epoch, observe what actually happened, and solve a fresh twin-rollout problem there. In short: **long closed-loop rollout for evaluation, one-step commitment in reality**. This is receding-horizon or model-predictive execution: the rollout judges the first move by its downstream consequences without precommitting the real system to an entire predicted future.

**Step 8 — Rule discovery (steady-state GA), as in XCS but selecting on LCB.** Periodically, within each (role type × action) niche: select parents with probability increasing in LCB(Δ̄), cross over and mutate condition intervals, insert offspring with inherited-but-discounted statistics, delete population-wide among low-LCB, high-numerosity-redundant rules, and apply subsumption — a rule may subsume a more specific same-action rule only if its own effect estimate is confidently positive and at least as large (within tolerance). Generalization pressure thus pushes toward the *widest condition over which the effect holds*, which is precisely the natural-language shape of a heuristic.

### 11. The fitness mathematics, precisely

Each rule *r* maintains `(Δ̄_r, σ²_r, n_r)` over its received effect samples. Its selection score is a lower confidence bound:

```
LCB(r) = Δ̄_r − z · σ_r / √n_r        (z ≈ 1.0–1.96; a tunable pessimism knob)
```

Three design notes, each answering a real failure mode:

**Why LCB and not the mean.** Effect samples are noisy even after pairing; young rules have tiny *n*. Selecting on the raw mean lets lucky junk reproduce; the LCB makes a rule prove its effect against its own uncertainty before it breeds — the same logic as optimism-under-uncertainty bandits, run in reverse because reproduction is a commitment.

**What the estimate *is*.** Conditional on rule *r*'s predicates holding, Δ̄_r estimates the conditional average **policy-mediated action advantage** of *r*'s initial action versus the contender mix it was tested against: "choose this action now, then continue under frozen policy `π(P_t)`" versus "choose the contender now, then continue under that same policy." It is not the isolated physical effect of the first maneuver with all later actions suppressed. This is intentional: an initial action is valuable partly because it leads the rule system to need fewer, smaller, or safer corrections later. It also means a rule's measured value depends on the quality of the rest of the population; as that population changes across real epochs, the same initial action can acquire a different long-run value.

Two further caveats belong in any write-up. First, exploration is not uniformly random, so effect estimates carry selection pressure from the behavior policy; within-match-set randomization on explore steps (choose among advocated actions uniformly at random) keeps the initial-action comparison interpretable. Exploration used *inside* rollouts must still obey the frozen-policy and coupled-randomness requirements of Step 5. Second, the contender is the *current* best rival, so fitness is **nonstationary**: as the population improves, the bar rises — an advantage measured against a stronger baseline shrinks. This is the same self-play-like nonstationarity as any advantage-based method and is mostly benign (surviving rules are those that beat *good* alternatives), but it means absolute effect sizes drift. The stabilizer: periodically re-run a fraction of twin rollouts against a fixed `no-op` arm regardless of rivals, maintaining an absolutely grounded second statistic per rule. Report both.

**Sample sharing.** All co-advocates in the match set share each Δ sample. This is the classifier-system analogue of every-visit credit and is what lets general rules accumulate *n* fast. Its known pathology — an over-general rule harvesting credit from a subregion where the effect is real and coasting elsewhere — is handled by the decorrelating generator (Part V): if the generator visits the rule's whole condition region, the over-general rule's Δ samples mix in the zero-effect region, its mean drops, its variance rises, its LCB collapses, and the GA's specialization pressure carves out the subregion where the effect actually lives.

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
                           action and all later actions selected by frozen π(P_t))
  + w₄ · global residual:  small-weight throughput term over the window,
                           insurance against effects escaping the k-window
```

Defaults to start: k = 3, H = the k-th trailer's threshold crossing plus one slot, w₁ : w₂ : w₃ : w₄ ≈ 1 : 1 : 0.3 : 0.1.

**The separation penalty must be graded, not catastrophic.** Score margin erosion piecewise (full margin → partial erosion → violation → go-around) rather than as a single huge negative constant. With a cliff penalty, Δ samples in risky regions are all-or-nothing and their variance swamps the estimator; with a graded score, "this swap ate 40% of the trailing pair's margin" is a smooth, learnable signal that points in the same direction as the rare disaster. (Fuel, workload-as-entropy, and monitorability terms remain deliberately dropped from the outcome — they dilute the credit signal and are second-paper material. Parsimony stays because it is the pressure toward sparse, controller-like intervention patterns.)

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

These traces are the proof-of-concept in prose: one positive action rule (the easy case, which validates the machinery) and one prohibition (the hard case, which *required* the advantage-fitness revision). Numbers are illustrative but of realistic magnitude.

### 14. Trace 1 — "Small error → speed, large error → path"

**Target.** The delay-absorption heuristic decomposes into two rules over a leader–follower pair, anchored on the follower:

```
R_speed:  spacing_deviation ∈ [−90 s, 0)  ∧  required_delay/speed_capacity ≤ 1
          ⇒ (F, speed, absorb 0–1.5 min)
R_path:   spacing_deviation ∈ [−300 s, −90 s)  ∧  required_delay/path_capacity ≤ 1
          ⇒ (F, stretch, absorb 2–4 min)
```

Both conditions are single intervals on Tier 3 ratios plus one on the raw deviation — expressible as *one rule each* only because the ratio library exists (§7). Without Tier 3, the same causal content shatters into a tiling of boxes over (deviation × track-miles × speed-margin) space.

**A learning event, concretely.** Epoch t: pair (L, F), predicted deviation −60 s at the merge, F has speed capacity ≈ 90 s. The follower-role match set contains `R_speed` (advocating speed), a covering-born rival `R_junk` (advocating stretch under similar conditions), and a no-op advocate. Exploration picks speed. Contender: stretch (strongest rival). Freeze `P_t`, then run both temporary branches under `π(P_t)`: arm A (speed) closes the deviation to −5 s, trailing aircraft unaffected, one small intervention charged → y_A ≈ +0.8. Arm B (stretch) also closes the gap but the wider geometry disturbs the third aircraft in trail and causes the frozen policy to make a larger downstream correction → y_B ≈ +0.4. Sample Δ = +0.4 flows to `R_speed`; −0.4 flows to `R_junk`; the downstream correction's advocate receives no update from this initial-action comparison. Both branches are then discarded, and only the initial speed action is applied in reality. Over dozens of such matched events, `R_speed`'s LCB climbs, `R_junk`'s collapses, the GA breeds variants of `R_speed`, and subsumption widens its intervals to the largest region where the advantage holds — which is exactly the ratio boundary.

**The confound test.** The natural confounder of "error is small" is "close to final" (small errors are usually *detected* late under naturalistic traffic, so the two co-occur). A confounded rule `distance_to_final < θ ⇒ speed` predicts success well observationally. The generator's registered decorrelation produces late-detected *large* errors; the confounded rule is eligible there and, when its speed candidate is selected, speed fails to absorb the error, the pair term of y_A goes negative, and the rule inherits negative Δ samples it cannot escape. It dies not because it predicted badly — it predicted fine on the naturalistic majority — but because its advocated action performed badly where its predicate and the true cause came apart. That sentence is the entire epistemic difference from accuracy-based fitness and from distillation, exhibited on one rule.

**Expected artifact.** Two surviving rules whose load-bearing predicates are the slack-ratio features — i.e., the *slack* concept, emerged and certified. The vanilla-XCS baseline run on identical experience is predicted to retain the distance-to-final confound; that contrast is the Phase 0 headline figure.

### 15. Trace 2 — "Freeze the sequence near final"

**Target.** A prohibition: when commitment is high, do not reorder (and mostly do not act). Two encodings exist, and the framework should produce **both** and show they coincide:

```
Encoding A (explicit rule):
  time_to_final < θ  ∧  intercept_established  ∧  path_freedom low
  ⇒ (—, no-op, —)
Encoding B (population boundary):
  swap-advocating rules simply fail to survive in the high-commitment
  region; the heuristic is the survival frontier of the swap population.
```

**Why the original fitness definition failed here.** Under "credit = outcome-with-lever minus outcome-with-no-op," a rule whose lever *is* no-op has identically zero fitness forever. As originally written, the framework could represent the profession's most characteristic heuristic only as an absence (Encoding B), which is fragile — it requires dense coverage of the region by attempted swaps before the frontier is readable, and an absence is an awkward deliverable to hand a controller. Under the revised **advantage fitness**, when a high-commitment match set contains a swap advocate and a no-op advocate, the twin rollout arms are *swap vs no-op*, and the no-op rule earns +Δ exactly when acting hurts. Prohibitions become storable, reportable, first-class rules. This single heuristic is what forced the fitness redefinition, and it is the honest way to motivate that change in a paper.

**Why the effect is detectable.** A late swap's damage is a chain effect: the swapped pair compresses, the compression propagates rearward, and the cost lands on the third and fourth aircraft in trail, tens of seconds later. Three pieces of the design earn their keep simultaneously here. The *windowed multi-aircraft outcome* (k ≥ 3) is what sees the damage at all — a pair-only outcome frequently scores a late swap as fine while the trailers eat the go-around risk. The *twin rollout* is a textbook blocking intervention, and the cascade-identifiability results are the theoretical warrant that this comparison recovers structure that observational scoring provably cannot. And *graded margin scoring* turns "this swap consumed 40% of the trailing pair's margin" into a smooth signal aligned with the rare catastrophe.

**A learning event.** High-commitment epoch: F established on intercept, 4 min to threshold, a tempting order improvement available. Exploration (region-scheduled — this cell is protected from annealing precisely so this event happens) selects the swap as the initial action; contender is no-op. Under the same frozen `π(P_t)`, arm A's sequence order improves on paper, but trailers 2–3 compress, one margin drops 45%, and the policy issues an extra corrective intervention → y_A ≈ −0.9. In arm B, nothing is done initially and the stream lands as built → y_B ≈ +0.1. Δ = −1.0: the initial swap advocates absorb −1.0, the initial no-op advocate absorbs +1.0; any rule responsible only for the later correction receives no sample from this comparison. Both futures are discarded after scoring, and only the initially selected swap is tried in the real simulator. Repeated across the decorrelated distribution, swap rules retreat from the high-commitment region (Encoding B emerges) while the explicit no-op rule's LCB climbs (Encoding A emerges). **Reporting both and showing the no-op rule's condition coincides with the swap-survival frontier is an internal-validity check no distillation baseline can offer.**

**The confound test.** High commitment co-occurs naturally with high pressure (streams get committed *because* they are built under demand). A correlational learner therefore tends to learn `pressure high ⇒ don't swap` — which is wrong: under *low* commitment, swapping under pressure is often exactly right ("insert into a gap"). The registered decorrelation produces high-commitment/low-pressure and low-commitment/high-pressure situations; interventionally, swaps in the second quadrant show *positive* Δ, so the pressure predicate cannot survive in the prohibition, while the commitment signature (time-to-final × path freedom × established geometry — the *commitment* concept) can and does.

---

## Part VII — Evaluation, staging, and diagnostics

### 16. What counts as success

The north star is "the learned concepts are real," so evaluation targets the rule population, not single-scenario control scores.

**Causal load-bearing (primary).** For each surviving rule and each of its predicates: use the *generator* to manufacture matched situations where that predicate is flipped while the rest of the condition holds, run the rule's action, and measure the effect. A load-bearing predicate's flip should erase or reverse the advantage; an inert predicate's flip should not change it — and inert predicates should already have been stripped by subsumption, so finding one is a bug report. This is predicate ablation done interventionally, closing the loop on the method's own claim.

**Generalization.** Train on one region of generator-parameter space; test surviving rules' effects on held-out regions (different demand levels, different geometry variants). A heuristic, by definition, transfers.

**Concept correspondence.** Because Tier 2/3 features are the operationalized concept vocabulary, this is a direct readout: report which concepts' features appear as load-bearing predicates in surviving rules, against the target table (flow, leader–follower, pressure, slack, commitment, intervention region, reservoir, gap, compression risk). Include the frontier readouts (Encoding B objects) alongside explicit rules.

**Baseline beat.** Two concrete baselines on *identical experience streams*: (a) vanilla accuracy-based XCSR — isolates the contribution of the credit signal alone, everything else held fixed; (b) a trained policy (GNN or tabular) plus decision-tree distillation — the existing interpretable-AMAN recipe. The predicted, falsifiable contrasts: both baselines retain the registered confounds (distance-to-final in H1, pressure in H2); the interventional population does not; and interventional rules transfer better across generator regions.

### 17. The staged plan

**Phase 0 — one pair, three actions (weeks, not months).** A single leader–follower pair on a simple merge geometry; levers {speed, stretch, no-op}, two bands; the H1 decorrelation registered and audited. Success: H1 recovered as ≤ 3 rules whose load-bearing predicates are the slack ratios, while vanilla XCS on the same runs keeps the confound. This is the minimum publishable signal that interventional fitness does something accuracy cannot, and every later phase inherits its infrastructure (twin rollouts, windowed outcome, audit).

**Phase 1 — three-aircraft chain, add {swap}.** Target: H2 by both encodings, plus the coincidence check between the no-op rule and the swap-survival frontier; the distillation baseline demonstrably learns the pressure confound. Region-scheduled exploration becomes necessary here and should be validated (measure sampling density in the high-commitment cell with and without it).

**Phase 2 — full distribution.** Remaining levers ({hold, meter-upstream}), three bands per lever via the refinement operator, hyperedge conditions (conjunctions across multiple node types) enabled, the full concept-correspondence table, and the generalization evaluation. Multi-anchor simultaneous interventions and oblique predicates remain listed as extensions beyond even this phase.

### 18. Failure modes and their designed answers

| Failure mode | Symptom | Designed answer |
|---|---|---|
| Effect drowned in noise | LCBs never separate from zero | Twin rollouts (§10.5) + windowed outcome (§12); check w₄ is small |
| Concept fragmentation | Many small same-action rules tiling a diagonal | Tier 3 ratio features (§7); phase-2 oblique predicates |
| Confound survives | Registered pair predicate persists in rules | Generator audit failed — fix decorrelation, retrain (§13) |
| Prohibitions unlearnable | No surviving no-op rules where freezing is right | Verify advantage fitness (not do-vs-no-op) is implemented (§10.4) |
| Exploration dies early | High-commitment cells unsampled; no late-swap data | Region-scheduled exploration with per-cell coverage floors (§10.3) |
| Lucky junk reproduces | Volatile population, young rules breeding | Select on LCB, not mean; raise z; raise GA experience threshold (§11) |
| Over-general credit harvesting | Broad rule with bimodal Δ samples | Generator coverage of full condition region + specialization pressure (§11) |
| Cliff-penalty variance | Δ variance explodes near separation limits | Graded margin scoring, not catastrophic constants (§12) |
| Band data starvation | Per-(lever,band) n too small | Two coarse bands to start; refine only confident rules (§8, §17) |
| Fitness drift | Effect sizes shrink as population improves | Expected (advantage vs. improving rivals); keep the no-op-grounded second statistic (§11) |

### 19. Starting parameter sheet

All values are starting points, not conclusions: population size 1,000–2,000 (Phase 0) rising with phases; learning-rate β = 0.1–0.2 for online stats; LCB z = 1.0 exploratory, 1.96 for reproduction eligibility; GA experience threshold ≈ 20 effect samples; k = 3 trailers; horizon H = k-th trailer's threshold crossing + one slot; outcome weights 1 : 1 : 0.3 : 0.1; bands = 2 per lever; decorrelation audit threshold |ρ| ≤ 0.3; exploration coverage floor ≈ 200 visits per registered concept-axis cell before annealing; fraction of twin rollouts re-grounded against no-op ≈ 10%.

### 20. What was deliberately dropped, and the open questions that remain

Dropped as decisions, unchanged from the consolidated note: autonomous procedure design (second paper); fuel/workload/monitorability reward terms (dilute the signal); concepts-as-embedding-clusters (replaced by the interventional-survival definition — one definition, not three); distillation as a method (kept only as the baseline it is meant to beat); neuro-symbolic-from-scratch (one mechanism, not a menu).

Genuinely open after the revisions, in rough order of risk: the constants of the windowed outcome (k, H, weights) and their sensitivity; whether sample sharing plus generator coverage suffices against over-general harvesting or a per-rule heterogeneity test (variance decomposition of a rule's Δ samples over its condition region) is needed; the arbitration rule when several anchors want interventions in one epoch; how far the nonstationary-contender drift can go before the no-op re-grounding fraction must rise; and the one remaining literature sweep in causal-ML/online-RL before claiming the intersection is open.

---

## Appendix — Glossary

**Anchor** — a graph node/edge where a rule may be eligible; a match provisionally binds it, but an intervention is physically applied there only if its anchor–action candidate wins arbitration. **Advantage fitness** — a rule's credit: paired-rollout outcome gap between its advocated initial action and the strongest contender in its match set, followed in both arms by the same frozen policy. **Band** — coarse magnitude slot of an intervention tuple. **Blocking intervention** — withholding/replacing an action in a cascade system; the theoretical basis for identifiability here. **CATE** — conditional average treatment effect; here, the policy-mediated action advantage that Δ̄ estimates conditional on a rule's predicates. **Candidate application** — a rule match paired with its particular anchor and advocated action; it is eligible for arbitration, not an executed action. **Commitment** — accumulated cost of changing an aircraft's role/order; a target concept. **Contender** — the counterfactual initial action in a twin rollout: strongest rival action at the selected anchor, fallback no-op. **Covering** — XCS mechanism creating a rule when a situation lacks an advocate. **Ego-graph** — the local subgraph rooted at a candidate anchor over which a rule's predicates are evaluated. **Encoding B / frontier** — a prohibition read off as the survival boundary of an action's rule population. **Epoch** — event-driven decision point of the semi-MDP. **Frozen policy** — the snapshot `π(P_t)` used without learning or structural population changes throughout both rollout arms; the same mapping may produce different later actions in different arm states. **LCB** — lower confidence bound on a rule's mean effect; the selection score. **Lever** — the action verb of a tuple: speed, stretch, hold, swap, meter-upstream, no-op. **Match set** — rules whose conditions hold for one particular candidate anchor; membership means eligibility, not execution. **Minimum-time map** — per-aircraft earliest-feasible times at downstream points; basis of spacing deviation, slack, pressure. **Niche** — (role type × action) context within which the GA competes rules. **Receding-horizon execution** — use a long simulated future to evaluate a choice, commit only its first action, then recompute at the next real epoch. **Region-scheduled exploration** — per-cell coverage floors over concept axes, preventing premature annealing. **Role-centric matching** — evaluating rule conditions per candidate anchor rather than on a global state vector; one rule can create several candidate applications but cannot execute at more than one anchor in an epoch. **SEQP** — the lower-level realization layer that turns tuples into legal geometry. **Slack** — absorbable delay; speed-slack and path-slack variants. **Subsumption** — absorption of a specific rule by a confidently-at-least-as-good more general one. **Tier 3 features** — derived ratio/comparison features; the operationalized concept vocabulary. **Twin rollout** — same-seed temporary simulator forks that differ in the initial action and then use the same frozen policy; the interventional estimator. **Windowed outcome** — semi-local outcome: pair term + k-trailer propagation + parsimony + small global residual.

---
# Complete Example
Here's one complete epoch, run with actual numbers from start to finish. Same machinery as before, but now nothing is abstract — every step produces a concrete quantity you can follow into the next step.

**The scene.** Four arrivals converging on merge point M for runway 25, in predicted order: AAL12 (a heavy), then UAL88 (a 737), then DAL34 (an A320), then JBU77 well behind. The graph holds three leader–follower edges — (AAL12→UAL88), (UAL88→DAL34), (DAL34→JBU77) — one flow node grouping all four, and the aircraft–runway pairs.

**The trigger.** At sim time 12:02:10, UAL88 crosses entry fix KODIA. That's a discrete event, so an epoch fires. Its minimum-time map is recomputed, and the prediction now shows trouble ahead.

---

**Step 1 — Enumerate anchors, compute features.** The system walks the graph and computes each anchor's ego-graph features. The two that matter this epoch:

*Anchor 1, edge (AAL12→UAL88), anchored on follower UAL88.* Required separation at M for a heavy→medium pair: 120 s. Predicted interval: 55 s. So spacing deviation = 55 − 120 = **−65 s** — compression; if nothing is done, UAL88 arrives 65 s too close behind the heavy. Required delay: 65 s. UAL88's speed-absorption capacity (from its envelope and remaining track-miles): 80 s. Path-absorption capacity: 240 s. So the Tier-3 ratios: `required_delay/speed_capacity` = 65/80 = **0.81** (speed alone can just barely absorb it), `required_delay/path_capacity` = **0.27**. Time-to-final: 14 min, intercept not established, path freedom high — low commitment.

*Anchor 2, edge (UAL88→DAL34).* Predicted deviation **+45 s** — a modest gap, wasted capacity. Some rules see an opportunity here.

**Step 2 — Build match sets per anchor.** On anchor 1, four rules match:

```
r_23 (speed, 0–1.5 min):  Δ̄=+0.51, σ²=0.16, n=48  →  LCB = 0.51 − 0.40/√48 ≈ +0.45
r_87 (speed, 0–1.5 min):  Δ̄=+0.60, σ²=0.49, n=9   →  LCB = 0.60 − 0.70/√9  ≈ +0.37
r_61 (stretch, 2–4 min):  Δ̄=+0.22, σ²=0.36, n=25  →  LCB = 0.22 − 0.60/√25 ≈ +0.10
r_09 (no-op):             Δ̄=+0.02, σ²=0.09, n=40  →  LCB ≈ −0.03
```

(Using z = 1 for selection. Notice r_87 has a *higher mean* than r_23 but a *lower* LCB — nine noisy samples buy less trust than forty-eight consistent ones. That's the LCB doing its job.) On anchor 2, `r_44` (swap, LCB +0.12) and a no-op advocate (+0.01) match. Suppose the covering policy notices anchor 2's cell has no `hold` advocate — it mints `r_cov112` on the spot, intervals centered on anchor 2's current features and randomly widened, statistics zeroed. It exists now but has no standing.

**Step 3 — Arbitrate one intervention.** First, explore or exploit? The region scheduler checks this epoch's cell — (low commitment × moderate pressure × medium error) — visit count 640, well above the 200 floor. So: exploit. Pool advocates into per-action scores on each anchor (fitness-weighted): anchor 1 gives speed ≈ +0.44 (r_23 and r_87 fused), stretch +0.10, no-op −0.03; anchor 2 gives swap +0.12. The global arbitration compares those *anchor–action* candidates: **+0.44 for (anchor 1 / UAL88, speed) wins**. Thus SEQP receives one bound tuple, `(UAL88, speed, 0–1.5 min)`. The swap candidate on anchor 2 is deferred — not cancelled; its edge is still live next epoch.

If, in this same epoch, r_23 also happened to match a different edge — say (UAL88→DAL34), where its follower binding would be DAL34 — that would create a second candidate for r_23, not a second speed command. Since r_23's candidate on anchor 1 won, only UAL88 is acted on. The DAL34 candidate would have to survive recomputation and win a new arbitration at a later epoch before SEQP could act on it.

**Step 4 — Choose the contender.** Within anchor 1's match set, the strongest rival to speed is stretch at +0.10 (beats no-op's −0.03). So c = (UAL88, stretch, absorb 2–4 min). Note the contender is *not* the swap from anchor 2 — the comparison is always within the match set at the selected anchor, because that is the counterfactual "what else could I have done *here*."

**Step 5 — Freeze and run the twin rollout.** Snapshot the current population as `P_t`, then fork the simulator from 12:02:10: identical exogenous seed, identical pending event stream (JBU77's entry at 12:04:30, a wind shift event at 12:06:00 — both will happen in both arms). Arm A initially realizes speed — UAL88 reduces from 250 to 210 kt, absorbing ~70 s. Arm B initially realizes stretch — a heading-240 vector worth ~2.5 min of delay. Horizon H: the third trailer behind the intervention is JBU77, so both temporary arms roll to JBU77's threshold crossing plus one slot — about 12 minutes of simulated time. During those minutes, epochs keep firing and both arms use exactly the same frozen rule population, scores, arbitration, and tie-breaking. No learning, covering, or GA runs inside either arm. The states nevertheless diverge, so different rules may match and later actions may differ. That closed-loop divergence is *part of the effect being measured*; it does not mean those later actions will be committed to reality.

**Step 6 — Score and distribute.** Windowed outcome per arm, with weights w = (1, 1, 0.3, 0.1):

*Arm A (speed):* pair (AAL12, UAL88) lands at −8 s deviation — nearly closed, graded pair term **+0.85**. Trailers: DAL34's 45 s gap comfortably absorbs UAL88's slowdown, JBU77 untouched — propagation **+0.75**. The frozen policy needs no later correction, so only the initial small intervention is charged — parsimony **−0.4**. Residual +0.1.
y_A = 1(0.85) + 1(0.75) + 0.3(−0.4) + 0.1(0.1) = **+1.49**

*Arm B (stretch):* also closes the pair deviation, pair term +0.80. But the vector swings UAL88 wide, eroding 20% of the (UAL88→DAL34) margin; the same frozen policy now matches a different situation and issues a corrective speed cut to DAL34 — propagation drops to **+0.40**, and parsimony charges *two* simulated interventions, one of them large: **−1.3**. Residual +0.1.
y_B = 1(0.80) + 1(0.40) + 0.3(−1.3) + 0.1(0.1) = **+0.82**

**Δ = 1.49 − 0.82 = +0.67.** Distribution: every advocate of the initially selected speed action gets +Δ, every advocate of the initial stretch contender gets −Δ, and everyone else gets nothing. In particular, a rule that selected DAL34's corrective speed cut inside arm B does not update from this comparison; that hypothetical correction affects the score but is not a separately credited learning event.

```
r_23: n 48→49, Δ̄ 0.51 → 0.51 + (0.67−0.51)/49 ≈ 0.513   LCB ticks up
r_87: n 9→10,  Δ̄ 0.60 → 0.607                            SE shrinks, LCB → ≈ +0.39
r_61: n 25→26, Δ̄ 0.22 → 0.22 + (−0.67−0.22)/26 ≈ 0.186   variance up, LCB → ≈ +0.06
r_09, r_44, r_cov112: nothing
```

One epoch has moved three ledgers by a few hundredths. The heuristic "small-ish error with adequate speed capacity → use speed, not geometry" is being carved one Δ at a time.

**Step 7 — Commit one action; reality continues.** Neither 12-minute branch becomes the real timeline. Delete both temporary arms, return to the still-untouched real state at 12:02:10, and issue only Arm A's first command: UAL88 flies 210 kt. The simulated later no-ops or corrections are not queued. The real simulator now advances from 12:02:10 using actual events. When DAL34 crosses a fix at 12:03:40, a new epoch begins at Step 1 and the system recomputes from the observed state. Anchor 2's swap opportunity, deferred 90 seconds earlier, is re-arbitrated against whatever the world actually looks like after the speed reduction.

**Step 8 — Periodic GA (fires on its own schedule, say every 50 epochs).** Inside the niche (leader–follower role × speed action): parents drawn with probability increasing in LCB — r_23 is selected, crossed with another speed rule, a mutation nudges one interval boundary, offspring r_101 inserted with discounted inherited stats. Deletion culls a low-LCB redundant rule elsewhere in the population. Subsumption check: r_23's condition strictly contains r_87's, and r_23's effect is confidently positive and at least as large within tolerance — so r_23 *absorbs* r_87: r_87 is deleted, r_23's numerosity increments. The population just got one rule simpler while losing nothing it had learned.

---

Three things this concrete run makes visible that the abstract version hides. First, where the "one intervention" rule bit: the swap candidate on anchor 2 had a positive score but lost global arbitration and was made to wait ~90 seconds — the cost of clean credit was a minor deferral, because events are dense. Second, why the contender choice mattered: had the contender been no-op instead of stretch, Δ would have been measured against "do nothing" (y ≈ +0.3 as the compression partially persists), giving speed a bigger but less informative advantage — "+1.2 versus doing nothing" says less than "+0.67 versus the best alternative treatment." Third, the propagation term earned its keep in arm B: a pair-only outcome would have scored stretch nearly as well as speed (+0.80 vs +0.85), and the real difference — the disturbance to DAL34, the extra corrective intervention — lives entirely in the trailing window and the parsimony charge. Delete w₂ and w₃ and this epoch teaches almost nothing.

---
# Notes about Updates

**What updated immediately: the rules' scores, and those scores are global to the rule, not local to the selected anchor.** r_23's ledger is attached to r_23 the *pattern*, not to the (AAL12→UAL88) edge where its candidate won. At the next epoch, when the system evaluates edge (DAL34→JBU77), some different pair tomorrow, or the same UAL88 edge again — anywhere r_23's condition intervals hold — arbitration uses r_23's post-update LCB, +0.513-ish and slightly tighter. One Δ sample from the selected anchor shifts r_23's standing at every future anchor where it may match. That is a **knowledge update**, not a physical broadcast of the original speed command: matching r_23 elsewhere still creates only candidates, each of which must independently win a later arbitration before any aircraft is acted on. This evidence amortization across the condition region is the economy of the design (the contrast with MCTS node statistics from earlier). It cuts both ways, of course — a rule burned on one anchor becomes more cautious everywhere, including in subregions where it might actually be fine, and it's the GA's specialization pressure that eventually splits the condition if those subregions genuinely differ.

**What did *not* update: the conditions themselves.** Within an ordinary epoch, no rule's IF-part changes — intervals only move via the GA (mutation, crossover), covering (new rules), and subsumption (deleted rules), which run on their own schedule. So *matching* in the strict sense — which rules appear in which match sets — is determined by conditions and features, and this epoch left conditions untouched (except that r_cov112 now exists and will start appearing in match sets, and after the GA pass, r_87 is gone and r_101 is new). What changed for the very next epoch is *selection within* the match sets: same candidates, different scores, so the prediction arrays, the arbitration winner, and the choice of contender can all come out differently. Concretely: if the deferred swap on anchor 2 gets re-arbitrated next epoch, it's now competing against a speed action whose pooled score just went *up* — the bar it must clear rose because of an experiment it wasn't even part of.

**And a second, independent channel: the world itself changed.** Only Arm A's *first action* is committed, so UAL88 is physically flying 210 kt; the rest of Arm A's 12-minute simulated action sequence was discarded. Next epoch's anchors are enumerated from the real world produced by that first command and whatever actually occurred afterward: the (AAL12→UAL88) deviation may now be near zero, so r_23's compression predicate may no longer hold there and it can drop out of that match set; the (UAL88→DAL34) gap may have shrunk as UAL88 slid back toward DAL34, so anchor 2's features can move and a different set of rules may match it. Thus two effects stack: on the *knowledge* side, every future match set containing the initially credited rules sees updated scores; on the *world* side, the one committed action reshapes future states and therefore which anchors and conditions actually appear.

The one-sentence version: rule statistics propagate instantly and globally through the population's scores; rule conditions propagate slowly through the GA; and the real world propagates from the single committed first action — while the longer arm-A and arm-B futures remain temporary evidence used only to evaluate that choice.
