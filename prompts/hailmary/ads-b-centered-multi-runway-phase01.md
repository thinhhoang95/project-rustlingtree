ADS-B-Centered Multi-Runway Redesign of Hailmary Phases 0–1
Summary
Replace the two-aircraft, one-runway factorial experiment with independent ADS-B-centered traffic snapshots:
Generate half-open one-hour windows [start, start + 3600s) every 20 minutes, counted by reconstructed 50-NM terminal entry.
Include every runway represented by valid arrivals in each window and every aircraft in the snapshot.
Use scale 1.0 in Phase 0; Phase 1 applies one global scale to every window and every runway-scoped arrival cluster. Scaling is never applied independently to a runway total.
Retain pairwise decisions, but define pairs as adjacent aircraft committed to the same inferred directed route segment—not by runway-threshold order.
Remove factorial scenario generation, correlation gates, and controlled commitment/error/pressure/time-to-final levels entirely. These quantities may remain naturally observed learner features.
Phase 0 certifies both scenario/topology fidelity and held-out learner improvement; Phase 1 adds scaled traffic and downstream-trailer credit.
The current dataset contains 1,026 catalogued arrivals over nine runway labels, while existing Hailmary cluster artifacts cover only RW18R. The offline pipeline must therefore build runway-scoped cluster/template artifacts for the complete eligible ADS-B corpus.
Implementation Changes
Demand windows and traffic scaling
Introduce DemandWindowConfig(width_s=3600, stride_s=1200), TrafficScaleConfig, TrafficScenario, and TrafficScenarioBatch.

Add the raw ADS-B directory to data_manifest.json; reconstruct terminal entries and qualify cluster IDs as (airport, runway, cluster). In the formulas below, w identifies the window, r the runway, and c the arrival cluster within that runway.

Treat overlapping windows as independent scenarios. Preserve observed terminal-entry timestamps and cluster/runway counts at scale 1.0; duplicated source flights across overlapping snapshots are expected.

Compute each target with deterministic half-up rounding:
target_count[w,r,c] = floor(global_scale × observed_count[w,r,c] + 0.5).
Thus 1.05 × 34 → 36; a zero baseline count remains zero.

Scaling is performed independently for each (airport, runway, arrival_cluster) count. A runway target is not scaled or apportioned separately; its resulting count is derived as the sum of its rounded cluster targets:
target_count[w,r] = sum_c target_count[w,r,c].
Because rounding occurs per cluster, this derived runway count may differ slightly from floor(global_scale × observed_count[w,r] + 0.5). This is intentional: cluster-level scaling preserves the ADS-B-observed arrival-direction mixture.

For scale above 1.0, retain all baseline arrivals and draw exactly the required additions from a cluster/time-of-day empirical non-homogeneous Poisson intensity conditioned on the target count. Fit bandwidths on training data, with runway-level and then airport-level intensity fallback for sparse clusters.

For scale below 1.0, select the exact target through seeded uniform thinning without replacement. All random streams derive from stable names containing dataset, window, cluster, scale, and replicate.

Use the observed cluster medoid path for every materialized flight. Jointly sample terminal-entry ground speed and altitude from one real member of that cluster, record the donor ID, and use the existing dynamics/compiler to produce the remaining feasible profile.

Build a deterministic singleton-medoid fallback for sparse runways or clusters rather than excluding their traffic. Flights lacking a reconstructable terminal entry remain rejected with explicit reason counts and do not receive invented release times.

Geometry-derived route graph and true pairs
Create a versioned RouteGraphArtifact containing stable directed segment IDs, merge/diverge nodes, runway endpoints, source cluster membership, geometry, thresholds, and provenance.
Infer topology from continuous medoid polylines rather than waypoint equality:resample by arc length;
match sustained same-direction corridors using lateral distance, tangent alignment, and monotone station mapping;
use local within-cluster dispersion for corridor width, with initial defaults of a 0.5-NM floor, 15° tangent tolerance, and 5-NM minimum common length;
apply hysteresis so curves do not fragment into repeated merge/split nodes;
keep parallel runway finals distinct unless a genuine common corridor is established.

Use fix sequences only to label or corroborate inferred nodes. Exact, missing, duplicated, or collinear waypoints never determine connectivity.
Add typed SegmentTraversalDefinition records to each flight and segment-entry/exit resources to its trajectory variant.
Replace implicit single-threshold anchor construction with build_current_segment_anchors(). Pair detection must not sort every aircraft by runway ETA or by ETA at arbitrary medoid waypoints. Each maximal directed segment has one canonical entry gate and one canonical exit gate, obtained from the medoid-derived route graph and mapped to stations on every traversing template.

At every decision epoch, build the queue for each segment S as follows:
consider only active flights whose fixed SegmentTraversalDefinition contains S and which have not crossed S's exit gate;
classify each eligible flight as an occupant if it has crossed S's entry gate, otherwise as a committed future entrant;
order occupants first by current along-segment progress from most downstream to most upstream, using the current active trajectory variant rather than baseline elapsed time;
order future entrants after the occupants by current ETA at S's entry gate, breaking exact ETA ties by stable flight ID;
concatenate the occupant and future-entrant lists and create leader-follower anchors only for adjacent flights in that queue;
bind a flight pair to its earliest unpassed common segment at an epoch, preventing duplicate anchors on its downstream common suffix;
compute predicted pair spacing, required delay, and rollout outcome at S's common exit gate, but do not use exit-gate ETA to reorder aircraft already physically established on S.

The separation between ordering and evaluation is intentional. Entry-gate ETA decides the planned merge sequence for aircraft still on different inbound segments. Physical progress decides the established sequence after entry. Exit-gate ETA measures whether the established or planned pair will have adequate downstream spacing. If a trailing occupant has an earlier predicted exit than the aircraft physically ahead, record a catch-up or loss-of-separation condition; do not silently reinterpret it as overtaking and reverse the pair.

Rebuild segment queues and anchors at every real decision epoch from current trajectory variants, so committed speed or path-stretch actions affect future ETAs and may change the planned merge order. During common-root action evaluation, freeze the original anchor and outcome cohort across rollout arms to avoid post-treatment pair selection; score each arm using its actual changed crossing times. After committing the selected action, normal dynamic pairing resumes at the next epoch.

Segment membership is static for the current action vocabulary. It is inferred once from the medoid template and inherited by speed and path-stretch variants. A path stretch is a temporary upstream detour that must rejoin the base route before the relevant shared segment; it changes station mappings and entry/exit ETAs, not the ordered SegmentTraversalDefinition. Only a future true reroute or runway reassignment may replace segment membership explicitly.

This allows different clusters to become a pair before a merge when both are committed to the first common outgoing segment. Aircraft that merely land consecutively, cross geometrically, use parallel finals, or never share a directed segment cannot become a pair.

Illustrative pairing example
Consider three arrival clusters whose routes converge onto shared segment S, plus a fourth cluster that remains on nearby parallel segment P before reaching the same runway threshold:

C1 ----\
C2 ----- MERGE ===== shared segment S =====> RW18R
C3 ----/
C5 ---------------- parallel segment P ----> RW18R

Suppose A(C3) is already on S. B(C1), C(C1), and D(C2) remain on inbound segments, with entry-gate ETAs that order them B, C, D. The queue for S is therefore A, B, C, D, and the valid adjacent pairs are A->B, B->C, and C->D. A and D are not a pair because B and C intervene. If E(C5) is the next aircraft in runway-threshold order after D, D->E is still not a pair: D and E do not traverse the same directed segment. E is ordered only against other aircraft committed to P. C and D may become an actionable pair before either reaches the merge because both committed routes already contain S and their predicted entries are adjacent.

For a numerical example at 10:00, let A already occupy S with exit ETA 10:04:30; let C approach the merge with entry/exit ETAs 10:01:20/10:06:50; and let B approach with entry/exit ETAs 10:02:00/10:07:00. The segment queue is A, C, B. Pair spacing at the exit is 140 seconds for A->C and 10 seconds for C->B. If an upstream path stretch delays C's entry ETA to 10:03:20, the next real decision epoch rebuilds the queue as A, B, C and the pairs as A->B and B->C, while all three flights retain the same membership in S.

The repository contains a real instance of this distinction. In the 2026-04-01 10:20-11:20 UTC RW18R window, clusters 1, 2, and 3 use medoids with approximately 13-15 NM of common aligned downstream geometry. Cluster 5's medoid is about 1.48 NM away from that corridor at both 5 NM and 10 NM from the threshold, despite ending at the same runway. Threshold order places ENY4020M1a34c68 (cluster 2, 10:54:59 UTC) immediately before JIA5113M1a67b75 (cluster 5, 11:07:59 UTC), but the segment graph must not emit ENY4020M1a34c68->JIA5113M1a67b75. Within the shared segment, the real order AAL2149M1ad385f (cluster 3), AAL2563M1aa1be0 (cluster 1), NKS1673M1a9cc41 (cluster 1), and ENY4020M1a34c68 (cluster 2) illustrates legitimate cross-cluster adjacency after convergence.

Require explicit flight/runway or segment resources throughout actions, features, outcomes, and rulebook evaluation. Remove every dependency on “the sole configured runway threshold.”
Learning and phase contracts
Remove generate_factorial, FactorialCondition, FactorialScenario, FactorialScenarioBatch, correlation-audit APIs, registered factor pairs, their tests, and factorial benchmark orchestration.
Change Phase0ExperimentRunner to accept TrafficScenarioBatch and permit full multi-aircraft, multi-runway snapshots. Windows without actionable shared-segment pairs contribute to fidelity reporting but are skipped for training with a recorded reason.
Build candidate contexts for all current segment anchors; retain the existing global scheduler so one common-root intervention is selected per decision epoch.
Compute spacing, pressure, and capacity relative to the anchor’s shared segment exit. Path-stretch actions must rejoin before that segment; resource crossings must be remapped after every trajectory splice.
Phase 0 uses pair-only outcome credit (trailer_count=0) while simulating the complete traffic snapshot and retaining whole-snapshot conflict/runway safety gates.
Phase 1 uses the same generator with one configured global traffic scale and restores trailer-aware semi-local credit on the same segment flow.
Add exact categorical rule predicates for airport, runway, segment, and leader/follower cluster scope. Replace numeric cluster_index interval matching so approach-specific specialization is meaningful rather than dependent on arbitrary category ordering.
Rewrite Phase 0’s scientific claim: paired rollouts may establish action benefit on naturalistic demand, but the removed factorial design can no longer claim identification of a deliberately decorrelated causal predicate.
Public Interfaces and Artifacts
Add scenario schema v2 fields for demand-window provenance, scale, observed/target cluster counts, segment traversals, and synthetic-flight donor provenance.
Add feature/rule schema v2 with exact categorical predicates alongside continuous intervals.
Add route-graph and traffic-batch build commands; both emit canonical hashes and reproducible audit reports.
Keep generic ScenarioGenerator.generate() and simulator primitives. Factorial APIs receive no compatibility shim because the design is intentionally removed.
Update Phase 0, architecture, data-flow, and benchmark documentation to describe multi-runway snapshots, segment anchors, and the Phase 0/Phase 1 distinction.
Test and Acceptance Plan
Demand tests:verify half-open 60-minute windows and 20-minute stride at boundaries;
verify scale 1.0 exactly reproduces observed per-window/per-cluster counts and timestamps;
verify 34 × 1.05 = 36, exact thinning, zero-count behavior, deterministic seeds, and input-order independence;
verify each runway total equals the sum of its rounded cluster targets and is never independently rescaled or re-apportioned;
verify sampled speed/altitude always comes jointly from a recorded member of the correct cluster.

Topology tests:differently sampled or collinear representations of the same corridor merge correctly;
nearby parallel finals, brief crossings, opposite-direction overlaps, and short overlaps remain separate;
different inbound clusters mix only on their first common outgoing segment;
unrelated aircraft adjacent at a runway threshold never become a pair;
an aircraft already occupying a segment precedes all committed future entrants, occupants are ordered by physical progress, and future entrants are ordered by entry-gate ETA with stable flight-ID ties;
an apparent exit-ETA inversion between two occupants is reported as catch-up/loss-of-separation without reversing their physical leader-follower order;
every emitted pair passes shared-segment, adjacency, ordering, and no-intervening-flight invariants.

Limited real-data pairing fixture:
check in a compact immutable fixture extracted from artifacts/hailmary/medoid_tracks_6.json, artifacts/hailmary/clusters_6.json, and the 2026-04-01 10:20-11:20 UTC demand window; retain exact source flight IDs, medoid coordinates, terminal-entry times, runway times, cluster assignments, source artifact hashes, and extraction parameters;
include medoids for clusters 0, 1, 2, 3, and 5 and the eight arrivals in that window, rather than loading the full daily dataset during unit tests;
assert that clusters 1, 2, and 3 share the reviewed common downstream segment, while cluster 5 remains on a distinct parallel segment under the registered 0.5-NM, 15-degree, and 5-NM gates;
assert the expected adjacent pairs AAL2149M1ad385f->AAL2563M1aa1be0, AAL2563M1aa1be0->NKS1673M1a9cc41, and NKS1673M1a9cc41->ENY4020M1a34c68 on the common segment;
assert that ENY4020M1a34c68->JIA5113M1a67b75 is absent even though those flights are adjacent in runway-threshold order; assert the independent same-segment pairs JIA5113M1a67b75->JIA5463M1a71aa5 for cluster 5 and JIA5066M1a7cfe6->AAL1286M1ac0dd9 for cluster 0; and assert that the threshold-adjacent JIA5463M1a71aa5->JIA5066M1a7cfe6 cross-segment pair is absent;
derive the expected graph and pair list from a manually reviewed fixture manifest, not by calling the production graph builder during fixture creation inside the test, so the test is not self-confirming;
emit a deterministic diagnostic SVG or GeoJSON overlay showing medoids, inferred segments, merge nodes, flight order, accepted pairs, and rejected threshold-only adjacency. Store the structured expected graph/pair manifest as the CI assertion; use the overlay for human review when the topology algorithm or thresholds change;
require an explicit fixture-version and source-hash update when the real artifacts or registered geometry thresholds change, preventing silent golden-test drift.

Integration tests:run snapshots containing several aircraft and multiple runway resources through actions, forks, rollouts, outcomes, training, and frozen rulebook replay;
verify segment crossings remain correct after speed and path-stretch variants;
verify an upstream path stretch can change the next entry-gate ordering and adjacent pairs while preserving the flight's SegmentTraversalDefinition byte-for-byte;
verify common-root rollout arms retain the same frozen original anchor and cohort even when one arm changes predicted order, then verify the committed real state rebuilds the new queue at the following decision epoch;
verify Phase 0 scores only the bound pair while global safety gates still detect externalities;
verify Phase 1 uses one identical global scale across all windows and clusters and includes downstream trailers.

Leakage control:partition training and held-out demand by source day where available, otherwise by contiguous time blocks;
embargo at least one full window between blocks so no ADS-B arrival appears in both sets;
fit NHPP intensity and clustering fallbacks only from the permitted training corpus.

Phase 0 passes only when:all count/provenance and pair-validity invariants pass;
scale-1 runway, cluster, and terminal-entry distributions match the ADS-B baseline exactly;
the frozen learned policy’s paired block-bootstrap 95% lower confidence bound against permanent no-op is above zero on held-out windows;
whole-snapshot conflict and runway-safety results are not degraded.

Assumptions
“Cluster median” means the repository’s observed-track medoid, avoiding a potentially non-flyable coordinate-wise median.
All runways means all runway labels represented by arrivals with valid terminal-entry reconstruction, including sparse singleton fallbacks.
Pair eligibility is anticipatory: both routes must be committed to the shared segment, but both aircraft need not already occupy it.
Phase 0 always uses scale 1.0; Phase 1 owns the single global scale value.
Repository work uses the local uv environment at ./.venv with dependencies locked by uv.lock, as required by AGENTS.md.
