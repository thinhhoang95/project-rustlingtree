"""Structured smoke and simulator-backed benchmarks for causal learning.

The default invocation runs only dependency-free learning microbenchmarks:

    python benchmarks/hailmary/benchmark_learning.py

List every benchmark contract and any proposed, non-enforcing target:

    python benchmarks/hailmary/benchmark_learning.py --list

Select individual cases with repeated --case arguments. Production contracts
default to a deterministic in-repo representative scenario and execute the
real simulator, action runtime, feature cache, rollout, and trainer paths.
Site-specific production fixtures may override those runners through
``--fixture-module``. No case asserts a machine-specific performance threshold.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import gc
from importlib import import_module
import json
import math
from time import perf_counter
import tracemalloc
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from hailmary.actions.models import ActionLever
from hailmary.config import LearningConfig, StretchConfig
from hailmary.evaluation import simulator_outcome_plan
from hailmary.features import (
    build_current_segment_anchors,
    leader_follower_feature_schema,
    simulator_state_vector,
)
from hailmary.features.schema import FeatureField, FeatureSchema
from hailmary.learning.artifacts import EvaluationSnapshot, TrainingCheckpoint
from hailmary.learning.conditions import Interval, RuleCondition
from hailmary.learning.evolution import mutate_condition, select_parent
from hailmary.learning.matching import AnchorContext
from hailmary.learning.population import Population
from hailmary.learning.rulebook import FrozenRulebookPolicy, SimulatorRulebookPolicy
from hailmary.learning.rules import MutableRule, RuleAction
from hailmary.learning.statistics import OnlineMoments
from hailmary.learning.trainer import CausalTrainer
from hailmary.rollout import three_arm_simulator_rollout
from hailmary.runtime import ActionRuntime, build_action_runtime
from hailmary.scenario import (
    ActionStationDefinition,
    FlightDefinition,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
    SegmentTraversalDefinition,
)
from hailmary.templates import TrajectoryVariant


SCHEMA = FeatureSchema(
    "benchmark.learning.v1",
    (
        FeatureField("x", lower_bound=0.0, upper_bound=10.0),
        FeatureField("y", lower_bound=0.0, upper_bound=1.0),
    ),
)
ACTION = RuleAction(ActionLever.SPEED, "light")
BenchmarkStatus = Literal["ok", "fixture_required"]
MetricValue = int | float
Runner = Callable[["BenchmarkContext"], Mapping[str, MetricValue]]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    description: str
    runner: Runner | None = None
    fixture_requirement: str | None = None
    proposed_targets: Mapping[str, MetricValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("benchmark case name must be a non-empty token")
        if self.runner is None and self.fixture_requirement is None:
            raise ValueError(
                "a benchmark case needs a runner or an external fixture requirement"
            )
        if (
            self.fixture_requirement is not None
            and not self.fixture_requirement.strip()
        ):
            raise ValueError("fixture requirement cannot be blank")
        targets: dict[str, MetricValue] = {}
        for name, value in self.proposed_targets.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("proposed target names must be non-empty strings")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("proposed targets must be finite numbers")
            targets[name] = value
        object.__setattr__(
            self,
            "proposed_targets",
            MappingProxyType(dict(sorted(targets.items()))),
        )

    @property
    def default_status(self) -> BenchmarkStatus:
        return "ok" if self.runner is not None else "fixture_required"

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "default_status": self.default_status,
            "fixture_requirement": self.fixture_requirement,
            "proposed_targets": dict(self.proposed_targets),
        }


@dataclass(slots=True)
class BenchmarkContext:
    population_size: int
    iterations: int
    population: Population
    rng: np.random.Generator
    condition: RuleCondition
    current: Mapping[str, float]
    checkpoint: TrainingCheckpoint
    simulator_fixtures: dict[int, "SimulatorFixture"]


@dataclass(frozen=True, slots=True)
class SimulatorFixture:
    """One real, deterministic decision state shared by measured operations."""

    runtime: ActionRuntime
    simulator: Any
    event_batch: Any
    anchor: Any
    candidates: tuple[Any, ...]
    outcome_plan: Any
    continuation_policy: SimulatorRulebookPolicy
    active_flights: int


def _population(size: int) -> Population:
    rules: list[MutableRule] = []
    for index in range(size):
        lower = 9.0 * index / max(size, 1)
        upper = min(10.0, lower + 1.0)
        rules.append(
            MutableRule(
                condition=RuleCondition(
                    "leader_follower",
                    SCHEMA.schema_hash,
                    {"x": Interval(lower, upper)},
                ),
                action=ACTION,
                evolution=OnlineMoments(n=20, mean=1.0 + index / size, m2=1.9),
                deployment=OnlineMoments(n=20, mean=0.5, m2=1.9),
                provenance={"benchmark_index": index},
            )
        )
    return Population(rules, max_size=max(size * 2, 1))


def _context(population_size: int, iterations: int) -> BenchmarkContext:
    if population_size < 2:
        raise ValueError("population_size must be at least two")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    population = _population(population_size)
    rng = np.random.default_rng(17)
    condition = RuleCondition(
        "leader_follower",
        SCHEMA.schema_hash,
        {"x": Interval(2.0, 8.0)},
    )
    current = MappingProxyType({"x": 5.0, "y": 0.5})
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA.schema_hash,
        certification_generation=1,
    )
    evaluation = EvaluationSnapshot(
        rulebook,
        config_hash="benchmark-config-v1",
    )
    checkpoint = TrainingCheckpoint(
        scenario_definition_hash="benchmark-scenario",
        simulator_snapshot={
            "definition_hash": "benchmark-scenario",
            "state_id": "benchmark-root",
        },
        population=population,
        evaluation_snapshot=evaluation,
        epoch=1,
        rng_state=rng,
        exploration_counts={"global": iterations},
    )
    return BenchmarkContext(
        population_size=population_size,
        iterations=iterations,
        population=population,
        rng=rng,
        condition=condition,
        current=current,
        checkpoint=checkpoint,
        simulator_fixtures={},
    )


def _average_seconds(operation: Callable[[], object], iterations: int) -> float:
    started = perf_counter()
    for _ in range(iterations):
        operation()
    return (perf_counter() - started) / iterations


def _parent_selection(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    seconds = _average_seconds(
        lambda: select_parent(
            context.population.rules,
            niche=("leader_follower", ACTION),
            rng=context.rng,
            min_experience=2,
        ),
        context.iterations,
    )
    return {"average_us": seconds * 1_000_000.0}


def _condition_mutation(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    seconds = _average_seconds(
        lambda: mutate_condition(
            context.condition,
            SCHEMA,
            rng=context.rng,
            current_state=context.current,
        ),
        context.iterations,
    )
    return {"average_us": seconds * 1_000_000.0}


def _checkpoint_serialization(
    context: BenchmarkContext,
) -> Mapping[str, MetricValue]:
    measured_iterations = max(1, context.iterations // 10)
    seconds = _average_seconds(
        context.checkpoint.to_json,
        measured_iterations,
    )
    return {
        "average_ms": seconds * 1_000.0,
        "measured_iterations": measured_iterations,
    }


def _representative_variant() -> TrajectoryVariant:
    """Build a physical constant-speed arrival suitable for every action lever."""

    station_count = 101
    stations_m = np.linspace(0.0, 100_000.0, station_count, dtype=np.float64)
    speed_mps = np.full(station_count, 100.0, dtype=np.float64)
    return TrajectoryVariant.from_kinematic_profile(
        template_id="BENCHMARK_TEMPLATE",
        cluster_id="BENCHMARK_CLUSTER",
        s_m=stations_m,
        east_m=stations_m,
        north_m=np.zeros(station_count, dtype=np.float64),
        altitude_m=np.zeros(station_count, dtype=np.float64),
        cas_mps=speed_mps,
        tas_mps=speed_mps,
        ground_speed_mps=speed_mps,
        command_cas_mps=speed_mps,
        reference_command_cas_mps=speed_mps,
        lower_cas_mps=np.full(station_count, 70.0, dtype=np.float64),
        upper_cas_mps=np.full(station_count, 130.0, dtype=np.float64),
        threshold_resource_id="RWY",
    )


def _representative_definition(flight_count: int) -> ScenarioDefinition:
    if flight_count < 2:
        raise ValueError("representative benchmark needs at least two flights")
    variant = _representative_variant()
    target_stations = (
        ActionStationDefinition(0, 90_000.0, "speed"),
        ActionStationDefinition(0, 90_000.0, "path_stretch"),
    )
    crossings = (
        ResourceCrossingDefinition("FINAL:entry", 50_000.0),
        ResourceCrossingDefinition("RWY", 0.0),
    )
    traversals = (
        SegmentTraversalDefinition(
            0, "FINAL", "FINAL:entry", "RWY", 50_000.0, 0.0
        ),
    )
    flights = tuple(
        FlightDefinition(
            flight_id=f"F{index:03d}",
            release_time_s=0.0,
            baseline_variant_id=variant.variant_id,
            cluster_id="BENCHMARK_CLUSTER",
            action_stations=target_stations if index == 1 else (),
            resource_crossings=crossings,
            segment_traversals=traversals,
        )
        for index in range(flight_count)
    )
    return ScenarioDefinition(
        scenario_id=f"LEARNING_BENCHMARK_{flight_count}",
        seed=17,
        flights=flights,
        resources=(
            ResourceDefinition("RWY", required_interval_s=90.0),
            ResourceDefinition("FINAL:entry", kind="segment_entry"),
        ),
        variants=(variant,),
    )


def _active_flight_count(simulator: Any) -> int:
    return sum(
        str(getattr(flight.lifecycle, "value", flight.lifecycle)).lower() == "active"
        for flight in simulator.state.flights
    )


def _simulator_fixture(
    context: BenchmarkContext,
    *,
    flight_count: int,
) -> SimulatorFixture:
    cached = context.simulator_fixtures.get(flight_count)
    if cached is not None:
        return cached

    runtime = build_action_runtime(
        stretch_config=StretchConfig(max_turn_deg=120.0),
    )
    simulator = runtime.create_simulator(_representative_definition(flight_count))
    event_batch = None
    while True:
        event_batch = simulator.advance_next()
        if event_batch is None:
            raise RuntimeError("benchmark scenario never reached its action station")
        if any(
            event.flight_id == "F001"
            and str(getattr(event.kind, "value", event.kind))
            == "ACTION_STATION_CROSSED"
            and event.payload_dict.get("station_type") == "speed"
            for event in event_batch.events
        ):
            break

    active_flights = _active_flight_count(simulator)
    if active_flights != flight_count:
        raise RuntimeError(
            "representative decision state does not contain the requested "
            "number of active flights"
        )
    anchor = next(
        item
        for item in build_current_segment_anchors(simulator).leader_follower
        if item.follower_id == "F001"
    )
    candidates = runtime.catalog.enumerate_for_batch(
        simulator,
        event_batch,
        anchor_id=anchor.anchor_id,
        bound_flight_id=anchor.follower_id,
        resource_id=anchor.resource_id,
        segment_id=anchor.segment_id,
    )
    identities = tuple((item.lever.value, item.band) for item in candidates)
    expected = (
        ("no_op", "no_op"),
        ("speed", "light"),
        ("speed", "medium"),
        ("speed", "heavy"),
        ("path_stretch", "oracle_short_medium_long"),
    )
    if identities != expected:
        raise RuntimeError("representative fixture did not expose the five actions")

    schema = leader_follower_feature_schema()
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=schema.schema_hash,
        action_configuration=runtime.vocabulary.payload,
    )
    continuation_policy = SimulatorRulebookPolicy(
        rulebook,
        template_config=runtime.template_config,
        schema=schema,
        runtime_configuration_hash=runtime.runtime_configuration_hash,
    )
    fixture = SimulatorFixture(
        runtime=runtime,
        simulator=simulator,
        event_batch=event_batch,
        anchor=anchor,
        candidates=tuple(candidates),
        outcome_plan=simulator_outcome_plan(simulator, anchor),
        continuation_policy=continuation_policy,
        active_flights=active_flights,
    )
    context.simulator_fixtures[flight_count] = fixture
    return fixture


def _candidate(
    fixture: SimulatorFixture,
    lever: ActionLever,
    band: str | None = None,
) -> Any:
    return next(
        candidate
        for candidate in fixture.candidates
        if candidate.lever is lever and (band is None or candidate.band == band)
    )


def _bounded_iterations(context: BenchmarkContext, maximum: int) -> int:
    return max(1, min(context.iterations, maximum))


def _fork_latency(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    fixture = _simulator_fixture(context, flight_count=100)
    parent_hash = fixture.simulator.dynamic_content_hash
    started = perf_counter()
    for index in range(context.iterations):
        child = fixture.simulator.fork(label=f"benchmark-fork:{index}")
        if child.dynamic_content_hash != parent_hash:
            raise RuntimeError("a benchmark fork did not preserve parent content")
    elapsed = perf_counter() - started
    return {
        "average_ms": elapsed * 1_000.0 / context.iterations,
        "measured_iterations": context.iterations,
        "active_flights": fixture.active_flights,
    }


def _feature_candidates(fixture: SimulatorFixture) -> tuple[Any, ...]:
    """Keep cache timing separate from the nested stretch-oracle benchmark.

    The real no-op and three physical speed candidates still exercise forked
    reachability. The exact stretch action is timed by ``three_arm_stretch_oracle``.
    """

    return tuple(
        candidate
        for candidate in fixture.candidates
        if candidate.lever is not ActionLever.PATH_STRETCH
    )


def _feature_vector(
    fixture: SimulatorFixture,
    *,
    cache_namespace: str | None,
) -> Any:
    return simulator_state_vector(
        fixture.simulator,
        fixture.anchor,
        action_candidates=_feature_candidates(fixture),
        action_applier=fixture.runtime.action_applier,
        reachability_cache_namespace=cache_namespace,
    )


def _feature_cache_hit(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    fixture = _simulator_fixture(context, flight_count=100)
    namespace = "benchmark-warmed-feature-vector"
    _feature_vector(fixture, cache_namespace=namespace)
    measured_iterations = _bounded_iterations(context, 50)
    seconds = _average_seconds(
        lambda: _feature_vector(fixture, cache_namespace=namespace),
        measured_iterations,
    )
    return {
        "average_ms": seconds * 1_000.0,
        "measured_iterations": measured_iterations,
        "active_flights": fixture.active_flights,
        "evaluated_candidates": len(_feature_candidates(fixture)),
        "stretch_oracle_measured_separately": 1,
    }


def _feature_cache_miss(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    fixture = _simulator_fixture(context, flight_count=100)
    measured_iterations = _bounded_iterations(context, 5)
    seconds = _average_seconds(
        lambda: _feature_vector(fixture, cache_namespace=None),
        measured_iterations,
    )
    return {
        "average_ms": seconds * 1_000.0,
        "measured_iterations": measured_iterations,
        "active_flights": fixture.active_flights,
        "evaluated_candidates": len(_feature_candidates(fixture)),
        "stretch_oracle_measured_separately": 1,
    }


def _three_arm(
    context: BenchmarkContext,
    *,
    selected_lever: ActionLever,
    selected_band: str,
    contender_lever: ActionLever,
    contender_band: str,
    maximum_iterations: int,
) -> Mapping[str, MetricValue]:
    fixture = _simulator_fixture(context, flight_count=5)
    selected = _candidate(fixture, selected_lever, selected_band)
    contender = _candidate(fixture, contender_lever, contender_band)
    no_op = _candidate(fixture, ActionLever.NO_OP, "no_op")
    measured_iterations = _bounded_iterations(context, maximum_iterations)
    latest = None
    started = perf_counter()
    for _ in range(measured_iterations):
        latest = three_arm_simulator_rollout(
            fixture.simulator,
            selected_action=selected,
            contender_action=contender,
            no_op_action=no_op,
            frozen_policy=fixture.continuation_policy,
            outcome_plan=fixture.outcome_plan,
        )
    elapsed = perf_counter() - started
    if (
        latest is None
        or latest.parent_dynamic_content_hash != fixture.simulator.dynamic_content_hash
    ):
        raise RuntimeError("native three-arm benchmark did not preserve its parent")
    return {
        "average_ms": elapsed * 1_000.0 / measured_iterations,
        "measured_iterations": measured_iterations,
        "active_flights": fixture.active_flights,
        "arms": 3,
        "horizon_s": float(fixture.outcome_plan.horizon_s),
        "delta_rival": float(latest.delta_rival),
    }


def _three_arm_no_op(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    return _three_arm(
        context,
        selected_lever=ActionLever.NO_OP,
        selected_band="no_op",
        contender_lever=ActionLever.SPEED,
        contender_band="light",
        maximum_iterations=5,
    )


def _three_arm_speed(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    return _three_arm(
        context,
        selected_lever=ActionLever.SPEED,
        selected_band="light",
        contender_lever=ActionLever.NO_OP,
        contender_band="no_op",
        maximum_iterations=5,
    )


def _three_arm_stretch(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    return _three_arm(
        context,
        selected_lever=ActionLever.PATH_STRETCH,
        selected_band="oracle_short_medium_long",
        contender_lever=ActionLever.SPEED,
        contender_band="light",
        maximum_iterations=2,
    )


def _training_epoch(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    fixture = _simulator_fixture(context, flight_count=5)
    schema = leader_follower_feature_schema()
    config = LearningConfig(
        population_limit=100,
        ga_interval=10_000,
        ga_min_experience=2,
        certification_interval=10_000,
        action_min_noop_samples=2,
        veto_min_rival_samples=2,
        contender_min_samples=2,
        exploration_coverage_floor=1,
        base_exploration_rate=0.0,
        random_seed=23,
    )
    seed_rule = MutableRule(
        condition=RuleCondition("leader_follower", schema.schema_hash, {}),
        action=RuleAction(ActionLever.SPEED, "light"),
        evolution=OnlineMoments(n=2, mean=1.0, m2=0.0),
        deployment=OnlineMoments(n=2, mean=1.0, m2=0.0),
        provenance={"kind": "benchmark_seed"},
    )
    population = Population((seed_rule,), max_size=config.population_limit)
    seed_trainer = CausalTrainer(
        fixture.runtime,
        population=population,
        config=config,
        schema=schema,
    )
    seed_trainer._publish_if_due(epoch=0)
    evaluation = seed_trainer.evaluation_snapshot
    trainer = CausalTrainer(
        fixture.runtime,
        population=population,
        config=config,
        schema=schema,
        evaluation_snapshot=evaluation,
    )
    vector = simulator_state_vector(
        fixture.simulator,
        fixture.anchor,
        action_candidates=fixture.candidates,
        action_applier=fixture.runtime.action_applier,
        template_config=fixture.runtime.template_config,
        schema=schema,
    )
    training_context = AnchorContext(
        role_type="leader_follower",
        schema_hash=schema.schema_hash,
        anchor_id=fixture.anchor.anchor_id,
        vector=vector,
        candidates=fixture.candidates,
    )
    # Complete, rather than merely select, the region's coverage experiments
    # so timed epochs measure frozen-rulebook exploitation rather than initial
    # allocation. A selection alone increments visits but leaves the action's
    # completed-experiment count at zero.
    speed_action = RuleAction(ActionLever.SPEED, "light")
    cell = trainer.scheduler.cell_for(training_context)
    while trainer.scheduler.visit_count(
        cell
    ) < config.exploration_coverage_floor or any(
        trainer.scheduler.experiment_count(cell, action) == 0
        for action in training_context.candidate_actions
    ):
        decision = trainer.scheduler.choose(
            training_context,
            exploit_action=speed_action,
        )
        trainer.scheduler.record_experiment(decision)
    root_snapshot = fixture.simulator.snapshot()
    measured_iterations = _bounded_iterations(context, 2)
    committed = 0
    started = perf_counter()
    for _ in range(measured_iterations):
        root = fixture.runtime.resume_simulator(
            fixture.simulator.definition,
            root_snapshot,
        )
        result = trainer.process_epoch(root, fixture.event_batch)
        if not result.committed:
            raise RuntimeError("complete training benchmark did not commit Arm A")
        if result.trace.exploration_reason != "exploit":
            raise RuntimeError(
                "complete training benchmark did not measure rulebook exploitation"
            )
        committed += 1
    elapsed = perf_counter() - started
    return {
        "epochs_per_second": committed / elapsed,
        "average_ms": elapsed * 1_000.0 / committed,
        "measured_iterations": measured_iterations,
        "committed_epochs": committed,
        "exploitation_epochs": committed,
        "active_flights": fixture.active_flights,
        "scenario_count": 1,
    }


def _temporary_fork_memory(context: BenchmarkContext) -> Mapping[str, MetricValue]:
    fixture = _simulator_fixture(context, flight_count=100)
    measured_iterations = context.iterations
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    gc.collect()
    baseline_bytes = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    started = perf_counter()
    for index in range(measured_iterations):
        child = fixture.simulator.fork(label=f"benchmark-memory:{index}")
        child.run_until(fixture.simulator.state.sim_time_s + 1.0)
        del child
        if (index + 1) % 25 == 0:
            gc.collect()
    elapsed = perf_counter() - started
    gc.collect()
    retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()
    return {
        "peak_growth_bytes": max(0, peak_bytes - baseline_bytes),
        "retained_growth_bytes": max(0, retained_bytes - baseline_bytes),
        "average_us": elapsed * 1_000_000.0 / measured_iterations,
        "temporary_forks": measured_iterations,
        "active_flights": fixture.active_flights,
    }


_CASES = (
    BenchmarkCase(
        "parent_selection",
        "Seeded evolution-ledger LCB parent selection in one exact niche.",
        runner=_parent_selection,
    ),
    BenchmarkCase(
        "condition_mutation",
        "Bounded interval mutation while preserving a current-state match.",
        runner=_condition_mutation,
    ),
    BenchmarkCase(
        "checkpoint_serialization",
        "Canonical JSON serialization of a detached training checkpoint.",
        runner=_checkpoint_serialization,
    ),
    BenchmarkCase(
        "fork_latency_100_flights",
        "Simulator fork latency with exactly 100 active flights.",
        runner=_fork_latency,
        fixture_requirement=(
            "optional site-specific root Simulator with exactly 100 active flights "
            "and its production action runtime attached"
        ),
        proposed_targets={"average_ms_lt": 1.0},
    ),
    BenchmarkCase(
        "feature_vector_cache_hit",
        (
            "Leader-follower vector latency from a warmed speed-reachability "
            "cache; nested stretch-oracle cost is benchmarked separately."
        ),
        runner=_feature_cache_hit,
        fixture_requirement=(
            "optional site-specific 100-flight state and anchor whose production "
            "feature cache is explicitly warmed before measurement"
        ),
        proposed_targets={"average_ms_lt": 2.0},
    ),
    BenchmarkCase(
        "feature_vector_cache_miss",
        (
            "Leader-follower vector latency after speed-reachability cache "
            "invalidation; nested stretch-oracle cost is benchmarked separately."
        ),
        runner=_feature_cache_miss,
        fixture_requirement=(
            "optional site-specific 100-flight state and anchor with a fresh or "
            "explicitly invalidated production feature cache"
        ),
    ),
    BenchmarkCase(
        "three_arm_no_op",
        "Native selected/contender/no-op rollout when selected is no-op.",
        runner=_three_arm_no_op,
        fixture_requirement=(
            "optional site-specific scenario root, frozen rulebook, actions, "
            "outcome plan, and fixed rollout horizon"
        ),
    ),
    BenchmarkCase(
        "three_arm_speed",
        "Native selected/contender/no-op rollout with a physical speed band.",
        runner=_three_arm_speed,
        fixture_requirement=(
            "optional site-specific root, frozen rulebook, feasible speed/rival/"
            "no-op candidates, outcome plan, and fixed horizon"
        ),
    ),
    BenchmarkCase(
        "three_arm_stretch_oracle",
        "Three-arm rollout whose selected path stretch runs the inner oracle.",
        runner=_three_arm_stretch,
        fixture_requirement=(
            "optional site-specific ActionRuntime with PathStretchRealizer and "
            "short/medium/long oracle geometries plus the frozen rollout inputs"
        ),
    ),
    BenchmarkCase(
        "complete_training_epochs_per_second",
        "End-to-end committed training decisions per second.",
        runner=_training_epoch,
        fixture_requirement=(
            "optional site-specific trainer, reproducible decision source, "
            "scenario suite, frozen action runtime, and three-arm horizon"
        ),
    ),
    BenchmarkCase(
        "temporary_fork_memory_growth",
        "Peak and retained memory across repeated temporary simulator forks.",
        runner=_temporary_fork_memory,
        fixture_requirement=(
            "optional site-specific 100-flight root and fork/rollout workload; "
            "measurement must force collection between batches"
        ),
    ),
)
BENCHMARK_CASES: Mapping[str, BenchmarkCase] = MappingProxyType(
    {case.name: case for case in _CASES}
)
DEFAULT_SMOKE_CASES = (
    "parent_selection",
    "condition_mutation",
    "checkpoint_serialization",
)


def list_cases() -> tuple[dict[str, Any], ...]:
    return tuple(case.metadata() for case in BENCHMARK_CASES.values())


def run_cases(
    names: Sequence[str],
    *,
    population_size: int = 200,
    iterations: int = 500,
    fixture_runners: Mapping[str, Runner] | None = None,
) -> tuple[dict[str, Any], ...]:
    if isinstance(names, (str, bytes)):
        raise TypeError("benchmark case names must be a sequence")
    unknown = tuple(name for name in names if name not in BENCHMARK_CASES)
    if unknown:
        raise ValueError(f"unknown benchmark cases: {unknown}")
    injected = {} if fixture_runners is None else dict(fixture_runners)
    unknown_injected = tuple(sorted(set(injected) - set(BENCHMARK_CASES)))
    if unknown_injected:
        raise ValueError(f"unknown fixture benchmark runners: {unknown_injected}")
    if any(not callable(runner) for runner in injected.values()):
        raise TypeError("fixture benchmark runners must be callable")

    runnable = any(
        BENCHMARK_CASES[name].runner is not None or name in injected for name in names
    )
    context = _context(population_size, iterations) if runnable else None
    results: list[dict[str, Any]] = []
    for name in names:
        case = BENCHMARK_CASES[name]
        runner = injected.get(name, case.runner)
        if runner is None:
            results.append(
                {
                    "name": case.name,
                    "status": "fixture_required",
                    "description": case.description,
                    "fixture_requirement": case.fixture_requirement,
                    "proposed_targets": dict(case.proposed_targets),
                    "metrics": {},
                }
            )
            continue
        assert context is not None
        metrics = dict(runner(context))
        if not metrics:
            raise ValueError(f"benchmark {name!r} returned no metrics")
        for metric_name, metric_value in metrics.items():
            if not isinstance(metric_name, str) or not metric_name.strip():
                raise ValueError("benchmark metric names must be non-empty strings")
            if (
                isinstance(metric_value, bool)
                or not isinstance(metric_value, (int, float))
                or not math.isfinite(float(metric_value))
            ):
                raise ValueError(
                    f"benchmark metric {metric_name!r} must be a finite number"
                )
        results.append(
            {
                "name": case.name,
                "status": "ok",
                "description": case.description,
                "fixture_requirement": case.fixture_requirement,
                "proposed_targets": dict(case.proposed_targets),
                "metrics": metrics,
            }
        )
    return tuple(results)


def load_fixture_runners(module_name: str) -> Mapping[str, Runner]:
    """Load fixture-backed benchmark closures from ``BENCHMARK_RUNNERS``."""

    name = str(module_name).strip()
    if not name:
        raise ValueError("fixture module name cannot be empty")
    runners = getattr(import_module(name), "BENCHMARK_RUNNERS", None)
    if not isinstance(runners, Mapping):
        raise TypeError("fixture module must expose a BENCHMARK_RUNNERS mapping")
    return dict(runners)


def run_benchmark(
    *,
    population_size: int = 200,
    iterations: int = 500,
) -> dict[str, float | int]:
    """Backward-compatible aggregate for the default runnable smoke cases."""

    results = {
        result["name"]: result["metrics"]
        for result in run_cases(
            DEFAULT_SMOKE_CASES,
            population_size=population_size,
            iterations=iterations,
        )
    }
    return {
        "population_size": population_size,
        "iterations": iterations,
        "parent_selection_us": float(results["parent_selection"]["average_us"]),
        "condition_mutation_us": float(results["condition_mutation"]["average_us"]),
        "checkpoint_serialization_ms": float(
            results["checkpoint_serialization"]["average_ms"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(BENCHMARK_CASES),
        help="benchmark case to run; repeat to select multiple",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run or report every registered benchmark case",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list benchmark contracts and fixture requirements",
    )
    parser.add_argument(
        "--fixture-module",
        help="import BENCHMARK_RUNNERS from this module for simulator-backed cases",
    )
    arguments = parser.parse_args()
    fixture_runners = (
        {}
        if arguments.fixture_module is None
        else load_fixture_runners(arguments.fixture_module)
    )

    if arguments.list:
        output: Any = {"cases": list_cases()}
    else:
        selected = (
            tuple(BENCHMARK_CASES)
            if arguments.all
            else tuple(arguments.case or DEFAULT_SMOKE_CASES)
        )
        output = {
            "population_size": arguments.population_size,
            "iterations": arguments.iterations,
            "cases": run_cases(
                selected,
                population_size=arguments.population_size,
                iterations=arguments.iterations,
                fixture_runners=fixture_runners,
            ),
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
