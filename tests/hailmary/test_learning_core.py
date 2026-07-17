from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from hailmary.actions import action_vocabulary
from hailmary.config import LearningConfig
from hailmary.learning import (
    CertificationThresholds,
    FrozenActionRule,
    FrozenVetoRule,
    Interval,
    MutableRule,
    OnlineMoments,
    Population,
    RuleAction,
    RuleCondition,
    certify_rule,
)


SCHEMA_HASH = "schema-learning-core-v1"


def _condition(**intervals: tuple[float | None, float | None]) -> RuleCondition:
    return RuleCondition("leader_follower", SCHEMA_HASH, intervals)


def test_online_moments_is_exact_welford_and_round_trips() -> None:
    moments = OnlineMoments().extend((1.0, 2.0, 3.0, 4.0))

    assert moments.n == 4
    assert moments.mean == pytest.approx(2.5)
    assert moments.m2 == pytest.approx(5.0)
    assert moments.sample_variance == pytest.approx(5.0 / 3.0)
    assert moments.standard_error(1.0e-6) == pytest.approx(math.sqrt((5.0 / 3.0) / 4.0))
    assert moments.lcb(1.96, 1.0e-6) == pytest.approx(
        2.5 - 1.96 * math.sqrt((5.0 / 3.0) / 4.0)
    )
    assert OnlineMoments.from_dict(moments.to_dict()) == moments


def test_sparse_moments_have_finite_conservative_uncertainty() -> None:
    empty = OnlineMoments()
    assert empty.precision(1.0e-6) == 0.0
    assert math.isfinite(empty.standard_error(1.0e-6))
    assert empty.lcb(1.96, 1.0e-6) < 0.0

    singleton = OnlineMoments().update(3.0)
    assert singleton.sample_variance == 0.0
    assert 0.0 < singleton.precision(1.0e-6) < math.inf
    assert math.isfinite(singleton.lcb(1.96, 1.0e-6))

    with pytest.raises(ValueError, match="finite"):
        singleton.update(float("nan"))
    with pytest.raises(ValueError, match="positive"):
        singleton.precision(0.0)


def test_conditions_match_contain_mutate_and_serialize_none_bounds() -> None:
    general = _condition(x=(None, 10.0), y=(0.0, 1.0))
    specific = _condition(x=(-2.0, 4.0), y=(0.2, 0.8), z=(5.0, 6.0))

    assert general.matches(
        {"x": 4.0, "y": 0.5},
        role_type="leader_follower",
        schema_hash=SCHEMA_HASH,
    )
    assert not general.matches(
        {"x": 11.0, "y": 0.5},
        schema_hash=SCHEMA_HASH,
    )
    assert not general.matches(
        {"x": 4.0, "y": 0.5},
        schema_hash="different-schema",
    )
    assert general.contains(specific)
    assert not specific.contains(general)
    assert general.shifted("y", 1.0).intervals["y"] == Interval(1.0, 2.0)
    assert general.widened("y", 0.5).intervals["y"] == Interval(-0.5, 1.5)
    assert general.without_interval("x").contains(general)

    payload = general.to_dict()
    assert payload["intervals"]["x"] == [None, 10.0]
    assert RuleCondition.from_dict(payload) == general
    with pytest.raises(TypeError):
        general.intervals["x"] = Interval(0.0, 1.0)  # type: ignore[index]


def test_rule_action_delegates_to_catalog_vocabulary() -> None:
    vocabulary = action_vocabulary()
    actions = tuple(
        RuleAction(identity.lever, identity.band) for identity in vocabulary.identities
    )

    assert tuple(action.key for action in actions) == tuple(
        f"{identity.lever.value}/{identity.band}" for identity in vocabulary.identities
    )
    with pytest.raises(ValueError, match="ActionCatalog vocabulary"):
        RuleAction("speed", "not-a-catalog-band")


def test_mutable_rule_population_matching_merging_and_bounds() -> None:
    condition = _condition(x=(0.0, 1.0))
    first = MutableRule(condition, RuleAction("speed", "light"))
    first.update_rival(1.0)
    first.update_noop(0.5)
    duplicate = MutableRule(condition, RuleAction("speed", "light"))
    duplicate.update_rival(3.0)
    duplicate.update_noop(1.5)

    population = Population(max_size=2)
    assert population.add(first) is first
    assert population.add(duplicate) is first
    assert population.macro_size == 1
    assert population.total_numerosity == 2
    assert first.evolution.n == 2
    assert first.evolution.mean == pytest.approx(2.0)
    assert population.match(
        {"x": 0.5},
        role_type="leader_follower",
        schema_hash=SCHEMA_HASH,
    ) == (first,)
    assert (
        population.match(
            {"x": 2.0},
            role_type="leader_follower",
            schema_hash=SCHEMA_HASH,
        )
        == ()
    )

    with pytest.raises(OverflowError, match="capacity"):
        population.add(
            MutableRule(_condition(x=(2.0, 3.0)), RuleAction("speed", "heavy"))
        )
    restored = Population.from_dict(population.to_dict())
    assert restored.to_dict() == population.to_dict()


def test_population_restore_preserves_equivalent_distinct_macro_rules() -> None:
    condition = _condition(x=(0.0, 1.0))
    first = MutableRule(
        condition,
        RuleAction("speed", "light"),
        provenance={"lineage": "first"},
    )
    second = MutableRule(
        condition,
        RuleAction("speed", "light"),
        provenance={"lineage": "second"},
    )
    first.evolution.update(1.0)
    second.evolution.update(3.0)

    population = Population(max_size=4)
    population.add(first, merge_equivalent=False)
    population.add(second, merge_equivalent=False)
    payload = population.to_dict()
    restored = Population.from_dict(payload)

    assert restored.macro_size == 2
    assert {rule.rule_id for rule in restored.rules} == {
        first.rule_id,
        second.rule_id,
    }
    assert restored.to_dict() == payload


def test_mutable_rule_provenance_is_detached_and_recursively_immutable() -> None:
    source = {
        "lineage": {
            "parent_ids": ["parent-a", "parent-b"],
            "audit": {"scores": [1.0, 2.0]},
        }
    }
    rule = MutableRule(
        _condition(x=(0.0, 1.0)),
        RuleAction("speed", "light"),
        provenance=source,
    )
    lineage = rule.provenance["lineage"]

    source["lineage"]["parent_ids"][0] = "mutated"
    source["lineage"]["audit"]["scores"].append(3.0)
    assert lineage["parent_ids"] == ("parent-a", "parent-b")
    assert lineage["audit"]["scores"] == (1.0, 2.0)

    with pytest.raises(TypeError):
        lineage["new_field"] = "forbidden"
    with pytest.raises(TypeError):
        lineage["audit"]["scores"][0] = 99.0


def test_certification_uses_dual_ledgers_and_detaches_snapshots() -> None:
    action_rule = MutableRule(
        _condition(x=(0.0, 1.0)),
        RuleAction("speed", "medium"),
    )
    action_rule.deployment.extend([0.5] * 30)
    action_rule.evolution.extend([0.2, 0.3, 0.4])
    thresholds = CertificationThresholds.from_config(LearningConfig())

    certified_action = certify_rule(action_rule, thresholds)
    assert isinstance(certified_action, FrozenActionRule)
    assert certified_action.w == pytest.approx(0.5)
    assert certified_action.rival_samples == 3
    frozen_rival_lcb = certified_action.rival_lcb
    frozen_condition = certified_action.condition.to_dict()

    action_rule.deployment.update(100.0)
    action_rule.evolution.update(100.0)
    action_rule.condition = action_rule.condition.with_interval("x", (5.0, 6.0))
    assert certified_action.w == pytest.approx(0.5)
    assert certified_action.rival_lcb == frozen_rival_lcb
    assert certified_action.condition.to_dict() == frozen_condition

    veto_rule = MutableRule(_condition(x=(0.0, 1.0)), RuleAction.no_op())
    veto_rule.evolution.extend([0.5] * 60)
    assert isinstance(certify_rule(veto_rule, thresholds), FrozenVetoRule)

    under_sampled = MutableRule(
        _condition(x=(0.0, 1.0)),
        RuleAction("path_stretch", "oracle_short_medium_long"),
    )
    under_sampled.deployment.extend([1.0] * 29)
    assert certify_rule(under_sampled, thresholds) is None


def _threshold_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "action_min_noop_samples": 30,
        "veto_min_rival_samples": 60,
        "contender_min_samples": 2,
        "action_lcb_z": 1.96,
        "veto_lcb_z": 1.96,
        "contender_lcb_z": 1.96,
        "variance_floor": 1.0e-6,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_certification_thresholds_require_exact_integer_sample_counts() -> None:
    for field_name in (
        "action_min_noop_samples",
        "veto_min_rival_samples",
        "contender_min_samples",
    ):
        for invalid_value in (True, 3.0, 3.5, "3"):
            with pytest.raises(ValueError, match="integer"):
                CertificationThresholds(**{field_name: invalid_value})
            with pytest.raises(ValueError, match="integer"):
                CertificationThresholds.from_config(
                    _threshold_config(**{field_name: invalid_value})
                )


def test_certification_thresholds_require_finite_non_bool_numeric_values() -> None:
    numeric_fields = (
        "action_lcb_z",
        "veto_lcb_z",
        "contender_lcb_z",
        "variance_floor",
    )
    for field_name in numeric_fields:
        for invalid_value in (True, "1.0", float("inf"), float("nan")):
            with pytest.raises(ValueError, match="numeric"):
                CertificationThresholds(**{field_name: invalid_value})
            with pytest.raises(ValueError, match="numeric"):
                CertificationThresholds.from_config(
                    _threshold_config(**{field_name: invalid_value})
                )

    for field_name in ("action_lcb_z", "veto_lcb_z", "contender_lcb_z"):
        with pytest.raises(ValueError, match="non-negative"):
            CertificationThresholds(**{field_name: -1.0})
    with pytest.raises(ValueError, match="positive"):
        CertificationThresholds(variance_floor=0.0)

    numeric_integers = CertificationThresholds(
        action_lcb_z=1,
        veto_lcb_z=1,
        contender_lcb_z=1,
        variance_floor=1,
    )
    assert numeric_integers.action_lcb_z == 1.0
    assert numeric_integers.variance_floor == 1.0


def test_frozen_action_rule_rejects_coerced_or_nonfinite_statistics() -> None:
    canonical = FrozenActionRule(
        source_rule_id="rule-a",
        condition=_condition(x=(0.0, 1.0)),
        action=RuleAction("speed", "light"),
        w=1.0,
        precision=2.0,
        rival_lcb=0.5,
        rival_samples=3,
    ).to_dict()

    for field_name in ("w", "precision", "rival_lcb"):
        for invalid_value in (True, "1.0", float("inf"), float("nan")):
            with pytest.raises(ValueError, match="numeric"):
                FrozenActionRule.from_dict({**canonical, field_name: invalid_value})
    with pytest.raises(ValueError, match="positive"):
        FrozenActionRule.from_dict({**canonical, "precision": 0.0})

    for invalid_samples in (True, 1.0, 1.5, "1", -1):
        with pytest.raises(ValueError, match="non-negative integer"):
            FrozenActionRule.from_dict({**canonical, "rival_samples": invalid_samples})


def test_frozen_rule_payloads_require_exact_canonical_fields() -> None:
    action_payload = FrozenActionRule(
        source_rule_id="rule-a",
        condition=_condition(x=(0.0, 1.0)),
        action=RuleAction("speed", "light"),
        w=1.0,
        precision=2.0,
        rival_lcb=0.5,
        rival_samples=3,
    ).to_dict()
    for field_name in tuple(action_payload):
        malformed = dict(action_payload)
        malformed.pop(field_name)
        with pytest.raises(ValueError, match="exactly"):
            FrozenActionRule.from_dict(malformed)
    with pytest.raises(ValueError, match="exactly"):
        FrozenActionRule.from_dict({**action_payload, "unknown": None})
    with pytest.raises(TypeError, match="string"):
        FrozenActionRule.from_dict({**action_payload, "source_rule_id": 7})

    veto_payload = FrozenVetoRule(
        source_rule_id="rule-veto",
        condition=_condition(x=(0.0, 1.0)),
    ).to_dict()
    for field_name in tuple(veto_payload):
        malformed = dict(veto_payload)
        malformed.pop(field_name)
        with pytest.raises(ValueError, match="exactly"):
            FrozenVetoRule.from_dict(malformed)
    with pytest.raises(ValueError, match="exactly"):
        FrozenVetoRule.from_dict({**veto_payload, "unknown": None})
    with pytest.raises(ValueError, match="veto=true"):
        FrozenVetoRule.from_dict({**veto_payload, "veto": False})
    with pytest.raises(TypeError, match="string"):
        FrozenVetoRule.from_dict({**veto_payload, "source_rule_id": 7})


def test_online_moments_deserialization_requires_exact_schema_and_integer_count() -> (
    None
):
    canonical = {"n": 0, "mean": 0.0, "m2": 0.0}

    missing = dict(canonical)
    missing.pop("m2")
    unknown = {**canonical, "extra": 0}
    for payload in (missing, unknown):
        with pytest.raises(ValueError, match="exactly"):
            OnlineMoments.from_dict(payload)

    for invalid_n in (True, 1.5, "1", -1):
        with pytest.raises(ValueError, match="non-negative integer"):
            OnlineMoments.from_dict({**canonical, "n": invalid_n})


def test_condition_and_action_deserialization_reject_missing_or_unknown_fields() -> (
    None
):
    condition = _condition(x=(0.0, 1.0)).to_dict()
    action = RuleAction("speed", "light").to_dict()

    for payload in (
        {key: value for key, value in condition.items() if key != "intervals"},
        {**condition, "extra": None},
    ):
        with pytest.raises(ValueError, match="exactly"):
            RuleCondition.from_dict(payload)

    for payload in (
        {key: value for key, value in action.items() if key != "band"},
        {**action, "extra": None},
    ):
        with pytest.raises(ValueError, match="exactly"):
            RuleAction.from_dict(payload)


def test_mutable_rule_deserialization_requires_exact_schema_and_integer_fields() -> (
    None
):
    canonical = MutableRule(
        _condition(x=(0.0, 1.0)),
        RuleAction("speed", "light"),
    ).to_dict()

    for field_name in tuple(canonical):
        missing = dict(canonical)
        missing.pop(field_name)
        with pytest.raises(ValueError, match="canonical MutableRule schema"):
            MutableRule.from_dict(missing)
    with pytest.raises(ValueError, match="canonical MutableRule schema"):
        MutableRule.from_dict({**canonical, "extra": None})

    for field_name in ("numerosity", "creation_epoch", "last_ga_epoch"):
        for invalid_value in (True, 1.5, "1"):
            malformed = {**canonical, field_name: invalid_value}
            with pytest.raises(ValueError, match="integer"):
                MutableRule.from_dict(malformed)


def test_population_deserialization_requires_exact_schema_and_integer_limit() -> None:
    canonical = Population(
        max_size=2,
        credit_mode="vanilla_accuracy",
    ).to_dict()

    for field_name in tuple(canonical):
        payload = dict(canonical)
        payload.pop(field_name)
        with pytest.raises(ValueError, match="exactly"):
            Population.from_dict(payload)
    with pytest.raises(ValueError, match="exactly"):
        Population.from_dict({**canonical, "extra": None})

    for invalid_max_size in (True, 1.5, "2", 0):
        with pytest.raises(ValueError, match="positive integer"):
            Population.from_dict({**canonical, "max_size": invalid_max_size})

    with pytest.raises(ValueError, match="unsupported"):
        Population.from_dict(
            {**canonical, "artifact_version": "hailmary.population.v999"}
        )
    for invalid_mode in (None, True, "causal ", "unsupported"):
        expected = TypeError if type(invalid_mode) is not str else ValueError
        with pytest.raises(expected, match="credit_mode"):
            Population.from_dict({**canonical, "credit_mode": invalid_mode})

    restored = Population.from_dict(canonical)
    assert restored.credit_mode == "vanilla_accuracy"
    with pytest.raises(AttributeError, match="immutable"):
        restored.credit_mode = "causal"  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        restored._credit_mode = "causal"  # type: ignore[attr-defined]


def test_population_constructor_requires_exact_integer_limit() -> None:
    for invalid_max_size in (True, 1.5, "2", 0):
        with pytest.raises(ValueError, match="positive integer"):
            Population(max_size=invalid_max_size)  # type: ignore[arg-type]
