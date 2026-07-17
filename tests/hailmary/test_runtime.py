from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from hailmary.actions import (
    ActionCatalog,
    ActionIdentity,
    ActionLever,
    ActionVocabulary,
    PathStretchRealizer,
    action_vocabulary,
    is_supported_action_identity,
)
from hailmary.config import (
    FeatureConfig,
    LearningConfig,
    OutcomeConfig,
    ScenarioConfig,
    StretchConfig,
    TemplateConfig,
)
from hailmary.errors import ConfigurationError
from hailmary.runtime import (
    ActionRuntime,
    ConfiguredActionApplier,
    build_action_runtime,
)
from hailmary.simulator import Simulator

from .test_engine import _definition


def test_action_vocabulary_has_exactly_the_five_catalog_identities() -> None:
    template_config = TemplateConfig()
    vocabulary = action_vocabulary(template_config)
    catalog = ActionCatalog(template_config)

    expected = [
        (ActionLever.NO_OP, "no_op"),
        (ActionLever.SPEED, "light"),
        (ActionLever.SPEED, "medium"),
        (ActionLever.SPEED, "heavy"),
        (ActionLever.PATH_STRETCH, "oracle_short_medium_long"),
    ]
    assert [(item.lever, item.band) for item in vocabulary.identities] == expected
    assert catalog.vocabulary == vocabulary
    assert len(set(vocabulary.identities)) == 5
    assert is_supported_action_identity(ActionIdentity(ActionLever.SPEED, "light"))
    assert not is_supported_action_identity(
        ActionIdentity(ActionLever.SPEED, "delay_90s")
    )


def test_action_vocabulary_constructor_rejects_forged_or_nonfinite_contracts() -> None:
    vocabulary = action_vocabulary()

    with pytest.raises(ValueError, match="exact five"):
        ActionVocabulary(
            template_schema_version=vocabulary.template_schema_version,
            identities=(ActionIdentity(ActionLever.NO_OP, "forged"),),
            speed_reductions_kts=(),
            min_effective_reduction_kts=2.0,
            max_speed_actions=2,
            max_path_stretches=1,
        )
    with pytest.raises(ValueError, match="finite ordered"):
        ActionVocabulary(
            template_schema_version=vocabulary.template_schema_version,
            identities=vocabulary.identities,
            speed_reductions_kts=(10.0, float("nan"), 20.0),
            min_effective_reduction_kts=2.0,
            max_speed_actions=2,
            max_path_stretches=1,
        )


def test_action_vocabulary_hash_is_stable_and_tracks_physical_configuration() -> None:
    first = action_vocabulary(TemplateConfig())
    second = action_vocabulary(TemplateConfig())
    changed = action_vocabulary(
        replace(
            TemplateConfig(),
            speed_reduction_kts=(11.0, 15.0, 20.0),
        )
    )

    assert first.serialized == second.serialized
    assert first.content_hash == second.content_hash
    assert json.loads(first.serialized)["actions"][1] == {
        "band": "light",
        "lever": "speed",
        "reduction_kts": 10.0,
    }
    assert changed.identities == first.identities
    assert changed.content_hash != first.content_hash


def test_runtime_shares_one_bound_applier_across_create_fork_and_resume() -> None:
    runtime = build_action_runtime()
    parent = runtime.create_simulator(_definition())
    child = parent.fork(label="runtime-child")
    resumed = runtime.resume_simulator(parent.definition, parent.snapshot())

    assert isinstance(runtime.action_applier, ConfiguredActionApplier)
    assert runtime.action_applier.stretch_realizer is runtime.stretch_realizer
    assert parent.action_applier is runtime.action_applier
    assert child.action_applier is runtime.action_applier
    assert resumed.action_applier is runtime.action_applier
    assert {
        parent.runtime_configuration_hash,
        child.runtime_configuration_hash,
        resumed.runtime_configuration_hash,
    } == {runtime.runtime_configuration_hash}
    assert runtime.action_vocabulary_hash == runtime.catalog.vocabulary.content_hash


def test_snapshot_persists_and_requires_exact_runtime_configuration_hash() -> None:
    runtime = build_action_runtime()
    parent = runtime.create_simulator(_definition())
    child = parent.fork(label="runtime-hash-child")
    snapshot = parent.snapshot()

    assert snapshot["schema_version"] == "hailmary.simulation-snapshot.v2"
    assert snapshot["runtime_configuration_hash"] == runtime.runtime_configuration_hash
    assert (
        child.snapshot()["runtime_configuration_hash"]
        == runtime.runtime_configuration_hash
    )

    changed_runtime = build_action_runtime(
        template_config=replace(
            TemplateConfig(),
            speed_reduction_kts=(11.0, 15.0, 20.0),
        )
    )
    with pytest.raises(ValueError, match="does not match the resume runtime"):
        changed_runtime.resume_simulator(parent.definition, snapshot)
    with pytest.raises(ValueError, match="does not match the resume runtime"):
        Simulator.resume(parent.definition, snapshot)

    tampered = dict(snapshot)
    tampered["runtime_configuration_hash"] = "forged-runtime"
    with pytest.raises(ValueError, match="does not match the resume runtime"):
        runtime.resume_simulator(parent.definition, tampered)


def test_configured_runtime_routes_speed_and_stretch_through_one_realizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object, dict[str, object]]] = []

    def fake_apply(simulator, action, **kwargs):
        calls.append((simulator, action, kwargs))
        return {"applied": action["lever"]}

    monkeypatch.setattr("hailmary.runtime.apply_action", fake_apply)
    runtime = build_action_runtime()
    simulator = runtime.create_simulator(_definition())

    simulator.apply({"lever": "speed", "band": "light"})
    simulator.apply(
        {
            "lever": "path_stretch",
            "band": "oracle_short_medium_long",
        }
    )

    assert [action["lever"] for _, action, _ in calls] == [
        "speed",
        "path_stretch",
    ]
    assert all(
        kwargs["config"] is runtime.template_config
        and kwargs["stretch_realizer"] is runtime.stretch_realizer
        for _, _, kwargs in calls
    )


def test_runtime_allows_explicit_catalog_realizer_and_applier_injection() -> None:
    template_config = TemplateConfig()
    catalog = ActionCatalog(template_config)
    realizer = PathStretchRealizer(template_config=template_config)
    calls: list[object] = []

    def custom_applier(_simulator, action):
        calls.append(action)
        return None

    runtime = build_action_runtime(
        catalog=catalog,
        stretch_realizer=realizer,
        action_applier=custom_applier,
        runtime_fingerprint="recording-applier-v1",
    )
    simulator = runtime.create_simulator(_definition())
    simulator.apply({"lever": "speed", "band": "medium"})

    assert runtime.catalog is catalog
    assert runtime.stretch_realizer is realizer
    assert runtime.action_applier is custom_applier
    assert calls == [{"lever": "speed", "band": "medium"}]


def test_runtime_rejects_opaque_custom_applier_without_fingerprint() -> None:
    def custom_applier(_simulator, _action):
        return None

    with pytest.raises(ConfigurationError, match="explicit runtime_fingerprint"):
        build_action_runtime(action_applier=custom_applier)


def test_runtime_direct_constructor_rejects_opaque_applier_without_fingerprint() -> (
    None
):
    def custom_applier(_simulator, _action):
        return None

    catalog = ActionCatalog()
    realizer = PathStretchRealizer()
    with pytest.raises(ConfigurationError, match="explicit runtime_fingerprint"):
        ActionRuntime(
            catalog=catalog,
            stretch_realizer=realizer,
            action_applier=custom_applier,
        )


@pytest.mark.parametrize(
    "applier_realizer",
    (
        PathStretchRealizer(
            config=replace(StretchConfig(), max_turn_deg=120.0),
        ),
        PathStretchRealizer(
            other_medoid_polylines_m=(np.asarray(((0.0, 0.0), (1_000.0, 1_000.0))),),
        ),
    ),
)
def test_runtime_rejects_configured_applier_bound_to_different_realizer(
    applier_realizer: PathStretchRealizer,
) -> None:
    runtime_realizer = PathStretchRealizer()
    configured_applier = ConfiguredActionApplier(
        template_config=TemplateConfig(),
        stretch_realizer=applier_realizer,
    )

    with pytest.raises(ConfigurationError, match="different stretch realizers"):
        build_action_runtime(
            stretch_realizer=runtime_realizer,
            action_applier=configured_applier,
        )


def test_runtime_configuration_hash_tracks_all_physical_realization_inputs() -> None:
    baseline = build_action_runtime()
    changed_template = build_action_runtime(
        template_config=replace(
            TemplateConfig(),
            speed_reduction_kts=(11.0, 15.0, 20.0),
        )
    )
    changed_stretch = build_action_runtime(
        stretch_config=replace(StretchConfig(), max_turn_deg=120.0)
    )
    changed_geometry = build_action_runtime(
        stretch_realizer=PathStretchRealizer(
            other_medoid_polylines_m=(np.asarray(((0.0, 0.0), (1_000.0, 1_000.0))),),
        )
    )

    hashes = {
        baseline.runtime_configuration_hash,
        changed_template.runtime_configuration_hash,
        changed_stretch.runtime_configuration_hash,
        changed_geometry.runtime_configuration_hash,
    }
    assert len(hashes) == 4


def test_runtime_configuration_hash_tracks_explicit_custom_runtime_token() -> None:
    def custom_applier(_simulator, _action):
        return None

    first = build_action_runtime(
        action_applier=custom_applier,
        runtime_fingerprint="custom-applier-v1",
    )
    second = build_action_runtime(
        action_applier=custom_applier,
        runtime_fingerprint="custom-applier-v2",
    )

    assert first.runtime_configuration_hash != second.runtime_configuration_hash


def test_runtime_rejects_mismatched_action_configuration() -> None:
    changed = replace(
        TemplateConfig(),
        speed_reduction_kts=(11.0, 15.0, 20.0),
    )
    with pytest.raises(ConfigurationError, match="catalog"):
        build_action_runtime(
            template_config=TemplateConfig(),
            catalog=ActionCatalog(changed),
        )


def test_learning_config_exposes_phase_zero_defaults() -> None:
    config = LearningConfig()

    assert config.population_limit == 1_000
    assert config.ga_interval == 50
    assert config.ga_min_experience == 20
    assert config.exploration_coverage_floor == 200
    assert config.certification_interval == 500
    assert config.action_min_noop_samples == 30
    assert config.veto_min_rival_samples == 60
    assert config.action_lcb_z == 1.96
    assert config.veto_lcb_z == 1.96
    assert config.contender_min_samples == 2
    assert config.variance_floor > 0.0
    assert config.random_seed == 17


@pytest.mark.parametrize(
    "changes",
    [
        {"population_limit": 0},
        {"population_limit": True},
        {"ga_interval": 1.5},
        {"ga_min_experience": 1},
        {"contender_min_samples": 1},
        {"action_lcb_z": float("nan")},
        {"action_lcb_z": True},
        {"mutation_probability": float("nan")},
        {"mutation_probability": True},
        {"offspring_evidence_discount": True},
        {"base_exploration_rate": 1.01},
        {
            "covering_width_min_fraction": 0.4,
            "covering_width_max_fraction": 0.2,
        },
        {"offspring_evidence_discount": 0.0},
        {"variance_floor": 0.0},
        {"random_seed": -1},
    ],
)
def test_learning_config_rejects_invalid_values(changes: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        LearningConfig(**changes)


@pytest.mark.parametrize(
    ("config_type", "changes"),
    [
        (TemplateConfig, {"commitment_gate_nm": float("nan")}),
        (TemplateConfig, {"speed_reduction_kts": (10.0, float("inf"), 20.0)}),
        (TemplateConfig, {"min_effective_reduction_kts": float("nan")}),
        (TemplateConfig, {"max_speed_actions": 1.5}),
        (TemplateConfig, {"max_historical_clamp_kts": float("nan")}),
        (TemplateConfig, {"max_timing_error_fraction": float("nan")}),
        (TemplateConfig, {"max_timing_error_s": float("nan")}),
        (TemplateConfig, {"payload_kg": float("nan")}),
        (TemplateConfig, {"zero_wind": False}),
        (StretchConfig, {"added_distance_nm": (2.0, float("nan"), 9.0)}),
        (StretchConfig, {"variant_names": ("a", "b", "c")}),
        (StretchConfig, {"candidate_azimuth_count": 4.5}),
        (StretchConfig, {"max_turn_deg": float("nan")}),
        (StretchConfig, {"minimum_medoid_clearance_nm": float("nan")}),
        (ScenarioConfig, {"separation_s": float("nan")}),
        (FeatureConfig, {"time_weight": float("nan")}),
        (OutcomeConfig, {"pair_weight": float("nan")}),
        (OutcomeConfig, {"trailer_count": 1.5}),
    ],
)
def test_physical_configs_reject_nonfinite_or_fractional_values(
    config_type: (
        type[TemplateConfig]
        | type[StretchConfig]
        | type[ScenarioConfig]
        | type[FeatureConfig]
        | type[OutcomeConfig]
    ),
    changes: dict[str, object],
) -> None:
    with pytest.raises(ConfigurationError):
        config_type(**changes)
