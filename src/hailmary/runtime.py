"""Reproducible action catalog and realization wiring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping

from hailmary.actions import ActionCatalog, PathStretchRealizer, apply_action
from hailmary.actions.vocabulary import ActionVocabulary
from hailmary.config import StretchConfig, TemplateConfig
from hailmary.errors import ConfigurationError
from hailmary.ids import content_hash
from hailmary.scenario.models import ScenarioDefinition
from hailmary.simulator.engine import Simulator


ActionApplier = Callable[[Simulator, Any], Any]
StretchOutcomeEvaluator = Callable[[Any], float]


def _callable_type(value: Any | None) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


@dataclass(frozen=True, slots=True)
class ConfiguredActionApplier:
    """One inspectable callable binding both physical action realizers."""

    template_config: TemplateConfig
    stretch_realizer: PathStretchRealizer
    stretch_outcome_evaluator: StretchOutcomeEvaluator | None = None
    stretch_selector: str | None = None

    def __post_init__(self) -> None:
        if self.stretch_realizer.template_config != self.template_config:
            raise ConfigurationError(
                "stretch realizer and configured action applier use different template settings"
            )

    def __call__(self, simulator: Simulator, action: Any) -> Any:
        return apply_action(
            simulator,
            action,
            config=self.template_config,
            stretch_realizer=self.stretch_realizer,
            stretch_outcome_evaluator=self.stretch_outcome_evaluator,
            stretch_selector=self.stretch_selector,
        )


@dataclass(frozen=True, slots=True)
class ActionRuntime:
    """Fixed catalog/applier bundle shared by real, forked, and resumed simulators."""

    catalog: ActionCatalog
    stretch_realizer: PathStretchRealizer
    action_applier: ActionApplier
    custom_runtime_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.catalog.config != self.stretch_realizer.template_config:
            raise ConfigurationError(
                "action catalog and stretch realizer use different template settings"
            )
        fingerprint = str(self.custom_runtime_fingerprint).strip()
        configured = (
            self.action_applier
            if isinstance(self.action_applier, ConfiguredActionApplier)
            else None
        )
        if (
            configured is not None
            and configured.stretch_realizer is not self.stretch_realizer
        ):
            raise ConfigurationError(
                "configured action applier and action runtime use different stretch realizers"
            )
        opaque_runtime_component = bool(
            configured is None
            or self.stretch_realizer.validator is not None
            or configured.stretch_outcome_evaluator is not None
        )
        if opaque_runtime_component and not fingerprint:
            raise ConfigurationError(
                "custom runtime callables require an explicit runtime_fingerprint"
            )
        if self.custom_runtime_fingerprint and not fingerprint:
            raise ConfigurationError("custom runtime fingerprint cannot be blank")
        object.__setattr__(self, "custom_runtime_fingerprint", fingerprint)

    @property
    def template_config(self) -> TemplateConfig:
        return self.catalog.config

    @property
    def vocabulary(self) -> ActionVocabulary:
        return self.catalog.vocabulary

    @property
    def action_vocabulary_hash(self) -> str:
        return self.vocabulary.content_hash

    @property
    def action_vocabulary_serialized(self) -> str:
        return self.vocabulary.serialized

    @property
    def realization_configuration(self) -> Mapping[str, Any]:
        configured = (
            self.action_applier
            if isinstance(self.action_applier, ConfiguredActionApplier)
            else None
        )
        return {
            "schema_version": "hailmary.action_runtime.v1",
            "action_vocabulary": self.vocabulary.payload,
            "template_config": asdict(self.template_config),
            "stretch": {
                "config": asdict(self.stretch_realizer.config),
                "other_medoid_polylines_m": self.stretch_realizer.other_medoid_polylines_m,
                "boundary_polylines_m": self.stretch_realizer.boundary_polylines_m,
                "validator_type": _callable_type(self.stretch_realizer.validator),
                "selector": (
                    configured.stretch_selector or self.stretch_realizer.config.selector
                    if configured is not None
                    else self.stretch_realizer.config.selector
                ),
                "outcome_evaluator_type": (
                    _callable_type(configured.stretch_outcome_evaluator)
                    if configured is not None
                    else None
                ),
            },
            "action_applier_type": _callable_type(self.action_applier),
            "custom_runtime_fingerprint": self.custom_runtime_fingerprint or None,
        }

    @property
    def runtime_configuration_hash(self) -> str:
        return content_hash(
            self.realization_configuration,
            namespace="hailmary.action_runtime.v1",
        )

    def create_simulator(self, definition: ScenarioDefinition) -> Simulator:
        return Simulator(
            definition,
            action_applier=self.action_applier,
            runtime_configuration_hash=self.runtime_configuration_hash,
        )

    def resume_simulator(
        self,
        definition: ScenarioDefinition,
        snapshot: Mapping[str, Any],
    ) -> Simulator:
        return Simulator.resume(
            definition,
            snapshot,
            action_applier=self.action_applier,
            runtime_configuration_hash=self.runtime_configuration_hash,
        )


def build_action_runtime(
    *,
    template_config: TemplateConfig | None = None,
    stretch_config: StretchConfig | None = None,
    catalog: ActionCatalog | None = None,
    stretch_realizer: PathStretchRealizer | None = None,
    action_applier: ActionApplier | None = None,
    runtime_fingerprint: str | None = None,
    stretch_outcome_evaluator: StretchOutcomeEvaluator | None = None,
    stretch_selector: str | None = None,
    variant_validator: object | None = None,
) -> ActionRuntime:
    """Build one action runtime, with explicit injection points for adapters/tests."""

    if template_config is None:
        if catalog is not None:
            resolved_template = catalog.config
        elif stretch_realizer is not None:
            resolved_template = stretch_realizer.template_config
        else:
            resolved_template = TemplateConfig()
    else:
        resolved_template = template_config

    resolved_catalog = ActionCatalog(resolved_template) if catalog is None else catalog
    if resolved_catalog.config != resolved_template:
        raise ConfigurationError(
            "injected action catalog does not match the requested template settings"
        )

    if stretch_realizer is None:
        resolved_realizer = PathStretchRealizer(
            config=StretchConfig() if stretch_config is None else stretch_config,
            template_config=resolved_template,
            validator=variant_validator,  # type: ignore[arg-type]
        )
    else:
        if stretch_realizer.template_config != resolved_template:
            raise ConfigurationError(
                "injected stretch realizer does not match the requested template settings"
            )
        if stretch_config is not None and stretch_realizer.config != stretch_config:
            raise ConfigurationError(
                "injected stretch realizer does not match the requested stretch settings"
            )
        resolved_realizer = stretch_realizer
        if (
            variant_validator is not None
            and resolved_realizer.validator is not variant_validator
        ):
            resolved_realizer = replace(
                resolved_realizer,
                validator=variant_validator,  # type: ignore[arg-type]
            )

    if action_applier is not None and (
        stretch_outcome_evaluator is not None or stretch_selector is not None
    ):
        raise ConfigurationError(
            "stretch evaluator/selector options cannot accompany a custom action applier"
        )
    opaque_runtime_component = bool(
        (
            action_applier is not None
            and not isinstance(action_applier, ConfiguredActionApplier)
        )
        or stretch_outcome_evaluator is not None
        or variant_validator is not None
        or resolved_realizer.validator is not None
    )
    resolved_fingerprint = (
        "" if runtime_fingerprint is None else str(runtime_fingerprint).strip()
    )
    if opaque_runtime_component and not resolved_fingerprint:
        raise ConfigurationError(
            "custom runtime callables require an explicit runtime_fingerprint"
        )
    resolved_applier = action_applier or ConfiguredActionApplier(
        template_config=resolved_template,
        stretch_realizer=resolved_realizer,
        stretch_outcome_evaluator=stretch_outcome_evaluator,
        stretch_selector=stretch_selector,
    )
    return ActionRuntime(
        catalog=resolved_catalog,
        stretch_realizer=resolved_realizer,
        action_applier=resolved_applier,
        custom_runtime_fingerprint=resolved_fingerprint,
    )


__all__ = [
    "ActionApplier",
    "ActionRuntime",
    "ConfiguredActionApplier",
    "StretchOutcomeEvaluator",
    "build_action_runtime",
]
