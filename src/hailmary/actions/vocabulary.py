"""Scenario-independent action vocabulary and stable serialization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from hailmary.actions.models import ActionIdentity, ActionLever
from hailmary.config import TemplateConfig
from hailmary.ids import canonical_json, content_hash


ACTION_VOCABULARY_SCHEMA_VERSION = "hailmary.action_vocabulary.v1"
NO_OP_BAND = "no_op"
PATH_STRETCH_MACRO_BAND = "oracle_short_medium_long"


@dataclass(frozen=True, slots=True)
class ActionVocabulary:
    """Immutable physical action keys and the catalog settings behind them."""

    template_schema_version: str
    identities: tuple[ActionIdentity, ...]
    speed_reductions_kts: tuple[float, ...]
    min_effective_reduction_kts: float
    max_speed_actions: int
    max_path_stretches: int
    schema_version: str = ACTION_VOCABULARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.template_schema_version != "hailmary.template.v1":
            raise ValueError("unsupported template schema version")
        if self.schema_version != ACTION_VOCABULARY_SCHEMA_VERSION:
            raise ValueError("unsupported action vocabulary schema version")
        expected_identities = (
            ActionIdentity(ActionLever.NO_OP, NO_OP_BAND),
            ActionIdentity(ActionLever.SPEED, "light"),
            ActionIdentity(ActionLever.SPEED, "medium"),
            ActionIdentity(ActionLever.SPEED, "heavy"),
            ActionIdentity(ActionLever.PATH_STRETCH, PATH_STRETCH_MACRO_BAND),
        )
        if self.identities != expected_identities:
            raise ValueError(
                "action vocabulary must contain the exact five Hailmary identities"
            )
        if (
            len(self.speed_reductions_kts) != 3
            or any(
                isinstance(value, bool) or not math.isfinite(value) or value <= 0.0
                for value in self.speed_reductions_kts
            )
            or tuple(sorted(self.speed_reductions_kts)) != self.speed_reductions_kts
        ):
            raise ValueError(
                "speed reductions must be three finite ordered positive values"
            )
        if (
            isinstance(self.min_effective_reduction_kts, bool)
            or not math.isfinite(self.min_effective_reduction_kts)
            or self.min_effective_reduction_kts <= 0.0
        ):
            raise ValueError(
                "minimum effective speed reduction must be finite and positive"
            )
        if (
            isinstance(self.max_speed_actions, bool)
            or not isinstance(self.max_speed_actions, int)
            or isinstance(self.max_path_stretches, bool)
            or not isinstance(self.max_path_stretches, int)
            or self.max_speed_actions < 1
            or self.max_path_stretches < 1
        ):
            raise ValueError("action limits must be positive integers")

    @property
    def payload(self) -> dict[str, Any]:
        reductions = iter(self.speed_reductions_kts)
        actions: list[dict[str, Any]] = []
        for identity in self.identities:
            item: dict[str, Any] = identity.to_dict()
            if identity.lever is ActionLever.SPEED:
                item["reduction_kts"] = float(next(reductions))
            actions.append(item)
        return {
            "schema_version": self.schema_version,
            "template_schema_version": self.template_schema_version,
            "actions": actions,
            "availability": {
                "min_effective_reduction_kts": self.min_effective_reduction_kts,
                "max_speed_actions": self.max_speed_actions,
                "max_path_stretches": self.max_path_stretches,
            },
        }

    @property
    def serialized(self) -> str:
        return canonical_json(self.payload)

    @property
    def content_hash(self) -> str:
        return content_hash(self.payload, namespace=self.schema_version)


def action_vocabulary(
    template_config: TemplateConfig | None = None,
) -> ActionVocabulary:
    """Build the only supported action vocabulary from template configuration."""

    config = TemplateConfig() if template_config is None else template_config
    identities = (
        ActionIdentity(ActionLever.NO_OP, NO_OP_BAND),
        *(ActionIdentity(ActionLever.SPEED, band) for band in config.speed_band_names),
        ActionIdentity(ActionLever.PATH_STRETCH, PATH_STRETCH_MACRO_BAND),
    )
    return ActionVocabulary(
        template_schema_version=config.schema_version,
        identities=identities,
        speed_reductions_kts=tuple(
            float(value) for value in config.speed_reduction_kts
        ),
        min_effective_reduction_kts=float(config.min_effective_reduction_kts),
        max_speed_actions=int(config.max_speed_actions),
        max_path_stretches=int(config.max_path_stretches),
    )


def is_supported_action_identity(
    identity: ActionIdentity | tuple[ActionLever | str, str] | object,
    template_config: TemplateConfig | None = None,
) -> bool:
    """Return whether an action key belongs to the configured vocabulary."""

    try:
        if isinstance(identity, ActionIdentity):
            normalized = identity
        elif isinstance(identity, tuple) and len(identity) == 2:
            normalized = ActionIdentity(identity[0], identity[1])
        else:
            normalized = ActionIdentity(
                getattr(identity, "lever"),
                getattr(identity, "band"),
            )
    except (AttributeError, TypeError, ValueError):
        return False
    return normalized in action_vocabulary(template_config).identities


__all__ = [
    "ACTION_VOCABULARY_SCHEMA_VERSION",
    "ActionVocabulary",
    "NO_OP_BAND",
    "PATH_STRETCH_MACRO_BAND",
    "action_vocabulary",
    "is_supported_action_identity",
]
