"""Mutable training rules and stable action identities."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from hailmary.actions.models import ActionIdentity, ActionLever
from hailmary.actions.vocabulary import is_supported_action_identity
from hailmary.ids import canonical_data, stable_id
from hailmary.learning.conditions import RuleCondition
from hailmary.learning.statistics import OnlineMoments


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class RuleAction:
    """One exact action identity from the existing Hailmary catalog."""

    lever: ActionLever | str
    band: str

    def __post_init__(self) -> None:
        try:
            identity = ActionIdentity(self.lever, str(self.band).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported Hailmary action lever {self.lever!r}"
            ) from exc
        if not is_supported_action_identity(identity):
            raise ValueError(
                f"action identity {identity.lever.value}/{identity.band} "
                "is not in ActionCatalog vocabulary"
            )
        object.__setattr__(self, "lever", identity.lever)
        object.__setattr__(self, "band", identity.band)

    @property
    def identity(self) -> ActionIdentity:
        return ActionIdentity(self.lever, self.band)

    @property
    def key(self) -> str:
        return f"{self.lever.value}/{self.band}"

    @property
    def action_key(self) -> str:
        return self.key

    @property
    def is_no_op(self) -> bool:
        return self.lever is ActionLever.NO_OP

    @classmethod
    def no_op(cls) -> "RuleAction":
        return cls(ActionLever.NO_OP, "no_op")

    @classmethod
    def from_candidate(cls, candidate: Any) -> "RuleAction":
        if isinstance(candidate, cls):
            return candidate
        if isinstance(candidate, Mapping):
            return cls(candidate["lever"], str(candidate["band"]))
        lever = getattr(candidate, "lever", None)
        band = getattr(candidate, "band", None)
        if lever is None or band is None:
            raise TypeError("candidate must expose lever and band")
        return cls(lever, str(band))

    def to_dict(self) -> dict[str, str]:
        return {"lever": self.lever.value, "band": self.band}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleAction":
        if not isinstance(payload, Mapping):
            raise TypeError("action payload must be a mapping")
        expected_fields = {"lever", "band"}
        if set(payload) != expected_fields:
            raise ValueError("action payload fields must be exactly lever and band")
        lever = payload["lever"]
        band = payload["band"]
        if type(lever) is not str or type(band) is not str:
            raise TypeError("serialized action lever and band must be strings")
        return cls(lever, band)


@dataclass
class MutableRule:
    """One XCSR classifier with independent rival and no-op evidence."""

    condition: RuleCondition
    action: RuleAction
    evolution: OnlineMoments = field(default_factory=OnlineMoments)
    deployment: OnlineMoments = field(default_factory=OnlineMoments)
    numerosity: int = 1
    creation_epoch: int = 0
    last_ga_epoch: int = 0
    parent_ids: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    rule_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.condition, RuleCondition):
            raise TypeError("condition must be RuleCondition")
        if not isinstance(self.action, RuleAction):
            self.action = RuleAction.from_candidate(self.action)
        if not isinstance(self.evolution, OnlineMoments) or not isinstance(
            self.deployment, OnlineMoments
        ):
            raise TypeError("rule ledgers must be OnlineMoments")
        if isinstance(self.numerosity, bool) or int(self.numerosity) < 1:
            raise ValueError("rule numerosity must be positive")
        self.numerosity = int(self.numerosity)
        if self.creation_epoch < 0 or self.last_ga_epoch < 0:
            raise ValueError("rule epochs cannot be negative")
        self.creation_epoch = int(self.creation_epoch)
        self.last_ga_epoch = int(self.last_ga_epoch)
        self.parent_ids = tuple(str(item) for item in self.parent_ids)
        if any(not item for item in self.parent_ids):
            raise ValueError("parent rule IDs cannot be empty")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("rule provenance must be a mapping")
        normalized_provenance = canonical_data(dict(self.provenance))
        self.provenance = _deep_freeze(normalized_provenance)
        computed_id = stable_id(
            "rule",
            {
                "condition": self.condition.to_dict(),
                "action": self.action.to_dict(),
                "creation_epoch": self.creation_epoch,
                "parent_ids": self.parent_ids,
                "provenance": self.provenance,
            },
            length=32,
        )
        if self.rule_id and not str(self.rule_id).strip():
            raise ValueError("rule_id cannot be blank")
        self.rule_id = str(self.rule_id) if self.rule_id else computed_id

    @property
    def role_type(self) -> str:
        return self.condition.role_type

    @property
    def schema_hash(self) -> str:
        return self.condition.schema_hash

    @property
    def action_key(self) -> str:
        return self.action.key

    @property
    def rival_ledger(self) -> OnlineMoments:
        return self.evolution

    @property
    def evolution_ledger(self) -> OnlineMoments:
        return self.evolution

    @property
    def noop_ledger(self) -> OnlineMoments:
        return self.deployment

    @property
    def deployment_ledger(self) -> OnlineMoments:
        return self.deployment

    def matches(self, values: Mapping[str, float] | Any, **kwargs: Any) -> bool:
        return self.condition.matches(values, **kwargs)

    def update_rival(self, value: float) -> None:
        self.evolution.update(value)

    def update_noop(self, value: float) -> None:
        if self.action.is_no_op:
            raise ValueError("no-op rules receive veto evidence in the rival ledger")
        self.deployment.update(value)

    update_evolution = update_rival
    update_deployment = update_noop

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "condition": self.condition.to_dict(),
            "action": self.action.to_dict(),
            "evolution": self.evolution.to_dict(),
            "deployment": self.deployment.to_dict(),
            "numerosity": self.numerosity,
            "creation_epoch": self.creation_epoch,
            "last_ga_epoch": self.last_ga_epoch,
            "parent_ids": list(self.parent_ids),
            "provenance": canonical_data(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MutableRule":
        if not isinstance(payload, Mapping):
            raise TypeError("rule payload must be a mapping")
        expected_fields = {
            "rule_id",
            "condition",
            "action",
            "evolution",
            "deployment",
            "numerosity",
            "creation_epoch",
            "last_ga_epoch",
            "parent_ids",
            "provenance",
        }
        if set(payload) != expected_fields:
            raise ValueError(
                "rule payload fields must exactly match the canonical MutableRule schema"
            )
        numerosity = payload["numerosity"]
        creation_epoch = payload["creation_epoch"]
        last_ga_epoch = payload["last_ga_epoch"]
        if type(numerosity) is not int or numerosity < 1:
            raise ValueError("rule numerosity must be a positive integer")
        if (
            type(creation_epoch) is not int
            or creation_epoch < 0
            or type(last_ga_epoch) is not int
            or last_ga_epoch < 0
        ):
            raise ValueError("rule epochs must be non-negative integers")
        parent_ids = payload["parent_ids"]
        if (
            isinstance(parent_ids, (str, bytes))
            or not isinstance(parent_ids, Sequence)
            or any(type(item) is not str for item in parent_ids)
        ):
            raise TypeError("serialized parent_ids must be a sequence of strings")
        rule_id = payload["rule_id"]
        if type(rule_id) is not str:
            raise TypeError("serialized rule_id must be a string")
        return cls(
            condition=RuleCondition.from_dict(payload["condition"]),
            action=RuleAction.from_dict(payload["action"]),
            evolution=OnlineMoments.from_dict(payload["evolution"]),
            deployment=OnlineMoments.from_dict(payload["deployment"]),
            numerosity=numerosity,
            creation_epoch=creation_epoch,
            last_ga_epoch=last_ga_epoch,
            parent_ids=tuple(parent_ids),
            provenance=payload["provenance"],
            rule_id=rule_id,
        )


__all__ = ["MutableRule", "RuleAction"]
