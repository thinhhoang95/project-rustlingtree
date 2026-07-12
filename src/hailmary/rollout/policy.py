"""Frozen downstream-policy protocol and mutation audits."""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from hailmary.ids import content_hash


@runtime_checkable
class FrozenPolicy(Protocol):
    """A rollout policy may select actions but must not learn or mutate."""

    def select_action(self, context: Any) -> Any | None: ...


@dataclass(frozen=True)
class NoOpPolicy:
    """Stateless policy useful for fixed-horizon and inner comparisons."""

    action: Any | None = None

    def select_action(self, context: Any) -> Any | None:
        del context
        return self.action


@dataclass(frozen=True)
class CallableFrozenPolicy:
    """Explicitly mark a side-effect-free callable as a frozen policy."""

    selector: Callable[[Any], Any | None]
    name: str | None = None

    def select_action(self, context: Any) -> Any | None:
        return self.selector(context)

    def policy_fingerprint(self) -> str:
        return content_hash(
            {
                "kind": "callable_frozen_policy",
                "name": self.name,
                "module": getattr(self.selector, "__module__", ""),
                "qualname": getattr(self.selector, "__qualname__", repr(self.selector)),
            },
            namespace="hailmary.policy",
        )


def policy_fingerprint(policy: Any) -> str:
    """Fingerprint visible policy state before and after temporary rollouts."""

    explicit = getattr(policy, "policy_fingerprint", None)
    if callable(explicit):
        result = explicit()
        if not isinstance(result, str) or not result:
            raise TypeError("policy_fingerprint() must return a non-empty string")
        return result

    if inspect.isfunction(policy) or inspect.ismethod(policy):
        payload = {
            "module": getattr(policy, "__module__", ""),
            "qualname": getattr(policy, "__qualname__", repr(policy)),
            "defaults": getattr(policy, "__defaults__", None),
        }
        return content_hash(payload, namespace="hailmary.policy")
    if dataclasses.is_dataclass(policy) and not isinstance(policy, type):
        return content_hash(policy, namespace="hailmary.policy")
    if hasattr(policy, "__dict__"):
        try:
            return content_hash(vars(policy), namespace="hailmary.policy")
        except (TypeError, ValueError):
            # This fallback still detects ordinary visible-state mutation while
            # allowing policies to hold opaque immutable service handles.
            payload = tuple(sorted((str(key), repr(value)) for key, value in vars(policy).items()))
            return content_hash(payload, namespace="hailmary.policy_repr")
    return content_hash(
        {
            "type": f"{type(policy).__module__}.{type(policy).__qualname__}",
            "repr": repr(policy),
        },
        namespace="hailmary.policy_repr",
    )


__all__ = [
    "CallableFrozenPolicy",
    "FrozenPolicy",
    "NoOpPolicy",
    "policy_fingerprint",
]
