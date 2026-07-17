"""Canonical identities for scientifically distinct learning modes."""

from __future__ import annotations

from typing import Any


CAUSAL_CREDIT_MODE = "causal"
VANILLA_ACCURACY_CREDIT_MODE = "vanilla_accuracy"
SUPPORTED_CREDIT_MODES = frozenset({CAUSAL_CREDIT_MODE, VANILLA_ACCURACY_CREDIT_MODE})


def validate_credit_mode(value: Any, *, name: str = "credit_mode") -> str:
    """Return one exact mode identity without coercing serialized data."""

    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if value not in SUPPORTED_CREDIT_MODES:
        raise ValueError(
            f"{name} must be {CAUSAL_CREDIT_MODE!r} or {VANILLA_ACCURACY_CREDIT_MODE!r}"
        )
    return value


__all__ = [
    "CAUSAL_CREDIT_MODE",
    "SUPPORTED_CREDIT_MODES",
    "VANILLA_ACCURACY_CREDIT_MODE",
    "validate_credit_mode",
]
