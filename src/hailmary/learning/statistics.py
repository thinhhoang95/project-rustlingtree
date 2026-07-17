"""Finite online statistics used by causal learning ledgers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping


def _finite(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_floor(value: float) -> float:
    floor = _finite(value, name="variance_floor")
    if floor <= 0.0:
        raise ValueError("variance_floor must be positive")
    return floor


@dataclass
class OnlineMoments:
    """Welford sample moments with finite, conservative uncertainty helpers.

    The object is intentionally mutable: a training rule owns two instances
    and updates them after each real three-arm experiment.  Serialization uses
    the exact sufficient statistics ``n``, ``mean``, and ``m2``.

    A sample variance is mathematically undefined below two observations.  We
    expose ``0.0`` for that descriptive property, while uncertainty and
    precision methods always apply a positive variance floor.  Consequently an
    empty ledger has zero precision and a one-sample ledger can never acquire
    infinite precision.
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError("n must be a non-negative integer")
        self.mean = _finite(self.mean, name="mean")
        self.m2 = _finite(self.m2, name="m2")
        if self.m2 < 0.0:
            raise ValueError("m2 cannot be negative")
        if self.n == 0 and (self.mean != 0.0 or self.m2 != 0.0):
            raise ValueError("an empty accumulator must have mean=m2=0")
        if self.n == 1 and self.m2 != 0.0:
            raise ValueError("a one-sample accumulator must have m2=0")

    @property
    def sample_variance(self) -> float:
        if self.n < 2:
            return 0.0
        variance = self.m2 / float(self.n - 1)
        if not math.isfinite(variance) or variance < 0.0:
            raise ArithmeticError("sample variance is not finite and non-negative")
        return float(variance)

    @property
    def variance(self) -> float:
        """Alias retained for concise learning equations."""

        return self.sample_variance

    def effective_variance(self, variance_floor: float = 1.0) -> float:
        return max(self.sample_variance, _positive_floor(variance_floor))

    def standard_error(self, variance_floor: float = 1.0) -> float:
        # Treat an empty accumulator as one hypothetical uncertain observation
        # so diagnostics and LCBs remain finite and conservative.
        denominator = max(self.n, 1)
        result = math.sqrt(self.effective_variance(variance_floor) / denominator)
        if not math.isfinite(result):
            raise ArithmeticError("standard error overflowed")
        return float(result)

    def lcb(self, z: float = 1.96, variance_floor: float = 1.0) -> float:
        multiplier = _finite(z, name="z")
        if multiplier < 0.0:
            raise ValueError("z must be non-negative")
        center = self.mean if self.n else 0.0
        result = center - multiplier * self.standard_error(variance_floor)
        if not math.isfinite(result):
            raise ArithmeticError("lower confidence bound overflowed")
        return float(result)

    def precision(self, variance_floor: float = 1.0) -> float:
        if self.n == 0:
            return 0.0
        result = float(self.n) / self.effective_variance(variance_floor)
        if not math.isfinite(result):
            raise ArithmeticError("precision overflowed")
        return result

    def update(self, value: float) -> "OnlineMoments":
        observation = _finite(value, name="observation")
        next_n = self.n + 1
        delta = observation - self.mean
        next_mean = self.mean + delta / float(next_n)
        delta2 = observation - next_mean
        next_m2 = self.m2 + delta * delta2
        # Roundoff can produce a tiny negative value for otherwise constant
        # data.  It is safe to clamp only at machine-scale tolerance.
        tolerance = 1e-15 * max(1.0, abs(self.m2), abs(delta * delta2))
        if next_m2 < 0.0 and next_m2 >= -tolerance:
            next_m2 = 0.0
        if not math.isfinite(next_mean) or not math.isfinite(next_m2) or next_m2 < 0.0:
            raise ArithmeticError("Welford update produced invalid moments")
        self.n = next_n
        self.mean = float(next_mean)
        self.m2 = float(next_m2)
        return self

    def extend(self, values: Iterable[float]) -> "OnlineMoments":
        for value in values:
            self.update(value)
        return self

    def updated(self, value: float) -> "OnlineMoments":
        return self.copy().update(value)

    def combine(self, other: "OnlineMoments") -> "OnlineMoments":
        """Merge another accumulator using the parallel Welford equation."""

        if not isinstance(other, OnlineMoments):
            raise TypeError("other must be OnlineMoments")
        if other.n == 0:
            return self
        if self.n == 0:
            self.n, self.mean, self.m2 = other.n, other.mean, other.m2
            return self
        total_n = self.n + other.n
        delta = other.mean - self.mean
        combined_mean = self.mean + delta * other.n / float(total_n)
        combined_m2 = (
            self.m2 + other.m2 + delta * delta * self.n * other.n / float(total_n)
        )
        if not math.isfinite(combined_mean) or not math.isfinite(combined_m2):
            raise ArithmeticError("combined moments overflowed")
        self.n = total_n
        self.mean = float(combined_mean)
        self.m2 = max(0.0, float(combined_m2))
        return self

    def copy(self) -> "OnlineMoments":
        return OnlineMoments(n=self.n, mean=self.mean, m2=self.m2)

    def to_dict(self) -> dict[str, int | float]:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OnlineMoments":
        if not isinstance(payload, Mapping):
            raise TypeError("online moments payload must be a mapping")
        expected_fields = {"n", "mean", "m2"}
        if set(payload) != expected_fields:
            raise ValueError(
                "online moments payload fields must be exactly n, mean, and m2"
            )
        n = payload["n"]
        if type(n) is not int or n < 0:
            raise ValueError("n must be a non-negative integer")
        return cls(
            n=n,
            mean=payload["mean"],
            m2=payload["m2"],
        )


__all__ = ["OnlineMoments"]
