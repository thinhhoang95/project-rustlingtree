"""Runway-resource spacing primitives and bounded scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from hailmary.features.anchors import LeaderFollowerAnchor


@runtime_checkable
class SeparationTable(Protocol):
    def required_interval_s(self, leader_id: str, follower_id: str, resource_id: str) -> float: ...


@dataclass(frozen=True)
class HomogeneousSeparationTable:
    """Version-1 homogeneous A320 threshold separation."""

    interval_s: float = 90.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.interval_s) or self.interval_s <= 0.0:
            raise ValueError("interval_s must be finite and positive")

    def required_interval_s(self, leader_id: str, follower_id: str, resource_id: str) -> float:
        del leader_id, follower_id, resource_id
        return float(self.interval_s)


@dataclass(frozen=True)
class SpacingState:
    leader_eta_s: float
    follower_eta_s: float
    predicted_interval_s: float
    required_interval_s: float
    spacing_deviation_s: float
    required_delay_s: float


def compute_spacing(
    *,
    leader_eta_s: float,
    follower_eta_s: float,
    required_interval_s: float = 90.0,
) -> SpacingState:
    leader = float(leader_eta_s)
    follower = float(follower_eta_s)
    required = float(required_interval_s)
    if not all(np.isfinite(value) for value in (leader, follower, required)):
        raise ValueError("spacing inputs must be finite")
    if required <= 0.0:
        raise ValueError("required_interval_s must be positive")
    predicted = follower - leader
    deviation = predicted - required
    return SpacingState(
        leader_eta_s=leader,
        follower_eta_s=follower,
        predicted_interval_s=float(predicted),
        required_interval_s=required,
        spacing_deviation_s=float(deviation),
        required_delay_s=float(max(0.0, -deviation)),
    )


def spacing_for_anchor(
    anchor: LeaderFollowerAnchor,
    *,
    eta_by_flight: dict[str, float],
    separation_table: SeparationTable | None = None,
) -> SpacingState:
    table = HomogeneousSeparationTable() if separation_table is None else separation_table
    return compute_spacing(
        leader_eta_s=eta_by_flight[anchor.leader_id],
        follower_eta_s=eta_by_flight[anchor.follower_id],
        required_interval_s=table.required_interval_s(
            anchor.leader_id,
            anchor.follower_id,
            anchor.resource_id,
        ),
    )


def spacing_ratio_score(
    predicted_interval_s: float,
    required_interval_s: float = 90.0,
    *,
    dynamically_feasible: bool = True,
) -> float:
    """Bounded continuous score from the SEQD design.

    A go-around or dynamically infeasible completion receives ``-1`` without
    attempting to interpret its nominal interval.
    """

    predicted = float(predicted_interval_s)
    required = float(required_interval_s)
    if not np.isfinite(predicted) or not np.isfinite(required):
        raise ValueError("spacing score inputs must be finite")
    if required <= 0.0:
        raise ValueError("required_interval_s must be positive")
    if not dynamically_feasible:
        return -1.0

    ratio = predicted / required
    if ratio <= 0.0:
        return -1.0
    if ratio < 1.0:
        return float(-1.0 + 2.0 * ratio)
    if ratio <= 1.25:
        return 1.0
    if ratio < 2.0:
        return float(1.0 - 2.0 * (ratio - 1.25))
    return -0.5


__all__ = [
    "HomogeneousSeparationTable",
    "SeparationTable",
    "SpacingState",
    "compute_spacing",
    "spacing_for_anchor",
    "spacing_ratio_score",
]
