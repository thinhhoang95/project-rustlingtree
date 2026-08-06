"""Shared action schema and stale-epoch guards."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hailmary.errors import InfeasibleActionError, StaleActionError
from hailmary.ids import stable_id


class ActionLever(StrEnum):
    NO_OP = "no_op"
    SPEED = "speed"
    PATH_STRETCH = "path_stretch"


@dataclass(frozen=True, slots=True)
class ActionIdentity:
    """Scenario-independent action key used by catalogs and learned rules."""

    lever: ActionLever | str
    band: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lever", ActionLever(self.lever))
        if not self.band:
            raise ValueError("action identity band cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return {"lever": self.lever.value, "band": self.band}


@dataclass(frozen=True)
class ActionCandidate:
    anchor_id: str
    bound_flight_id: str
    resource_id: str
    segment_id: str
    lever: ActionLever | str
    band: str
    state_id: str
    state_version: int
    epoch_index: int
    station_index: int
    s_m: float
    dynamic_content_hash: str = ""
    feasible: bool = True
    reason: str = ""
    realization_metadata: tuple[tuple[str, Any], ...] = ()
    action_id: str = ""

    def __post_init__(self) -> None:
        normalized_lever = ActionLever(self.lever)
        object.__setattr__(self, "lever", normalized_lever)
        if (
            not self.anchor_id
            or not self.bound_flight_id
            or not self.resource_id
            or not self.segment_id
            or not self.state_id
        ):
            raise ValueError("anchor, flight, segment, resource, and state identities cannot be empty")
        if self.state_version < 0 or self.epoch_index < 1 or self.station_index < 0:
            raise ValueError("state, epoch, and station indices are invalid")
        if not np.isfinite(self.s_m) or self.s_m < 0.0:
            raise ValueError("action station must be finite and non-negative")
        if not self.band:
            raise ValueError("action band cannot be empty")
        computed = stable_id(
            "action",
            {
                "anchor_id": self.anchor_id,
                "bound_flight_id": self.bound_flight_id,
                "resource_id": self.resource_id,
                "segment_id": self.segment_id,
                "lever": normalized_lever.value,
                "band": self.band,
                "bound_dynamic_content": self.dynamic_content_hash or self.state_id,
                "state_version": self.state_version,
                "epoch_index": self.epoch_index,
                "station_index": self.station_index,
                "s_m": self.s_m,
                "dynamic_content_hash": self.dynamic_content_hash,
                "realization_metadata": self.realization_metadata,
            },
            length=32,
        )
        if self.action_id and self.action_id != computed:
            raise ValueError("action_id does not match the bound candidate")
        object.__setattr__(self, "action_id", computed)

    def assert_applicable(self, simulator: object) -> None:
        if not self.feasible:
            raise InfeasibleActionError(self.reason or "action candidate is infeasible")
        state = getattr(simulator, "state", None)
        if state is None:
            raise TypeError("simulator must expose state")
        same_state = getattr(state, "state_id", None) == self.state_id
        same_fork_content = bool(
            self.dynamic_content_hash
            and getattr(state, "parent_state_id", None) == self.state_id
            and getattr(state, "dynamic_content_hash", None)
            == self.dynamic_content_hash
        )
        if (not same_state and not same_fork_content) or getattr(
            state, "version", None
        ) != self.state_version:
            raise StaleActionError("action was computed for a stale simulation state")
        if getattr(state, "decision_epoch_index", None) != self.epoch_index:
            raise StaleActionError("action eligibility expired with its decision epoch")


@dataclass(frozen=True)
class ActionRealization:
    action: ActionCandidate
    variant_id: str | None
    realized_delay_s: float
    intervention_magnitude: float
    audit: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not np.isfinite(self.realized_delay_s) or self.realized_delay_s < -1e-8:
            raise ValueError("realized action delay must be finite and non-negative")
        if (
            not np.isfinite(self.intervention_magnitude)
            or self.intervention_magnitude < 0.0
        ):
            raise ValueError("intervention magnitude must be finite and non-negative")
