from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, Mapping

from hailmary.scenario.models import FrozenPayload, freeze_payload, thaw_payload


class EventKind(StrEnum):
    EXOGENOUS_DISTURBANCE = "EXOGENOUS_DISTURBANCE"
    FLIGHT_RELEASED = "FLIGHT_RELEASED"
    ACTION_STATION_CROSSED = "ACTION_STATION_CROSSED"
    RESOURCE_CROSSED = "RESOURCE_CROSSED"
    FLIGHT_COMPLETED = "FLIGHT_COMPLETED"


EVENT_PRIORITY: dict[EventKind, int] = {
    EventKind.EXOGENOUS_DISTURBANCE: 0,
    EventKind.FLIGHT_RELEASED: 1,
    EventKind.ACTION_STATION_CROSSED: 2,
    EventKind.RESOURCE_CROSSED: 3,
    EventKind.FLIGHT_COMPLETED: 4,
}


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    time_s: float
    kind: EventKind
    event_id: str
    insertion_sequence: int
    flight_id: str = ""
    station_index: int = -1
    resource_id: str = ""
    payload: FrozenPayload | Mapping[str, Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EventKind(self.kind))
        if not math.isfinite(self.time_s):
            raise ValueError("event time_s must be finite")
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if self.insertion_sequence < 0:
            raise ValueError("insertion_sequence must be non-negative")
        object.__setattr__(self, "payload", freeze_payload(self.payload))

    @property
    def priority(self) -> int:
        return EVENT_PRIORITY[self.kind]  # type: ignore[index]

    @property
    def sort_key(self) -> tuple[float, int, str, int, int]:
        return (
            float(self.time_s),
            self.priority,
            self.flight_id,
            int(self.station_index),
            int(self.insertion_sequence),
        )

    @property
    def payload_dict(self) -> dict[str, Any]:
        return thaw_payload(self.payload)  # type: ignore[arg-type]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ScheduledEvent):
            return NotImplemented
        return self.sort_key < other.sort_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "kind": self.kind.value,
            "event_id": self.event_id,
            "insertion_sequence": self.insertion_sequence,
            "flight_id": self.flight_id,
            "station_index": self.station_index,
            "resource_id": self.resource_id,
            "payload": self.payload_dict,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScheduledEvent":
        return cls(
            time_s=float(payload["time_s"]),
            kind=EventKind(str(payload["kind"])),
            event_id=str(payload["event_id"]),
            insertion_sequence=int(payload["insertion_sequence"]),
            flight_id=str(payload.get("flight_id", "")),
            station_index=int(payload.get("station_index", -1)),
            resource_id=str(payload.get("resource_id", "")),
            payload=payload.get("payload", {}),
        )


@dataclass(frozen=True, slots=True)
class DecisionEpoch:
    epoch_index: int
    time_s: float
    state_version: int
    state_id: str
    trigger_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EventBatchResult:
    time_s: float
    events: tuple[ScheduledEvent, ...]
    decision_epoch: DecisionEpoch | None
    state_id_before: str
    state_id_after: str


def event_sort_key(event: ScheduledEvent) -> tuple[float, int, str, int, int]:
    return event.sort_key


__all__ = [
    "DecisionEpoch",
    "EVENT_PRIORITY",
    "EventBatchResult",
    "EventKind",
    "ScheduledEvent",
    "event_sort_key",
]
