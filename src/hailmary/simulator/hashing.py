from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hailmary.ids import content_hash as _shared_content_hash
from hailmary.ids import provenance_state_id as _shared_provenance_state_id


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any, *, prefix: str = "") -> str:
    # Route all public IDs through the package-wide canonical identifier layer.
    return _shared_content_hash(
        canonicalize(value),
        namespace=prefix or "hailmary.simulator",
    )


def canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical hashes do not permit non-finite floats")
        # JSON's shortest round-trippable float representation is deterministic.
        return value
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: canonicalize(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, tuple | list):
        return [canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        normalized = [canonicalize(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))

    explicit_hash = getattr(value, "content_hash", None)
    if isinstance(explicit_hash, str) and explicit_hash:
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "content_hash": explicit_hash,
        }
    # Opaque objects (for example an immutable weather provider) are identified
    # by public scalar attributes.  Never include object IDs or default reprs.
    public_values = {
        name: item
        for name, item in vars(value).items()
        if not name.startswith("_") and isinstance(item, str | bool | int | float | type(None))
    } if hasattr(value, "__dict__") else {}
    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "attributes": canonicalize(public_values),
    }


def scenario_definition_hash(definition: Any) -> str:
    return content_hash(definition, prefix="hailmary-scenario-definition-v1")


def simulation_dynamic_payload(state: Any) -> dict[str, Any]:
    events = sorted(state.event_heap, key=lambda event: event.sort_key)
    flights = sorted(state.flights, key=lambda flight: flight.flight_id)
    metrics = sorted(state.metrics, key=lambda item: item[0])
    return {
        "schema": "hailmary-simulation-dynamic-v1",
        "definition_hash": state.definition.definition_hash,
        "sim_time_s": state.sim_time_s,
        "event_sequence": state.event_sequence,
        "flights": flights,
        "pending_events": events,
        "rng_state": json.loads(state.rng_state_json),
        "metrics": metrics,
        "action_log": state.action_log,
        "exogenous_state": state.exogenous_state,
        "exogenous_event_log": state.exogenous_event_log,
        "decision_epoch_index": state.decision_epoch_index,
    }


def dynamic_content_hash(state: Any) -> str:
    return content_hash(simulation_dynamic_payload(state), prefix="hailmary-dynamic-state-v1")


def provenance_state_id(
    dynamic_hash: str,
    *,
    parent_state_id: str | None,
    branch_label: str,
    transition: str,
    lineage_version: int,
) -> str:
    return _shared_provenance_state_id(
        dynamic_hash,
        parent_state_id=parent_state_id,
        branch_label=f"{branch_label}:{transition}",
        lineage_sequence=lineage_version,
    )


__all__ = [
    "canonical_json_bytes",
    "canonicalize",
    "content_hash",
    "dynamic_content_hash",
    "provenance_state_id",
    "scenario_definition_hash",
    "simulation_dynamic_payload",
]
