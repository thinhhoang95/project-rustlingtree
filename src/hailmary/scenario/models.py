from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np

from hailmary.ids import canonical_data


JSONPrimitive: TypeAlias = str | int | float | bool | None
FrozenPayload: TypeAlias = tuple[tuple[str, str], ...]


def _array_has_immutable_storage(array: np.ndarray) -> bool:
    """Return whether an array ultimately views an immutable byte buffer."""

    owner: object = array
    seen: set[int] = set()
    while isinstance(owner, np.ndarray):
        if id(owner) in seen:
            return False
        seen.add(id(owner))
        if owner.base is None:
            return False
        owner = owner.base
    return isinstance(owner, bytes)


def _freeze_array(values: np.ndarray) -> np.ndarray:
    """Copy an array onto immutable bytes so ``setflags(write=True)`` cannot thaw it."""

    if values.dtype.hasobject:
        raise TypeError("scenario variants cannot contain object-dtype arrays")
    if values.flags.c_contiguous and not values.flags.writeable and _array_has_immutable_storage(values):
        return values
    contiguous = np.ascontiguousarray(values)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    frozen.setflags(write=False)
    return frozen


def _freeze_variant_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _freeze_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_variant_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, tuple):
        return tuple(_freeze_variant_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_variant_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_variant_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        if params is None or not params.frozen:
            return FrozenVariant.from_object(value)
        updates = {
            item.name: _freeze_variant_value(getattr(value, item.name))
            for item in fields(value)
            if item.init
        }
        if all(getattr(value, name) is item for name, item in updates.items()):
            return value
        return replace(value, **updates)
    raise TypeError(
        "scenario variant fields must be immutable JSON values, arrays, mappings, or frozen dataclasses; "
        f"got {type(value).__qualname__}"
    )


@dataclass(frozen=True, slots=True)
class FrozenVariant(Mapping[str, Any]):
    """Attribute-compatible snapshot used for mapping or mutable variant inputs."""

    source_type: str
    attributes: tuple[tuple[str, Any], ...]

    @classmethod
    def from_object(cls, variant: object) -> "FrozenVariant":
        if isinstance(variant, FrozenVariant):
            return variant
        if isinstance(variant, Mapping):
            raw = dict(variant)
        elif is_dataclass(variant) and not isinstance(variant, type):
            raw = {item.name: getattr(variant, item.name) for item in fields(variant)}
        elif hasattr(variant, "__dict__"):
            raw = {
                str(name): value
                for name, value in vars(variant).items()
                if not str(name).startswith("_")
            }
        else:
            raise TypeError(
                "trajectory variants must be mappings, dataclass instances, or expose public attributes"
            )
        frozen = tuple(
            (str(name), _freeze_variant_value(value))
            for name, value in sorted(raw.items(), key=lambda pair: str(pair[0]))
        )
        result = cls(
            source_type=f"{type(variant).__module__}.{type(variant).__qualname__}",
            attributes=frozen,
        )
        trajectory_variant_id(result)
        return result

    def __getitem__(self, key: str) -> Any:
        for name, value in self.attributes:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self.attributes)

    def __len__(self) -> int:
        return len(self.attributes)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def freeze_variant(variant: object) -> object:
    """Defensively snapshot a trajectory variant at the scenario boundary."""

    if isinstance(variant, FrozenVariant):
        return variant
    if is_dataclass(variant) and not isinstance(variant, type):
        params = getattr(type(variant), "__dataclass_params__", None)
        if params is not None and params.frozen:
            return _freeze_variant_value(variant)
    return FrozenVariant.from_object(variant)


@dataclass(frozen=True, slots=True)
class FrozenWeatherRealization:
    """Canonical immutable snapshot of an arbitrary weather realization."""

    source_type: str
    payload_json: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        payload = json.loads(self.payload_json)
        canonical = _canonical_json(payload)
        object.__setattr__(self, "payload_json", canonical)
        digest = hashlib.sha256(
            f"{self.source_type}\0{canonical}".encode("utf-8")
        ).hexdigest()
        if self.content_hash and self.content_hash != digest:
            raise ValueError("weather content_hash does not match its canonical payload")
        object.__setattr__(self, "content_hash", digest)

    @property
    def payload(self) -> Any:
        return json.loads(self.payload_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "payload": self.payload,
            "content_hash": self.content_hash,
        }


def freeze_weather(weather: object | None) -> FrozenWeatherRealization | None:
    if weather is None or isinstance(weather, FrozenWeatherRealization):
        return weather
    try:
        payload = canonical_data(weather)
    except TypeError:
        if not hasattr(weather, "__dict__"):
            raise TypeError("weather realizations must be canonically serializable") from None
        public = {
            str(name): value
            for name, value in vars(weather).items()
            if not str(name).startswith("_")
        }
        if not public:
            raise TypeError("weather realizations must expose immutable public state") from None
        payload = canonical_data(public)
    return FrozenWeatherRealization(
        source_type=f"{type(weather).__module__}.{type(weather).__qualname__}",
        payload_json=_canonical_json(payload),
    )


def freeze_payload(payload: Mapping[str, Any] | FrozenPayload | None) -> FrozenPayload:
    """Return an immutable, canonically ordered JSON payload.

    Values are stored as canonical JSON strings.  This keeps scenario definitions
    genuinely immutable even when callers pass nested dictionaries or lists.
    """

    if payload is None:
        return ()
    if isinstance(payload, tuple):
        normalized: list[tuple[str, str]] = []
        for key, value in payload:
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("frozen payload entries must be (str, canonical-json str) pairs")
            # Validate that an alleged frozen value is JSON and canonicalize it.
            decoded = json.loads(value)
            normalized.append((key, _canonical_json(decoded)))
        return tuple(sorted(normalized))
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping, frozen payload, or None")
    return tuple(
        (str(key), _canonical_json(value))
        for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
    )


def thaw_payload(payload: FrozenPayload) -> dict[str, Any]:
    return {key: json.loads(value) for key, value in payload}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def trajectory_variant_id(variant: object) -> str:
    if isinstance(variant, Mapping):
        for name in ("variant_id", "content_hash", "id"):
            value = variant.get(name)
            if value is not None and str(value):
                return str(value)
    for name in ("variant_id", "content_hash", "id"):
        value = getattr(variant, name, None)
        if value is not None and str(value):
            return str(value)
    raise ValueError("trajectory variant must expose variant_id, content_hash, or id")


@dataclass(frozen=True, slots=True)
class ResourceDefinition:
    resource_id: str
    kind: str = "runway_threshold"
    required_interval_s: float = 90.0
    metadata: FrozenPayload | Mapping[str, Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        if not self.kind:
            raise ValueError("resource kind must be non-empty")
        if not math.isfinite(self.required_interval_s) or self.required_interval_s <= 0.0:
            raise ValueError("required_interval_s must be finite and positive")
        object.__setattr__(self, "metadata", freeze_payload(self.metadata))

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return thaw_payload(self.metadata)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ActionStationDefinition:
    station_index: int
    s_m: float
    station_type: str = "speed"

    def __post_init__(self) -> None:
        if self.station_index < 0:
            raise ValueError("station_index must be non-negative")
        if not math.isfinite(self.s_m) or self.s_m < 0.0:
            raise ValueError("action-station s_m must be finite and non-negative")
        if not self.station_type:
            raise ValueError("station_type must be non-empty")


@dataclass(frozen=True, slots=True)
class ResourceCrossingDefinition:
    resource_id: str
    s_m: float
    station_index: int = 0

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        if self.station_index < 0:
            raise ValueError("station_index must be non-negative")
        if not math.isfinite(self.s_m) or self.s_m < 0.0:
            raise ValueError("resource-crossing s_m must be finite and non-negative")


@dataclass(frozen=True, slots=True, order=True)
class SegmentTraversalDefinition:
    """Static membership of a flight in one directed route segment.

    Trajectory station values are remaining distance: the entry gate therefore
    has the larger station and the exit gate has the smaller station.
    """

    ordinal: int
    segment_id: str
    entry_resource_id: str
    exit_resource_id: str
    entry_s_m: float
    exit_s_m: float

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("segment traversal ordinal must be non-negative")
        if not self.segment_id or not self.entry_resource_id or not self.exit_resource_id:
            raise ValueError("segment traversal identities must be non-empty")
        if self.entry_resource_id == self.exit_resource_id:
            raise ValueError("segment entry and exit resources must be distinct")
        if not math.isfinite(self.entry_s_m) or not math.isfinite(self.exit_s_m):
            raise ValueError("segment traversal stations must be finite")
        if self.exit_s_m < 0.0 or self.entry_s_m <= self.exit_s_m:
            raise ValueError("segment traversal requires entry_s_m > exit_s_m >= 0")


@dataclass(frozen=True, slots=True)
class FlightDefinition:
    flight_id: str
    release_time_s: float
    baseline_variant_id: str
    cluster_id: str = ""
    callsign: str = ""
    icao24: str = ""
    runway: str = ""
    observed_release_time_s: float | None = None
    release_offset_s: float = 0.0
    action_stations: tuple[ActionStationDefinition, ...] = ()
    resource_crossings: tuple[ResourceCrossingDefinition, ...] = ()
    segment_traversals: tuple[SegmentTraversalDefinition, ...] = ()
    metadata: FrozenPayload | Mapping[str, Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.flight_id:
            raise ValueError("flight_id must be non-empty")
        if not self.baseline_variant_id:
            raise ValueError("baseline_variant_id must be non-empty")
        if not math.isfinite(self.release_time_s):
            raise ValueError("release_time_s must be finite")
        if self.observed_release_time_s is not None and not math.isfinite(self.observed_release_time_s):
            raise ValueError("observed_release_time_s must be finite when supplied")
        if not math.isfinite(self.release_offset_s):
            raise ValueError("release_offset_s must be finite")
        object.__setattr__(self, "action_stations", tuple(self.action_stations))
        object.__setattr__(self, "resource_crossings", tuple(self.resource_crossings))
        object.__setattr__(self, "segment_traversals", tuple(self.segment_traversals))
        object.__setattr__(self, "metadata", freeze_payload(self.metadata))

        station_keys = [(item.station_type, item.station_index) for item in self.action_stations]
        if len(station_keys) != len(set(station_keys)):
            raise ValueError(f"flight {self.flight_id!r} has duplicate action-station identities")
        resource_keys = [item.resource_id for item in self.resource_crossings]
        if len(resource_keys) != len(set(resource_keys)):
            raise ValueError(f"flight {self.flight_id!r} has duplicate resource crossings")
        segment_ids = [item.segment_id for item in self.segment_traversals]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError(f"flight {self.flight_id!r} has duplicate segment traversals")
        ordinals = [item.ordinal for item in self.segment_traversals]
        if ordinals != list(range(len(ordinals))):
            raise ValueError(
                f"flight {self.flight_id!r} segment traversal ordinals must be contiguous"
            )
        crossing_ids = set(resource_keys)
        for traversal in self.segment_traversals:
            if traversal.entry_resource_id not in crossing_ids:
                raise ValueError(
                    f"flight {self.flight_id!r} lacks segment entry crossing "
                    f"{traversal.entry_resource_id!r}"
                )
            if traversal.exit_resource_id not in crossing_ids:
                raise ValueError(
                    f"flight {self.flight_id!r} lacks segment exit crossing "
                    f"{traversal.exit_resource_id!r}"
                )

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return thaw_payload(self.metadata)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class MaterializedExogenousEvent:
    event_id: str
    time_s: float
    stream_name: str = "exogenous"
    payload: FrozenPayload | Mapping[str, Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not self.stream_name:
            raise ValueError("stream_name must be non-empty")
        if not math.isfinite(self.time_s):
            raise ValueError("exogenous-event time_s must be finite")
        object.__setattr__(self, "payload", freeze_payload(self.payload))

    @property
    def payload_dict(self) -> dict[str, Any]:
        return thaw_payload(self.payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    scenario_id: str
    seed: int
    flights: tuple[FlightDefinition, ...]
    resources: tuple[ResourceDefinition, ...]
    variants: tuple[object, ...]
    exogenous_events: tuple[MaterializedExogenousEvent, ...] = ()
    decision_trigger_kinds: tuple[str, ...] = (
        "EXOGENOUS_DISTURBANCE",
        "FLIGHT_RELEASED",
        "ACTION_STATION_CROSSED",
        "RESOURCE_CROSSED",
    )
    weather: FrozenWeatherRealization | object | None = None
    metadata: FrozenPayload | Mapping[str, Any] = field(default_factory=tuple)
    schema_version: str = "hailmary.scenario.v1"
    _definition_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not self.schema_version:
            raise ValueError("schema_version must be non-empty")
        object.__setattr__(self, "flights", tuple(self.flights))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "variants", tuple(freeze_variant(item) for item in self.variants))
        object.__setattr__(self, "exogenous_events", tuple(self.exogenous_events))
        object.__setattr__(self, "decision_trigger_kinds", tuple(str(item) for item in self.decision_trigger_kinds))
        object.__setattr__(self, "weather", freeze_weather(self.weather))
        object.__setattr__(self, "metadata", freeze_payload(self.metadata))

        _require_unique("flight_id", [flight.flight_id for flight in self.flights])
        _require_unique("resource_id", [resource.resource_id for resource in self.resources])
        _require_unique("variant_id", [trajectory_variant_id(variant) for variant in self.variants])
        _require_unique("exogenous event_id", [event.event_id for event in self.exogenous_events])

        variant_ids = set(self.variant_ids)
        resource_ids = set(self.resource_ids)
        for flight in self.flights:
            if flight.baseline_variant_id not in variant_ids:
                raise ValueError(
                    f"flight {flight.flight_id!r} references unknown variant {flight.baseline_variant_id!r}"
                )
            unknown_resources = {
                crossing.resource_id
                for crossing in flight.resource_crossings
                if crossing.resource_id not in resource_ids
            }
            if unknown_resources:
                raise ValueError(
                    f"flight {flight.flight_id!r} references unknown resources {sorted(unknown_resources)!r}"
                )
        from hailmary.simulator.hashing import scenario_definition_hash

        object.__setattr__(self, "_definition_hash", scenario_definition_hash(self))

    @property
    def variant_ids(self) -> tuple[str, ...]:
        return tuple(trajectory_variant_id(variant) for variant in self.variants)

    @property
    def resource_ids(self) -> tuple[str, ...]:
        return tuple(resource.resource_id for resource in self.resources)

    def variant(self, variant_id: str) -> object:
        for variant in self.variants:
            if trajectory_variant_id(variant) == variant_id:
                return variant
        raise KeyError(f"unknown trajectory variant {variant_id!r}")

    def flight(self, flight_id: str) -> FlightDefinition:
        for flight in self.flights:
            if flight.flight_id == flight_id:
                return flight
        raise KeyError(f"unknown flight {flight_id!r}")

    def resource(self, resource_id: str) -> ResourceDefinition:
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        raise KeyError(f"unknown resource {resource_id!r}")

    @property
    def definition_hash(self) -> str:
        return self._definition_hash

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return thaw_payload(self.metadata)  # type: ignore[arg-type]

    @property
    def weather_payload(self) -> Any | None:
        weather = self.weather
        if weather is None:
            return None
        if not isinstance(weather, FrozenWeatherRealization):  # narrowed after __post_init__
            raise TypeError("scenario weather was not frozen")
        return weather.payload

    @property
    def weather_hash(self) -> str | None:
        weather = self.weather
        if weather is None:
            return None
        if not isinstance(weather, FrozenWeatherRealization):
            raise TypeError("scenario weather was not frozen")
        return weather.content_hash


def _require_unique(label: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}: {duplicates!r}")


__all__ = [
    "ActionStationDefinition",
    "FlightDefinition",
    "FrozenVariant",
    "FrozenWeatherRealization",
    "FrozenPayload",
    "MaterializedExogenousEvent",
    "ResourceCrossingDefinition",
    "ResourceDefinition",
    "SegmentTraversalDefinition",
    "ScenarioDefinition",
    "freeze_payload",
    "freeze_variant",
    "freeze_weather",
    "thaw_payload",
    "trajectory_variant_id",
]
