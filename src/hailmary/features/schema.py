"""Versioned feature schemas and finite learning vectors.

The learning boundary is deliberately stricter than the diagnostic boundary:
named diagnostics may retain optional/raw values, while every value handed to
the learner is a finite ``float64`` in a stable, hashed order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from hailmary._arrays import readonly_float64
from hailmary.ids import content_hash


@dataclass(frozen=True)
class FeatureField:
    """One ordered scalar in a :class:`FeatureSchema`."""

    name: str
    unit: str = "1"
    normalization: str = "identity"
    lower_bound: float | None = None
    upper_bound: float | None = None
    missingness_mask: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("feature names must be non-empty stripped strings")
        if self.lower_bound is not None and not np.isfinite(self.lower_bound):
            raise ValueError(f"{self.name}.lower_bound must be finite")
        if self.upper_bound is not None and not np.isfinite(self.upper_bound):
            raise ValueError(f"{self.name}.upper_bound must be finite")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError(f"{self.name} has reversed bounds")


@dataclass(frozen=True)
class FeatureSchema:
    """An immutable ordered schema whose hash changes with its contract."""

    schema_version: str
    fields: tuple[FeatureField, ...]
    schema_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version cannot be empty")
        if not self.fields:
            raise ValueError("a feature schema requires at least one field")
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        name_set = set(names)
        for feature in self.fields:
            if feature.missingness_mask is not None and feature.missingness_mask not in name_set:
                raise ValueError(
                    f"{feature.name} references unknown missingness mask {feature.missingness_mask!r}"
                )
        object.__setattr__(
            self,
            "schema_hash",
            content_hash(
                {
                    "schema_version": self.schema_version,
                    "fields": self.fields,
                },
                namespace="hailmary.feature_schema",
            ),
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.fields)

    def encode(
        self,
        named_values: Mapping[str, float],
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "FeatureVector":
        missing = [name for name in self.names if name not in named_values]
        if missing:
            raise ValueError(f"missing feature values: {missing}")
        extras = sorted(set(named_values) - set(self.names))
        if extras:
            raise ValueError(f"unknown feature values: {extras}")

        values = np.asarray([float(named_values[name]) for name in self.names], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            bad = [name for name, value in zip(self.names, values, strict=True) if not np.isfinite(value)]
            raise ValueError(f"learning vector contains non-finite values: {bad}")

        tolerance = 1e-12
        for feature, value in zip(self.fields, values, strict=True):
            if feature.lower_bound is not None and value < feature.lower_bound - tolerance:
                raise ValueError(f"{feature.name}={value} is below {feature.lower_bound}")
            if feature.upper_bound is not None and value > feature.upper_bound + tolerance:
                raise ValueError(f"{feature.name}={value} is above {feature.upper_bound}")

        return FeatureVector(
            schema=self,
            values=values,
            named=named_values,
            diagnostics={} if diagnostics is None else diagnostics,
        )


@dataclass(frozen=True)
class FeatureVector:
    """Finite ordered values plus immutable named/debug views."""

    schema: FeatureSchema
    values: np.ndarray
    named: Mapping[str, float]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = readonly_float64(self.values, name="feature vector")
        if len(values) != len(self.schema.fields):
            raise ValueError(
                f"feature vector length {len(values)} does not match schema length {len(self.schema.fields)}"
            )
        named = {name: float(self.named[name]) for name in self.schema.names}
        if not all(np.isfinite(value) for value in named.values()):
            raise ValueError("named feature values must be finite")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "named", MappingProxyType(named))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def schema_hash(self) -> str:
        return self.schema.schema_hash

    def value(self, name: str) -> float:
        return float(self.named[name])


def leader_follower_feature_schema(
    schema_version: str = "hailmary.features.leader_follower.v1",
) -> FeatureSchema:
    """Return the version-1 leader/follower learning-vector contract."""

    nonnegative = 0.0
    unit_interval = (0.0, 1.0)
    fields = (
        FeatureField("spacing_deviation_s", "s"),
        FeatureField("abs_spacing_deviation_s", "s", lower_bound=nonnegative),
        FeatureField("required_delay_s", "s", lower_bound=nonnegative),
        FeatureField("predicted_interval_s", "s"),
        FeatureField("required_interval_s", "s", lower_bound=nonnegative),
        FeatureField("follower_time_to_resource_s", "s"),
        FeatureField("leader_time_to_resource_s", "s"),
        FeatureField("follower_distance_to_resource_m", "m", lower_bound=nonnegative),
        FeatureField("leader_distance_to_resource_m", "m", lower_bound=nonnegative),
        FeatureField("follower_cas_kts", "kt", lower_bound=nonnegative),
        FeatureField("follower_cas_lower_kts", "kt", lower_bound=nonnegative),
        FeatureField("follower_cas_margin_kts", "kt"),
        FeatureField("speed_capacity_s", "s", lower_bound=nonnegative),
        FeatureField("path_capacity_s", "s", lower_bound=nonnegative),
        FeatureField(
            "required_delay_over_speed_capacity",
            "1",
            lower_bound=0.0,
            upper_bound=10.0,
            missingness_mask="speed_capacity_undefined_mask",
        ),
        FeatureField(
            "required_delay_over_path_capacity",
            "1",
            lower_bound=0.0,
            upper_bound=10.0,
            missingness_mask="path_capacity_undefined_mask",
        ),
        FeatureField("commitment_fraction", "1", lower_bound=unit_interval[0], upper_bound=unit_interval[1]),
        FeatureField(
            "intercept_or_final_gate_flag",
            "1",
            lower_bound=unit_interval[0],
            upper_bound=unit_interval[1],
        ),
        FeatureField(
            "remaining_action_station_fraction",
            "1",
            lower_bound=unit_interval[0],
            upper_bound=unit_interval[1],
        ),
        FeatureField("local_flow_count", "count", lower_bound=nonnegative),
        FeatureField("pressure_ratio", "1", lower_bound=nonnegative),
        FeatureField(
            "trailing_min_spacing_margin_s",
            "s",
            missingness_mask="trailing_spacing_undefined_mask",
        ),
        FeatureField("cluster_index", "category_index", lower_bound=nonnegative),
        FeatureField("speed_capacity_undefined_mask", "1", lower_bound=0.0, upper_bound=1.0),
        FeatureField("path_capacity_undefined_mask", "1", lower_bound=0.0, upper_bound=1.0),
        FeatureField("trailing_spacing_undefined_mask", "1", lower_bound=0.0, upper_bound=1.0),
    )
    return FeatureSchema(schema_version=schema_version, fields=fields)


__all__ = [
    "FeatureField",
    "FeatureSchema",
    "FeatureVector",
    "leader_follower_feature_schema",
]
