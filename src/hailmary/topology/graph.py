"""Infer stable directed segments from continuous medoid geometry.

The builder intentionally ignores waypoint identity.  It compares resampled
polylines by lateral distance and tangent direction, requires a sustained
corridor, and applies a small hysteresis gap before creating merge boundaries.
The station convention matches :class:`TrajectoryVariant`: zero is the runway
end and values increase upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from hailmary.config import M_PER_NM
from hailmary.geometry.frame import LocalFrame
from hailmary.ids import canonical_data, content_hash, stable_id
from hailmary.scenario.models import (
    ResourceCrossingDefinition,
    ResourceDefinition,
    SegmentTraversalDefinition,
)


@dataclass(frozen=True, slots=True)
class RouteGraphConfig:
    lateral_floor_nm: float = 0.5
    tangent_tolerance_deg: float = 15.0
    minimum_common_length_nm: float = 5.0
    resample_step_nm: float = 0.25
    hysteresis_gap_nm: float = 0.75
    required_interval_s: float = 90.0
    schema_version: str = "hailmary.route_graph.config.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "hailmary.route_graph.config.v1":
            raise ValueError("unsupported route-graph configuration schema")
        positive = (
            self.lateral_floor_nm,
            self.tangent_tolerance_deg,
            self.minimum_common_length_nm,
            self.resample_step_nm,
            self.required_interval_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("route-graph thresholds must be finite and positive")
        if self.tangent_tolerance_deg >= 90.0:
            raise ValueError("tangent_tolerance_deg must be below 90 degrees")
        if not math.isfinite(self.hysteresis_gap_nm) or self.hysteresis_gap_nm < 0.0:
            raise ValueError("hysteresis_gap_nm must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MedoidRoute:
    dataset_id: str
    airport: str
    runway: str
    cluster_id: str
    lat_deg: tuple[float, ...]
    lon_deg: tuple[float, ...]
    dispersion_m: float = 0.0
    medoid_flight_id: str = ""
    source_hash: str = ""

    def __post_init__(self) -> None:
        identities = (self.dataset_id, self.airport, self.runway, self.cluster_id)
        if any(not str(value).strip() for value in identities):
            raise ValueError("medoid route identities must be non-empty")
        lat = tuple(float(value) for value in self.lat_deg)
        lon = tuple(float(value) for value in self.lon_deg)
        if len(lat) != len(lon) or len(lat) < 2:
            raise ValueError("medoid route coordinates require equal lengths of at least two")
        if any(not math.isfinite(value) or abs(value) > 90.0 for value in lat):
            raise ValueError("medoid route contains invalid latitude")
        if any(not math.isfinite(value) or abs(value) > 180.0 for value in lon):
            raise ValueError("medoid route contains invalid longitude")
        if not math.isfinite(self.dispersion_m) or self.dispersion_m < 0.0:
            raise ValueError("medoid route dispersion must be finite and non-negative")
        object.__setattr__(self, "airport", str(self.airport).strip().upper())
        runway = str(self.runway).strip().upper()
        object.__setattr__(self, "runway", runway if runway.startswith("RW") else f"RW{runway}")
        object.__setattr__(self, "lat_deg", lat)
        object.__setattr__(self, "lon_deg", lon)

    @property
    def qualified_cluster_id(self) -> str:
        return f"{self.airport}:{self.runway}:{self.cluster_id}"


@dataclass(frozen=True, slots=True, order=True)
class RouteGraphNode:
    node_id: str
    kind: str
    airport: str
    runway: str
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True, slots=True, order=True)
class RouteSegment:
    segment_id: str
    airport: str
    runway: str
    entry_node_id: str
    exit_node_id: str
    cluster_ids: tuple[str, ...]
    lat_deg: tuple[float, ...]
    lon_deg: tuple[float, ...]
    length_m: float
    corridor_width_m: float

    @property
    def entry_resource_id(self) -> str:
        return f"{self.segment_id}:entry"

    @property
    def exit_resource_id(self) -> str:
        return f"{self.segment_id}:exit"


@dataclass(frozen=True, slots=True, order=True)
class ClusterSegmentTraversal:
    qualified_cluster_id: str
    ordinal: int
    segment_id: str
    entry_s_m: float
    exit_s_m: float


@dataclass(frozen=True, slots=True)
class RouteGraphArtifact:
    dataset_id: str
    config: RouteGraphConfig
    nodes: tuple[RouteGraphNode, ...]
    segments: tuple[RouteSegment, ...]
    traversals: tuple[ClusterSegmentTraversal, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "hailmary.route_graph.v1"
    artifact_content_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != "hailmary.route_graph.v1":
            raise ValueError("unsupported route-graph artifact schema")
        if not str(self.dataset_id).strip():
            raise ValueError("route graph dataset_id must be non-empty")
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes)))
        object.__setattr__(self, "segments", tuple(sorted(self.segments)))
        object.__setattr__(self, "traversals", tuple(sorted(self.traversals)))
        if len({item.node_id for item in self.nodes}) != len(self.nodes):
            raise ValueError("route graph contains duplicate node IDs")
        if len({item.segment_id for item in self.segments}) != len(self.segments):
            raise ValueError("route graph contains duplicate segment IDs")
        segment_ids = {item.segment_id for item in self.segments}
        if any(item.segment_id not in segment_ids for item in self.traversals):
            raise ValueError("route graph traversal references an unknown segment")
        payload = self._content_payload()
        computed = content_hash(payload, namespace="hailmary.route_graph.v1")
        if self.artifact_content_hash and self.artifact_content_hash != computed:
            raise ValueError("route graph artifact_content_hash does not match content")
        object.__setattr__(self, "provenance", dict(sorted(self.provenance.items())))
        object.__setattr__(self, "artifact_content_hash", computed)

    def _content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "config": canonical_data(self.config),
            "nodes": canonical_data(tuple(sorted(self.nodes))),
            "segments": canonical_data(tuple(sorted(self.segments))),
            "traversals": canonical_data(tuple(sorted(self.traversals))),
            "provenance": canonical_data(dict(sorted(self.provenance.items()))),
        }

    def segment(self, segment_id: str) -> RouteSegment:
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise KeyError(segment_id)

    def traversals_for(self, qualified_cluster_id: str) -> tuple[SegmentTraversalDefinition, ...]:
        records = sorted(
            (
                item
                for item in self.traversals
                if item.qualified_cluster_id == qualified_cluster_id
            ),
            key=lambda item: item.ordinal,
        )
        return tuple(
            SegmentTraversalDefinition(
                ordinal=index,
                segment_id=item.segment_id,
                entry_resource_id=self.segment(item.segment_id).entry_resource_id,
                exit_resource_id=self.segment(item.segment_id).exit_resource_id,
                entry_s_m=item.entry_s_m,
                exit_s_m=item.exit_s_m,
            )
            for index, item in enumerate(records)
        )

    def resources(self) -> tuple[ResourceDefinition, ...]:
        result: list[ResourceDefinition] = []
        for segment in self.segments:
            metadata = {
                "airport": segment.airport,
                "runway": segment.runway,
                "segment_id": segment.segment_id,
            }
            result.extend(
                (
                    ResourceDefinition(
                        segment.entry_resource_id,
                        kind="segment_entry",
                        required_interval_s=self.config.required_interval_s,
                        metadata={**metadata, "gate": "entry"},
                    ),
                    ResourceDefinition(
                        segment.exit_resource_id,
                        kind="segment_exit",
                        required_interval_s=self.config.required_interval_s,
                        metadata={**metadata, "gate": "exit"},
                    ),
                )
            )
        return tuple(sorted(result, key=lambda item: item.resource_id))

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_payload(), "artifact_content_hash": self.artifact_content_hash}

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouteGraphArtifact":
        return cls(
            dataset_id=str(payload["dataset_id"]),
            config=RouteGraphConfig(**dict(payload["config"])),
            nodes=tuple(RouteGraphNode(**item) for item in payload["nodes"]),
            segments=tuple(
                RouteSegment(
                    **{
                        **item,
                        "cluster_ids": tuple(item["cluster_ids"]),
                        "lat_deg": tuple(item["lat_deg"]),
                        "lon_deg": tuple(item["lon_deg"]),
                    }
                )
                for item in payload["segments"]
            ),
            traversals=tuple(ClusterSegmentTraversal(**item) for item in payload["traversals"]),
            provenance=dict(payload.get("provenance", {})),
            schema_version=str(payload.get("schema_version", "")),
            artifact_content_hash=str(payload.get("artifact_content_hash", "")),
        )

    @classmethod
    def read(cls, path: str | Path) -> "RouteGraphArtifact":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class _SampledRoute:
    source: MedoidRoute
    stations_m: np.ndarray
    points_m: np.ndarray
    tangents: np.ndarray
    frame: LocalFrame

    @property
    def length_m(self) -> float:
        return float(self.stations_m[-1])

    def point(self, station_m: float) -> np.ndarray:
        station = float(np.clip(station_m, 0.0, self.length_m))
        return np.asarray(
            [
                np.interp(station, self.stations_m, self.points_m[:, 0]),
                np.interp(station, self.stations_m, self.points_m[:, 1]),
            ]
        )

    def lat_lon(self, stations_m: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
        east = np.interp(stations_m, self.stations_m, self.points_m[:, 0])
        north = np.interp(stations_m, self.stations_m, self.points_m[:, 1])
        lat, lon = self.frame.unproject(east, north)
        return tuple(float(value) for value in lat), tuple(float(value) for value in lon)


def _sample_route(route: MedoidRoute, frame: LocalFrame, step_m: float) -> _SampledRoute:
    points = frame.project_points(route.lat_deg, route.lon_deg)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment > 1.0e-3))
    points = points[keep]
    if len(points) < 2:
        raise ValueError(f"medoid route {route.qualified_cluster_id!r} has zero length")
    progress = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    total = float(progress[-1])
    remaining = total - progress
    count = max(2, int(math.ceil(total / step_m)) + 1)
    stations = np.linspace(0.0, total, count)
    east = np.interp(stations, remaining[::-1], points[::-1, 0])
    north = np.interp(stations, remaining[::-1], points[::-1, 1])
    sampled = np.column_stack((east, north))
    # Derivatives point upstream for increasing remaining-distance station.  A
    # common sign reversal leaves the angle between flight-direction tangents unchanged.
    tangent = np.gradient(sampled, stations, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    tangent = tangent / np.maximum(norm[:, None], 1.0e-12)
    return _SampledRoute(route, stations, sampled, tangent, frame)


def _fill_short_false_gaps(mask: np.ndarray, maximum_samples: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        stop = index
        while stop < len(result) and not result[stop]:
            stop += 1
        if index > 0 and stop < len(result) and stop - index <= maximum_samples:
            result[index:stop] = True
        index = stop
    return result


def _drop_short_true_runs(mask: np.ndarray, minimum_samples: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    index = 0
    while index < len(result):
        if not result[index]:
            index += 1
            continue
        stop = index
        while stop < len(result) and result[stop]:
            stop += 1
        if stop - index < minimum_samples:
            result[index:stop] = False
        index = stop
    return result


def _components(active_ids: tuple[str, ...], edges: set[tuple[str, str]]) -> dict[str, frozenset[str]]:
    remaining = set(active_ids)
    result: dict[str, frozenset[str]] = {}
    while remaining:
        root = min(remaining)
        component = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            neighbors = {
                right if left == current else left
                for left, right in edges
                if left == current or right == current
            }
            for neighbor in sorted(neighbors - component):
                component.add(neighbor)
                frontier.append(neighbor)
        frozen = frozenset(component)
        for member in component:
            result[member] = frozen
        remaining.difference_update(component)
    return result


def build_route_graph(
    routes: Iterable[MedoidRoute],
    *,
    config: RouteGraphConfig | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> RouteGraphArtifact:
    """Build a deterministic graph without consulting waypoint sequences."""

    cfg = RouteGraphConfig() if config is None else config
    sources = tuple(sorted(routes, key=lambda item: item.qualified_cluster_id))
    if not sources:
        raise ValueError("route graph requires at least one medoid route")
    if len({item.qualified_cluster_id for item in sources}) != len(sources):
        raise ValueError("route graph medoid cluster identities must be unique")
    dataset_ids = {item.dataset_id for item in sources}
    if len(dataset_ids) != 1:
        raise ValueError("one route graph artifact cannot mix datasets")

    all_nodes: dict[str, RouteGraphNode] = {}
    all_segments: list[RouteSegment] = []
    all_traversals: list[ClusterSegmentTraversal] = []
    step_m = cfg.resample_step_nm * M_PER_NM
    minimum_samples = max(2, int(math.ceil(cfg.minimum_common_length_nm / cfg.resample_step_nm)))
    gap_samples = int(math.floor(cfg.hysteresis_gap_nm / cfg.resample_step_nm))

    partitions: dict[tuple[str, str], list[MedoidRoute]] = {}
    for route in sources:
        partitions.setdefault((route.airport, route.runway), []).append(route)

    for (airport, runway), partition in sorted(partitions.items()):
        origin_lat = float(np.median([item.lat_deg[-1] for item in partition]))
        origin_lon = float(np.median([item.lon_deg[-1] for item in partition]))
        frame = LocalFrame(origin_lat, origin_lon)
        sampled = {
            route.qualified_cluster_id: _sample_route(route, frame, step_m)
            for route in partition
        }
        maximum = max(item.length_m for item in sampled.values())
        grid = np.arange(0.0, maximum + step_m * 0.5, step_m)
        relations_by_index: list[set[tuple[str, str]]] = [set() for _ in grid]
        cluster_ids = tuple(sorted(sampled))
        for left_index, left_id in enumerate(cluster_ids):
            left = sampled[left_id]
            for right_id in cluster_ids[left_index + 1 :]:
                right = sampled[right_id]
                common_count = int(np.searchsorted(grid, min(left.length_m, right.length_m), side="right"))
                if common_count < 2:
                    continue
                common_grid = grid[:common_count]
                left_points = np.column_stack(
                    (
                        np.interp(common_grid, left.stations_m, left.points_m[:, 0]),
                        np.interp(common_grid, left.stations_m, left.points_m[:, 1]),
                    )
                )
                right_points = np.column_stack(
                    (
                        np.interp(common_grid, right.stations_m, right.points_m[:, 0]),
                        np.interp(common_grid, right.stations_m, right.points_m[:, 1]),
                    )
                )
                left_tx = np.interp(common_grid, left.stations_m, left.tangents[:, 0])
                left_ty = np.interp(common_grid, left.stations_m, left.tangents[:, 1])
                right_tx = np.interp(common_grid, right.stations_m, right.tangents[:, 0])
                right_ty = np.interp(common_grid, right.stations_m, right.tangents[:, 1])
                dot = np.clip(left_tx * right_tx + left_ty * right_ty, -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))
                width = max(
                    cfg.lateral_floor_nm * M_PER_NM,
                    left.source.dispersion_m,
                    right.source.dispersion_m,
                )
                raw = (np.linalg.norm(left_points - right_points, axis=1) <= width) & (
                    angle <= cfg.tangent_tolerance_deg
                )
                sustained = _drop_short_true_runs(
                    _fill_short_false_gaps(raw, gap_samples), minimum_samples
                )
                pair = (left_id, right_id)
                for index in np.flatnonzero(sustained):
                    relations_by_index[int(index)].add(pair)

        membership_by_cluster: dict[str, list[frozenset[str] | None]] = {
            cluster_id: [] for cluster_id in cluster_ids
        }
        for index, station in enumerate(grid):
            active = tuple(
                cluster_id
                for cluster_id in cluster_ids
                if station <= sampled[cluster_id].length_m + 1.0e-6
            )
            components = _components(active, relations_by_index[index])
            for cluster_id in cluster_ids:
                membership_by_cluster[cluster_id].append(components.get(cluster_id))

        emitted: dict[tuple[frozenset[str], int, int], RouteSegment] = {}
        traversal_ranges: dict[str, list[tuple[float, float, RouteSegment]]] = {
            cluster_id: [] for cluster_id in cluster_ids
        }
        for cluster_id in cluster_ids:
            memberships = membership_by_cluster[cluster_id]
            start = 0
            while start < len(grid):
                membership = memberships[start]
                if membership is None:
                    break
                stop = start + 1
                while stop < len(grid) and memberships[stop] == membership:
                    stop += 1
                low = float(grid[start])
                route_length = sampled[cluster_id].length_m
                high = min(float(grid[min(stop, len(grid) - 1)]), route_length)
                if stop == len(grid) or memberships[stop] is None:
                    high = route_length
                if high - low > 1.0:
                    key = (membership, start, stop)
                    segment = emitted.get(key)
                    if segment is None:
                        representative_id = min(membership)
                        representative = sampled[representative_id]
                        geometry_stations = np.linspace(high, low, max(2, int(math.ceil((high - low) / step_m)) + 1))
                        lat, lon = representative.lat_lon(geometry_stations)
                        entry_payload = {
                            "airport": airport,
                            "runway": runway,
                            "clusters": tuple(sorted(membership)),
                            "station_m": round(high, 3),
                            "lat": round(lat[0], 8),
                            "lon": round(lon[0], 8),
                        }
                        exit_payload = {
                            "airport": airport,
                            "runway": runway,
                            "clusters": tuple(sorted(membership)),
                            "station_m": round(low, 3),
                            "lat": round(lat[-1], 8),
                            "lon": round(lon[-1], 8),
                        }
                        entry_node_id = stable_id("route_node", entry_payload, length=24)
                        exit_node_id = stable_id("route_node", exit_payload, length=24)
                        downstream_membership = memberships[start - 1] if start > 0 else None
                        upstream_membership = memberships[stop] if stop < len(memberships) else None
                        entry_kind = "merge" if upstream_membership != membership else "corridor"
                        exit_kind = (
                            "runway_endpoint"
                            if low <= 1.0
                            else "merge"
                            if downstream_membership != membership
                            else "corridor"
                        )
                        all_nodes[entry_node_id] = RouteGraphNode(
                            entry_node_id, entry_kind, airport, runway, lat[0], lon[0]
                        )
                        all_nodes[exit_node_id] = RouteGraphNode(
                            exit_node_id, exit_kind, airport, runway, lat[-1], lon[-1]
                        )
                        corridor_width = max(
                            cfg.lateral_floor_nm * M_PER_NM,
                            *(sampled[item].source.dispersion_m for item in membership),
                        )
                        segment_payload = {
                            "airport": airport,
                            "runway": runway,
                            "entry_node_id": entry_node_id,
                            "exit_node_id": exit_node_id,
                            "cluster_ids": tuple(sorted(membership)),
                            "geometry": tuple(
                                (round(a, 8), round(b, 8))
                                for a, b in zip(lat, lon, strict=True)
                            ),
                        }
                        segment = RouteSegment(
                            segment_id=stable_id("segment", segment_payload, length=28),
                            airport=airport,
                            runway=runway,
                            entry_node_id=entry_node_id,
                            exit_node_id=exit_node_id,
                            cluster_ids=tuple(sorted(membership)),
                            lat_deg=lat,
                            lon_deg=lon,
                            length_m=high - low,
                            corridor_width_m=corridor_width,
                        )
                        emitted[key] = segment
                        all_segments.append(segment)
                    traversal_ranges[cluster_id].append((high, low, segment))
                start = stop

        for cluster_id, ranges in traversal_ranges.items():
            for ordinal, (high, low, segment) in enumerate(
                sorted(ranges, key=lambda item: (-item[0], -item[1], item[2].segment_id))
            ):
                all_traversals.append(
                    ClusterSegmentTraversal(cluster_id, ordinal, segment.segment_id, high, low)
                )

    return RouteGraphArtifact(
        dataset_id=next(iter(dataset_ids)),
        config=cfg,
        nodes=tuple(all_nodes.values()),
        segments=tuple(all_segments),
        traversals=tuple(all_traversals),
        provenance={} if provenance is None else provenance,
    )


def attach_route_graph(spec: Any, artifact: RouteGraphArtifact) -> Any:
    """Return a flight generation spec with explicit segment resources attached."""

    airport = str(spec.metadata.get("airport", "")).strip().upper()
    runway = str(spec.runway).strip().upper()
    runway = runway if runway.startswith("RW") else f"RW{runway}"
    cluster_id = str(spec.cluster_id)
    qualified = cluster_id if cluster_id.count(":") >= 2 else f"{airport}:{runway}:{cluster_id}"
    traversals = artifact.traversals_for(qualified)
    existing = {item.resource_id: item for item in spec.resource_crossings}
    for traversal in traversals:
        existing[traversal.entry_resource_id] = ResourceCrossingDefinition(
            traversal.entry_resource_id, traversal.entry_s_m
        )
        existing[traversal.exit_resource_id] = ResourceCrossingDefinition(
            traversal.exit_resource_id, traversal.exit_s_m
        )
    return replace(
        spec,
        cluster_id=qualified,
        resource_crossings=tuple(sorted(existing.values(), key=lambda item: item.resource_id)),
        segment_traversals=traversals,
    )


__all__ = [
    "ClusterSegmentTraversal",
    "MedoidRoute",
    "RouteGraphArtifact",
    "RouteGraphConfig",
    "RouteGraphNode",
    "RouteSegment",
    "attach_route_graph",
    "build_route_graph",
]
