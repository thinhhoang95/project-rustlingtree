"""Infer stable airport-wide flow corridors from continuous medoid geometry.

The builder ignores waypoint identity and destination-runway partitions. It
matches nearby co-directional medoid samples in a common airport frame and
turns sustained matches into directed corridors. A route may therefore join,
split from, and later rejoin a route terminating at a different runway.

Stations follow :class:`TrajectoryVariant`: zero is the runway end and values
increase upstream. Segment geometry and traversals are stored in flight order
from the upstream entry gate to the downstream exit gate.
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
    schema_version: str = "hailmary.route_graph.config.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "hailmary.route_graph.config.v2":
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
            raise ValueError(
                "medoid route coordinates require equal lengths of at least two"
            )
        if any(not math.isfinite(value) or abs(value) > 90.0 for value in lat):
            raise ValueError("medoid route contains invalid latitude")
        if any(not math.isfinite(value) or abs(value) > 180.0 for value in lon):
            raise ValueError("medoid route contains invalid longitude")
        if not math.isfinite(self.dispersion_m) or self.dispersion_m < 0.0:
            raise ValueError("medoid route dispersion must be finite and non-negative")
        object.__setattr__(self, "airport", str(self.airport).strip().upper())
        runway = str(self.runway).strip().upper()
        object.__setattr__(
            self, "runway", runway if runway.startswith("RW") else f"RW{runway}"
        )
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
    runway_ids: tuple[str, ...]
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True, slots=True, order=True)
class RouteSegment:
    segment_id: str
    airport: str
    runway_ids: tuple[str, ...]
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
    schema_version: str = "hailmary.route_graph.v2"
    artifact_content_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != "hailmary.route_graph.v2":
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
        node_ids = {item.node_id for item in self.nodes}
        segment_ids = {item.segment_id for item in self.segments}
        if any(
            item.entry_node_id not in node_ids or item.exit_node_id not in node_ids
            for item in self.segments
        ):
            raise ValueError("route graph segment references an unknown node")
        if any(item.segment_id not in segment_ids for item in self.traversals):
            raise ValueError("route graph traversal references an unknown segment")
        for segment in self.segments:
            if not segment.runway_ids:
                raise ValueError("route graph segment must serve at least one runway")
            if segment.length_m <= 0.0 or segment.corridor_width_m <= 0.0:
                raise ValueError("route graph segment dimensions must be positive")
        traversals_by_cluster: dict[str, list[ClusterSegmentTraversal]] = {}
        for traversal in self.traversals:
            if traversal.entry_s_m <= traversal.exit_s_m:
                raise ValueError(
                    "route graph traversal must point toward decreasing station"
                )
            traversals_by_cluster.setdefault(traversal.qualified_cluster_id, []).append(
                traversal
            )
        for records in traversals_by_cluster.values():
            ordered = sorted(records, key=lambda item: item.ordinal)
            if [item.ordinal for item in ordered] != list(range(len(ordered))):
                raise ValueError("route graph traversal ordinals must be contiguous")
            if any(
                not math.isclose(left.exit_s_m, right.entry_s_m, abs_tol=1.0e-6)
                for left, right in zip(ordered, ordered[1:], strict=False)
            ):
                raise ValueError("route graph traversals must cover a contiguous route")
        payload = self._content_payload()
        computed = content_hash(payload, namespace="hailmary.route_graph.v2")
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

    def traversals_for(
        self, qualified_cluster_id: str
    ) -> tuple[SegmentTraversalDefinition, ...]:
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
                "runway_ids": list(segment.runway_ids),
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
        return {
            **self._content_payload(),
            "artifact_content_hash": self.artifact_content_hash,
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouteGraphArtifact":
        return cls(
            dataset_id=str(payload["dataset_id"]),
            config=RouteGraphConfig(**dict(payload["config"])),
            nodes=tuple(
                RouteGraphNode(**{**item, "runway_ids": tuple(item["runway_ids"])})
                for item in payload["nodes"]
            ),
            segments=tuple(
                RouteSegment(
                    **{
                        **item,
                        "runway_ids": tuple(item["runway_ids"]),
                        "cluster_ids": tuple(item["cluster_ids"]),
                        "lat_deg": tuple(item["lat_deg"]),
                        "lon_deg": tuple(item["lon_deg"]),
                    }
                )
                for item in payload["segments"]
            ),
            traversals=tuple(
                ClusterSegmentTraversal(**item) for item in payload["traversals"]
            ),
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

    def lat_lon(
        self, stations_m: np.ndarray
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        east = np.interp(stations_m, self.stations_m, self.points_m[:, 0])
        north = np.interp(stations_m, self.stations_m, self.points_m[:, 1])
        lat, lon = self.frame.unproject(east, north)
        return tuple(float(value) for value in lat), tuple(
            float(value) for value in lon
        )


@dataclass(frozen=True, slots=True)
class _RouteRun:
    run_id: int
    cluster_id: str
    membership: frozenset[str]
    start_index: int
    stop_index: int
    low_s_m: float
    high_s_m: float


@dataclass(frozen=True, slots=True)
class _SegmentDraft:
    segment_id: str
    airport: str
    runway_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    lat_deg: tuple[float, ...]
    lon_deg: tuple[float, ...]
    length_m: float
    corridor_width_m: float
    ranges: tuple[tuple[str, float, float], ...]


class _UnionFind:
    def __init__(self, items: Iterable[Any]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: Any) -> Any:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: Any, right: Any) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if repr(left_root) <= repr(right_root):
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root

    def groups(self) -> dict[Any, set[Any]]:
        result: dict[Any, set[Any]] = {}
        for item in self.parent:
            result.setdefault(self.find(item), set()).add(item)
        return result


def _sample_route(
    route: MedoidRoute, frame: LocalFrame, step_m: float
) -> _SampledRoute:
    points = frame.project_points(route.lat_deg, route.lon_deg)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment > 1.0e-3))
    points = points[keep]
    if len(points) < 2:
        raise ValueError(f"medoid route {route.qualified_cluster_id!r} has zero length")
    progress = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    total = float(progress[-1])
    remaining = total - progress
    count = max(2, int(math.ceil(total / step_m)) + 1)
    stations = np.linspace(0.0, total, count)
    east = np.interp(stations, remaining[::-1], points[::-1, 0])
    north = np.interp(stations, remaining[::-1], points[::-1, 1])
    sampled = np.column_stack((east, north))
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


def _fill_short_membership_gaps(
    memberships: list[frozenset[str]], maximum_samples: int
) -> list[frozenset[str]]:
    """Apply the same corridor hysteresis to transient membership changes."""

    result = list(memberships)
    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(result):
            stop = index + 1
            while stop < len(result) and result[stop] == result[index]:
                stop += 1
            if (
                index > 0
                and stop < len(result)
                and stop - index <= maximum_samples
                and result[index - 1] == result[stop]
            ):
                result[index:stop] = [result[index - 1]] * (stop - index)
                changed = True
            index = stop
    return result


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return nearest target indices without allocating one huge distance matrix."""

    result = np.empty(len(source), dtype=int)
    for start in range(0, len(source), 512):
        stop = min(start + 512, len(source))
        delta = source[start:stop, None, :] - target[None, :, :]
        result[start:stop] = np.argmin(np.einsum("ijk,ijk->ij", delta, delta), axis=1)
    return result


def _directed_sustained_matches(
    source: _SampledRoute,
    target: _SampledRoute,
    *,
    width_m: float,
    config: RouteGraphConfig,
) -> set[tuple[int, int]]:
    nearest = _nearest_indices(source.points_m, target.points_m)
    target_points = target.points_m[nearest]
    target_tangents = target.tangents[nearest]
    distance = np.linalg.norm(source.points_m - target_points, axis=1)
    dot = np.clip(np.sum(source.tangents * target_tangents, axis=1), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    raw = (distance <= width_m) & (angle <= config.tangent_tolerance_deg)
    gap_samples = int(math.floor(config.hysteresis_gap_nm / config.resample_step_nm))
    mask = _fill_short_false_gaps(raw, gap_samples)
    minimum_m = config.minimum_common_length_nm * M_PER_NM
    matches: set[tuple[int, int]] = set()
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        stop = index + 1
        while stop < len(mask) and mask[stop]:
            if nearest[stop] < nearest[stop - 1]:
                break
            stop += 1
        source_span = float(source.stations_m[stop - 1] - source.stations_m[index])
        target_span = float(
            target.stations_m[nearest[stop - 1]] - target.stations_m[nearest[index]]
        )
        if source_span >= minimum_m and target_span >= minimum_m:
            matches.update((item, int(nearest[item])) for item in range(index, stop))
        index = stop
    return matches


def _pair_matches(
    left: _SampledRoute,
    right: _SampledRoute,
    config: RouteGraphConfig,
) -> set[tuple[int, int]]:
    width_m = max(
        config.lateral_floor_nm * M_PER_NM,
        left.source.dispersion_m,
        right.source.dispersion_m,
    )
    forward = _directed_sustained_matches(left, right, width_m=width_m, config=config)
    reverse = _directed_sustained_matches(right, left, width_m=width_m, config=config)
    # Requiring reciprocal evidence prevents one sample on a diverging branch
    # from being pulled into a long corridor merely because it is the nearest
    # point for several samples on the other route.
    reciprocal = forward & {
        (right_index, left_index) for left_index, right_index in reverse
    }
    ordered = sorted(reciprocal)
    densified = set(reciprocal)
    maximum_gap = max(
        1, int(math.floor(config.hysteresis_gap_nm / config.resample_step_nm)) + 1
    )
    for (left_a, right_a), (left_b, right_b) in zip(ordered, ordered[1:], strict=False):
        left_gap = left_b - left_a
        right_gap = right_b - right_a
        if not (0 < left_gap <= maximum_gap and 0 <= right_gap <= maximum_gap):
            continue
        for left_index in range(left_a + 1, left_b):
            fraction = (left_index - left_a) / left_gap
            right_index = int(round(right_a + fraction * right_gap))
            densified.add((left_index, right_index))
        if right_gap:
            for right_index in range(right_a + 1, right_b):
                fraction = (right_index - right_a) / right_gap
                left_index = int(round(left_a + fraction * left_gap))
                densified.add((left_index, right_index))
    return densified


def _run_boundaries(route: _SampledRoute, start: int, stop: int) -> tuple[float, float]:
    low = (
        0.0
        if start == 0
        else float(0.5 * (route.stations_m[start - 1] + route.stations_m[start]))
    )
    high = (
        route.length_m
        if stop == len(route.stations_m)
        else float(0.5 * (route.stations_m[stop - 1] + route.stations_m[stop]))
    )
    return low, high


def _node_kind(incoming: set[str], outgoing: set[str]) -> str:
    if not incoming:
        return "route_entry"
    if not outgoing:
        return "runway_endpoint"
    if len(incoming) > 1 and len(outgoing) > 1:
        return "merge_split"
    if len(incoming) > 1:
        return "merge"
    if len(outgoing) > 1:
        return "split"
    return "corridor"


def _build_airport_graph(
    airport: str,
    routes: list[MedoidRoute],
    config: RouteGraphConfig,
) -> tuple[list[RouteGraphNode], list[RouteSegment], list[ClusterSegmentTraversal]]:
    step_m = config.resample_step_nm * M_PER_NM
    origin_lat = float(np.median([item.lat_deg[-1] for item in routes]))
    origin_lon = float(np.median([item.lon_deg[-1] for item in routes]))
    frame = LocalFrame(origin_lat, origin_lon)
    sampled = {
        route.qualified_cluster_id: _sample_route(route, frame, step_m)
        for route in routes
    }
    cluster_ids = tuple(sorted(sampled))

    sample_tokens = [
        (cluster_id, index)
        for cluster_id in cluster_ids
        for index in range(len(sampled[cluster_id].stations_m))
    ]
    sample_union = _UnionFind(sample_tokens)
    match_edges: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for left_position, left_id in enumerate(cluster_ids):
        for right_id in cluster_ids[left_position + 1 :]:
            for left_index, right_index in sorted(
                _pair_matches(sampled[left_id], sampled[right_id], config)
            ):
                left_token = (left_id, left_index)
                right_token = (right_id, right_index)
                sample_union.union(left_token, right_token)
                match_edges.append((left_token, right_token))

    membership_by_token: dict[tuple[str, int], frozenset[str]] = {}
    for members in sample_union.groups().values():
        membership = frozenset(item[0] for item in members)
        for token in members:
            membership_by_token[token] = membership

    runs: list[_RouteRun] = []
    run_by_token: dict[tuple[str, int], int] = {}
    runs_by_cluster: dict[str, list[int]] = {
        cluster_id: [] for cluster_id in cluster_ids
    }
    for cluster_id in cluster_ids:
        route = sampled[cluster_id]
        memberships = _fill_short_membership_gaps(
            [
                membership_by_token[(cluster_id, sample_index)]
                for sample_index in range(len(route.stations_m))
            ],
            int(math.floor(config.hysteresis_gap_nm / config.resample_step_nm)),
        )
        index = 0
        while index < len(route.stations_m):
            membership = memberships[index]
            stop = index + 1
            while stop < len(route.stations_m) and memberships[stop] == membership:
                stop += 1
            low, high = _run_boundaries(route, index, stop)
            run = _RouteRun(
                run_id=len(runs),
                cluster_id=cluster_id,
                membership=membership,
                start_index=index,
                stop_index=stop,
                low_s_m=low,
                high_s_m=high,
            )
            runs.append(run)
            runs_by_cluster[cluster_id].append(run.run_id)
            for sample_index in range(index, stop):
                run_by_token[(cluster_id, sample_index)] = run.run_id
            index = stop

    run_union = _UnionFind(range(len(runs)))
    for left_token, right_token in match_edges:
        left_run = run_by_token[left_token]
        right_run = run_by_token[right_token]
        if runs[left_run].membership == runs[right_run].membership:
            run_union.union(left_run, right_run)

    drafts: list[_SegmentDraft] = []
    segment_by_run: dict[int, str] = {}
    for run_ids in sorted(run_union.groups().values(), key=lambda items: min(items)):
        grouped = [runs[item] for item in sorted(run_ids)]
        grouped_clusters = [item.cluster_id for item in grouped]
        if len(set(grouped_clusters)) != len(grouped_clusters):
            details = tuple(
                (item.run_id, item.cluster_id, item.start_index, item.stop_index)
                for item in grouped
            )
            raise ValueError(
                "ambiguous corridor alignment repeats one route in a segment: "
                f"{details!r}"
            )
        cluster_scope = tuple(sorted(grouped_clusters))
        runways = tuple(sorted({sampled[item].source.runway for item in cluster_scope}))
        representative = min(grouped, key=lambda item: item.cluster_id)
        representative_route = sampled[representative.cluster_id]
        geometry_stations = np.linspace(
            representative.high_s_m,
            representative.low_s_m,
            max(
                2,
                int(
                    math.ceil(
                        (representative.high_s_m - representative.low_s_m) / step_m
                    )
                )
                + 1,
            ),
        )
        lat, lon = representative_route.lat_lon(geometry_stations)
        ranges = tuple(
            sorted((item.cluster_id, item.high_s_m, item.low_s_m) for item in grouped)
        )
        segment_payload = {
            "airport": airport,
            "runway_ids": runways,
            "cluster_ids": cluster_scope,
            "ranges_m": tuple(
                (cluster_id, round(high, 3), round(low, 3))
                for cluster_id, high, low in ranges
            ),
            "geometry": tuple(
                (round(a, 8), round(b, 8)) for a, b in zip(lat, lon, strict=True)
            ),
        }
        segment_id = stable_id("segment", segment_payload, length=28)
        draft = _SegmentDraft(
            segment_id=segment_id,
            airport=airport,
            runway_ids=runways,
            cluster_ids=cluster_scope,
            lat_deg=lat,
            lon_deg=lon,
            length_m=float(np.mean([item.high_s_m - item.low_s_m for item in grouped])),
            corridor_width_m=max(
                config.lateral_floor_nm * M_PER_NM,
                *(sampled[item].source.dispersion_m for item in cluster_scope),
            ),
            ranges=ranges,
        )
        drafts.append(draft)
        for run in grouped:
            segment_by_run[run.run_id] = segment_id

    endpoint_tokens = [
        (draft.segment_id, gate) for draft in drafts for gate in ("entry", "exit")
    ]
    endpoint_union = _UnionFind(endpoint_tokens)
    for route_run_ids in runs_by_cluster.values():
        ordered = sorted(route_run_ids, key=lambda item: -runs[item].high_s_m)
        for upstream_run, downstream_run in zip(ordered, ordered[1:], strict=False):
            endpoint_union.union(
                (segment_by_run[upstream_run], "exit"),
                (segment_by_run[downstream_run], "entry"),
            )

    endpoint_points: dict[tuple[str, str], list[np.ndarray]] = {}
    endpoint_runways: dict[tuple[str, str], set[str]] = {}
    for run in runs:
        segment_id = segment_by_run[run.run_id]
        route = sampled[run.cluster_id]
        for gate, station in (("entry", run.high_s_m), ("exit", run.low_s_m)):
            token = (segment_id, gate)
            endpoint_points.setdefault(token, []).append(route.point(station))
            endpoint_runways.setdefault(token, set()).add(route.source.runway)

    nodes: list[RouteGraphNode] = []
    node_by_endpoint: dict[tuple[str, str], str] = {}
    for endpoint_group in endpoint_union.groups().values():
        incoming = {segment_id for segment_id, gate in endpoint_group if gate == "exit"}
        outgoing = {
            segment_id for segment_id, gate in endpoint_group if gate == "entry"
        }
        runway_ids = tuple(
            sorted(
                {
                    runway
                    for token in endpoint_group
                    for runway in endpoint_runways.get(token, set())
                }
            )
        )
        coordinates = [
            point
            for token in sorted(endpoint_group)
            for point in endpoint_points.get(token, ())
        ]
        center = np.mean(np.asarray(coordinates), axis=0)
        lat, lon = frame.unproject(np.asarray([center[0]]), np.asarray([center[1]]))
        node_lat = round(float(lat[0]), 8)
        node_lon = round(float(lon[0]), 8)
        kind = _node_kind(incoming, outgoing)
        node_payload = {
            "airport": airport,
            "runway_ids": runway_ids,
            "kind": kind,
            "incoming": tuple(sorted(incoming)),
            "outgoing": tuple(sorted(outgoing)),
            "lat": node_lat,
            "lon": node_lon,
        }
        node_id = stable_id("route_node", node_payload, length=24)
        nodes.append(
            RouteGraphNode(
                node_id=node_id,
                kind=kind,
                airport=airport,
                runway_ids=runway_ids,
                lat_deg=node_lat,
                lon_deg=node_lon,
            )
        )
        for token in endpoint_group:
            node_by_endpoint[token] = node_id

    segments = [
        RouteSegment(
            segment_id=draft.segment_id,
            airport=draft.airport,
            runway_ids=draft.runway_ids,
            entry_node_id=node_by_endpoint[(draft.segment_id, "entry")],
            exit_node_id=node_by_endpoint[(draft.segment_id, "exit")],
            cluster_ids=draft.cluster_ids,
            lat_deg=draft.lat_deg,
            lon_deg=draft.lon_deg,
            length_m=draft.length_m,
            corridor_width_m=draft.corridor_width_m,
        )
        for draft in drafts
    ]

    traversals: list[ClusterSegmentTraversal] = []
    for cluster_id, route_run_ids in runs_by_cluster.items():
        ordered = sorted(route_run_ids, key=lambda item: -runs[item].high_s_m)
        for ordinal, run_id in enumerate(ordered):
            run = runs[run_id]
            traversals.append(
                ClusterSegmentTraversal(
                    qualified_cluster_id=cluster_id,
                    ordinal=ordinal,
                    segment_id=segment_by_run[run_id],
                    entry_s_m=run.high_s_m,
                    exit_s_m=run.low_s_m,
                )
            )
    return nodes, segments, traversals


def build_route_graph(
    routes: Iterable[MedoidRoute],
    *,
    config: RouteGraphConfig | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> RouteGraphArtifact:
    """Build a deterministic airport-wide graph without waypoint identities."""

    cfg = RouteGraphConfig() if config is None else config
    sources = tuple(sorted(routes, key=lambda item: item.qualified_cluster_id))
    if not sources:
        raise ValueError("route graph requires at least one medoid route")
    if len({item.qualified_cluster_id for item in sources}) != len(sources):
        raise ValueError("route graph medoid cluster identities must be unique")
    dataset_ids = {item.dataset_id for item in sources}
    if len(dataset_ids) != 1:
        raise ValueError("one route graph artifact cannot mix datasets")

    partitions: dict[str, list[MedoidRoute]] = {}
    for route in sources:
        partitions.setdefault(route.airport, []).append(route)

    all_nodes: list[RouteGraphNode] = []
    all_segments: list[RouteSegment] = []
    all_traversals: list[ClusterSegmentTraversal] = []
    for airport, airport_routes in sorted(partitions.items()):
        nodes, segments, traversals = _build_airport_graph(airport, airport_routes, cfg)
        all_nodes.extend(nodes)
        all_segments.extend(segments)
        all_traversals.extend(traversals)

    return RouteGraphArtifact(
        dataset_id=next(iter(dataset_ids)),
        config=cfg,
        nodes=tuple(all_nodes),
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
    qualified = (
        cluster_id if cluster_id.count(":") >= 2 else f"{airport}:{runway}:{cluster_id}"
    )
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
        resource_crossings=tuple(
            sorted(existing.values(), key=lambda item: item.resource_id)
        ),
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
