"""Runway-partitioned preparation of raw ADS-B tracks for clustering.

The lower-level clustering artifact builder intentionally accepts already
projected and resampled tracks.  This module owns the data-domain boundary in
front of it: matching catalog arrivals to raw tracks, reconstructing observed
terminal-entry releases, aligning tracks to one runway threshold, and retaining
the raw source used by every selected medoid.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from hailmary.config import ClusteringConfig, M_PER_NM
from hailmary.data.adsb import RawADSBTrack, TerminalEntry
from hailmary.data.catalog import CatalogArrival, normalize_runway
from hailmary.errors import ArtifactValidationError, NoEligibleArrivalsError
from hailmary.geometry.frame import LocalFrame
from hailmary.geometry.polyline import (
    orient_upstream_to_threshold,
    readonly_float64,
    remove_duplicate_neighbors,
    resample_polyline,
)

from .artifact import ClusterLibrary, build_cluster_library


class TrackRejectionReason(StrEnum):
    """Stable, machine-readable reasons an arrival was not clusterable."""

    MISSING_RAW_TRACK = "missing_raw_track"
    MISSING_TRACK = "missing_raw_track"
    DUPLICATE_CATALOG_FLIGHT_ID = "duplicate_catalog_flight_id"
    DUPLICATE_RAW_TRACK_ID = "duplicate_raw_track_id"
    DUPLICATE_TRACK_ID = "duplicate_raw_track_id"
    THRESHOLD_MISMATCH = "threshold_mismatch"
    NO_INBOUND_CROSSING = "no_inbound_crossing"
    NO_TERMINAL_ENTRY = "no_inbound_crossing"
    LANDING_PRECEDES_ENTRY = "landing_precedes_entry"
    INSUFFICIENT_POINTS = "insufficient_points"
    TOO_FEW_POINTS = "insufficient_points"
    ENDPOINT_OUTSIDE_CAPTURE_RADIUS = "endpoint_outside_capture_radius"
    OUTSIDE_THRESHOLD_CAPTURE_RADIUS = "endpoint_outside_capture_radius"
    INVALID_GEOMETRY = "invalid_geometry"


# A concise public alias for callers that do not need the track qualifier.
RejectionReason = TrackRejectionReason


@dataclass(frozen=True)
class TrackRejection:
    """One catalog arrival rejected before clustering."""

    flight_id: str
    reason: TrackRejectionReason
    detail: str

    def __post_init__(self) -> None:
        flight_id = str(self.flight_id).strip()
        if not flight_id:
            raise ValueError("rejected flight_id must be nonempty")
        reason = self.reason
        if not isinstance(reason, TrackRejectionReason):
            reason = TrackRejectionReason(str(reason))
        object.__setattr__(self, "flight_id", flight_id)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "detail", str(self.detail).strip())

    def to_dict(self) -> dict[str, str]:
        return {
            "flight_id": self.flight_id,
            "reason": self.reason.value,
            "detail": self.detail,
        }


# Compatibility spelling that reads naturally in result annotations.
ADSBTrackRejection = TrackRejection


def _readonly_source(values: object, *, name: str) -> np.ndarray:
    return readonly_float64(values, name=name, ndim=1)


@dataclass(frozen=True)
class PreparedADSBTrack:
    """One accepted arrival and its clustering/raw-source representations.

    ``source_*`` spans the interpolated terminal-entry crossing through the
    observed landing endpoint.  It deliberately does not invent a timestamp
    for the appended runway threshold.  ``aligned_points_m`` is the geometric
    source with that exact threshold attached, while ``resampled_points_m`` is
    the fixed-width clustering input.
    """

    catalog_arrival: CatalogArrival
    raw_track: RawADSBTrack
    terminal_entry: TerminalEntry
    source_time_s: np.ndarray
    source_lat_deg: np.ndarray
    source_lon_deg: np.ndarray
    source_geoaltitude_m: np.ndarray
    aligned_points_m: np.ndarray
    resampled_points_m: np.ndarray

    def __post_init__(self) -> None:
        if self.catalog_arrival.flight_id != self.raw_track.flight_id:
            raise ValueError("catalog and raw track flight IDs must match")
        if self.terminal_entry.flight_id != self.raw_track.flight_id:
            raise ValueError("terminal entry and raw track flight IDs must match")
        arrays = {
            "source_time_s": _readonly_source(self.source_time_s, name="source_time_s"),
            "source_lat_deg": _readonly_source(self.source_lat_deg, name="source_lat_deg"),
            "source_lon_deg": _readonly_source(self.source_lon_deg, name="source_lon_deg"),
            "source_geoaltitude_m": _readonly_source(
                self.source_geoaltitude_m,
                name="source_geoaltitude_m",
            ),
        }
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
            raise ValueError("clipped source arrays must have equal lengths of at least two")
        if np.any(np.diff(arrays["source_time_s"]) <= 0.0):
            raise ValueError("clipped source times must be strictly increasing")
        aligned = readonly_float64(self.aligned_points_m, name="aligned_points_m", ndim=2)
        resampled = readonly_float64(self.resampled_points_m, name="resampled_points_m", ndim=2)
        if aligned.shape[1:] != (2,) or len(aligned) < 2:
            raise ValueError("aligned_points_m must have shape (n, 2), n >= 2")
        if resampled.shape[1:] != (2,) or len(resampled) < 2:
            raise ValueError("resampled_points_m must have shape (n, 2), n >= 2")
        if not np.isclose(arrays["source_time_s"][0], self.terminal_entry.time_s):
            raise ValueError("clipped source must start at the terminal entry time")
        if not np.allclose(aligned[-1], np.zeros(2), atol=1.0e-7):
            raise ValueError("aligned ADS-B geometry must end at the runway threshold")
        if not np.allclose(resampled[-1], np.zeros(2), atol=1.0e-7):
            raise ValueError("resampled ADS-B geometry must end at the runway threshold")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "aligned_points_m", aligned)
        object.__setattr__(self, "resampled_points_m", resampled)

    @property
    def flight_id(self) -> str:
        return self.raw_track.flight_id

    @property
    def observed_release_time_s(self) -> float:
        return float(self.terminal_entry.time_s)

    @property
    def release_time_s(self) -> float:
        return self.observed_release_time_s

    @property
    def points_m(self) -> np.ndarray:
        """Alias for the fixed-width clustering input."""

        return self.resampled_points_m

    @property
    def clipped_points_m(self) -> np.ndarray:
        """Alias for the aligned, pre-resampling geometry."""

        return self.aligned_points_m

    def to_medoid_track(self) -> Any:
        """Return the executable raw medoid source with an exact threshold end.

        When the last observed sample is merely within the capture radius, its
        coordinate is snapped at the same timestamp. If raw coverage ends
        before the catalog landing event, an explicit threshold sample is
        appended at that event time with the last observed altitude. This
        choice is deterministic and visible in the cluster/template provenance.
        """

        from hailmary.templates.compiler import MedoidTrack
        from hailmary.templates.speed import finite_difference_ground_speed

        time_s = np.array(self.source_time_s, dtype=np.float64, copy=True)
        lat_deg = np.array(self.source_lat_deg, dtype=np.float64, copy=True)
        lon_deg = np.array(self.source_lon_deg, dtype=np.float64, copy=True)
        altitude_m = np.array(self.source_geoaltitude_m, dtype=np.float64, copy=True)
        threshold_lat = float(self.catalog_arrival.threshold_lat_deg)
        threshold_lon = float(self.catalog_arrival.threshold_lon_deg)
        event_time = float(self.catalog_arrival.event_time_s)
        speed_frame = LocalFrame(threshold_lat, threshold_lon)
        source_east_m, source_north_m = speed_frame.project(
            lat_deg,
            lon_deg,
        )
        # Derive observed ground speed before exact-threshold alignment.  The
        # catalog threshold is authoritative geometry, but snapping a sparse
        # final ADS-B point at the same timestamp must not create a fictitious
        # terminal speed spike.
        try:
            ground_speed_mps: np.ndarray | None = finite_difference_ground_speed(
                time_s,
                source_east_m,
                source_north_m,
            )
        except ArtifactValidationError:
            # Clustering remains geometry-only.  An unrealistic synthetic or
            # poor-quality raw speed series is retained as unavailable here so
            # the template compiler can apply its normal quality rejection
            # rather than making the medoid source itself unreadable.
            ground_speed_mps = None
        if event_time > float(time_s[-1]) + 1e-9:
            time_s = np.append(time_s, event_time)
            lat_deg = np.append(lat_deg, threshold_lat)
            lon_deg = np.append(lon_deg, threshold_lon)
            altitude_m = np.append(altitude_m, altitude_m[-1])
            if ground_speed_mps is not None:
                ground_speed_mps = np.append(ground_speed_mps, ground_speed_mps[-1])
        else:
            lat_deg[-1] = threshold_lat
            lon_deg[-1] = threshold_lon
        return MedoidTrack(
            flight_id=self.flight_id,
            time_s=time_s,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            altitude_m=altitude_m,
            ground_speed_mps=ground_speed_mps,
        )


@dataclass(frozen=True)
class ADSBPreparationResult:
    """Accepted and rejected records for exactly one airport/runway."""

    airport: str
    runway: str
    frame: LocalFrame
    prepared_tracks: tuple[PreparedADSBTrack, ...]
    rejections: tuple[TrackRejection, ...]
    terminal_radius_nm: float
    threshold_capture_radius_nm: float
    n_resample: int

    def __post_init__(self) -> None:
        airport = str(self.airport).strip().upper()
        runway = normalize_runway(self.runway)
        if not airport:
            raise ValueError("airport must be nonempty")
        prepared = tuple(sorted(self.prepared_tracks, key=lambda item: item.flight_id))
        rejected = tuple(sorted(self.rejections, key=lambda item: (item.flight_id, item.reason.value)))
        accepted_ids = [item.flight_id for item in prepared]
        rejected_ids = [item.flight_id for item in rejected]
        if len(set(accepted_ids)) != len(accepted_ids):
            raise ValueError("prepared track flight IDs must be unique")
        if len(set(rejected_ids)) != len(rejected_ids):
            raise ValueError("rejected track flight IDs must be unique")
        if set(accepted_ids).intersection(rejected_ids):
            raise ValueError("a flight cannot be both prepared and rejected")
        if any(item.catalog_arrival.runway != runway for item in prepared):
            raise ValueError("prepared tracks must belong to the declared runway")
        if any(len(item.resampled_points_m) != int(self.n_resample) for item in prepared):
            raise ValueError("prepared tracks must have exactly n_resample stations")
        if not np.isfinite(self.terminal_radius_nm) or self.terminal_radius_nm <= 0.0:
            raise ValueError("terminal_radius_nm must be finite and positive")
        if (
            not np.isfinite(self.threshold_capture_radius_nm)
            or self.threshold_capture_radius_nm <= 0.0
        ):
            raise ValueError("threshold_capture_radius_nm must be finite and positive")
        if isinstance(self.n_resample, bool) or int(self.n_resample) != self.n_resample or self.n_resample < 2:
            raise ValueError("n_resample must be an integer at least two")
        object.__setattr__(self, "airport", airport)
        object.__setattr__(self, "runway", runway)
        object.__setattr__(self, "prepared_tracks", prepared)
        object.__setattr__(self, "rejections", rejected)
        object.__setattr__(self, "terminal_radius_nm", float(self.terminal_radius_nm))
        object.__setattr__(self, "threshold_capture_radius_nm", float(self.threshold_capture_radius_nm))
        object.__setattr__(self, "n_resample", int(self.n_resample))

    @property
    def accepted_tracks(self) -> tuple[PreparedADSBTrack, ...]:
        return self.prepared_tracks

    @property
    def accepted(self) -> tuple[PreparedADSBTrack, ...]:
        return self.prepared_tracks

    @property
    def resampled_tracks_m(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({item.flight_id: item.resampled_points_m for item in self.prepared_tracks})

    @property
    def tracks_by_flight_id(self) -> Mapping[str, PreparedADSBTrack]:
        return MappingProxyType({item.flight_id: item for item in self.prepared_tracks})

    @property
    def observed_release_times_s(self) -> Mapping[str, float]:
        return MappingProxyType(
            {item.flight_id: item.observed_release_time_s for item in self.prepared_tracks}
        )

    @property
    def release_times_s(self) -> Mapping[str, float]:
        return self.observed_release_times_s

    @property
    def selected_arrival_count(self) -> int:
        return len(self.prepared_tracks) + len(self.rejections)


@dataclass(frozen=True)
class ADSBClusterBuildResult:
    """Cluster artifact plus the release and raw-medoid data it came from."""

    library: ClusterLibrary
    preparation: ADSBPreparationResult
    raw_medoid_tracks: Mapping[str, RawADSBTrack]

    def __post_init__(self) -> None:
        sources = dict(self.raw_medoid_tracks)
        expected_ids = {item.medoid_flight_id for item in self.library.medoids}
        if set(sources) != expected_ids:
            raise ValueError("raw medoid sources must exactly match cluster medoids")
        if any(key != value.flight_id for key, value in sources.items()):
            raise ValueError("raw medoid source keys must match raw track flight IDs")
        object.__setattr__(self, "raw_medoid_tracks", MappingProxyType(dict(sorted(sources.items()))))

    @property
    def cluster_library(self) -> ClusterLibrary:
        return self.library

    @property
    def artifact(self) -> ClusterLibrary:
        return self.library

    @property
    def rejections(self) -> tuple[TrackRejection, ...]:
        return self.preparation.rejections

    @property
    def observed_release_times_s(self) -> Mapping[str, float]:
        return self.preparation.observed_release_times_s

    @property
    def raw_medoid_sources(self) -> Mapping[str, RawADSBTrack]:
        return self.raw_medoid_tracks

    @property
    def template_medoid_tracks(self) -> Mapping[str, Any]:
        prepared = self.preparation.tracks_by_flight_id
        return MappingProxyType(
            {
                medoid.medoid_flight_id: prepared[medoid.medoid_flight_id].to_medoid_track()
                for medoid in self.library.medoids
            }
        )


@dataclass(frozen=True)
class ADSBMultiRunwayBuildResult:
    """Complete eligible runway corpus with per-partition artifacts."""

    dataset_id: str
    airport: str
    runway_results: tuple[ADSBClusterBuildResult, ...]
    rejection_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        results = tuple(sorted(self.runway_results, key=lambda item: item.library.runway))
        runways = [item.library.runway for item in results]
        if len(runways) != len(set(runways)):
            raise ValueError("multi-runway build contains duplicate runway partitions")
        object.__setattr__(self, "runway_results", results)
        object.__setattr__(self, "rejection_counts", tuple(sorted(self.rejection_counts)))

    @property
    def libraries(self) -> tuple[ClusterLibrary, ...]:
        return tuple(item.library for item in self.runway_results)

    @property
    def accepted_track_count(self) -> int:
        return sum(len(item.preparation.prepared_tracks) for item in self.runway_results)


def _coerce_pipeline_config(config: object | None) -> object:
    resolved = ClusteringConfig() if config is None else config
    required = ("n_resample", "terminal_radius_nm", "threshold_capture_radius_nm")
    missing = [name for name in required if not hasattr(resolved, name)]
    if missing:
        raise TypeError(f"clustering config is missing ADS-B preparation fields: {missing}")
    n_resample = getattr(resolved, "n_resample")
    terminal_radius_nm = float(getattr(resolved, "terminal_radius_nm"))
    capture_radius_nm = float(getattr(resolved, "threshold_capture_radius_nm"))
    if isinstance(n_resample, bool) or int(n_resample) != n_resample or int(n_resample) < 2:
        raise ValueError("config.n_resample must be an integer at least two")
    if not np.isfinite(terminal_radius_nm) or terminal_radius_nm <= 0.0:
        raise ValueError("config.terminal_radius_nm must be finite and positive")
    if not np.isfinite(capture_radius_nm) or capture_radius_nm <= 0.0:
        raise ValueError("config.threshold_capture_radius_nm must be finite and positive")
    return resolved


def _truncate_at_landing(
    track: RawADSBTrack,
    event_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Clip raw samples at the catalog landing event, interpolating if needed."""

    landing_time_s = min(float(event_time_s), float(track.time_s[-1]))
    if landing_time_s <= float(track.time_s[0]):
        return None
    insertion = int(np.searchsorted(track.time_s, landing_time_s, side="left"))
    if insertion < len(track.time_s) and np.isclose(
        track.time_s[insertion],
        landing_time_s,
        rtol=0.0,
        atol=1.0e-9,
    ):
        stop = insertion + 1
        return (
            track.time_s[:stop],
            track.lat_deg[:stop],
            track.lon_deg[:stop],
            track.geoaltitude_m[:stop],
        )
    if insertion >= len(track.time_s):
        return track.time_s, track.lat_deg, track.lon_deg, track.geoaltitude_m
    left = insertion - 1
    span_s = float(track.time_s[insertion] - track.time_s[left])
    fraction = (landing_time_s - float(track.time_s[left])) / span_s

    def interpolate(values: np.ndarray) -> float:
        return float(values[left] + fraction * (values[insertion] - values[left]))

    return (
        readonly_float64(
            np.concatenate((track.time_s[:insertion], [landing_time_s])),
            name="landing-clipped time",
            ndim=1,
        ),
        readonly_float64(
            np.concatenate((track.lat_deg[:insertion], [interpolate(track.lat_deg)])),
            name="landing-clipped latitude",
            ndim=1,
        ),
        readonly_float64(
            np.concatenate((track.lon_deg[:insertion], [interpolate(track.lon_deg)])),
            name="landing-clipped longitude",
            ndim=1,
        ),
        readonly_float64(
            np.concatenate(
                (track.geoaltitude_m[:insertion], [interpolate(track.geoaltitude_m)])
            ),
            name="landing-clipped altitude",
            ndim=1,
        ),
    )


def _circle_crossing_fraction(p0: np.ndarray, p1: np.ndarray, radius_m: float) -> float:
    delta = p1 - p0
    a = float(np.dot(delta, delta))
    if a <= 0.0:
        raise ValueError("terminal-boundary crossing segment has zero length")
    b = 2.0 * float(np.dot(p0, delta))
    c = float(np.dot(p0, p0) - radius_m * radius_m)
    discriminant = b * b - 4.0 * a * c
    if discriminant < -1.0e-6:
        raise ValueError("terminal-boundary crossing segment does not intersect the boundary")
    root = np.sqrt(max(0.0, discriminant))
    candidates = sorted(
        value
        for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
        if -1.0e-12 <= value <= 1.0 + 1.0e-12
    )
    if not candidates:
        raise ValueError("terminal-boundary intersection lies outside its source segment")
    # An outside-to-inside segment has one relevant root.  Selecting the last
    # in-range root also handles a start point lying exactly on the boundary.
    return float(np.clip(candidates[-1], 0.0, 1.0))


def _final_inbound_entry(
    track: RawADSBTrack,
    frame: LocalFrame,
    *,
    radius_m: float,
    time_s: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    altitude_m: np.ndarray,
) -> tuple[TerminalEntry, int, float, np.ndarray] | None:
    points_m = frame.project_points(lat_deg, lon_deg)
    distances_m = np.linalg.norm(points_m, axis=1)
    tolerance_m = max(1.0e-7 * radius_m, 1.0e-6)
    inbound_indices = [
        index
        for index in range(len(points_m) - 1)
        if distances_m[index] >= radius_m - tolerance_m
        and distances_m[index + 1] <= radius_m + tolerance_m
        and (
            distances_m[index] > radius_m + tolerance_m
            or distances_m[index + 1] < radius_m - tolerance_m
        )
    ]
    if not inbound_indices:
        return None
    index = inbound_indices[-1]
    fraction = _circle_crossing_fraction(points_m[index], points_m[index + 1], radius_m)
    crossing_m = points_m[index] + fraction * (points_m[index + 1] - points_m[index])
    crossing_lat, crossing_lon = frame.unproject(crossing_m[0], crossing_m[1])

    def interpolate(values: np.ndarray) -> float:
        return float(values[index] + fraction * (values[index + 1] - values[index]))

    entry = TerminalEntry(
        flight_id=track.flight_id,
        time_s=interpolate(time_s),
        lat_deg=float(crossing_lat),
        lon_deg=float(crossing_lon),
        geoaltitude_m=interpolate(altitude_m),
        segment_index=index,
        segment_fraction=fraction,
        radius_m=radius_m,
    )
    return entry, index, fraction, points_m


def _source_from_crossing(
    entry: TerminalEntry,
    index: int,
    fraction: float,
    *,
    time_s: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    altitude_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # When the root is exactly the segment's second endpoint, avoid adding the
    # same source sample twice (which would violate strict source time order).
    start = index + 2 if np.isclose(fraction, 1.0, rtol=0.0, atol=1.0e-12) else index + 1
    return (
        readonly_float64(
            np.concatenate(([entry.time_s], time_s[start:])),
            name="source_time_s",
            ndim=1,
        ),
        readonly_float64(
            np.concatenate(([entry.lat_deg], lat_deg[start:])),
            name="source_lat_deg",
            ndim=1,
        ),
        readonly_float64(
            np.concatenate(([entry.lon_deg], lon_deg[start:])),
            name="source_lon_deg",
            ndim=1,
        ),
        readonly_float64(
            np.concatenate(([entry.geoaltitude_m], altitude_m[start:])),
            name="source_geoaltitude_m",
            ndim=1,
        ),
    )


def _partition_arrivals(
    catalog_arrivals: Iterable[CatalogArrival],
    *,
    airport: str,
    runway: str,
) -> tuple[CatalogArrival, ...]:
    selected = [
        arrival
        for arrival in catalog_arrivals
        if arrival.runway == runway
        and (arrival.airport is None or str(arrival.airport).strip().upper() == airport)
    ]
    return tuple(sorted(selected, key=lambda item: (item.flight_id, item.event_time_s)))


def _reference_arrival(arrivals: tuple[CatalogArrival, ...]) -> CatalogArrival:
    """Choose the modal threshold deterministically, protecting against an outlier."""

    by_threshold: dict[tuple[float, float], list[CatalogArrival]] = defaultdict(list)
    for arrival in arrivals:
        key = (round(arrival.threshold_lat_deg, 9), round(arrival.threshold_lon_deg, 9))
        by_threshold[key].append(arrival)
    _, candidates = sorted(
        by_threshold.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[0]
    return min(candidates, key=lambda item: (item.flight_id, item.event_time_s))


def prepare_adsb_tracks_for_clustering(
    catalog_arrivals: Iterable[CatalogArrival],
    raw_adsb_tracks: Iterable[RawADSBTrack],
    *,
    airport: str,
    runway: str,
    config: object | None = None,
    threshold_consistency_tolerance_m: float = 25.0,
) -> ADSBPreparationResult:
    """Prepare a single runway partition and retain typed rejection records."""

    resolved_config = _coerce_pipeline_config(config)
    airport_token = str(airport).strip().upper()
    runway_token = normalize_runway(runway)
    if not airport_token:
        raise ValueError("airport must be nonempty")
    if (
        not np.isfinite(threshold_consistency_tolerance_m)
        or threshold_consistency_tolerance_m < 0.0
    ):
        raise ValueError("threshold_consistency_tolerance_m must be finite and nonnegative")

    selected = _partition_arrivals(
        tuple(catalog_arrivals),
        airport=airport_token,
        runway=runway_token,
    )
    if not selected:
        raise ValueError(f"catalog contains no arrivals for {airport_token}/{runway_token}")
    reference = _reference_arrival(selected)
    frame = LocalFrame(reference.threshold_lat_deg, reference.threshold_lon_deg)

    raw_tracks = tuple(raw_adsb_tracks)
    raw_by_id: dict[str, list[RawADSBTrack]] = defaultdict(list)
    for track in raw_tracks:
        raw_by_id[track.flight_id].append(track)
    catalog_counts = Counter(item.flight_id for item in selected)
    duplicate_catalog_ids = {item for item, count in catalog_counts.items() if count > 1}

    terminal_radius_nm = float(getattr(resolved_config, "terminal_radius_nm"))
    capture_radius_nm = float(getattr(resolved_config, "threshold_capture_radius_nm"))
    n_resample = int(getattr(resolved_config, "n_resample"))
    terminal_radius_m = terminal_radius_nm * M_PER_NM
    capture_radius_m = capture_radius_nm * M_PER_NM
    prepared: list[PreparedADSBTrack] = []
    rejected: list[TrackRejection] = []

    for arrival in selected:
        flight_id = arrival.flight_id
        if flight_id in duplicate_catalog_ids:
            if not any(item.flight_id == flight_id for item in rejected):
                rejected.append(
                    TrackRejection(
                        flight_id,
                        TrackRejectionReason.DUPLICATE_CATALOG_FLIGHT_ID,
                        "the selected runway partition contains multiple catalog records",
                    )
                )
            continue
        candidates = raw_by_id.get(flight_id, [])
        if not candidates:
            rejected.append(
                TrackRejection(
                    flight_id,
                    TrackRejectionReason.MISSING_RAW_TRACK,
                    "no raw ADS-B track has the catalog flight_id",
                )
            )
            continue
        if len(candidates) > 1:
            rejected.append(
                TrackRejection(
                    flight_id,
                    TrackRejectionReason.DUPLICATE_RAW_TRACK_ID,
                    "multiple raw ADS-B tracks have the catalog flight_id",
                )
            )
            continue

        threshold_xy = frame.project_points(
            [arrival.threshold_lat_deg],
            [arrival.threshold_lon_deg],
        )[0]
        threshold_delta_m = float(np.linalg.norm(threshold_xy))
        if threshold_delta_m > threshold_consistency_tolerance_m:
            rejected.append(
                TrackRejection(
                    flight_id,
                    TrackRejectionReason.THRESHOLD_MISMATCH,
                    (
                        f"catalog threshold differs from the partition threshold by "
                        f"{threshold_delta_m:.3f} m"
                    ),
                )
            )
            continue

        raw_track = candidates[0]
        truncated = _truncate_at_landing(raw_track, arrival.event_time_s)
        if truncated is None or len(truncated[0]) < 2:
            rejected.append(
                TrackRejection(
                    flight_id,
                    TrackRejectionReason.LANDING_PRECEDES_ENTRY,
                    "the catalog landing event does not leave two raw source samples",
                )
            )
            continue
        time_s, lat_deg, lon_deg, altitude_m = truncated
        try:
            crossing = _final_inbound_entry(
                raw_track,
                frame,
                radius_m=terminal_radius_m,
                time_s=time_s,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                altitude_m=altitude_m,
            )
        except ValueError as exc:
            rejected.append(
                TrackRejection(
                    flight_id,
                    TrackRejectionReason.INVALID_GEOMETRY,
                    str(exc),
                )
            )
            continue
        if crossing is None:
            rejected.append(
                TrackRejection(
                    flight_id,
                    TrackRejectionReason.NO_INBOUND_CROSSING,
                    f"no outside-to-inside crossing of the {terminal_radius_nm:g}-NM boundary",
                )
            )
            continue
        entry, crossing_index, fraction, _ = crossing
        source = _source_from_crossing(
            entry,
            crossing_index,
            fraction,
            time_s=time_s,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            altitude_m=altitude_m,
        )
        if len(source[0]) < 2:
            rejected.append(
                TrackRejection(
                    flight_id,
                    TrackRejectionReason.INSUFFICIENT_POINTS,
                    "the final inbound crossing has no later landing sample",
                )
            )
            continue
        source_points_m = frame.project_points(source[1], source[2])
        endpoint_distance_m = float(np.linalg.norm(source_points_m[-1]))
        if endpoint_distance_m > capture_radius_m + 1.0e-7:
            rejected.append(
                TrackRejection(
                    flight_id,
                    TrackRejectionReason.ENDPOINT_OUTSIDE_CAPTURE_RADIUS,
                    (
                        f"landing endpoint is {endpoint_distance_m:.3f} m from the threshold; "
                        f"capture radius is {capture_radius_m:.3f} m"
                    ),
                )
            )
            continue

        if endpoint_distance_m <= 1.0e-7:
            aligned = np.array(source_points_m, dtype=np.float64, copy=True)
            aligned[-1] = 0.0
        else:
            aligned = np.vstack((source_points_m, np.zeros(2, dtype=np.float64)))
        try:
            aligned = orient_upstream_to_threshold(aligned)
            aligned = remove_duplicate_neighbors(aligned, tolerance_m=1.0e-7)
            if len(aligned) < 2:
                raise ValueError("aligned path has fewer than two distinct points")
            sampled = resample_polyline(aligned, n_points=n_resample)
        except ValueError as exc:
            rejected.append(
                TrackRejection(
                    flight_id,
                    TrackRejectionReason.INSUFFICIENT_POINTS,
                    str(exc),
                )
            )
            continue
        prepared.append(
            PreparedADSBTrack(
                catalog_arrival=arrival,
                raw_track=raw_track,
                terminal_entry=entry,
                source_time_s=source[0],
                source_lat_deg=source[1],
                source_lon_deg=source[2],
                source_geoaltitude_m=source[3],
                aligned_points_m=aligned,
                resampled_points_m=sampled.points_m,
            )
        )

    return ADSBPreparationResult(
        airport=airport_token,
        runway=runway_token,
        frame=frame,
        prepared_tracks=tuple(prepared),
        rejections=tuple(rejected),
        terminal_radius_nm=terminal_radius_nm,
        threshold_capture_radius_nm=capture_radius_nm,
        n_resample=n_resample,
    )


def _release_provenance(preparation: ADSBPreparationResult) -> list[dict[str, Any]]:
    return [
        {
            "flight_id": item.flight_id,
            "observed_release_time_s": item.observed_release_time_s,
            "crossing_segment_index": item.terminal_entry.segment_index,
            "crossing_segment_fraction": item.terminal_entry.segment_fraction,
            "terminal_radius_m": item.terminal_entry.radius_m,
            "raw_source_point_count": item.raw_track.point_count,
            "clipped_source_point_count": len(item.source_time_s),
        }
        for item in preparation.prepared_tracks
    ]


def build_cluster_library_from_adsb(
    catalog_arrivals: Iterable[CatalogArrival],
    raw_adsb_tracks: Iterable[RawADSBTrack],
    *,
    dataset_id: str,
    airport: str,
    runway: str,
    config: object | None = None,
    acceptance_quantile: float | None = None,
    acceptance_multiplier: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    threshold_consistency_tolerance_m: float = 25.0,
) -> ADSBClusterBuildResult:
    """Prepare raw arrivals, build their cluster library, and retain sources."""

    resolved_config = _coerce_pipeline_config(config)
    preparation = prepare_adsb_tracks_for_clustering(
        catalog_arrivals,
        raw_adsb_tracks,
        airport=airport,
        runway=runway,
        config=resolved_config,
        threshold_consistency_tolerance_m=threshold_consistency_tolerance_m,
    )
    if not preparation.prepared_tracks:
        reason_counts = Counter(item.reason.value for item in preparation.rejections)
        raise NoEligibleArrivalsError(
            "no ADS-B arrivals were accepted for clustering; "
            f"rejection_counts={dict(sorted(reason_counts.items()))}"
        )

    releases = {
        item.flight_id: item.observed_release_time_s
        for item in preparation.prepared_tracks
    }
    rejection_payload = [item.to_dict() for item in preparation.rejections]
    adsb_metadata: dict[str, Any] = {
        "source": "raw_adsb",
        "terminal_radius_nm": preparation.terminal_radius_nm,
        "threshold_capture_radius_nm": preparation.threshold_capture_radius_nm,
        "accepted_track_count": len(preparation.prepared_tracks),
        "rejected_track_count": len(preparation.rejections),
        "observed_releases": _release_provenance(preparation),
        "rejections": rejection_payload,
        "raw_source_flight_ids": [item.flight_id for item in preparation.prepared_tracks],
        "template_threshold_alignment": (
            "snap_last_sample_or_append_at_catalog_event_with_last_observed_altitude"
        ),
    }
    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        {
            "input_source": "raw_adsb",
            "observed_release_times_s": releases,
            "track_rejections": rejection_payload,
            "adsb_pipeline": adsb_metadata,
        }
    )
    library = build_cluster_library(
        preparation.resampled_tracks_m,
        dataset_id=dataset_id,
        airport=preparation.airport,
        runway=preparation.runway,
        projection=preparation.frame.to_dict(),
        config=resolved_config,
        acceptance_quantile=acceptance_quantile,
        acceptance_multiplier=acceptance_multiplier,
        metadata=artifact_metadata,
    )
    prepared_by_id = preparation.tracks_by_flight_id
    raw_medoid_tracks = {
        medoid.medoid_flight_id: prepared_by_id[medoid.medoid_flight_id].raw_track
        for medoid in library.medoids
    }
    return ADSBClusterBuildResult(
        library=library,
        preparation=preparation,
        raw_medoid_tracks=raw_medoid_tracks,
    )


def build_all_runway_cluster_libraries_from_adsb(
    catalog_arrivals: Iterable[CatalogArrival],
    raw_adsb_tracks: Iterable[RawADSBTrack],
    *,
    dataset_id: str,
    airport: str,
    config: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ADSBMultiRunwayBuildResult:
    """Build every runway represented by reconstructable ADS-B arrivals.

    The same immutable raw corpus is reused for each runway partition. Sparse
    partitions, including a single accepted flight, flow through the existing
    deterministic one-cluster KMeans fallback and therefore keep their medoid
    instead of being excluded.
    """

    arrivals = tuple(catalog_arrivals)
    tracks = tuple(raw_adsb_tracks)
    runway_ids = tuple(sorted({normalize_runway(item.runway) for item in arrivals}))
    results: list[ADSBClusterBuildResult] = []
    rejected: Counter[str] = Counter()
    for runway in runway_ids:
        try:
            result = build_cluster_library_from_adsb(
                arrivals,
                tracks,
                dataset_id=dataset_id,
                airport=airport,
                runway=runway,
                config=config,
                metadata={
                    **({} if metadata is None else dict(metadata)),
                    "multi_runway_build": True,
                },
            )
        except NoEligibleArrivalsError:
            rejected[f"{runway}:no_reconstructable_terminal_entry"] += 1
            continue
        results.append(result)
        rejected.update(
            f"{runway}:{item.reason.value}" for item in result.preparation.rejections
        )
    if not results:
        raise ValueError("no runway partition produced an eligible cluster library")
    return ADSBMultiRunwayBuildResult(
        dataset_id=dataset_id,
        airport=str(airport).strip().upper(),
        runway_results=tuple(results),
        rejection_counts=tuple(sorted(rejected.items())),
    )


# Short aliases for callers that already established the clustering context.
prepare_adsb_tracks = prepare_adsb_tracks_for_clustering
prepare_adsb_clustering_inputs = prepare_adsb_tracks_for_clustering


__all__ = [
    "ADSBClusterBuildResult",
    "ADSBMultiRunwayBuildResult",
    "ADSBPreparationResult",
    "ADSBTrackRejection",
    "PreparedADSBTrack",
    "RejectionReason",
    "TrackRejection",
    "TrackRejectionReason",
    "build_cluster_library_from_adsb",
    "build_all_runway_cluster_libraries_from_adsb",
    "prepare_adsb_clustering_inputs",
    "prepare_adsb_tracks",
    "prepare_adsb_tracks_for_clustering",
]
