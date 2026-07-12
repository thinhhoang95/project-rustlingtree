"""Runway-local geometry used by Hailmary offline and runtime artifacts."""

from .frame import LocalFrame, LocalProjection
from .dogleg import (
    DoglegGeometry,
    construct_runway_away_dogleg,
    point_segment_distance,
    polyline_min_distance,
    segment_distance,
    segments_intersect,
)
from .polyline import (
    ExecutablePolyline,
    ResampledPolyline,
    align_terminal_track,
    arc_length_resample_points,
    clip_to_terminal_radius,
    cumulative_lengths,
    cumulative_lengths_m,
    orient_upstream_to_threshold,
    polyline_length,
    polyline_length_m,
    prepare_track_for_clustering,
    readonly_float64,
    remove_duplicate_neighbors,
    resample_polyline,
    to_executable_station_order,
)

__all__ = [
    "ExecutablePolyline",
    "DoglegGeometry",
    "LocalFrame",
    "LocalProjection",
    "ResampledPolyline",
    "align_terminal_track",
    "arc_length_resample_points",
    "clip_to_terminal_radius",
    "cumulative_lengths",
    "cumulative_lengths_m",
    "construct_runway_away_dogleg",
    "orient_upstream_to_threshold",
    "polyline_length",
    "polyline_length_m",
    "prepare_track_for_clustering",
    "point_segment_distance",
    "polyline_min_distance",
    "readonly_float64",
    "remove_duplicate_neighbors",
    "resample_polyline",
    "segment_distance",
    "segments_intersect",
    "to_executable_station_order",
]
