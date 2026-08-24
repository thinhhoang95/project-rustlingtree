"""Causal splice of replay-compiled physical profiles onto a live parent."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from hailmary.errors import InfeasibleActionError
from hailmary.templates.models import ResourceCrossing, TrajectoryVariant


_VARIANT_PROFILE_NAMES = (
    "lat_deg",
    "lon_deg",
    "east_m",
    "north_m",
    "altitude_m",
    "cas_mps",
    "tas_mps",
    "ground_speed_mps",
    "command_cas_mps",
    "reference_command_cas_mps",
    "lower_cas_mps",
    "upper_cas_mps",
)


def preserve_compiled_live_prefix(
    parent: TrajectoryVariant,
    compiled: TrajectoryVariant,
    *,
    parent_anchor_s_m: float,
    child_anchor_s_m: float,
    station_mapping_m: tuple[tuple[float, float], ...] | None = None,
    envelope_tolerance_mps: float = 0.5,
) -> TrajectoryVariant:
    """Keep history from ``parent`` and the replay future from ``compiled``.

    A full-path deterministic replay can differ slightly before an action even
    when its commands only change downstream.  A live intervention cannot
    rewrite that already-flown prefix.  This helper maps the parent profiles to
    the child station axis, retains them from the action point upstream, and
    shifts the replay clock suffix to meet the parent at the exact splice.
    """

    parent_anchor = float(parent_anchor_s_m)
    child_anchor = float(child_anchor_s_m)
    compiled_s = np.asarray(compiled.s_m, dtype=np.float64)
    if station_mapping_m is None:
        parent_s = np.asarray(parent.s_m, dtype=np.float64)
        mapped_child_s = parent_s
    else:
        mapping = np.asarray(station_mapping_m, dtype=np.float64)
        if mapping.ndim != 2 or mapping.shape[1:] != (2,) or len(mapping) < 2:
            raise InfeasibleActionError("compiled splice station mapping is invalid")
        parent_s = mapping[:, 0]
        mapped_child_s = mapping[:, 1]
        if np.any(np.diff(parent_s) <= 0.0) or np.any(np.diff(mapped_child_s) <= 0.0):
            raise InfeasibleActionError("compiled splice station mapping is not monotone")

    tolerance_m = max(1.0e-7, float(compiled.path_length_m) * 1.0e-12)
    # Retain the replay grid only downstream of the live point.  Upstream, use
    # every parent knot (mapped onto the child's station axis), plus an exact
    # anchor knot.  Merely sampling the parent profile onto replay knots is not
    # sufficient: the simulator compares both trajectories on the union of
    # their *time* knots, so two different spatial grids can otherwise imply a
    # slightly different piecewise-linear history between matching samples.
    future = compiled_s < child_anchor - tolerance_m
    parent_stations = np.asarray(parent.s_m, dtype=np.float64)
    upstream_parent = parent_stations > parent_anchor + tolerance_m
    prefix_parent_s = np.concatenate(
        (np.asarray([parent_anchor], dtype=np.float64), parent_stations[upstream_parent])
    )
    prefix_child_s = np.interp(prefix_parent_s, parent_s, mapped_child_s)
    prefix_child_s[0] = child_anchor
    child_s = np.concatenate((compiled_s[future], prefix_child_s))
    if len(child_s) < 2 or np.any(np.diff(child_s) <= 0.0):
        raise InfeasibleActionError(
            "causal replay splice produced a non-monotone station grid"
        )

    profiles: dict[str, np.ndarray] = {}
    for name in _VARIANT_PROFILE_NAMES:
        values = np.concatenate(
            (
                np.asarray(getattr(compiled, name), dtype=np.float64)[future],
                np.interp(
                    prefix_parent_s,
                    parent_stations,
                    getattr(parent, name),
                ),
            )
        )
        profiles[name] = values

    prefix_elapsed = np.interp(
        prefix_parent_s,
        parent_stations,
        parent.elapsed_time_s,
    )
    parent_anchor_elapsed = float(prefix_elapsed[0])
    compiled_anchor_elapsed = float(
        np.interp(child_anchor, compiled_s, compiled.elapsed_time_s)
    )
    future_elapsed = (
        parent_anchor_elapsed
        + np.asarray(compiled.elapsed_time_s, dtype=np.float64)[future]
        - compiled_anchor_elapsed
    )
    elapsed = np.concatenate((future_elapsed, prefix_elapsed))
    if np.any(np.diff(elapsed) >= -1.0e-10):
        raise InfeasibleActionError("causal replay splice produced a non-monotone clock")

    lower = profiles["lower_cas_mps"]
    upper = profiles["upper_cas_mps"]
    envelope_tolerance = float(envelope_tolerance_mps)
    if not np.isfinite(envelope_tolerance) or envelope_tolerance < 0.0:
        raise ValueError("envelope_tolerance_mps must be finite and nonnegative")
    if np.any(profiles["cas_mps"] < lower - envelope_tolerance - 1.0e-9) or np.any(
        profiles["cas_mps"] > upper + envelope_tolerance + 1.0e-9
    ):
        raise InfeasibleActionError(
            "parent physical CAS is incompatible with the compiled child envelope"
        )

    crossings = tuple(
        ResourceCrossing(
            resource_id=crossing.resource_id,
            s_m=crossing.s_m,
            elapsed_time_s=float(np.interp(crossing.s_m, child_s, elapsed)),
        )
        for crossing in compiled.resource_crossings
    )
    details = dict(compiled.diagnostics.details)
    details.update(
        {
            "live_splice_parent_anchor_s_m": parent_anchor,
            "live_splice_child_anchor_s_m": child_anchor,
            "live_splice_parent_elapsed_s": parent_anchor_elapsed,
            "live_splice_replay_elapsed_s": compiled_anchor_elapsed,
            "live_splice_prefix_source": "parent_physical_profile",
            "live_splice_suffix_source": "simap_public_coupled_replay",
            "live_splice_parent_prefix_knots": int(len(prefix_parent_s)),
        }
    )
    timing_reference = details.get("simap_kinematic_duration_s")
    compiled_duration = float(elapsed[0])
    details["live_splice_compiled_duration_s"] = compiled_duration
    signed_error = (
        compiled.diagnostics.signed_timing_error_s
        if timing_reference is None
        else compiled_duration - float(timing_reference)
    )
    diagnostics = replace(
        compiled.diagnostics,
        compiled_duration_s=compiled_duration,
        signed_timing_error_s=signed_error,
        absolute_timing_error_s=None if signed_error is None else abs(float(signed_error)),
        details=tuple(sorted(details.items())),
    )
    return TrajectoryVariant(
        template_id=compiled.template_id,
        cluster_id=compiled.cluster_id,
        s_m=child_s,
        lat_deg=profiles["lat_deg"],
        lon_deg=profiles["lon_deg"],
        east_m=profiles["east_m"],
        north_m=profiles["north_m"],
        altitude_m=profiles["altitude_m"],
        cas_mps=profiles["cas_mps"],
        tas_mps=profiles["tas_mps"],
        ground_speed_mps=profiles["ground_speed_mps"],
        command_cas_mps=profiles["command_cas_mps"],
        reference_command_cas_mps=profiles["reference_command_cas_mps"],
        lower_cas_mps=profiles["lower_cas_mps"],
        upper_cas_mps=profiles["upper_cas_mps"],
        elapsed_time_s=elapsed,
        resource_crossings=crossings,
        diagnostics=diagnostics,
        action_provenance=compiled.action_provenance,
        schema_version=compiled.schema_version,
    )


__all__ = ["preserve_compiled_live_prefix"]
