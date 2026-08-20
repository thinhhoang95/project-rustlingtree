"""Pointwise slowdown realization composed from the current branch variant."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np

from hailmary.config import MPS_PER_KNOT
from hailmary.errors import InfeasibleActionError
from hailmary.actions.splice import preserve_compiled_live_prefix
from hailmary.templates.models import ActionProvenance, TrajectoryVariant, VariantDiagnostics
from hailmary.templates.speed import cas_to_tas


_PROFILE_NAMES = (
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


class VariantValidator(Protocol):
    def validate(self, variant: TrajectoryVariant) -> VariantDiagnostics: ...


def _profiles_with_exact_anchor(
    current: TrajectoryVariant,
    station: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    distances = np.abs(current.s_m - station)
    nearest = int(np.argmin(distances))
    if distances[nearest] <= 1e-6:
        return current.s_m, {name: getattr(current, name) for name in _PROFILE_NAMES}, nearest
    insertion_index = int(np.searchsorted(current.s_m, station))
    if insertion_index <= 0 or insertion_index >= len(current.s_m):
        raise InfeasibleActionError("speed anchor is outside the current trajectory grid")
    stations = np.insert(current.s_m, insertion_index, station)
    profiles = {
        name: np.insert(
            getattr(current, name),
            insertion_index,
            np.interp(station, current.s_m, getattr(current, name)),
        )
        for name in _PROFILE_NAMES
    }
    return stations, profiles, insertion_index


def realize_speed_variant(
    current: TrajectoryVariant,
    *,
    anchor_s_m: float,
    band: str,
    reduction_kts: float,
    min_effective_reduction_kts: float = 2.0,
    validator: VariantValidator | None = None,
) -> TrajectoryVariant:
    """Compile one slowdown without ever relaxing an earlier command."""

    station = float(anchor_s_m)
    if not 0.0 <= station <= current.path_length_m:
        raise InfeasibleActionError("speed anchor is outside the current trajectory")
    if reduction_kts <= 0.0:
        raise InfeasibleActionError("speed reduction must be positive")
    stations, profiles, anchor_index = _profiles_with_exact_anchor(current, station)
    # The command at the exact splice belongs to both the historical and the
    # replacement trajectory, so retain it verbatim.  The intervention starts
    # at the first represented point strictly downstream of the anchor.
    downstream = stations < station - 1e-8
    if not np.any(downstream):
        raise InfeasibleActionError("speed anchor has no downstream trajectory")
    delta_mps = float(reduction_kts * MPS_PER_KNOT)
    command = np.array(profiles["command_cas_mps"], dtype=float, copy=True)
    requested = profiles["reference_command_cas_mps"] - delta_mps
    command[downstream] = np.minimum(command[downstream], requested[downstream])
    command[downstream] = np.maximum(command[downstream], profiles["lower_cas_mps"][downstream])

    first_downstream_index = int(np.flatnonzero(downstream)[-1])
    effective = float(
        profiles["command_cas_mps"][first_downstream_index]
        - command[first_downstream_index]
    )
    if effective < min_effective_reduction_kts * MPS_PER_KNOT - 1e-9:
        raise InfeasibleActionError("speed-envelope clamping leaves less than the effective reduction floor")
    if np.any(command > profiles["command_cas_mps"] + 1e-9):
        raise AssertionError("composed speed action relaxed the current branch command")
    if np.any(command > profiles["reference_command_cas_mps"] + 1e-9):
        raise AssertionError("speed action exceeds the medoid reference")
    if np.any(np.diff(command) < -1e-8):
        raise InfeasibleActionError(
            "speed-envelope clamping would require a downstream commanded acceleration"
        )

    physical_cas = np.array(profiles["cas_mps"], dtype=float, copy=True)
    physical_cas[downstream] = command[downstream]
    tas = np.array(profiles["tas_mps"], dtype=float, copy=True)
    tas[downstream] = cas_to_tas(
        command[downstream],
        profiles["altitude_m"][downstream],
    )
    # Preserve the compiled variant's wind/along-track relationship. Under the
    # version-1 zero-wind compiler this equals TAS; it also keeps analytic and
    # externally compiled fixtures slowdown-monotone.
    ground = np.array(profiles["ground_speed_mps"], dtype=float, copy=True)
    ground[downstream] = (
        profiles["ground_speed_mps"][downstream]
        * command[downstream]
        / np.maximum(profiles["command_cas_mps"][downstream], 1e-9)
    )
    resources = tuple(
        (crossing.resource_id, crossing.s_m)
        for crossing in current.resource_crossings
        if abs(crossing.s_m) > 1e-8
    )
    threshold_resource = current.threshold_resource_id or "runway_threshold"
    diagnostics = VariantDiagnostics(
        feasible=True,
        message=f"deterministic {band} slowdown",
        speed_source=current.diagnostics.speed_source,
        wind_model=current.diagnostics.wind_model,
        cas_derivation=current.diagnostics.cas_derivation,
        observed_duration_s=current.diagnostics.observed_duration_s,
        max_command_envelope_excursion_mps=0.0,
        clamped_fraction=float(
            np.count_nonzero(requested[downstream] < profiles["lower_cas_mps"][downstream])
            / max(1, np.count_nonzero(downstream))
        ),
        details=(
            ("requested_reduction_kts", float(reduction_kts)),
            ("effective_downstream_reduction_kts", effective / MPS_PER_KNOT),
        ),
    )
    variant = TrajectoryVariant.from_kinematic_profile(
        template_id=current.template_id,
        cluster_id=current.cluster_id,
        s_m=stations,
        lat_deg=profiles["lat_deg"],
        lon_deg=profiles["lon_deg"],
        east_m=profiles["east_m"],
        north_m=profiles["north_m"],
        altitude_m=profiles["altitude_m"],
        cas_mps=physical_cas,
        tas_mps=tas,
        ground_speed_mps=ground,
        command_cas_mps=command,
        reference_command_cas_mps=profiles["reference_command_cas_mps"],
        lower_cas_mps=profiles["lower_cas_mps"],
        upper_cas_mps=profiles["upper_cas_mps"],
        threshold_resource_id=threshold_resource,
        resource_stations_m=resources,
        diagnostics=diagnostics,
        action_provenance=ActionProvenance(
            lever="speed",
            band=band,
            parent_variant_id=current.variant_id,
            anchor_station_index=anchor_index,
            speed_reduction_mps=delta_mps,
            realization_metadata=(("effective_reduction_mps", effective),),
        ),
    )
    if validator is not None:
        compile_variant = getattr(validator, "compile", None)
        if callable(compile_variant):
            compiled = compile_variant(variant)
            if not isinstance(compiled, TrajectoryVariant):
                raise TypeError("variant compiler must return TrajectoryVariant")
            variant = preserve_compiled_live_prefix(
                current,
                compiled,
                parent_anchor_s_m=station,
                child_anchor_s_m=station,
                envelope_tolerance_mps=float(
                    getattr(validator, "physical_envelope_tolerance_mps", 0.5)
                ),
            )
            validated = variant.diagnostics
        else:
            validated = validator.validate(variant)
            variant = replace(variant, diagnostics=validated, variant_id="")
        if not validated.feasible:
            raise InfeasibleActionError(f"SIMAP validation failed: {validated.message}")
    delay = variant.duration_s - current.duration_s
    if delay < -1e-8:
        raise AssertionError("slowdown produced a negative delay")
    return variant
