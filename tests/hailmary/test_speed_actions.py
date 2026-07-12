from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hailmary.actions.speed import realize_speed_variant
from hailmary.config import MPS_PER_KNOT
from hailmary.errors import InfeasibleActionError
from hailmary.templates.models import TrajectoryVariant, VariantDiagnostics

from .test_templates import _straight_variant


class _RecordingCompiler:
    def __init__(self) -> None:
        self.calls = 0

    def compile(self, variant: TrajectoryVariant) -> TrajectoryVariant:
        self.calls += 1
        diagnostics = replace(
            variant.diagnostics,
            feasible=True,
            message="synthetic executable replay passed",
            compiled_duration_s=variant.duration_s,
            details=(("validator", "recording_compiler"),),
        )
        return replace(variant, diagnostics=diagnostics, variant_id="")

    def validate(self, variant: TrajectoryVariant) -> VariantDiagnostics:
        raise AssertionError(f"compile() should be preferred for {variant.variant_id}")


def test_speed_action_never_exceeds_reference_or_current_command() -> None:
    baseline = _straight_variant()
    anchor_s_m = 80_000.0

    slowed = realize_speed_variant(
        baseline,
        anchor_s_m=anchor_s_m,
        band="light",
        reduction_kts=10.0,
    )

    downstream = slowed.s_m < anchor_s_m - 1e-9
    splice_and_upstream = ~downstream
    assert np.all(slowed.command_cas_mps <= baseline.command_cas_mps + 1e-12)
    assert np.all(slowed.command_cas_mps <= slowed.reference_command_cas_mps + 1e-12)
    assert np.all(slowed.command_cas_mps >= slowed.lower_cas_mps - 1e-12)
    np.testing.assert_allclose(
        slowed.command_cas_mps[downstream],
        100.0 - 10.0 * MPS_PER_KNOT,
    )
    np.testing.assert_allclose(slowed.command_cas_mps[splice_and_upstream], 100.0)
    anchor_index = int(np.flatnonzero(np.isclose(slowed.s_m, anchor_s_m, atol=1e-8))[0])
    assert slowed.command_cas_mps[anchor_index] == baseline.command_cas_mps[anchor_index]
    assert slowed.duration_s > baseline.duration_s
    assert slowed.action_provenance.parent_variant_id == baseline.variant_id
    # Realization creates new immutable arrays and leaves its parent byte-stable.
    np.testing.assert_array_equal(baseline.command_cas_mps, np.full(len(baseline.s_m), 100.0))


def test_sequential_speed_actions_compose_from_current_variant_without_relaxation() -> None:
    baseline = _straight_variant()
    first = realize_speed_variant(
        baseline,
        anchor_s_m=80_000.0,
        band="light",
        reduction_kts=10.0,
    )
    second = realize_speed_variant(
        first,
        anchor_s_m=60_000.0,
        band="medium",
        reduction_kts=15.0,
    )

    assert np.all(second.command_cas_mps <= first.command_cas_mps + 1e-12)
    assert np.all(second.command_cas_mps <= baseline.command_cas_mps + 1e-12)
    assert second.duration_s >= first.duration_s >= baseline.duration_s
    assert second.action_provenance.parent_variant_id == first.variant_id
    assert second.variant_id != first.variant_id != baseline.variant_id
    at_70km = int(np.argmin(np.abs(second.s_m - 70_000.0)))
    at_50km = int(np.argmin(np.abs(second.s_m - 50_000.0)))
    assert second.command_cas_mps[at_70km] == pytest.approx(100.0 - 10.0 * MPS_PER_KNOT)
    assert second.command_cas_mps[at_50km] == pytest.approx(100.0 - 15.0 * MPS_PER_KNOT)


def test_speed_action_rejects_reduction_erased_by_lower_envelope() -> None:
    constrained = _straight_variant(lower_cas_mps=99.5)

    with pytest.raises(InfeasibleActionError, match="effective reduction floor"):
        realize_speed_variant(
            constrained,
            anchor_s_m=60_000.0,
            band="light",
            reduction_kts=10.0,
            min_effective_reduction_kts=2.0,
        )


def test_reapplying_a_lighter_command_cannot_undo_an_existing_slowdown() -> None:
    baseline = _straight_variant()
    heavy = realize_speed_variant(
        baseline,
        anchor_s_m=80_000.0,
        band="heavy",
        reduction_kts=20.0,
    )

    with pytest.raises(InfeasibleActionError, match="effective reduction floor"):
        realize_speed_variant(
            heavy,
            anchor_s_m=60_000.0,
            band="light",
            reduction_kts=10.0,
        )
    np.testing.assert_array_less(heavy.command_cas_mps, baseline.command_cas_mps + 1e-12)


def test_speed_action_prefers_executable_variant_compiler_when_supplied() -> None:
    baseline = _straight_variant()
    compiler = _RecordingCompiler()

    slowed = realize_speed_variant(
        baseline,
        anchor_s_m=80_000.0,
        band="light",
        reduction_kts=10.0,
        validator=compiler,
    )

    assert compiler.calls == 1
    assert slowed.diagnostics.feasible
    assert slowed.diagnostics.message == "synthetic executable replay passed"
    assert dict(slowed.diagnostics.details)["validator"] == "recording_compiler"
