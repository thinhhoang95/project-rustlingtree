from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from simap.descend_now_heuristics import DescendNowRequest, simulate_stitched_level_then_descend
from simap.units import m_to_ft, mps_to_fpm, mps_to_kts
from tactical import TacticalCommand, TacticalCondition
from tactical.builder import build_tactical_plan_request


LATERAL_PATH = "JUSST SWTCH THEMM TUSLE SEEVR BRDJE NUSSS YAHBT ZINGG RW17C"


def plot_descend_now_response(result) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), sharex=True)

    axes[0, 0].plot(result.t_s, m_to_ft(result.h_m), linewidth=2.0, color="#1f6f4a")
    axes[0, 0].set_title("Altitude")
    axes[0, 0].set_ylabel("Altitude [ft]")

    axes[0, 1].plot(result.t_s, mps_to_kts(result.v_cas_mps), linewidth=2.0, color="#b45f06", label="CAS")
    axes[0, 1].plot(
        result.t_s,
        mps_to_kts(result.target_cas_mps),
        linewidth=1.4,
        color="#73410d",
        linestyle="--",
        label="Mode target",
    )
    axes[0, 1].set_title("CAS Tracking")
    axes[0, 1].set_ylabel("CAS [kt]")
    axes[0, 1].legend()

    axes[1, 0].plot(result.t_s, np.rad2deg(result.pitch_rad), linewidth=2.0, color="#2454a6")
    axes[1, 0].set_title("PI Pitch Command")
    axes[1, 0].set_xlabel("Elapsed Time [s]")
    axes[1, 0].set_ylabel("Pitch proxy [deg]")

    axes[1, 1].plot(result.t_s, mps_to_fpm(result.vertical_speed_mps), linewidth=2.0, color="#7f3b08")
    axes[1, 1].axhline(-3_000.0, color="0.35", linestyle="--", linewidth=1.0, label="-3000 fpm limit")
    axes[1, 1].axhline(0.0, color="0.35", linestyle=":", linewidth=1.0, label="0 fpm limit")
    axes[1, 1].set_title("Vertical Speed")
    axes[1, 1].set_xlabel("Elapsed Time [s]")
    axes[1, 1].set_ylabel("Vertical speed [fpm]")
    axes[1, 1].legend()

    for ax in axes.flat:
        if result.tod_s_m is not None:
            tod_idx = int(np.argmin(np.abs(result.s_m - result.tod_s_m)))
            ax.axvline(result.t_s[tod_idx], color="0.25", linestyle="--", linewidth=1.0, label="TOD")
        ax.grid(True, alpha=0.3)

    fig.suptitle("A320 KDFW Stitched Level + Descend-Now Heuristic")
    fig.tight_layout()


def render_summary(console: Console, result) -> None:
    table = Table(title="Descend-now heuristic summary")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("success", str(result.success))
    table.add_row("message", result.message)
    if result.tod_s_m is not None:
        table.add_row("TOD s", f"{result.tod_s_m:,.1f} m")
        table.add_row("TOD from threshold", f"{result.tod_s_m / 1_852.0:,.2f} nmi")
    table.add_row("total distance", f"{result.descent_distance_m / 1_852.0:,.2f} nmi")
    table.add_row("total time", f"{result.descent_time_s / 60.0:,.2f} min")
    table.add_row("level distance", f"{result.level_distance_m / 1_852.0:,.2f} nmi")
    table.add_row("level time", f"{result.level_time_s / 60.0:,.2f} min")
    if result.descent_segment_distance_m is not None:
        table.add_row("descent segment distance", f"{result.descent_segment_distance_m / 1_852.0:,.2f} nmi")
    if result.descent_segment_time_s is not None:
        table.add_row("descent segment time", f"{result.descent_segment_time_s / 60.0:,.2f} min")
    table.add_row("start altitude", f"{m_to_ft(result.h_m[0]):,.0f} ft")
    table.add_row("final altitude", f"{m_to_ft(result.h_m[-1]):,.0f} ft")
    table.add_row("start CAS", f"{mps_to_kts(result.v_cas_mps[0]):,.1f} kt")
    table.add_row("final CAS", f"{mps_to_kts(result.v_cas_mps[-1]):,.1f} kt")
    table.add_row("min vertical speed", f"{mps_to_fpm(np.min(result.vertical_speed_mps)):,.0f} fpm")
    table.add_row("max vertical speed", f"{mps_to_fpm(np.max(result.vertical_speed_mps)):,.0f} fpm")
    table.add_row("min pitch", f"{np.rad2deg(np.min(result.pitch_rad)):,.2f} deg")
    table.add_row("max pitch", f"{np.rad2deg(np.max(result.pitch_rad)):,.2f} deg")
    console.print(table)


def render_input_summary(console: Console, bundle, descend_request: DescendNowRequest) -> None:
    table = Table(title="Descend-now heuristic inputs")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("route", " -> ".join(bundle.path.identifiers))
    table.add_row("reference length", f"{descend_request.reference_path.total_length_m:,.1f} m")
    table.add_row("start s", f"{descend_request.start_s_m:,.1f} m")
    table.add_row("start altitude", f"{m_to_ft(descend_request.start_h_m):,.0f} ft")
    table.add_row("target altitude", f"{m_to_ft(descend_request.target_h_m):,.0f} ft")
    table.add_row("start CAS", f"{mps_to_kts(descend_request.start_cas_mps):,.1f} kt")
    table.add_row("clean target", f"{mps_to_kts(descend_request.speed_targets.clean_cas_mps):,.1f} kt")
    table.add_row("approach target", f"{mps_to_kts(descend_request.speed_targets.approach_cas_mps):,.1f} kt")
    table.add_row("final target", f"{mps_to_kts(descend_request.speed_targets.final_cas_mps):,.1f} kt")
    if (
        descend_request.speed_targets.below_altitude_limit_h_m is not None
        and descend_request.speed_targets.below_altitude_limit_cas_mps is not None
    ):
        table.add_row(
            "altitude speed cap",
            (
                f"{mps_to_kts(descend_request.speed_targets.below_altitude_limit_cas_mps):,.0f} kt "
                f"below {m_to_ft(descend_request.speed_targets.below_altitude_limit_h_m):,.0f} ft"
            ),
        )
    table.add_row("vertical-speed lower", f"{mps_to_fpm(descend_request.controller.min_vertical_speed_mps):,.0f} fpm")
    table.add_row("vertical-speed upper", f"{mps_to_fpm(descend_request.controller.max_vertical_speed_mps):,.0f} fpm")
    console.print(table)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    console = Console()
    command = TacticalCommand(
        lateral_path=LATERAL_PATH,
        upstream=TacticalCondition(
            fix_identifier="JUSST",
            cas_kts=290.0,
            altitude_ft=26_000.0,
        ),
        altitude_constraints=(),
        runway_altitude_ft=620.0,
    )

    bundle = build_tactical_plan_request(
        command,
        fixes_csv=repo_root / "data/kdfw_procs/airport_related_fixes.csv",
    )

    descend_request = DescendNowRequest.from_coupled_request(bundle.request, dt_s=0.5)
    render_input_summary(console, bundle, descend_request)
    result = simulate_stitched_level_then_descend(descend_request)

    render_summary(console, result)
    plot_descend_now_response(result)

    print(result.to_pandas().head())
    print(result.to_pandas().tail())
    print(
        {
            "route": bundle.path.identifiers,
            "success": result.success,
            "message": result.message,
            "tod_s_m": result.tod_s_m,
            "tod_nmi_from_threshold": None if result.tod_s_m is None else result.tod_s_m / 1_852.0,
            "total_distance_m": result.descent_distance_m,
            "total_distance_nmi": result.descent_distance_m / 1_852.0,
            "total_time_s": result.descent_time_s,
            "total_time_min": result.descent_time_s / 60.0,
            "level_distance_m": result.level_distance_m,
            "level_time_s": result.level_time_s,
            "descent_segment_distance_m": result.descent_segment_distance_m,
            "descent_segment_time_s": result.descent_segment_time_s,
            "start_cas_kts": float(mps_to_kts(result.v_cas_mps[0])),
            "final_cas_kts": float(mps_to_kts(result.v_cas_mps[-1])),
            "start_altitude_ft": float(m_to_ft(result.h_m[0])),
            "final_altitude_ft": float(m_to_ft(result.h_m[-1])),
            "min_pitch_deg": float(np.rad2deg(np.min(result.pitch_rad))),
            "max_pitch_deg": float(np.rad2deg(np.max(result.pitch_rad))),
        }
    )
    plt.show()


if __name__ == "__main__":
    main()
