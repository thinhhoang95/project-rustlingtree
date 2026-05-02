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

from simap.fms import FMSRequest, HoldAwareFMSRequest, HoldInstruction
from simap.fms_bichannel import FMSBiChannelRequest, plan_fms_bichannel
from simap.lateral_dynamics import LateralGuidanceConfig
from simap.nlp_colloc.tactical import TacticalCommand, TacticalCondition
from simap.nlp_colloc.tactical.builder import build_tactical_plan_request
from simap.units import m_to_ft, mps_to_fpm, mps_to_kts


LATERAL_PATH = "JUSST SWTCH THEMM TUSLE SEEVR BRDJE NUSSS YAHBT ZINGG RW17C"
DEFAULT_HOLDS = (
    HoldInstruction(
        holding_altitude_ft=12_000.0,
        holding_speed_kts=None,
        holding_time_s=120.0,
    ),
)


def plot_bichannel_response(result) -> None:
    longitudinal = result.longitudinal
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.0))

    axes[0, 0].plot(longitudinal.t_s, m_to_ft(longitudinal.h_m), linewidth=2.0, color="#1f6f4a")
    axes[0, 0].set_title("Altitude")
    axes[0, 0].set_ylabel("Altitude [ft]")

    axes[0, 1].plot(longitudinal.t_s, mps_to_kts(longitudinal.v_cas_mps), linewidth=2.0, label="CAS")
    axes[0, 1].plot(
        longitudinal.t_s,
        mps_to_kts(longitudinal.target_cas_mps),
        linewidth=1.4,
        linestyle="--",
        label="Mode target",
    )
    axes[0, 1].set_title("CAS Tracking")
    axes[0, 1].set_ylabel("CAS [kt]")
    axes[0, 1].legend()

    axes[1, 0].plot(longitudinal.t_s, mps_to_fpm(longitudinal.vertical_speed_mps), linewidth=2.0)
    axes[1, 0].set_title("Vertical Speed")
    axes[1, 0].set_ylabel("Vertical speed [fpm]")

    axes[1, 1].plot(result.east_m, result.north_m, linewidth=2.0, label="Simulated")
    axes[1, 1].set_title("Ground Track")
    axes[1, 1].set_xlabel("East [m]")
    axes[1, 1].set_ylabel("North [m]")
    axes[1, 1].axis("equal")
    axes[1, 1].legend()

    axes[2, 0].plot(longitudinal.t_s, result.cross_track_m, linewidth=2.0)
    axes[2, 0].set_title("Cross-Track Error")
    axes[2, 0].set_xlabel("Elapsed Time [s]")
    axes[2, 0].set_ylabel("Cross-track [m]")

    axes[2, 1].plot(longitudinal.t_s, np.rad2deg(result.phi_rad), linewidth=2.0, label="Bank")
    axes[2, 1].plot(longitudinal.t_s, np.rad2deg(result.phi_req_rad), linewidth=1.4, label="Command")
    axes[2, 1].set_title("Bank Response")
    axes[2, 1].set_xlabel("Elapsed Time [s]")
    axes[2, 1].set_ylabel("Bank [deg]")
    axes[2, 1].legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle("A320 KDFW FMS Bichannel Response")
    fig.tight_layout()


def render_summary(console: Console, result) -> None:
    longitudinal = result.longitudinal
    table = Table(title="FMS bichannel summary")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("success", str(result.success))
    table.add_row("message", result.message)
    table.add_row("total distance", f"{longitudinal.descent_distance_m / 1_852.0:,.2f} nmi")
    table.add_row("total time", f"{longitudinal.descent_time_s / 60.0:,.2f} min")
    table.add_row("start altitude", f"{m_to_ft(longitudinal.h_m[0]):,.0f} ft")
    table.add_row("final altitude", f"{m_to_ft(longitudinal.h_m[-1]):,.0f} ft")
    table.add_row("start CAS", f"{mps_to_kts(longitudinal.v_cas_mps[0]):,.1f} kt")
    table.add_row("final CAS", f"{mps_to_kts(longitudinal.v_cas_mps[-1]):,.1f} kt")
    table.add_row("max cross-track", f"{result.max_abs_cross_track_m:,.2f} m")
    table.add_row("max track error", f"{np.rad2deg(result.max_abs_track_error_rad):,.2f} deg")
    table.add_row("max bank command ratio", f"{result.max_bank_command_ratio:,.3f}")
    table.add_row("final threshold error", f"{result.final_threshold_error_m:,.2f} m")
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
    fms_request = FMSRequest.from_coupled_request(bundle.request, dt_s=0.5)
    hold_request = HoldAwareFMSRequest(
        base_request=fms_request,
        holds=DEFAULT_HOLDS,
    )
    result = plan_fms_bichannel(
        FMSBiChannelRequest(
            base_request=hold_request,
            guidance=LateralGuidanceConfig(
                lookahead_m=2_500.0,
                cross_track_gain=1.0,
                track_error_gain=2.0,
            ),
        )
    )

    render_summary(console, result)
    plot_bichannel_response(result)
    print(result.to_pandas().head())
    print(result.to_pandas().tail())
    plt.show()


if __name__ == "__main__":
    main()
