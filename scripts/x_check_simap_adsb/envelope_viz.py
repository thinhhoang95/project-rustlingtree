from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.widgets import Slider

from simap.config import planned_cas_bounds_mps
from simap.fms import FMSRequest
from simap.fms_bichannel import FMSBiChannelRequest, FMSBiChannelResult, FMSBiChannelState, plan_fms_bichannel
from simap.lateral_dynamics import LateralGuidanceConfig, wrap_angle_rad
from simap.path_geometry import EARTH_RADIUS_M, ReferencePath
from simap.units import m_to_ft, mps_to_kts

UTC_TZ = timezone.utc
ADSB_COLOR = "#1b6b3a"
SIMAP_COLOR = "#c2410c"
CAS_COLOR = "#f59e0b"
GAMMA_COLOR = "#7c3aed"
BANK_COLOR = "#2563eb"
LOWER_STATE_COLOR = "#2563eb"
UPPER_STATE_COLOR = "#dc2626"
FREE_STATE_COLOR = "#6b7280"
UNAVAILABLE = "N/A"

CAS_STATE_ATOL_MPS = 0.25
GAMMA_STATE_ATOL_RAD = np.deg2rad(0.2)
BANK_STATE_ATOL_RAD = np.deg2rad(0.25)


class TrajectoryLike(Protocol):
    time_s: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    altitude_m: np.ndarray
    speed_mps: np.ndarray


class SeedLike(Protocol):
    time_s: int
    lat_deg: float
    lon_deg: float
    geoaltitude_m: float
    heading_deg: float | None
    ground_speed_mps: float


@dataclass(frozen=True)
class SampledTrajectory:
    available: bool
    time_s: float
    lat_deg: float | None = None
    lon_deg: float | None = None
    altitude_m: float | None = None
    speed_mps: float | None = None


def _fmt_maybe(value: float | None, *, suffix: str, decimals: int) -> str:
    if value is None or not np.isfinite(value):
        return UNAVAILABLE
    return f"{value:.{decimals}f}{suffix}"


def _fmt_unix_time(time_s: float) -> str:
    return datetime.fromtimestamp(float(time_s), tz=UTC_TZ).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_sample(name: str, sample: SampledTrajectory, *, speed_label: str) -> str:
    if not sample.available:
        return f"{name}: {UNAVAILABLE}"
    return (
        f"{name}: "
        f"lat {_fmt_maybe(sample.lat_deg, suffix='', decimals=5)}, "
        f"lon {_fmt_maybe(sample.lon_deg, suffix='', decimals=5)}, "
        f"alt {_fmt_maybe(m_to_ft(sample.altitude_m), suffix=' ft', decimals=1)}, "
        f"{speed_label} {_fmt_maybe(mps_to_kts(sample.speed_mps), suffix=' kt', decimals=1)}"
    )


def _sample_trajectory(trajectory: TrajectoryLike, time_s: float) -> SampledTrajectory:
    if len(trajectory.time_s) == 0:
        return SampledTrajectory(available=False, time_s=time_s)
    first_time = float(trajectory.time_s[0])
    last_time = float(trajectory.time_s[-1])
    if time_s < first_time or time_s > last_time:
        return SampledTrajectory(available=False, time_s=time_s)

    return SampledTrajectory(
        available=True,
        time_s=time_s,
        lat_deg=_interp_value(time_s, trajectory.time_s, trajectory.lat_deg),
        lon_deg=_interp_value(time_s, trajectory.time_s, trajectory.lon_deg),
        altitude_m=_interp_value(time_s, trajectory.time_s, trajectory.altitude_m),
        speed_mps=_interp_value(time_s, trajectory.time_s, trajectory.speed_mps),
    )


def _interp_value(time_s: float, times_s: np.ndarray, values: np.ndarray) -> float | None:
    if len(times_s) == 0 or time_s < float(times_s[0]) or time_s > float(times_s[-1]):
        return None
    index = int(np.searchsorted(times_s, time_s, side="left"))
    if index < len(times_s) and float(times_s[index]) == float(time_s):
        value = float(values[index])
        return value if np.isfinite(value) else None
    if index == 0 or index >= len(times_s):
        return None
    before = float(values[index - 1])
    after = float(values[index])
    if not np.isfinite(before) or not np.isfinite(after):
        return None
    t0 = float(times_s[index - 1])
    t1 = float(times_s[index])
    if t1 <= t0:
        return None
    fraction = (time_s - t0) / (t1 - t0)
    return before + fraction * (after - before)


def _latlon_to_ne(reference_path: ReferencePath, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0_rad = np.deg2rad(reference_path.origin_lat_deg)
    east_m = EARTH_RADIUS_M * np.cos(lat0_rad) * np.deg2rad(np.asarray(lon_deg, dtype=float) - reference_path.origin_lon_deg)
    north_m = EARTH_RADIUS_M * np.deg2rad(np.asarray(lat_deg, dtype=float) - reference_path.origin_lat_deg)
    return east_m, north_m


def _signed_cross_track(reference_path: ReferencePath, trajectory: TrajectoryLike) -> np.ndarray:
    if hasattr(trajectory, "cross_track_m"):
        return np.asarray(getattr(trajectory, "cross_track_m"), dtype=float)

    east_m, north_m = _latlon_to_ne(reference_path, trajectory.lat_deg, trajectory.lon_deg)
    ref_east_m = np.asarray(reference_path.east_m, dtype=float)
    ref_north_m = np.asarray(reference_path.north_m, dtype=float)
    ref_normals = reference_path.normal_hat_many(reference_path.s_m)

    delta_e = east_m[:, np.newaxis] - ref_east_m[np.newaxis, :]
    delta_n = north_m[:, np.newaxis] - ref_north_m[np.newaxis, :]
    nearest_idx = np.argmin(delta_e**2 + delta_n**2, axis=1)
    row_idx = np.arange(len(nearest_idx))
    return delta_e[row_idx, nearest_idx] * ref_normals[nearest_idx, 0] + delta_n[row_idx, nearest_idx] * ref_normals[nearest_idx, 1]


def _state_from_bounds(
    values: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    *,
    atol: float,
) -> np.ndarray:
    state = np.zeros_like(np.asarray(values, dtype=float), dtype=int)
    if lower is None and upper is None:
        return state

    if lower is None:
        assert upper is not None
        state[np.isfinite(upper) & (values >= upper - atol)] = 1
        return state

    if upper is None:
        state[np.isfinite(lower) & (values <= lower + atol)] = -1
        return state

    lower_active = np.isfinite(lower) & (values <= lower + atol)
    upper_active = np.isfinite(upper) & (values >= upper - atol)
    only_lower = lower_active & ~upper_active
    only_upper = upper_active & ~lower_active
    both = lower_active & upper_active
    state[only_lower] = -1
    state[only_upper] = 1
    if np.any(both):
        lower_gap = np.abs(values[both] - lower[both])
        upper_gap = np.abs(upper[both] - values[both])
        state[both] = np.where(lower_gap <= upper_gap, -1, 1)
    return state


def _state_summary(time_s: np.ndarray, state: np.ndarray) -> tuple[int, float | None]:
    active = np.flatnonzero(state != 0)
    if active.size == 0:
        return 0, None
    return int(active.size), float(time_s[int(active[0])])


def _first_active_position(trajectory: FMSBiChannelResult, state: np.ndarray) -> tuple[float, float] | None:
    active = np.flatnonzero(state != 0)
    if active.size == 0:
        return None
    idx = int(active[0])
    return float(trajectory.lon_deg[idx]), float(trajectory.lat_deg[idx])


def _effective_cas_bounds(request: Any, s_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    route_lower_mps, route_upper_mps = request.constraints.cas_bounds_many(s_m)
    mode_bounds = np.asarray([planned_cas_bounds_mps(request.cfg, float(s_val)) for s_val in s_m], dtype=float)
    return np.maximum(route_lower_mps, mode_bounds[:, 0]), np.minimum(route_upper_mps, mode_bounds[:, 1])


def _build_bichannel_request(bundle: Any, seed: SeedLike) -> FMSBiChannelRequest:
    reference_path = bundle.request.reference_path
    fms_request = FMSRequest.from_coupled_request(bundle.request, start_s_m=bundle.request.reference_path.total_length_m)

    h_m = max(float(seed.geoaltitude_m), 1.0)
    east_m, north_m = _latlon_to_ne(reference_path, np.asarray([seed.lat_deg]), np.asarray([seed.lon_deg]))
    heading_deg = getattr(seed, "heading_deg", None)
    if heading_deg is not None and np.isfinite(float(heading_deg)):
        psi_rad = wrap_angle_rad(np.deg2rad(90.0 - float(heading_deg)))
    else:
        psi_rad = float(reference_path.track_angle_rad(fms_request.start_s_m))

    initial_state = FMSBiChannelState(
        t_s=0.0,
        s_m=float(fms_request.start_s_m),
        h_m=h_m,
        v_tas_mps=float(fms_request.start_cas_mps),
        east_m=float(east_m[0]),
        north_m=float(north_m[0]),
        psi_rad=float(psi_rad),
        phi_rad=0.0,
    )
    return FMSBiChannelRequest(
        base_request=fms_request,
        guidance=LateralGuidanceConfig(),
        initial_state=initial_state,
    )


def build_bichannel_result(bundle: Any, seed: SeedLike) -> FMSBiChannelResult:
    request = _build_bichannel_request(bundle, seed)
    return plan_fms_bichannel(request)


def _plot_state_axis(ax: Axes, time_s: np.ndarray, state: np.ndarray, *, title: str, first_active_time_s: float | None, current_time_s: float | None = None) -> None:
    ax.step(time_s, state, where="mid", color="#111827", linewidth=1.2)
    if np.any(state == -1):
        ax.scatter(time_s[state == -1], state[state == -1], color=LOWER_STATE_COLOR, s=12, zorder=3, label="lower")
    if np.any(state == 1):
        ax.scatter(time_s[state == 1], state[state == 1], color=UPPER_STATE_COLOR, s=12, zorder=3, label="upper")
    if current_time_s is not None:
        current_value = _interp_value(current_time_s, time_s, state.astype(float))
        if current_value is not None:
            ax.plot([current_time_s], [current_value], "o", color="#111827", markersize=6, zorder=4)
    if first_active_time_s is not None:
        ax.axvline(first_active_time_s, color=UPPER_STATE_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(0.0, color=FREE_STATE_COLOR, linestyle=":", linewidth=1.0)
    ax.set_ylim(-1.25, 1.25)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["lower", "free", "upper"])
    ax.set_title(title)
    ax.set_ylabel("state")
    ax.grid(True, alpha=0.25)


def _hide_xticklabels(ax: Axes) -> None:
    ax.tick_params(labelbottom=False)


def _current_title(label: str, time_s: float, value_text: str) -> str:
    return f"{label} @ {_fmt_unix_time(time_s)}: {value_text}"


def plot_cross_check(
    adsb: TrajectoryLike,
    reference_path: ReferencePath,
    key: Any,
    *,
    bundle: Any,
    seed: SeedLike,
) -> None:
    bichannel = build_bichannel_result(bundle, seed)
    sim_time_s = float(seed.time_s) + np.asarray(bichannel.t_s, dtype=float)
    adsb_time_s = np.asarray(adsb.time_s, dtype=float)
    adsb_cross_track_m = _signed_cross_track(reference_path, adsb)
    sim_cross_track_m = np.asarray(bichannel.cross_track_m, dtype=float)
    sim_track_error_deg = np.rad2deg(np.asarray(bichannel.track_error_rad, dtype=float))
    sim_gamma_deg = np.rad2deg(np.asarray(bichannel.longitudinal.gamma_rad, dtype=float))
    sim_phi_deg = np.rad2deg(np.asarray(bichannel.phi_rad, dtype=float))
    sim_phi_req_deg = np.rad2deg(np.asarray(bichannel.phi_req_rad, dtype=float))
    sim_phi_max_deg = np.rad2deg(np.asarray(bichannel.phi_max_rad, dtype=float))

    cas_lower_mps, cas_upper_mps = _effective_cas_bounds(bundle.request, bichannel.s_m)
    gamma_lower_rad, gamma_upper_rad = bundle.request.constraints.gamma_bounds_many(bichannel.s_m)
    gamma_lower_deg = None if gamma_lower_rad is None else np.rad2deg(gamma_lower_rad)
    gamma_upper_deg = None if gamma_upper_rad is None else np.rad2deg(gamma_upper_rad)

    cas_state = _state_from_bounds(bichannel.longitudinal.v_cas_mps, cas_lower_mps, cas_upper_mps, atol=CAS_STATE_ATOL_MPS)
    gamma_state = _state_from_bounds(bichannel.longitudinal.gamma_rad, gamma_lower_rad, gamma_upper_rad, atol=GAMMA_STATE_ATOL_RAD)
    bank_state = _state_from_bounds(bichannel.phi_req_rad, -bichannel.phi_max_rad, bichannel.phi_max_rad, atol=BANK_STATE_ATOL_RAD)

    cas_active_count, cas_first_active = _state_summary(sim_time_s, cas_state)
    gamma_active_count, gamma_first_active = _state_summary(sim_time_s, gamma_state)
    bank_active_count, bank_first_active = _state_summary(sim_time_s, bank_state)

    fig = plt.figure(figsize=(18.0, 18.5))
    grid = fig.add_gridspec(6, 2, height_ratios=[1.22, 0.92, 0.92, 0.92, 0.92, 0.16])
    trajectory_ax = fig.add_subplot(grid[0, :])
    cross_track_ax = fig.add_subplot(grid[1, 0])
    track_error_ax = fig.add_subplot(grid[1, 1])
    cas_ax = fig.add_subplot(grid[2, 0])
    cas_state_ax = fig.add_subplot(grid[2, 1])
    gamma_ax = fig.add_subplot(grid[3, 0])
    gamma_state_ax = fig.add_subplot(grid[3, 1])
    bank_ax = fig.add_subplot(grid[4, 0])
    bank_state_ax = fig.add_subplot(grid[4, 1])
    slider_ax = fig.add_subplot(grid[5, :])

    trajectory_ax.plot(
        reference_path.lon_deg,
        reference_path.lat_deg,
        color="#6b7280",
        linewidth=2.0,
        linestyle=":",
        label="reference path",
        zorder=1,
    )
    trajectory_ax.plot(adsb.lon_deg, adsb.lat_deg, color=ADSB_COLOR, linewidth=2.0, label="ADS-B", zorder=2)
    trajectory_ax.plot(bichannel.lon_deg, bichannel.lat_deg, color=SIMAP_COLOR, linewidth=2.0, label="SIMAP", zorder=3)

    first_bank_position = _first_active_position(bichannel, bank_state)
    first_cas_position = _first_active_position(bichannel, cas_state)
    first_gamma_position = _first_active_position(bichannel, gamma_state)
    for position, marker, color, label in (
        (first_bank_position, "*", UPPER_STATE_COLOR, "first bank limit"),
        (first_cas_position, "D", CAS_COLOR, "first CAS limit"),
        (first_gamma_position, "s", GAMMA_COLOR, "first gamma limit"),
    ):
        if position is not None:
            trajectory_ax.scatter(
                [position[0]],
                [position[1]],
                s=90 if marker == "*" else 62,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
                label=label,
            )

    trajectory_ax.set_title(f"{key.callsign_segment} / {key.icao24} trajectory")
    trajectory_ax.set_xlabel("Longitude [deg]")
    trajectory_ax.set_ylabel("Latitude [deg]")
    trajectory_ax.grid(True, alpha=0.25)
    trajectory_ax.legend(loc="best", fontsize=8.5)
    trajectory_ax.axis("equal")

    cross_track_ax.plot(adsb_time_s, adsb_cross_track_m, color=ADSB_COLOR, linewidth=1.6, label="ADS-B")
    cross_track_ax.plot(sim_time_s, sim_cross_track_m, color=SIMAP_COLOR, linewidth=1.8, label="SIMAP")
    if bank_first_active is not None:
        cross_track_ax.axvline(bank_first_active, color=UPPER_STATE_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)
    if bank_active_count > 0:
        bank_active_times = sim_time_s[bank_state != 0]
        bank_active_values = sim_cross_track_m[bank_state != 0]
        cross_track_ax.scatter(bank_active_times, bank_active_values, color=UPPER_STATE_COLOR, s=12, zorder=3)
    cross_track_ax.set_title("Cross-track deviation")
    cross_track_ax.set_ylabel("cross-track [m]")
    cross_track_ax.grid(True, alpha=0.25)
    cross_track_ax.legend(loc="best", fontsize=8.5)
    _hide_xticklabels(cross_track_ax)

    track_error_ax.plot(sim_time_s, sim_track_error_deg, color="#0f766e", linewidth=1.8, label="SIMAP")
    track_error_ax.axhline(0.0, color=FREE_STATE_COLOR, linestyle=":", linewidth=1.0)
    if bank_first_active is not None:
        track_error_ax.axvline(bank_first_active, color=UPPER_STATE_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)
    track_error_ax.set_title("Track error")
    track_error_ax.set_ylabel("error [deg]")
    track_error_ax.grid(True, alpha=0.25)
    track_error_ax.legend(loc="best", fontsize=8.5)
    _hide_xticklabels(track_error_ax)

    cas_ax.plot(adsb_time_s, adsb.speed_mps, color=ADSB_COLOR, linewidth=1.6, label="ADS-B")
    cas_ax.plot(sim_time_s, bichannel.v_cas_mps, color=SIMAP_COLOR, linewidth=1.8, label="SIMAP")
    cas_ax.fill_between(
        sim_time_s,
        cas_lower_mps,
        cas_upper_mps,
        color=CAS_COLOR,
        alpha=0.14,
        label="CAS envelope",
    )
    if cas_first_active is not None:
        cas_ax.axvline(cas_first_active, color=CAS_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)
    if cas_active_count > 0:
        active = cas_state != 0
        cas_ax.scatter(sim_time_s[active], bichannel.v_cas_mps[active], color=CAS_COLOR, s=12, zorder=3)
    cas_ax.set_title("CAS")
    cas_ax.set_ylabel("CAS [m/s]")
    cas_ax.grid(True, alpha=0.25)
    cas_ax.legend(loc="best", fontsize=8.5)
    _hide_xticklabels(cas_ax)

    _plot_state_axis(
        cas_state_ax,
        sim_time_s,
        cas_state,
        title="CAS envelope enforcement",
        first_active_time_s=cas_first_active,
    )
    _hide_xticklabels(cas_state_ax)

    gamma_ax.plot(sim_time_s, sim_gamma_deg, color=GAMMA_COLOR, linewidth=1.8, label="SIMAP")
    if gamma_lower_deg is not None and gamma_upper_deg is not None:
        gamma_ax.fill_between(
            sim_time_s,
            gamma_lower_deg,
            gamma_upper_deg,
            color=GAMMA_COLOR,
            alpha=0.14,
            label="gamma envelope",
        )
    if gamma_first_active is not None:
        gamma_ax.axvline(gamma_first_active, color=GAMMA_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)
    if gamma_active_count > 0:
        active = gamma_state != 0
        gamma_ax.scatter(sim_time_s[active], sim_gamma_deg[active], color=GAMMA_COLOR, s=12, zorder=3)
    gamma_ax.set_title("Flight-path angle")
    gamma_ax.set_ylabel("gamma [deg]")
    gamma_ax.grid(True, alpha=0.25)
    gamma_ax.legend(loc="best", fontsize=8.5)
    _hide_xticklabels(gamma_ax)

    _plot_state_axis(
        gamma_state_ax,
        sim_time_s,
        gamma_state,
        title="Gamma envelope enforcement",
        first_active_time_s=gamma_first_active,
    )
    _hide_xticklabels(gamma_state_ax)

    bank_ax.plot(sim_time_s, sim_phi_deg, color=SIMAP_COLOR, linewidth=1.8, label="actual bank")
    bank_ax.plot(sim_time_s, sim_phi_req_deg, color=BANK_COLOR, linewidth=1.4, linestyle="--", label="requested bank")
    bank_ax.fill_between(
        sim_time_s,
        -sim_phi_max_deg,
        sim_phi_max_deg,
        color=BANK_COLOR,
        alpha=0.12,
        label="bank limit",
    )
    if bank_first_active is not None:
        bank_ax.axvline(bank_first_active, color=BANK_COLOR, linestyle="--", linewidth=1.0, alpha=0.75)
    if bank_active_count > 0:
        active = bank_state != 0
        bank_ax.scatter(sim_time_s[active], sim_phi_req_deg[active], color=BANK_COLOR, s=12, zorder=3)
    bank_ax.set_title("Bank angle")
    bank_ax.set_ylabel("bank [deg]")
    bank_ax.grid(True, alpha=0.25)
    bank_ax.legend(loc="best", fontsize=8.5)

    _plot_state_axis(
        bank_state_ax,
        sim_time_s,
        bank_state,
        title="Bank-angle envelope enforcement",
        first_active_time_s=bank_first_active,
    )

    start_time = min(float(adsb_time_s[0]), float(sim_time_s[0]))
    end_time = max(float(adsb_time_s[-1]), float(sim_time_s[-1]))
    initial_time = max(float(adsb_time_s[0]), float(sim_time_s[0]))

    summary_text = "\n".join(
        [
            _fmt_unix_time(initial_time),
            _fmt_sample("ADS-B", _sample_trajectory(adsb, initial_time), speed_label="ground speed"),
            _fmt_sample("SIMAP", _sample_trajectory(_ResultTrajectoryAdapter(bichannel, seed), initial_time), speed_label="CAS"),
        ]
    )
    trajectory_ax.text(
        0.02,
        0.98,
        summary_text,
        transform=trajectory_ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        linespacing=1.28,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#333333", "alpha": 0.93},
    )

    slider = Slider(
        ax=slider_ax,
        label="Time",
        valmin=start_time,
        valmax=end_time,
        valinit=initial_time,
        valstep=1.0,
    )

    current_adsb_point, = trajectory_ax.plot([], [], "o", color=ADSB_COLOR, markersize=7.5, zorder=7)
    current_simap_point, = trajectory_ax.plot([], [], "o", color=SIMAP_COLOR, markersize=7.5, zorder=7)
    current_cross_track_adsb_point, = cross_track_ax.plot([], [], "o", color=ADSB_COLOR, markersize=6.0, zorder=7)
    current_cross_track_simap_point, = cross_track_ax.plot([], [], "o", color=SIMAP_COLOR, markersize=6.0, zorder=7)
    current_track_error_point, = track_error_ax.plot([], [], "o", color="#0f766e", markersize=6.0, zorder=7)
    current_cas_adsb_point, = cas_ax.plot([], [], "o", color=ADSB_COLOR, markersize=6.0, zorder=7)
    current_cas_simap_point, = cas_ax.plot([], [], "o", color=SIMAP_COLOR, markersize=6.0, zorder=7)
    current_gamma_point, = gamma_ax.plot([], [], "o", color=GAMMA_COLOR, markersize=6.0, zorder=7)
    current_bank_actual_point, = bank_ax.plot([], [], "o", color=SIMAP_COLOR, markersize=6.0, zorder=7)
    current_bank_requested_point, = bank_ax.plot([], [], "o", color=BANK_COLOR, markersize=6.0, zorder=7)

    def _set_marker(marker, x_value: float | None, y_value: float | None) -> None:
        if x_value is None or y_value is None or not np.isfinite(x_value) or not np.isfinite(y_value):
            marker.set_visible(False)
            return
        marker.set_data([x_value], [y_value])
        marker.set_visible(True)

    def update(time_s: float) -> None:
        time_s = float(time_s)
        adsb_sample = _sample_trajectory(adsb, time_s)
        simap_sample = _sample_trajectory(_ResultTrajectoryAdapter(bichannel, seed), time_s)
        cross_track_sim_m = _interp_value(time_s, sim_time_s, sim_cross_track_m)
        track_error_deg = _interp_value(time_s, sim_time_s, sim_track_error_deg)
        cas_sim_mps = _interp_value(time_s, sim_time_s, bichannel.v_cas_mps)
        gamma_deg = _interp_value(time_s, sim_time_s, sim_gamma_deg)
        bank_actual_deg = _interp_value(time_s, sim_time_s, sim_phi_deg)
        bank_request_deg = _interp_value(time_s, sim_time_s, sim_phi_req_deg)

        _set_marker(current_adsb_point, adsb_sample.lon_deg, adsb_sample.lat_deg)
        _set_marker(current_simap_point, simap_sample.lon_deg, simap_sample.lat_deg)
        _set_marker(current_cross_track_adsb_point, adsb_sample.time_s, _interp_value(time_s, adsb_time_s, adsb_cross_track_m))
        _set_marker(current_cross_track_simap_point, time_s, cross_track_sim_m)
        _set_marker(current_track_error_point, time_s, track_error_deg)
        _set_marker(current_cas_adsb_point, adsb_sample.time_s, adsb_sample.speed_mps)
        _set_marker(current_cas_simap_point, time_s, cas_sim_mps)
        _set_marker(current_gamma_point, time_s, gamma_deg)
        _set_marker(current_bank_actual_point, time_s, bank_actual_deg)
        _set_marker(current_bank_requested_point, time_s, bank_request_deg)

        cross_track_ax.set_title(
            _current_title(
                "Cross-track deviation",
                time_s,
                f"SIMAP {UNAVAILABLE if cross_track_sim_m is None else f'{cross_track_sim_m:+.1f} m'}",
            )
        )
        track_error_ax.set_title(
            _current_title(
                "Track error",
                time_s,
                f"SIMAP {UNAVAILABLE if track_error_deg is None else f'{track_error_deg:+.2f} deg'}",
            )
        )
        cas_ax.set_title(
            _current_title(
                "CAS",
                time_s,
                f"SIMAP {UNAVAILABLE if cas_sim_mps is None else f'{mps_to_kts(cas_sim_mps):.1f} kt'}",
            )
        )
        gamma_ax.set_title(
            _current_title(
                "Flight-path angle",
                time_s,
                f"SIMAP {UNAVAILABLE if gamma_deg is None else f'{gamma_deg:+.2f} deg'}",
            )
        )
        bank_ax.set_title(
            _current_title(
                "Bank angle",
                time_s,
                (
                    "actual "
                    + (UNAVAILABLE if bank_actual_deg is None else f"{bank_actual_deg:+.2f} deg")
                    + ", req "
                    + (UNAVAILABLE if bank_request_deg is None else f"{bank_request_deg:+.2f} deg")
                ),
            )
        )
        trajectory_ax.set_title(f"{key.callsign_segment} / {key.icao24} trajectory @ {_fmt_unix_time(time_s)}")
        trajectory_ax.figure.canvas.draw_idle()

    slider.on_changed(update)
    update(initial_time)
    fig.tight_layout()
    if "agg" not in plt.get_backend().lower():
        plt.show()


@dataclass(frozen=True)
class _ResultTrajectoryAdapter:
    result: FMSBiChannelResult
    seed: SeedLike

    @property
    def time_s(self) -> np.ndarray:
        return float(self.seed.time_s) + np.asarray(self.result.t_s, dtype=float)

    @property
    def lat_deg(self) -> np.ndarray:
        return np.asarray(self.result.lat_deg, dtype=float)

    @property
    def lon_deg(self) -> np.ndarray:
        return np.asarray(self.result.lon_deg, dtype=float)

    @property
    def altitude_m(self) -> np.ndarray:
        return np.asarray(self.result.h_m, dtype=float)

    @property
    def speed_mps(self) -> np.ndarray:
        return np.asarray(self.result.v_cas_mps, dtype=float)
