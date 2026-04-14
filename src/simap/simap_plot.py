from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

from .lateral_dynamics import wrap_angle_rad
from .longitudinal_profiles import ScalarProfile
from .path_geometry import EARTH_RADIUS_M, ReferencePath


def _series(trajectory: Any, field: str) -> np.ndarray:
    if not hasattr(trajectory, field):
        raise AttributeError(f"trajectory does not provide required field '{field}'")
    values = np.asarray(getattr(trajectory, field), dtype=float)
    if values.ndim != 1:
        raise ValueError(f"trajectory field '{field}' must be one-dimensional")
    return values


def _match_time_and_values(t_s: np.ndarray, values: np.ndarray, label: str) -> None:
    if len(t_s) != len(values):
        raise ValueError(f"time series and {label} must have the same length")


def _require_cartopy() -> tuple[Any, Any]:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:  # pragma: no cover - exercised only when cartopy is absent
        raise ImportError("plot_trajectory_map_scrubber requires cartopy to be installed") from exc
    return ccrs, cfeature


def _plot_cartopy_trajectory(
    ax: Axes,
    *,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    plate: Any,
    cfeature: Any | None = None,
    reference_path: ReferencePath | None = None,
    show_reference_turning_points: bool = True,
    add_features: bool = True,
) -> None:
    lat_samples = [np.asarray(lat_deg, dtype=float)]
    lon_samples = [np.asarray(lon_deg, dtype=float)]
    if reference_path is not None:
        lat_samples.append(np.asarray(reference_path.lat_deg, dtype=float))
        lon_samples.append(np.asarray(reference_path.lon_deg, dtype=float))

    lat_all = np.concatenate(lat_samples)
    lon_all = np.concatenate(lon_samples)
    lat_min = float(np.min(lat_all))
    lat_max = float(np.max(lat_all))
    lon_min = float(np.min(lon_all))
    lon_max = float(np.max(lon_all))
    lat_pad = max(0.01, 0.08 * max(1e-9, lat_max - lat_min))
    lon_pad = max(0.01, 0.08 * max(1e-9, lon_max - lon_min))

    ax.set_extent([lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad], crs=plate)
    ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.35, linestyle=":")
    ax.plot(
        lon_deg,
        lat_deg,
        transform=plate,
        color="#6c757d",
        linewidth=1.6,
        alpha=0.9,
        zorder=2,
    )
    if reference_path is not None:
        ax.plot(
            reference_path.lon_deg,
            reference_path.lat_deg,
            transform=plate,
            color="#212529",
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
            zorder=2,
        )
        if show_reference_turning_points:
            ax.scatter(
                reference_path.waypoint_lon_deg,
                reference_path.waypoint_lat_deg,
                transform=plate,
                s=26.0,
                color="#f28e2b",
                edgecolors="none",
                zorder=4,
            )

    if add_features and cfeature is not None:
        try:
            ax.add_feature(cfeature.LAND, facecolor="#f1efe7", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="#dceaf7", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6, zorder=1)
        except Exception:
            # Natural Earth data may not be cached in all environments.
            pass


def _plot_cartopy_reference_path(
    ax: Axes,
    *,
    reference_path: ReferencePath,
    plate: Any,
    cfeature: Any | None = None,
    show_reference_turning_points: bool = True,
    add_features: bool = True,
) -> None:
    lat_deg = np.asarray(reference_path.lat_deg, dtype=float)
    lon_deg = np.asarray(reference_path.lon_deg, dtype=float)
    waypoint_lat_deg = np.asarray(reference_path.waypoint_lat_deg, dtype=float)
    waypoint_lon_deg = np.asarray(reference_path.waypoint_lon_deg, dtype=float)

    lat_min = float(np.min(np.concatenate([lat_deg, waypoint_lat_deg])))
    lat_max = float(np.max(np.concatenate([lat_deg, waypoint_lat_deg])))
    lon_min = float(np.min(np.concatenate([lon_deg, waypoint_lon_deg])))
    lon_max = float(np.max(np.concatenate([lon_deg, waypoint_lon_deg])))
    lat_pad = max(0.01, 0.08 * max(1e-9, lat_max - lat_min))
    lon_pad = max(0.01, 0.08 * max(1e-9, lon_max - lon_min))

    ax.set_extent([lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad], crs=plate)
    ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.35, linestyle=":")
    ax.plot(
        lon_deg,
        lat_deg,
        transform=plate,
        color="#212529",
        linewidth=1.8,
        alpha=0.95,
        zorder=2,
    )
    if show_reference_turning_points:
        ax.scatter(
            waypoint_lon_deg,
            waypoint_lat_deg,
            transform=plate,
            s=26.0,
            color="#f28e2b",
            edgecolors="none",
            zorder=4,
        )

    if add_features and cfeature is not None:
        try:
            ax.add_feature(cfeature.LAND, facecolor="#f1efe7", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="#dceaf7", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6, zorder=1)
        except Exception:
            # Natural Earth data may not be cached in all environments.
            pass


def _heading_uv_deg(
    lat_deg: np.ndarray,
    heading_rad: np.ndarray,
    *,
    arrow_length_m: float = 1500.0,
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat_deg, dtype=float)
    heading = np.asarray(heading_rad, dtype=float)
    lat0_rad = np.deg2rad(lat)
    cos_lat = np.clip(np.cos(lat0_rad), 1e-6, None)
    east_m = arrow_length_m * np.cos(heading)
    north_m = arrow_length_m * np.sin(heading)
    u = np.rad2deg(east_m / (EARTH_RADIUS_M * cos_lat))
    v = np.rad2deg(north_m / EARTH_RADIUS_M)
    return u, v


def _ground_track_rad(
    t_s: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    idx: int,
) -> float:
    if len(t_s) < 2:
        return 0.0
    if idx <= 0:
        i0, i1 = 0, 1
    elif idx >= len(t_s) - 1:
        i0, i1 = len(t_s) - 2, len(t_s) - 1
    else:
        i0, i1 = idx - 1, idx + 1
    lat_mid_rad = np.deg2rad(0.5 * (float(lat_deg[i0]) + float(lat_deg[i1])))
    east_m = EARTH_RADIUS_M * np.cos(lat_mid_rad) * np.deg2rad(float(lon_deg[i1]) - float(lon_deg[i0]))
    north_m = EARTH_RADIUS_M * np.deg2rad(float(lat_deg[i1]) - float(lat_deg[i0]))
    return float(np.arctan2(north_m, east_m))


def _optional_series(trajectory: Any, *fields: str) -> np.ndarray | None:
    for field in fields:
        if hasattr(trajectory, field):
            return _series(trajectory, field)
    return None


def _format_tracking_transition(
    unclamped_text: str | None,
    clamped_text: str | None,
    *,
    bracketed: bool,
) -> str:
    if unclamped_text is not None and clamped_text is not None:
        body = f"{unclamped_text} -> {clamped_text}"
    elif clamped_text is not None:
        body = f"-> {clamped_text}"
    elif unclamped_text is not None:
        body = f"{unclamped_text} ->"
    else:
        return ""
    return f"[{body}]" if bracketed else body


def _format_tracking_line(
    label: str,
    *,
    unclamped_target: str | None = None,
    clamped_target: str | None = None,
    unclamped_value: str | None = None,
    clamped_value: str | None = None,
) -> str:
    target_text = _format_tracking_transition(unclamped_target, clamped_target, bracketed=True)
    value_text = _format_tracking_transition(unclamped_value, clamped_value, bracketed=False)
    if target_text and value_text:
        return f"{label}: {target_text} {value_text}"
    if target_text:
        return f"{label}: {target_text}"
    if value_text:
        return f"{label}: {value_text}"
    return f"{label}: n/a"


def _tracking_status_text(
    *,
    trajectory: Any,
    reference_path: ReferencePath,
    idx: int,
) -> str:
    s_m = _series(trajectory, "s_m")
    t_s = _series(trajectory, "t_s")
    lat_deg = _series(trajectory, "lat_deg")
    lon_deg = _series(trajectory, "lon_deg")
    cross_track_series = _optional_series(trajectory, "cross_track_m")
    s_unclamped_series = _optional_series(trajectory, "s_unclamped_m")
    h_m = _optional_series(trajectory, "h_m")
    h_ref_series = _optional_series(trajectory, "h_ref_m")
    h_ref_unclamped_series = _optional_series(
        trajectory,
        "h_ref_unclamped_m",
        "h_ref_raw_m",
    )
    v_tas_mps = _optional_series(trajectory, "v_tas_mps")
    v_tas_unclamped_series = _optional_series(
        trajectory,
        "v_tas_unclamped_mps",
        "v_tas_raw_mps",
    )
    v_ref_tas_series = _optional_series(trajectory, "v_ref_tas_mps")
    v_ref_tas_unclamped_series = _optional_series(
        trajectory,
        "v_ref_tas_unclamped_mps",
        "v_ref_unclamped_tas_mps",
        "v_ref_raw_tas_mps",
    )
    heading_series = _optional_series(trajectory, "heading_rad", "psi_rad")
    heading_target_unclamped_series = _optional_series(
        trajectory,
        "heading_target_unclamped_rad",
        "psi_target_unclamped_rad",
    )
    heading_target_series = _optional_series(
        trajectory,
        "heading_target_rad",
        "psi_target_rad",
    )
    bank_rad = _optional_series(trajectory, "bank_rad", "phi_rad")
    phi_req_series = _optional_series(trajectory, "phi_req_rad", "bank_target_rad", "phi_target_rad")
    phi_req_unclamped_series = _optional_series(
        trajectory,
        "phi_req_unclamped_rad",
        "phi_req_unclipped_rad",
        "phi_req_raw_rad",
        "bank_target_unclamped_rad",
    )
    phi_max_series = _optional_series(trajectory, "phi_max_rad")

    optional_series = [
        cross_track_series,
        s_unclamped_series,
        h_m,
        h_ref_series,
        h_ref_unclamped_series,
        v_tas_mps,
        v_tas_unclamped_series,
        v_ref_tas_series,
        v_ref_tas_unclamped_series,
        heading_series,
        heading_target_unclamped_series,
        heading_target_series,
        bank_rad,
        phi_req_series,
        phi_req_unclamped_series,
        phi_max_series,
    ]
    for values in optional_series:
        if values is not None:
            _match_time_and_values(t_s, values, "tracking series")

    ref_track_rad = reference_path.track_angle_rad(float(s_m[idx]))
    if cross_track_series is not None:
        cross_track_m = float(cross_track_series[idx])
    else:
        ref_east_m, ref_north_m = reference_path.position_ne(float(s_m[idx]))
        east_m = EARTH_RADIUS_M * np.cos(np.deg2rad(reference_path.origin_lat_deg)) * np.deg2rad(
            float(lon_deg[idx]) - reference_path.origin_lon_deg
        )
        north_m = EARTH_RADIUS_M * np.deg2rad(float(lat_deg[idx]) - reference_path.origin_lat_deg)
        error_vector = np.asarray([east_m - ref_east_m, north_m - ref_north_m], dtype=float)
        cross_track_m = float(np.dot(error_vector, reference_path.normal_hat(float(s_m[idx]))))
    track_error_rad = wrap_angle_rad(_ground_track_rad(t_s, lat_deg, lon_deg, idx) - ref_track_rad)

    bank_target_unclamped_rad = float(phi_req_unclamped_series[idx]) if phi_req_unclamped_series is not None else None

    if phi_req_series is not None:
        bank_target_clamped_rad = float(phi_req_series[idx])
    elif phi_max_series is not None and bank_target_unclamped_rad is not None:
        phi_max_rad = float(phi_max_series[idx])
        bank_target_clamped_rad = float(np.clip(bank_target_unclamped_rad, -phi_max_rad, phi_max_rad))
    else:
        bank_target_clamped_rad = None

    def _fmt_speed(value_mps: float) -> str:
        return f"{value_mps:.2f} m/s ({value_mps / 0.514444:.1f} kt)"

    def _fmt_scalar_meter(value_m: float) -> str:
        return f"{value_m:.1f} m"

    def _fmt_error_meter(value_m: float) -> str:
        return f"{value_m:+.1f} m"

    def _fmt_angle_deg(value_rad: float) -> str:
        return f"{np.rad2deg(value_rad):+.2f} deg"

    reference_s_unclamped = float(s_unclamped_series[idx]) if s_unclamped_series is not None else None
    reference_s_clamped = float(s_m[idx])
    tas_target_unclamped = (
        _fmt_speed(float(v_ref_tas_unclamped_series[idx])) if v_ref_tas_unclamped_series is not None else None
    )
    tas_target_clamped = _fmt_speed(float(v_ref_tas_series[idx])) if v_ref_tas_series is not None else None
    tas_value_unclamped = _fmt_speed(float(v_tas_unclamped_series[idx])) if v_tas_unclamped_series is not None else None
    tas_value_clamped = _fmt_speed(float(v_tas_mps[idx])) if v_tas_mps is not None else None
    altitude_target_unclamped = _fmt_scalar_meter(float(h_ref_unclamped_series[idx])) if h_ref_unclamped_series is not None else None
    altitude_target_clamped = _fmt_scalar_meter(float(h_ref_series[idx])) if h_ref_series is not None else None
    altitude_value_unclamped = None
    altitude_value_clamped = _fmt_scalar_meter(float(h_m[idx])) if h_m is not None else None
    heading_target_unclamped = (
        _fmt_angle_deg(float(heading_target_unclamped_series[idx])) if heading_target_unclamped_series is not None else None
    )
    heading_target_clamped = _fmt_angle_deg(float(heading_target_series[idx])) if heading_target_series is not None else None
    heading_value_unclamped = None
    heading_value_clamped = _fmt_angle_deg(float(heading_series[idx])) if heading_series is not None else None
    bank_target_unclamped = _fmt_angle_deg(bank_target_unclamped_rad) if bank_target_unclamped_rad is not None else None
    bank_target_clamped = _fmt_angle_deg(bank_target_clamped_rad) if bank_target_clamped_rad is not None else None
    bank_value_unclamped = None
    bank_value_clamped = _fmt_angle_deg(float(bank_rad[idx])) if bank_rad is not None else None

    lines = [
        _format_tracking_line(
            "Reference s",
            unclamped_value=f"{reference_s_unclamped:.1f} m" if reference_s_unclamped is not None else None,
            clamped_value=f"{reference_s_clamped:.1f} m",
        ),
        _format_tracking_line(
            "TAS",
            unclamped_target=tas_target_unclamped,
            clamped_target=tas_target_clamped,
            unclamped_value=tas_value_unclamped,
            clamped_value=tas_value_clamped,
        ),
        _format_tracking_line(
            "Altitude",
            unclamped_target=altitude_target_unclamped,
            clamped_target=altitude_target_clamped,
            unclamped_value=altitude_value_unclamped,
            clamped_value=altitude_value_clamped,
        ),
        _format_tracking_line(
            "Cross-track error",
            clamped_target=_fmt_error_meter(0.0),
            clamped_value=_fmt_error_meter(cross_track_m),
        ),
        _format_tracking_line(
            "Track-error",
            clamped_target=_fmt_angle_deg(0.0),
            clamped_value=_fmt_angle_deg(track_error_rad),
        ),
        _format_tracking_line(
            "Heading",
            unclamped_target=heading_target_unclamped,
            clamped_target=heading_target_clamped,
            unclamped_value=heading_value_unclamped,
            clamped_value=heading_value_clamped,
        ),
        _format_tracking_line(
            "Bank angle",
            unclamped_target=bank_target_unclamped,
            clamped_target=bank_target_clamped,
            unclamped_value=bank_value_unclamped,
            clamped_value=bank_value_clamped,
        ),
    ]
    return "\n".join(lines)


def _plot(
    ax: Axes,
    t_s: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    color: str,
    reference: np.ndarray | None = None,
    reference_label: str | None = None,
) -> Axes:
    ax.plot(t_s, values, color=color, linewidth=1.8, label=None)
    if reference is not None:
        ax.plot(
            t_s,
            reference,
            color="#4d4d4d",
            linewidth=1.4,
            linestyle="--",
            label=reference_label,
        )
        if reference_label is not None:
            ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax


def plot_s_response(trajectory: Any, *, ax: Axes | None = None) -> Axes:
    t_s = _series(trajectory, "t_s")
    s_m = _series(trajectory, "s_m")
    _match_time_and_values(t_s, s_m, "s_m")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(axis, t_s, s_m, title="Along-track Distance", ylabel="s [m]", color="#1f77b4")


def plot_initial_profiles(
    reference_path: ReferencePath,
    altitude_profile: ScalarProfile,
    raw_speed_schedule_cas: ScalarProfile,
    *,
    feasible_speed_schedule_cas: ScalarProfile | None = None,
    figsize: tuple[float, float] = (12.0, 8.5),
    show_reference_turning_points: bool = True,
    add_features: bool = True,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot the geographic reference path and the initial scalar profiles.

    The top panel shows the planned route on a map. The bottom panel overlays
    altitude and raw CAS schedule against along-track distance. When provided,
    the feasible CAS profile is overlaid as a dashed line.
    """
    ccrs, cfeature = _require_cartopy()
    plate = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=(1.15, 1.0), hspace=0.35)
    map_ax = fig.add_subplot(gs[0], projection=plate)
    profile_ax = fig.add_subplot(gs[1])
    speed_ax = profile_ax.twinx()

    map_ax.set_title("Reference Path")
    _plot_cartopy_reference_path(
        map_ax,
        reference_path=reference_path,
        plate=plate,
        cfeature=cfeature,
        show_reference_turning_points=show_reference_turning_points,
        add_features=add_features,
    )

    profile_ax.plot(
        altitude_profile.s_m,
        altitude_profile.y,
        color="#2ca02c",
        linewidth=1.8,
        label="Altitude",
    )
    speed_ax.plot(
        raw_speed_schedule_cas.s_m,
        raw_speed_schedule_cas.y,
        color="#1f77b4",
        linewidth=1.8,
        label="Raw CAS",
    )
    if feasible_speed_schedule_cas is not None:
        speed_ax.plot(
            feasible_speed_schedule_cas.s_m,
            feasible_speed_schedule_cas.y,
            color="#1f77b4",
            linewidth=1.8,
            linestyle="--",
            label="Feasible CAS",
        )
    profile_ax.set_title("Initial Longitudinal Profiles")
    profile_ax.set_xlabel("Along-track distance [m]")
    profile_ax.set_ylabel("Altitude [m]", color="#2ca02c")
    speed_ax.set_ylabel("CAS [m/s]", color="#1f77b4")
    profile_ax.tick_params(axis="y", colors="#2ca02c")
    speed_ax.tick_params(axis="y", colors="#1f77b4")
    profile_ax.grid(True, alpha=0.3)
    s_nodes = [altitude_profile.s_m, raw_speed_schedule_cas.s_m]
    if feasible_speed_schedule_cas is not None:
        s_nodes.append(feasible_speed_schedule_cas.s_m)
    profile_ax.set_xlim(float(min(np.min(nodes) for nodes in s_nodes)), float(max(np.max(nodes) for nodes in s_nodes)))

    lines = profile_ax.get_lines() + speed_ax.get_lines()
    labels = [line.get_label() for line in lines]
    profile_ax.legend(lines, labels, loc="best")

    fig.suptitle("SIMAP Initial Profiles")
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.93)
    if show:
        plt.show()

    axes = np.asarray([map_ax, profile_ax, speed_ax], dtype=object)
    return fig, axes


def plot_lat_response(trajectory: Any, *, ax: Axes | None = None) -> Axes:
    t_s = _series(trajectory, "t_s")
    lat_deg = _series(trajectory, "lat_deg")
    _match_time_and_values(t_s, lat_deg, "lat_deg")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(axis, t_s, lat_deg, title="Latitude", ylabel="lat [deg]", color="#d62728")


def plot_lon_response(trajectory: Any, *, ax: Axes | None = None) -> Axes:
    t_s = _series(trajectory, "t_s")
    lon_deg = _series(trajectory, "lon_deg")
    _match_time_and_values(t_s, lon_deg, "lon_deg")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(axis, t_s, lon_deg, title="Longitude", ylabel="lon [deg]", color="#9467bd")


def plot_altitude_response(trajectory: Any, *, ax: Axes | None = None) -> Axes:
    t_s = _series(trajectory, "t_s")
    h_m = _series(trajectory, "h_m")
    _match_time_and_values(t_s, h_m, "h_m")
    h_ref_m = _series(trajectory, "h_ref_m") if hasattr(trajectory, "h_ref_m") else None
    if h_ref_m is not None:
        _match_time_and_values(t_s, h_ref_m, "h_ref_m")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(
        axis,
        t_s,
        h_m,
        title="Altitude",
        ylabel="h [m]",
        color="#2ca02c",
        reference=h_ref_m,
        reference_label="h_ref",
    )


def plot_tas_response(trajectory: Any, *, ax: Axes | None = None) -> Axes:
    t_s = _series(trajectory, "t_s")
    v_tas_mps = _series(trajectory, "v_tas_mps")
    _match_time_and_values(t_s, v_tas_mps, "v_tas_mps")
    v_ref_tas_mps = _series(trajectory, "v_ref_tas_mps") if hasattr(trajectory, "v_ref_tas_mps") else None
    if v_ref_tas_mps is not None:
        _match_time_and_values(t_s, v_ref_tas_mps, "v_ref_tas_mps")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(
        axis,
        t_s,
        v_tas_mps,
        title="TAS",
        ylabel="v_tas [m/s]",
        color="#1f77b4",
        reference=v_ref_tas_mps,
        reference_label="v_ref_tas",
    )


def plot_cas_response(trajectory: Any, *, ax: Axes | None = None) -> Axes:
    t_s = _series(trajectory, "t_s")
    v_cas_mps = _series(trajectory, "v_cas_mps")
    _match_time_and_values(t_s, v_cas_mps, "v_cas_mps")
    v_ref_cas_mps = _series(trajectory, "v_ref_cas_mps") if hasattr(trajectory, "v_ref_cas_mps") else None
    if v_ref_cas_mps is not None:
        _match_time_and_values(t_s, v_ref_cas_mps, "v_ref_cas_mps")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(
        axis,
        t_s,
        v_cas_mps,
        title="CAS",
        ylabel="v_cas [m/s]",
        color="#ff7f0e",
        reference=v_ref_cas_mps,
        reference_label="v_ref_cas",
    )


def plot_psi_response(
    trajectory: Any,
    *,
    ax: Axes | None = None,
    in_degrees: bool = True,
) -> Axes:
    t_s = _series(trajectory, "t_s")
    psi_rad = (
        _series(trajectory, "heading_rad")
        if hasattr(trajectory, "heading_rad")
        else _series(trajectory, "psi_rad")
    )
    _match_time_and_values(t_s, psi_rad, "heading/psi")
    values = np.rad2deg(psi_rad) if in_degrees else psi_rad
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(
        axis,
        t_s,
        values,
        title="Heading (psi)",
        ylabel="psi [deg]" if in_degrees else "psi [rad]",
        color="#ff7f0e",
    )


def plot_phi_response(
    trajectory: Any,
    *,
    ax: Axes | None = None,
    in_degrees: bool = True,
) -> Axes:
    t_s = _series(trajectory, "t_s")
    phi_rad = (
        _series(trajectory, "bank_rad") if hasattr(trajectory, "bank_rad") else _series(trajectory, "phi_rad")
    )
    _match_time_and_values(t_s, phi_rad, "bank/phi")
    values = np.rad2deg(phi_rad) if in_degrees else phi_rad
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(
        axis,
        t_s,
        values,
        title="Bank Angle (phi)",
        ylabel="phi [deg]" if in_degrees else "phi [rad]",
        color="#8c564b",
    )


def plot_state_overview(
    trajectory: Any,
    *,
    figsize: tuple[float, float] = (14.0, 10.0),
    in_degrees: bool = True,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """
    Draw the core state responses in one figure window.

    The six standard panels are: s, lat, lon, altitude, psi, phi.
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    plot_s_response(trajectory, ax=axes[0, 0])
    plot_lat_response(trajectory, ax=axes[0, 1])
    plot_lon_response(trajectory, ax=axes[1, 0])
    plot_altitude_response(trajectory, ax=axes[1, 1])
    plot_psi_response(trajectory, ax=axes[2, 0], in_degrees=in_degrees)
    plot_phi_response(trajectory, ax=axes[2, 1], in_degrees=in_degrees)
    fig.suptitle("SIMAP State Response")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_all_state_responses(
    trajectory: Any,
    *,
    reference_path: ReferencePath | None = None,
    show_reference_turning_points: bool = True,
    figsize: tuple[float, float] = (12.0, 9.5),
    in_degrees: bool = True,
    add_features: bool = True,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Draw the state and longitudinal response plots plus a cartopy map."""
    t_s = _series(trajectory, "t_s")
    s_m = _series(trajectory, "s_m")
    lat_deg = _series(trajectory, "lat_deg")
    lon_deg = _series(trajectory, "lon_deg")
    v_tas_mps = _series(trajectory, "v_tas_mps")
    v_cas_mps = _series(trajectory, "v_cas_mps")
    h_m = _series(trajectory, "h_m")
    _match_time_and_values(t_s, s_m, "s_m")
    _match_time_and_values(t_s, lat_deg, "lat_deg")
    _match_time_and_values(t_s, lon_deg, "lon_deg")
    _match_time_and_values(t_s, v_tas_mps, "v_tas_mps")
    _match_time_and_values(t_s, v_cas_mps, "v_cas_mps")
    _match_time_and_values(t_s, h_m, "h_m")

    ccrs, cfeature = _require_cartopy()
    plate = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, height_ratios=(1.0, 1.0, 1.0, 1.2), hspace=0.5, wspace=0.25)
    axes = np.empty((4, 2), dtype=object)

    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1], sharex=axes[0, 0])
    axes[1, 0] = fig.add_subplot(gs[1, 0], sharex=axes[0, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1], sharex=axes[0, 0])
    axes[2, 0] = fig.add_subplot(gs[2, 0], sharex=axes[0, 0])
    axes[2, 1] = fig.add_subplot(gs[2, 1], sharex=axes[0, 0])
    map_ax = fig.add_subplot(gs[3, :], projection=plate)
    axes[3, 0] = map_ax
    axes[3, 1] = map_ax

    plot_s_response(trajectory, ax=axes[0, 0])
    plot_tas_response(trajectory, ax=axes[0, 1])
    plot_cas_response(trajectory, ax=axes[1, 0])
    plot_altitude_response(trajectory, ax=axes[1, 1])
    plot_psi_response(trajectory, ax=axes[2, 0], in_degrees=in_degrees)
    plot_phi_response(trajectory, ax=axes[2, 1], in_degrees=in_degrees)

    map_ax.set_title("Trajectory Map")
    _plot_cartopy_trajectory(
        map_ax,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        plate=plate,
        cfeature=cfeature,
        reference_path=reference_path,
        show_reference_turning_points=show_reference_turning_points,
        add_features=add_features,
    )

    fig.suptitle("SIMAP State Response")
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.05, right=0.98, hspace=0.55, wspace=0.25)
    if show:
        plt.show()
    return fig, axes


def plot_trajectory_map_scrubber(
    trajectory: Any,
    *,
    reference_path: ReferencePath | None = None,
    show_reference_turning_points: bool = True,
    show_tracking_labels: bool = True,
    figsize: tuple[float, float] = (11.0, 8.0),
    initial_time_s: float | None = None,
    show: bool = True,
    add_features: bool = True,
    title: str = "Trajectory map scrubber",
) -> tuple[Figure, Axes, Slider, Any]:
    """Plot a cartopy map with a time scrubber and a movable aircraft arrow.

    The map, trajectory polyline, and any optional geographic features are
    rendered once. The slider callback only updates the marker position and the
    time annotation so the map itself is not redrawn.
    """
    t_s = _series(trajectory, "t_s")
    s_m = _series(trajectory, "s_m")
    lat_deg = _series(trajectory, "lat_deg")
    lon_deg = _series(trajectory, "lon_deg")
    heading_rad = (
        _series(trajectory, "heading_rad")
        if hasattr(trajectory, "heading_rad")
        else _series(trajectory, "psi_rad")
    )
    _match_time_and_values(t_s, s_m, "s_m")
    _match_time_and_values(t_s, lat_deg, "lat_deg")
    _match_time_and_values(t_s, lon_deg, "lon_deg")
    _match_time_and_values(t_s, heading_rad, "heading/psi")

    ccrs, cfeature = _require_cartopy()
    plate = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize)
    map_ax = fig.add_axes([0.05, 0.18, 0.9, 0.76], projection=plate)
    slider_ax = fig.add_axes([0.12, 0.07, 0.76, 0.04])

    map_ax.set_title(title)
    _plot_cartopy_trajectory(
        map_ax,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        plate=plate,
        cfeature=cfeature,
        reference_path=reference_path,
        show_reference_turning_points=show_reference_turning_points,
        add_features=add_features,
    )

    start_index = 0 if initial_time_s is None else int(np.argmin(np.abs(t_s - float(initial_time_s))))
    start_index = int(np.clip(start_index, 0, len(t_s) - 1))
    arrow_u, arrow_v = _heading_uv_deg(lat_deg, heading_rad)
    aircraft = map_ax.quiver(
        [lon_deg[start_index]],
        [lat_deg[start_index]],
        [arrow_u[start_index]],
        [arrow_v[start_index]],
        transform=plate,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0045,
        headwidth=4.5,
        headlength=6.0,
        headaxislength=5.0,
        color="#d62728",
        zorder=3,
    )
    reference_point = None
    if reference_path is not None:
        ref_lat0, ref_lon0 = reference_path.latlon(float(s_m[start_index]))
        reference_point, = map_ax.plot(
            [ref_lon0],
            [ref_lat0],
            marker="o",
            markersize=5.0,
            color="#2ca02c",
            linestyle="None",
            transform=plate,
            zorder=5,
        )
    status_text = None
    if show_tracking_labels and reference_path is not None:
        status_text = map_ax.text(
            0.02,
            0.34,
            _tracking_status_text(
                trajectory=trajectory,
                reference_path=reference_path,
                idx=start_index,
            ),
            transform=map_ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            zorder=6,
        )
    time_label = map_ax.text(
        0.02,
        0.98,
        f"t = {t_s[start_index]:.1f} s",
        transform=map_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    time_slider = Slider(
        ax=slider_ax,
        label="Time [s]",
        valmin=float(t_s[0]),
        valmax=float(t_s[-1]),
        valinit=float(t_s[start_index]),
    )

    def _update_time(val: float) -> None:
        idx = int(np.argmin(np.abs(t_s - float(val))))
        aircraft.set_offsets(np.asarray([[lon_deg[idx], lat_deg[idx]]], dtype=float))
        aircraft.set_UVC(np.asarray([arrow_u[idx]], dtype=float), np.asarray([arrow_v[idx]], dtype=float))
        if reference_point is not None:
            ref_lat, ref_lon = reference_path.latlon(float(s_m[idx]))
            reference_point.set_data([ref_lon], [ref_lat])
        if status_text is not None:
            status_text.set_text(
                _tracking_status_text(
                    trajectory=trajectory,
                    reference_path=reference_path,
                    idx=idx,
                )
            )
        time_label.set_text(f"t = {t_s[idx]:.1f} s")
        fig.canvas.draw_idle()

    time_slider.on_changed(_update_time)
    _update_time(float(t_s[start_index]))

    if show:
        plt.show()
    return fig, map_ax, time_slider, aircraft


__all__ = [
    "plot_all_state_responses",
    "plot_altitude_response",
    "plot_cas_response",
    "plot_initial_profiles",
    "plot_lat_response",
    "plot_lon_response",
    "plot_phi_response",
    "plot_psi_response",
    "plot_tas_response",
    "plot_s_response",
    "plot_state_overview",
    "plot_trajectory_map_scrubber",
]
