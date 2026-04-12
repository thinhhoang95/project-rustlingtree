from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

from .path_geometry import ReferencePath


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
    figsize: tuple[float, float] = (14.0, 10.0),
    in_degrees: bool = True,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Alias for plot_state_overview for clearer intent in analysis scripts."""
    return plot_state_overview(
        trajectory,
        figsize=figsize,
        in_degrees=in_degrees,
        show=show,
    )


def plot_trajectory_map_scrubber(
    trajectory: Any,
    *,
    reference_path: ReferencePath | None = None,
    show_reference_turning_points: bool = True,
    figsize: tuple[float, float] = (11.0, 8.0),
    initial_time_s: float | None = None,
    show: bool = True,
    add_features: bool = True,
    title: str = "Trajectory map scrubber",
) -> tuple[Figure, Axes, Slider, Line2D]:
    """Plot a cartopy map with a time scrubber and a movable aircraft marker.

    The map, trajectory polyline, and any optional geographic features are
    rendered once. The slider callback only updates the marker position and the
    time annotation so the map itself is not redrawn.
    """
    t_s = _series(trajectory, "t_s")
    lat_deg = _series(trajectory, "lat_deg")
    lon_deg = _series(trajectory, "lon_deg")
    _match_time_and_values(t_s, lat_deg, "lat_deg")
    _match_time_and_values(t_s, lon_deg, "lon_deg")

    ccrs, cfeature = _require_cartopy()
    plate = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize)
    map_ax = fig.add_axes([0.05, 0.18, 0.9, 0.76], projection=plate)
    slider_ax = fig.add_axes([0.12, 0.07, 0.76, 0.04])

    lat_samples = [lat_deg]
    lon_samples = [lon_deg]
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

    map_ax.set_title(title)
    map_ax.set_extent(
        [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
        crs=plate,
    )
    map_ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.35, linestyle=":")
    map_ax.plot(
        lon_deg,
        lat_deg,
        transform=plate,
        color="#6c757d",
        linewidth=1.6,
        alpha=0.9,
        zorder=2,
    )
    if reference_path is not None:
        map_ax.plot(
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
            map_ax.scatter(
                reference_path.waypoint_lon_deg,
                reference_path.waypoint_lat_deg,
                transform=plate,
                s=26.0,
                color="#f28e2b",
                edgecolors="none",
                zorder=4,
            )

    if add_features:
        try:
            map_ax.add_feature(cfeature.LAND, facecolor="#f1efe7", zorder=0)
            map_ax.add_feature(cfeature.OCEAN, facecolor="#dceaf7", zorder=0)
            map_ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=1)
            map_ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6, zorder=1)
        except Exception:
            # Natural Earth data may not be cached in all environments.
            pass

    start_index = 0 if initial_time_s is None else int(np.argmin(np.abs(t_s - float(initial_time_s))))
    start_index = int(np.clip(start_index, 0, len(t_s) - 1))
    marker, = map_ax.plot(
        [lon_deg[start_index]],
        [lat_deg[start_index]],
        marker="o",
        markersize=9,
        color="#d62728",
        linestyle="None",
        transform=plate,
        zorder=3,
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
        marker.set_data([lon_deg[idx]], [lat_deg[idx]])
        time_label.set_text(f"t = {t_s[idx]:.1f} s")
        fig.canvas.draw_idle()

    time_slider.on_changed(_update_time)
    _update_time(float(t_s[start_index]))

    if show:
        plt.show()
    return fig, map_ax, time_slider, marker


__all__ = [
    "plot_all_state_responses",
    "plot_altitude_response",
    "plot_lat_response",
    "plot_lon_response",
    "plot_phi_response",
    "plot_psi_response",
    "plot_s_response",
    "plot_state_overview",
    "plot_trajectory_map_scrubber",
]
