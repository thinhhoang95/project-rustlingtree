from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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


__all__ = [
    "plot_all_state_responses",
    "plot_altitude_response",
    "plot_lat_response",
    "plot_lon_response",
    "plot_phi_response",
    "plot_psi_response",
    "plot_s_response",
    "plot_state_overview",
]
