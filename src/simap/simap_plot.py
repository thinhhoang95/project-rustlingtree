from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .longitudinal_profiles import ConstraintEnvelope


def _series(values: Any, field: str) -> np.ndarray:
    if not hasattr(values, field):
        raise AttributeError(f"object does not provide required field '{field}'")
    data = np.asarray(getattr(values, field), dtype=float)
    if data.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    return data


def _plot(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> Axes:
    ax.plot(x, y, color=color, linewidth=1.8)
    if lower is not None and upper is not None:
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax


def _band(envelope: ConstraintEnvelope | None, s_m: np.ndarray, lower_field: str, upper_field: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if envelope is None:
        return None, None
    if lower_field == "h_lower_m" and upper_field == "h_upper_m":
        return envelope.h_bounds_many(s_m)
    if lower_field == "cas_lower_mps" and upper_field == "cas_upper_mps":
        return envelope.cas_bounds_many(s_m)
    if lower_field == "gamma_lower_rad" and upper_field == "gamma_upper_rad":
        return envelope.gamma_bounds_many(s_m)
    if lower_field == "thrust_lower_n" and upper_field == "thrust_upper_n":
        return envelope.thrust_bounds_many(s_m)
    raise ValueError(f"unsupported envelope band fields: {lower_field}, {upper_field}")


def plot_altitude_response(
    plan: Any,
    *,
    envelope: ConstraintEnvelope | None = None,
    ax: Axes | None = None,
) -> Axes:
    s_m = _series(plan, "s_m")
    h_m = _series(plan, "h_m")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    lower, upper = _band(envelope, s_m, "h_lower_m", "h_upper_m")
    return _plot(
        axis,
        s_m,
        h_m,
        title="Planned Altitude",
        xlabel="Distance From Threshold [m]",
        ylabel="h [m]",
        color="#2ca02c",
        lower=lower,
        upper=upper,
    )


def plot_tas_response(plan: Any, *, ax: Axes | None = None) -> Axes:
    s_m = _series(plan, "s_m")
    v_tas_mps = _series(plan, "v_tas_mps")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(
        axis,
        s_m,
        v_tas_mps,
        title="Planned TAS",
        xlabel="Distance From Threshold [m]",
        ylabel="v_tas [m/s]",
        color="#1f77b4",
    )


def plot_cas_response(
    plan: Any,
    *,
    envelope: ConstraintEnvelope | None = None,
    ax: Axes | None = None,
) -> Axes:
    s_m = _series(plan, "s_m")
    v_cas_mps = _series(plan, "v_cas_mps")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    lower, upper = _band(envelope, s_m, "cas_lower_mps", "cas_upper_mps")
    return _plot(
        axis,
        s_m,
        v_cas_mps,
        title="Planned CAS",
        xlabel="Distance From Threshold [m]",
        ylabel="v_cas [m/s]",
        color="#ff7f0e",
        lower=lower,
        upper=upper,
    )


def plot_gamma_response(
    plan: Any,
    *,
    envelope: ConstraintEnvelope | None = None,
    ax: Axes | None = None,
    in_degrees: bool = True,
) -> Axes:
    s_m = _series(plan, "s_m")
    gamma_rad = _series(plan, "gamma_rad")
    values = np.rad2deg(gamma_rad) if in_degrees else gamma_rad
    lower = None
    upper = None
    if envelope is not None and envelope.gamma_lower_rad is not None and envelope.gamma_upper_rad is not None:
        lower_band, upper_band = _band(envelope, s_m, "gamma_lower_rad", "gamma_upper_rad")
        if lower_band is not None and upper_band is not None:
            lower = np.rad2deg(lower_band) if in_degrees else lower_band
            upper = np.rad2deg(upper_band) if in_degrees else upper_band
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    return _plot(
        axis,
        s_m,
        values,
        title="Planned Flight-Path Angle",
        xlabel="Distance From Threshold [m]",
        ylabel="gamma [deg]" if in_degrees else "gamma [rad]",
        color="#9467bd",
        lower=lower,
        upper=upper,
    )


def plot_thrust_response(
    plan: Any,
    *,
    envelope: ConstraintEnvelope | None = None,
    ax: Axes | None = None,
) -> Axes:
    s_m = _series(plan, "s_m")
    thrust_n = _series(plan, "thrust_n")
    axis = ax if ax is not None else plt.subplots(1, 1, figsize=(6, 4))[1]
    lower = None
    upper = None
    if envelope is not None and envelope.thrust_lower_n is not None and envelope.thrust_upper_n is not None:
        lower, upper = _band(envelope, s_m, "thrust_lower_n", "thrust_upper_n")
    return _plot(
        axis,
        s_m,
        thrust_n,
        title="Planned Thrust",
        xlabel="Distance From Threshold [m]",
        ylabel="T [N]",
        color="#8c564b",
        lower=lower,
        upper=upper,
    )


def plot_constraint_envelope(
    envelope: ConstraintEnvelope,
    *,
    figsize: tuple[float, float] = (12.0, 4.5),
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _plot(
        axes[0],
        envelope.s_m,
        0.5 * (envelope.h_lower_m + envelope.h_upper_m),
        title="Altitude Envelope",
        xlabel="Distance From Threshold [m]",
        ylabel="h [m]",
        color="#2ca02c",
        lower=envelope.h_lower_m,
        upper=envelope.h_upper_m,
    )
    _plot(
        axes[1],
        envelope.s_m,
        0.5 * (envelope.cas_lower_mps + envelope.cas_upper_mps),
        title="CAS Envelope",
        xlabel="Distance From Threshold [m]",
        ylabel="v_cas [m/s]",
        color="#ff7f0e",
        lower=envelope.cas_lower_mps,
        upper=envelope.cas_upper_mps,
    )
    fig.suptitle("RNAV Constraint Envelope")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_longitudinal_plan(
    plan: Any,
    *,
    envelope: ConstraintEnvelope | None = None,
    figsize: tuple[float, float] = (12.0, 10.0),
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    plot_altitude_response(plan, envelope=envelope, ax=axes[0, 0])
    plot_cas_response(plan, envelope=envelope, ax=axes[0, 1])
    plot_gamma_response(plan, envelope=envelope, ax=axes[1, 0])
    plot_thrust_response(plan, envelope=envelope, ax=axes[1, 1])
    fig.suptitle("Authoritative RNAV Longitudinal Profile")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


__all__ = [
    "plot_altitude_response",
    "plot_cas_response",
    "plot_constraint_envelope",
    "plot_gamma_response",
    "plot_longitudinal_plan",
    "plot_tas_response",
    "plot_thrust_response",
]
