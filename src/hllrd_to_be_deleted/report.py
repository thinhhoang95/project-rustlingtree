from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hllrd_to_be_deleted.fit import HLLRDFitResult, event_summary


def write_event_summary_csv(path: Path, result: HLLRDFitResult) -> None:
    rows = event_summary(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event",
        "start",
        "end",
        "length",
        "peak_index",
        "active_count",
        "active_fraction",
        "raw_gain",
        "active_gain",
        "score",
        "explained_fraction",
        "local_simplifier_enabled",
        "local_simplifier_active_count",
        "local_simplifier_min_gain_per_point_m2",
        "local_simplifier_mean_points",
        "local_simplifier_median_points",
        "local_simplifier_max_points",
        "local_simplifier_initial_error_m2",
        "local_simplifier_residual_error_m2",
        "local_simplifier_reduced_error_m2",
        "local_simplifier_relative_reconstruction_loss",
        "local_simplifier_point_count_histogram",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_model_summary_json(path: Path, result: HLLRDFitResult) -> None:
    payload = {
        "K": len(result.events),
        "explained_fraction": result.explained_fraction,
        "sigma_hat": result.sigma_hat,
        "activation_energy_floor": result.activation_energy_floor,
        "activation_rms_floor": float(np.sqrt(max(0.0, result.activation_energy_floor))),
        "average_active_events_per_flight": _average_active_events_per_flight(result),
        "events": event_summary(result),
        "metadata": result.metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def plot_residual_energy(path: Path, X: np.ndarray, result: HLLRDFitResult | None = None) -> None:
    matrix = np.asarray(X, dtype=float)
    energy = np.mean(matrix * matrix, axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(matrix.shape[1]), energy, color="#1f77b4", linewidth=1.5, label="mean residual energy")
    if result is not None:
        for index, event in enumerate(result.events):
            ax.axvspan(event.start, event.end - 1, color="#ff7f0e", alpha=0.18)
            ax.text(
                (event.start + event.end - 1) / 2,
                float(np.max(energy)) if energy.size else 0.0,
                str(index),
                ha="center",
                va="top",
                fontsize=8,
            )
    ax.set_xlabel("station")
    ax.set_ylabel("mean squared normal residual (m^2)")
    ax.set_title("HLLRD residual energy and selected events")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reconstruction_heatmap(path: Path, result: HLLRDFitResult) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    images = [
        (result.reconstruction + result.residual, "centered residual matrix"),
        (result.reconstruction, "HLLRD reconstruction"),
        (result.residual, "final residual"),
    ]
    vmax = max(float(np.nanpercentile(np.abs(values), 98)) for values, _title in images)
    vmax = vmax if vmax > 0.0 else 1.0
    for ax, (values, title) in zip(axes, images, strict=True):
        im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_ylabel("flight")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    axes[-1].set_xlabel("station")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _average_active_events_per_flight(result: HLLRDFitResult) -> float:
    if not result.events:
        return 0.0
    active = np.zeros(result.residual.shape[0], dtype=int)
    for event in result.events:
        active += event.active_mask.astype(int)
    return float(np.mean(active))
