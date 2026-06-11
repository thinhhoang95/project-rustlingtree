from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeriesSimplification:
    values: np.ndarray
    retained_indices: np.ndarray
    initial_error_m2: float
    residual_error_m2: float
    point_gains_m2: tuple[float, ...]

    @property
    def approximation_points(self) -> int:
        return max(0, int(self.retained_indices.size) - 2)

    @property
    def reduced_error_m2(self) -> float:
        return self.initial_error_m2 - self.residual_error_m2


@dataclass(frozen=True)
class LocalBlockSimplification:
    values: np.ndarray
    active_mask: np.ndarray
    approximation_points: np.ndarray
    diagnostics: dict[str, object]


def simplify_series_by_gain(
    y: np.ndarray,
    *,
    min_gain_per_point_m2: float,
    max_approximation_points: int,
) -> SeriesSimplification:
    """Simplify one local normal-deviation series by marginal SSE reduction."""
    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if values.size <= 2:
        retained = np.arange(values.size, dtype=int)
        return SeriesSimplification(
            values=values.copy(),
            retained_indices=retained,
            initial_error_m2=0.0,
            residual_error_m2=0.0,
            point_gains_m2=(),
        )

    retained_points = [0, int(values.size - 1)]
    residual_error, simplified = _piecewise_linear_fit(values, retained_points)
    initial_error = residual_error
    point_gains: list[float] = []
    max_points = max(0, int(max_approximation_points))
    gain_floor = max(0.0, float(min_gain_per_point_m2))

    while len(retained_points) - 2 < max_points:
        best_gain = -np.inf
        best_index: int | None = None
        best_error = residual_error
        best_values = simplified
        retained_set = set(retained_points)
        for index in range(1, int(values.size - 1)):
            if index in retained_set:
                continue
            candidate_error, candidate_values = _piecewise_linear_fit(values, [*retained_points, index])
            gain = residual_error - candidate_error
            if gain > best_gain:
                best_gain = gain
                best_index = index
                best_error = candidate_error
                best_values = candidate_values
        if best_index is None or best_gain < gain_floor:
            break
        retained_points.append(best_index)
        retained_points.sort()
        residual_error = best_error
        simplified = best_values
        point_gains.append(float(best_gain))

    return SeriesSimplification(
        values=simplified,
        retained_indices=np.asarray(retained_points, dtype=int),
        initial_error_m2=float(initial_error),
        residual_error_m2=float(residual_error),
        point_gains_m2=tuple(point_gains),
    )


def simplify_local_deviation_block(
    local: np.ndarray,
    *,
    active_mask: np.ndarray,
    min_gain_per_point_m2: float,
    max_approximation_points: int,
) -> LocalBlockSimplification:
    """Simplify active rows of one local deviation block and zero inactive rows."""
    block = np.asarray(local, dtype=float)
    if block.ndim != 2:
        raise ValueError("local must be a two-dimensional matrix")
    mask = np.asarray(active_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != block.shape[0]:
        raise ValueError("active_mask must be one-dimensional and match local rows")

    simplified = np.zeros_like(block)
    active_indices = np.flatnonzero(mask)
    point_counts = np.zeros(active_indices.size, dtype=int)
    total_initial_error = 0.0
    total_residual_error = 0.0
    total_reduced_error = 0.0
    all_point_gains: list[float] = []

    for output_index, row_index in enumerate(active_indices):
        result = simplify_series_by_gain(
            block[row_index],
            min_gain_per_point_m2=min_gain_per_point_m2,
            max_approximation_points=max_approximation_points,
        )
        simplified[row_index] = result.values
        point_counts[output_index] = result.approximation_points
        total_initial_error += result.initial_error_m2
        total_residual_error += result.residual_error_m2
        total_reduced_error += result.reduced_error_m2
        all_point_gains.extend(result.point_gains_m2)

    diagnostics = {
        "active_count": int(active_indices.size),
        "min_gain_per_point_m2": float(min_gain_per_point_m2),
        "max_approximation_points": int(max_approximation_points),
        "mean_approximation_points": float(np.mean(point_counts)) if point_counts.size else 0.0,
        "median_approximation_points": float(np.median(point_counts)) if point_counts.size else 0.0,
        "max_observed_approximation_points": int(np.max(point_counts)) if point_counts.size else 0,
        "point_count_histogram": _point_count_histogram(point_counts, max_approximation_points),
        "initial_error_m2": float(total_initial_error),
        "residual_error_m2": float(total_residual_error),
        "reduced_error_m2": float(total_reduced_error),
        "mean_point_gain_m2": float(np.mean(all_point_gains)) if all_point_gains else 0.0,
    }
    return LocalBlockSimplification(
        values=simplified,
        active_mask=mask.copy(),
        approximation_points=point_counts,
        diagnostics=diagnostics,
    )


def _piecewise_linear_fit(y: np.ndarray, retained_indices: list[int]) -> tuple[float, np.ndarray]:
    values = np.asarray(y, dtype=float)
    retained = sorted(set(int(index) for index in retained_indices))
    if retained[0] != 0 or retained[-1] != values.size - 1:
        raise ValueError("retained_indices must include both endpoints")

    fitted = np.empty_like(values, dtype=float)
    x = np.arange(values.size, dtype=float)
    for start, end in zip(retained[:-1], retained[1:], strict=True):
        if end <= start:
            fitted[start] = values[start]
            continue
        fraction = (x[start : end + 1] - float(start)) / float(end - start)
        fitted[start : end + 1] = (1.0 - fraction) * values[start] + fraction * values[end]
    residual = values - fitted
    return float(np.sum(residual * residual)), fitted


def _point_count_histogram(point_counts: np.ndarray, max_approximation_points: int) -> dict[str, int]:
    maximum = max(0, int(max_approximation_points))
    return {str(index): int(np.count_nonzero(point_counts == index)) for index in range(maximum + 1)}
