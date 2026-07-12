"""Small NumPy ownership helpers used at artifact boundaries."""

from __future__ import annotations

import numpy as np

from .errors import ArtifactValidationError


def readonly_float64(
    values: object,
    *,
    name: str,
    ndim: int = 1,
    finite: bool = True,
) -> np.ndarray:
    """Return an owned, contiguous, read-only ``float64`` array.

    Writeable inputs are copied so freezing an artifact never mutates ownership
    expectations of its caller. Already frozen contiguous arrays are shared.
    """

    source = np.asarray(values)
    if source.dtype == np.float64 and source.flags.c_contiguous and not source.flags.writeable:
        array = source
    else:
        array = np.array(values, dtype=np.float64, order="C", copy=True)
        array.setflags(write=False)
    if array.ndim != ndim:
        raise ArtifactValidationError(f"{name} must have {ndim} dimensions; got shape {array.shape}")
    if finite and not np.all(np.isfinite(array)):
        raise ArtifactValidationError(f"{name} contains non-finite values")
    return array


def readonly_int64(
    values: object,
    *,
    name: str,
    ndim: int = 1,
) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype == np.int64 and source.flags.c_contiguous and not source.flags.writeable:
        array = source
    else:
        array = np.array(values, dtype=np.int64, order="C", copy=True)
        array.setflags(write=False)
    if array.ndim != ndim:
        raise ArtifactValidationError(f"{name} must have {ndim} dimensions; got shape {array.shape}")
    return array


def validate_same_length(expected: int, **arrays: np.ndarray) -> None:
    for name, array in arrays.items():
        if len(array) != expected:
            raise ArtifactValidationError(f"{name} has length {len(array)}; expected {expected}")


def strictly_increasing(values: np.ndarray, *, atol: float = 0.0) -> bool:
    return len(values) < 2 or bool(np.all(np.diff(values) > atol))


def strictly_decreasing(values: np.ndarray, *, atol: float = 0.0) -> bool:
    return len(values) < 2 or bool(np.all(np.diff(values) < -atol))
