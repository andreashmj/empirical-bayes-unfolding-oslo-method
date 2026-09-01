"""
Input and array checks used in the source code. 
"""

from __future__ import annotations
from typing import Any
import numpy as np


def require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    return bool(value)


def require_int(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if minimum is not None and value < int(minimum):
        raise ValueError(f"{name} must be >= {int(minimum)}.")
    return value




def require_float(
    value: Any,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if minimum is not None:
        minimum = float(minimum)
        if minimum_inclusive and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}.")
        if not minimum_inclusive and value <= minimum:
            raise ValueError(f"{name} must be > {minimum}.")

    if maximum is not None:
        maximum = float(maximum)
        if maximum_inclusive and value > maximum:
            raise ValueError(f"{name} must be <= {maximum}.")
        if not maximum_inclusive and value >= maximum:
            raise ValueError(f"{name} must be < {maximum}.")
    return value


def require_finite_array(
    values: Any,
    name: str,
    ndim: int | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if ndim is not None and array.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={int(ndim)}, got {array.ndim}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} contains negative values.")
    return array
