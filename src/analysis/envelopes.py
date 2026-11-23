
"""
Create simultaneous bands from adaptive maximum-deviation envelopes.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _draw_matrix(draws: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    if "sample" not in draws.dims or "Eg" not in draws.dims:
        raise ValueError("draws must have sample and Eg dimensions")

    axis = np.asarray(draws["Eg"].values, dtype=float)
    values = np.asarray(draws.transpose("sample", "Eg").values, dtype=float)

    if values.ndim != 2 or values.shape[0] < 4 or not np.all(np.isfinite(values)):
        raise ValueError("draws must contain at least four finite curves")

    return axis, values


def _order_statistic(values: np.ndarray, mass: float) -> float:
    k = int(np.ceil((len(values) + 1) * mass))
    if k > len(values):
        raise ValueError("Insufficient samples for the requested coverage")
    return float(np.partition(values, k - 1)[k - 1])


def adaptive_envelope(
    draws: xr.DataArray,
    mass: float = 0.95,
    trim: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if not 0.0 < mass < 1.0:
        raise ValueError("mass must lie in (0,1)")
    if not 0.0 <= trim < 1.0:
        raise ValueError("trim must lie in [0,1)")

    axis, values = _draw_matrix(draws)

    n_train = len(values) // 2
    train = values[:n_train]
    calibration = values[n_train:]

    center = train.mean(axis=0)
    deviations = np.abs(train - center)

    if trim > 0.0:
        scores = deviations.max(axis=1)
        threshold = _order_statistic(scores, 1.0 - trim)
        deviations = deviations[scores <= threshold]

    shape = np.maximum(deviations.max(axis=0), 1.0e-10)

    scores = np.max(np.abs(calibration - center) / shape, axis=1)

    cutoff = _order_statistic(scores, mass)

    lower = center - cutoff * shape
    upper = center + cutoff * shape

    if np.all(values >= 0.0):
        lower = np.maximum(lower, 0.0)

    mean = values.mean(axis=0)

    return axis, mean, lower, upper




