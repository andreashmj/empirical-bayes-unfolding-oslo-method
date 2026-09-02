"""
Global rank-envelope bands. Whole spectral curves are ranked by the extreme-rank-length ordering. The mean of samples are returned as the center of the curve.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..input_checks import require_float

def midranks(values: np.ndarray, atol: float = 0.0) -> np.ndarray:
    """
    Computes the midranks for a one-dimensional array.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    atol = require_float(atol, "atol", minimum=0.0)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=float)

    start = 0
    while start < values.size:
        stop = start + 1

        while stop < values.size and abs(sorted_values[stop] - sorted_values[start]) <= atol:
            stop += 1

        sorted_ranks[start:stop] = 0.5 * ((start + 1) + stop)
        start = stop

    ranks = np.empty(values.size, dtype=float)
    ranks[order] = sorted_ranks

    return ranks

def _draw_matrix(draws: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Different checks before returning the Eg axis and the matrix of draws on this axis.
    """

    if "sample" not in draws.dims or "Eg" not in draws.dims:
        raise ValueError("draws must have dimensions sample and Eg.")

    eg_axis = np.asarray(draws["Eg"].values, dtype=float)
    matrix = np.asarray(draws.transpose("sample", "Eg").values, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"draws must be two-dimensional, got shape {matrix.shape}.")
    if matrix.shape[0] < 2:
        raise ValueError("At least two samples are required.")
    if matrix.shape[1] != eg_axis.size:
        raise ValueError("Eg coordinate length does not match draw matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("draws contains non-finite values.")
    return eg_axis, matrix


def _least_extreme_indices(two_sided_ranks: np.ndarray, n_keep: int) -> np.ndarray:
    """
    Using ERL ordering to return least extreme curve indices.
    """
    sorted_ranks = np.sort(two_sided_ranks, axis=1)

    # np.lexsort uses the last key as primary. Reversing the columns gives
    # lexicographic ordering by smallest rank, then next smallest rank, etc.
    keys = tuple(
        sorted_ranks[:, column]
        for column in range(sorted_ranks.shape[1] - 1, -1, -1)
    )
    order = np.lexsort(keys)

    return order[-n_keep:]


def global_rank_envelope(
    draws: xr.DataArray,
    mass: float = 0.95,
    atol_ties: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The global rank envelopes are given by the 
    Eg, mean curve, lower band, and upper band.
    
    """

    mass = require_float(
        mass,
        "mass",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=True,
    )
    atol_ties = require_float(atol_ties, "atol_ties", minimum=0.0)

    eg_axis, matrix = _draw_matrix(draws)
    n_samples, n_eg = matrix.shape

    center = np.mean(matrix, axis=0)

    two_sided_ranks = np.empty((n_samples, n_eg), dtype=float)

    for eg_index in range(n_eg):
        ranks = midranks(matrix[:, eg_index], atol=atol_ties)
        two_sided_ranks[:, eg_index] = np.minimum(
            ranks,
            (n_samples + 1) - ranks,
        )

    n_keep = int(np.ceil(mass * n_samples))
    n_keep = max(1, min(n_samples, n_keep))

    keep = _least_extreme_indices(two_sided_ranks, n_keep=n_keep)
    kept_curves = matrix[keep, :]

    lower = np.min(kept_curves, axis=0)
    upper = np.max(kept_curves, axis=0)

    return eg_axis, center, lower, upper
