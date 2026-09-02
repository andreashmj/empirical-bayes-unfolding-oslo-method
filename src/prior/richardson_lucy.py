"""
Richardson--Lucy reference used for the empirical prior. 
The ON counts are used directly and a fixed background term is used in the forward model
    expected_on = x @ response_matrix + background_reference
"""

from __future__ import annotations

import numpy as np
from ..input_checks import require_finite_array, require_float, require_int


EPS = 1.0e-12

N_ITER_MAX = 500
ITERATION_WINDOW = 10
N_RESAMPLES = 50
RATIO_THRESHOLD = 2.0
MIN_CONSECUTIVE = 10
RESAMPLE_SEED = 123


def _response_matrix(response_matrix: np.ndarray) -> np.ndarray:
    response_matrix = require_finite_array(
        response_matrix,
        "response_matrix",
        ndim=2,
        nonnegative=True)

    if response_matrix.shape[0] < 1 or response_matrix.shape[1] < 1:
        raise ValueError("response_matrix must be non-empty.")

    return response_matrix


def _gamma_resolution_matrix(
    gamma_resolution_matrix: np.ndarray,
    nparams: int,
) -> np.ndarray:
    gamma_resolution_matrix = require_finite_array(
        gamma_resolution_matrix,
        "gamma_resolution_matrix",
        ndim=2,
        nonnegative=True)

    if gamma_resolution_matrix.shape != (nparams, nparams):
        raise ValueError(
            "gamma_resolution_matrix must have shape "
            f"({nparams}, {nparams}), got {gamma_resolution_matrix.shape}."
        )
    return gamma_resolution_matrix


def _on_counts(values: np.ndarray, n_out: int) -> np.ndarray:

    values = require_finite_array(
        values,
        "n_on",
        ndim=1,
        nonnegative=True,
    )

    if values.size != n_out:
        raise ValueError(f"n_on length must be {n_out}, got {values.size}.")

    return values


def _background_reference(values: np.ndarray | None, n_out: int) -> np.ndarray:
    if values is None:
        return np.zeros(n_out, dtype=float)

    values = require_finite_array(
        values,
        "background_reference",
        ndim=1,
        nonnegative=True,
    )

    if values.size != n_out:
        raise ValueError(
            f"background_reference length must be {n_out}, got {values.size}." )

    return values


def _initial_signal_scale(
    n_on: np.ndarray,
    background_reference: np.ndarray,
    nparams: int,
) -> float:
    signal_scale = float(np.sum(n_on) - np.sum(background_reference))

    if signal_scale <= 0.0:
        signal_scale = EPS * float(nparams)

    return signal_scale


def richardson_lucy_iterates(
    n_on: np.ndarray,
    response_matrix: np.ndarray,
    background_reference: np.ndarray | None = None,
    n_iter: int = 30,
) -> np.ndarray:
    """
    Run background-aware RL iterations and return the iterates.
    """

    n_iter = require_int(n_iter, "n_iter", minimum=0)

    response_matrix = _response_matrix(response_matrix)
    nparams, n_out = response_matrix.shape

    n_on = _on_counts(n_on, n_out)
    background_reference = _background_reference(background_reference, n_out)

    signal_scale = _initial_signal_scale(
        n_on=n_on,
        background_reference=background_reference,
        nparams=nparams,
    )

    estimate = np.full(nparams, signal_scale / float(nparams), dtype=float)
    iterates = np.empty((n_iter + 1, nparams), dtype=float)
    iterates[0] = estimate

    for iteration in range(1, n_iter + 1):
        expected_on = estimate @ response_matrix + background_reference
        ratio = n_on / (expected_on + EPS)
        back_projection = ratio @ response_matrix.T
        estimate = estimate * back_projection
        iterates[iteration] = estimate

    return iterates


def eta_relative_change(
    eta_iterates: np.ndarray,
    window: int = ITERATION_WINDOW,
) -> np.ndarray:
    """
    Return the relative change of eta_RL over a fixed iteration window."""

    eta_iterates = require_finite_array(
        eta_iterates,
        "eta_iterates",
        ndim=2,
        nonnegative=True,
    )

    window = require_int(window, "window", minimum=1)

    change = np.full(eta_iterates.shape[0], np.nan, dtype=float)

    for iteration in range(window, eta_iterates.shape[0]):
        numerator = np.linalg.norm(
            eta_iterates[iteration] - eta_iterates[iteration - window]
        )
        denominator = np.linalg.norm(eta_iterates[iteration]) + EPS
        change[iteration] = numerator / denominator

    return change


def eta_resampling_noise_level(
    n_on: np.ndarray,
    response_matrix: np.ndarray,
    gamma_resolution_matrix: np.ndarray,
    background_reference: np.ndarray | None = None,
    n_iter_max: int = N_ITER_MAX,
    n_resamples: int = N_RESAMPLES,
    rng_seed: int = RESAMPLE_SEED,
) -> np.ndarray:
    """
    Estimate the variability of eta_RL by Poisson resampling ON counts.
    """

    n_iter_max = require_int(n_iter_max, "n_iter_max", minimum=1)
    n_resamples = require_int(n_resamples, "n_resamples", minimum=2)
    rng_seed = require_int(rng_seed, "rng_seed", minimum=0)

    response_matrix = _response_matrix(response_matrix)
    nparams, n_out = response_matrix.shape

    gamma_resolution_matrix = _gamma_resolution_matrix(
        gamma_resolution_matrix,
        nparams=nparams,
    )

    n_on = _on_counts(n_on, n_out)
    background_reference = _background_reference(background_reference, n_out)
    rng = np.random.default_rng(rng_seed)
    mean_eta = None
    m2_eta = None

    for resample_index in range(n_resamples):
        n_on_resampled = rng.poisson(np.maximum(n_on, 0.0)).astype(np.float64)

        x_iterates = richardson_lucy_iterates(
            n_on=n_on_resampled,
            response_matrix=response_matrix,
            background_reference=background_reference,
            n_iter=n_iter_max,
        )

        eta_iterates = x_iterates @ gamma_resolution_matrix

        if mean_eta is None:
            mean_eta = np.zeros_like(eta_iterates, dtype=np.float64)
            m2_eta = np.zeros_like(eta_iterates, dtype=np.float64)

        delta = eta_iterates - mean_eta
        mean_eta += delta / float(resample_index + 1)
        delta2 = eta_iterates - mean_eta
        m2_eta += delta * delta2

    if mean_eta is None or m2_eta is None:
        raise RuntimeError("No resampling histories were generated.")

    eta_variance = m2_eta / float(n_resamples - 1)

    numerator = np.sqrt(np.sum(eta_variance, axis=1))
    denominator = np.linalg.norm(mean_eta, axis=1) + EPS
    noise_level = numerator / denominator
    noise_level[~np.isfinite(noise_level)] = np.nan

    return noise_level


def automatic_iteration_from_eta_noise(
    n_on: np.ndarray,
    response_matrix: np.ndarray,
    gamma_resolution_matrix: np.ndarray,
    background_reference: np.ndarray | None = None,
    n_iter_max: int = N_ITER_MAX,
    window: int = ITERATION_WINDOW,
    n_resamples: int = N_RESAMPLES,
    ratio_threshold: float = RATIO_THRESHOLD,
    min_consecutive: int = MIN_CONSECUTIVE,
    rng_seed: int = RESAMPLE_SEED,
) -> tuple[int, np.ndarray, dict]:
    """
    The iteration is the earliest t for which
        Delta_eta(t; window) / N_eta(t) <= ratio_threshold
    holds for min_consecutive iterations. If no such t is found,
    n_iter_max is used.
    """

    n_iter_max = require_int(n_iter_max, "n_iter_max", minimum=1)
    window = require_int(window, "window", minimum=1)
    n_resamples = require_int(n_resamples, "n_resamples", minimum=2)
    min_consecutive = require_int(min_consecutive, "min_consecutive", minimum=1)
    rng_seed = require_int(rng_seed, "rng_seed", minimum=0)
    ratio_threshold = require_float(
        ratio_threshold,
        "ratio_threshold",
        minimum=0.0,
        minimum_inclusive=False)

    x_iterates = richardson_lucy_iterates(
        n_on=n_on,
        response_matrix=response_matrix,
        background_reference=background_reference,
        n_iter=n_iter_max,
    )

    gamma_resolution_matrix = _gamma_resolution_matrix(
        gamma_resolution_matrix,
        nparams=x_iterates.shape[1],
    )

    eta_iterates = x_iterates@ gamma_resolution_matrix
    delta_eta = eta_relative_change(eta_iterates, window=window)

    noise_level = eta_resampling_noise_level(
        n_on=n_on,
        response_matrix=response_matrix,
        gamma_resolution_matrix=gamma_resolution_matrix,
        background_reference=background_reference,
        n_iter_max=n_iter_max,
        n_resamples=n_resamples,
        rng_seed=rng_seed,
    )

    ratio = delta_eta / (noise_level + EPS)
    ratio[~np.isfinite(ratio)] = np.nan

    iteration = None
    stop = ratio.size - min_consecutive + 1

    for candidate in range(window, stop):
        block = ratio[candidate : candidate + min_consecutive]

        if np.all(np.isfinite(block)) and np.all(block <= ratio_threshold):
            iteration = candidate
            break

    if iteration is None:
        iteration = n_iter_max
        status = "max_iter"
    else:
        status = "eta_noise_level"

    metadata = {
        "rl_iteration_rule": "eta_noise_level",
        "rl_iteration_status": status,
        "rl_iteration": int(iteration),
        "rl_delta_eta": float(delta_eta[iteration]),
        "rl_noise_level": float(noise_level[iteration]),
        "rl_change_noise_ratio": float(ratio[iteration]),
        "rl_n_iter_max": int(n_iter_max),
        "rl_iteration_window": int(window),
        "rl_n_resamples": int(n_resamples),
        "rl_ratio_threshold": float(ratio_threshold),
        "rl_min_consecutive": int(min_consecutive),
        "rl_resample_seed": int(rng_seed),
    }
    return int(iteration), x_iterates, metadata


def rl_reference(
    n_on: np.ndarray,
    response_matrix: np.ndarray,
    gamma_resolution_matrix: np.ndarray,
    background_reference: np.ndarray | None = None,
    rl_iterations: int | str = "auto",
) -> tuple[np.ndarray, dict]:
    """
    Return RL reference spectrum and metadata. 
    """

    if isinstance(rl_iterations, str):
        rl_iterations = rl_iterations.strip().lower()

        if rl_iterations != "auto":
            raise ValueError("rl_iterations must be an integer or auto.")

        iteration, x_iterates, metadata = automatic_iteration_from_eta_noise(
            n_on=n_on,
            response_matrix=response_matrix,
            gamma_resolution_matrix=gamma_resolution_matrix,
            background_reference=background_reference,
        )

    else:
        iteration = require_int(rl_iterations, "rl_iterations", minimum=0)

        x_iterates = richardson_lucy_iterates(
            n_on=n_on,
            response_matrix=response_matrix,
            background_reference=background_reference,
            n_iter=iteration,
        )

        metadata = {
            "rl_iteration_rule": "fixed",
            "rl_iteration_status": "fixed",
            "rl_iteration": int(iteration),
            "rl_delta_eta": np.nan,
            "rl_noise_level": np.nan,
            "rl_change_noise_ratio": np.nan }

    x_rl = x_iterates[iteration]
    x_rl = require_finite_array(
        x_rl,
        "x_rl",
        ndim=1,
        nonnegative=True,
    )

    metadata["rl_iterations_requested"] = (
        "auto" if isinstance(rl_iterations, str) else int(iteration)
    )

    return x_rl, metadata
