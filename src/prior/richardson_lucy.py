
"""
Calculation of the Richardson–Lucy reference or the empirical-Bayes prior.

"""

from __future__ import annotations

import numpy as np


epsilon = 1.0e-12

N_ITER_MAX = 500
ITERATION_WINDOW = 10
N_RESAMPLES = 50
RATIO_THRESHOLD = 2.0
MIN_CONSECUTIVE = 10
RESAMPLE_SEED = 123


def richardson_lucy_iterates(
    n_obs: np.ndarray,
    R: np.ndarray,
    background_reference: np.ndarray | None = None,
    n_iter: int = 30,
) -> np.ndarray:
    

    nparams = R.shape[0]

    if background_reference is None:
        background_reference = np.zeros(R.shape[1], dtype=float)

    signal_scale = float(np.sum(n_obs) - np.sum(background_reference))

    if signal_scale <= 0.0:
        signal_scale = epsilon * float(nparams)

    estimate = np.full(nparams, signal_scale / float(nparams), dtype=float)

    iterates = np.empty((n_iter + 1, nparams), dtype=float)
    iterates[0] = estimate

    for iteration in range(1, n_iter + 1):
        expected_on = estimate @ R + background_reference
        ratio = n_obs / (expected_on + epsilon)
        back_projection = ratio @ R.T
        estimate = estimate * back_projection
        iterates[iteration] = estimate

    return iterates


def eta_absolute_change(
    eta_iterates: np.ndarray,
    window: int = ITERATION_WINDOW,
) -> np.ndarray:
    

    change = np.full(eta_iterates.shape[0], np.nan, dtype=float)

    for iteration in range(window, eta_iterates.shape[0]):
        change[iteration] = np.linalg.norm(
            eta_iterates[iteration]
            - eta_iterates[iteration - window]
        )

    return change


def eta_resampling_noise_level(
    n_obs: np.ndarray,
    R: np.ndarray,
    G_g: np.ndarray,
    background_reference: np.ndarray | None = None,
    n_iter_max: int = N_ITER_MAX,
    n_resamples: int = N_RESAMPLES,
    rng_seed: int = RESAMPLE_SEED,
) -> np.ndarray:
    

    rng = np.random.default_rng(rng_seed)

    mean_eta = None
    m2_eta = None

    for resample_index in range(n_resamples):
        n_obs_resampled = rng.poisson(np.maximum(n_obs, 0.0)).astype(np.float64)

        x_iterates = richardson_lucy_iterates(
            n_obs=n_obs_resampled,
            R=R,
            background_reference=background_reference,
            n_iter=n_iter_max,
        )

        eta_iterates = x_iterates @ G_g

        if mean_eta is None:
            mean_eta = np.zeros_like( eta_iterates, dtype=np.float64)
            m2_eta = np.zeros_like(eta_iterates, dtype=np.float64)

        delta = eta_iterates - mean_eta
        mean_eta += delta / float(resample_index + 1)
        delta2 = eta_iterates - mean_eta
        m2_eta += delta * delta2

    eta_variance = m2_eta / float(n_resamples - 1)

    noise_level = np.sqrt(np.sum(eta_variance, axis=1))
    noise_level[~np.isfinite(noise_level)] = np.nan

    return noise_level


def automatic_iteration_from_eta_noise(
    n_obs: np.ndarray,
    R: np.ndarray,
    G_g: np.ndarray,
    background_reference: np.ndarray | None = None,
    n_iter_max: int = N_ITER_MAX,
    window: int = ITERATION_WINDOW,
    n_resamples: int = N_RESAMPLES,
    ratio_threshold: float = RATIO_THRESHOLD,
    min_consecutive: int = MIN_CONSECUTIVE,
    rng_seed: int = RESAMPLE_SEED,
) -> tuple[int, np.ndarray, dict[str, float | int]]:
   

    x_iterates = richardson_lucy_iterates(
        n_obs=n_obs,
        R=R,
        background_reference=background_reference,
        n_iter=n_iter_max,
    )

    eta_iterates = x_iterates @ G_g
    delta_eta = eta_absolute_change(eta_iterates, window=window)

    noise_level = eta_resampling_noise_level(
        n_obs=n_obs,
        R=R,
        G_g=G_g,
        background_reference=background_reference,
        n_iter_max=n_iter_max,
        n_resamples=n_resamples,
        rng_seed=rng_seed,
    )

    ratio = delta_eta / (noise_level + epsilon)
    ratio[~np.isfinite(ratio)] = np.nan

    iteration = None
    stop = ratio.size - min_consecutive + 1

    for candidate in range(window, stop):
        block = ratio[candidate : candidate + min_consecutive]

        if np.all(np.isfinite(block)) and np.all(
            block <= ratio_threshold
        ):
            iteration = candidate
            break

    if iteration is None:
        iteration = n_iter_max

    rl_details = {
        "rl_iteration": int(iteration),
        "rl_delta_eta": float(delta_eta[iteration]),
        "rl_noise_level": float(noise_level[iteration]),
        "rl_change_noise_ratio": float(ratio[iteration]),
    }

    return int(iteration), x_iterates, rl_details


def rl_reference(
    n_obs: np.ndarray,
    R: np.ndarray,
    G_g: np.ndarray,
    background_reference: np.ndarray | None = None,
    rl_iterations: int | str = "auto",
) -> tuple[np.ndarray, dict[str, float | int]]:

    if isinstance(rl_iterations, str):
        if rl_iterations.strip().lower() != "auto":
            raise ValueError("rl_iterations must be an integer or auto.")

        iteration, x_iterates, rl_details = (
            automatic_iteration_from_eta_noise(
                n_obs=n_obs,
                R=R,
                G_g=G_g,
                background_reference=background_reference,
            )
        )

    else:
        iteration = rl_iterations

        x_iterates = richardson_lucy_iterates(
            n_obs=n_obs,
            R=R,
            background_reference=background_reference,
            n_iter=iteration,
        )

        rl_details = {
            "rl_iteration": int(iteration),
            "rl_delta_eta": np.nan,
            "rl_noise_level": np.nan,
            "rl_change_noise_ratio": np.nan,
        }

    return x_iterates[iteration], rl_details

