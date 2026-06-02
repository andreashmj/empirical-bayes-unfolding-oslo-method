"""
Adaptive prior-width schedule for the RL-centered prior.

The prior width is controlled in the resolution-limited space,
    eta_rl = x_rl @ G_g.
For RL-based empirical priors, 'x_rl' should be the same RL reference that
enters the prior construction after any stability floor has been applied.
The schedule separates two effects:

1. The relative eta_rl shape determines where the empirical reference contains
   spectral structure.
2. The absolute local eta_rl count level determines how strongly that shape
   information is allowed to affect the prior width.

The final schedule is
    sigma_shape_j = sigma_min + (sigma_max - sigma_min) / (1 + eta_rl_j / eta_mean)
    activation_j = eta_rl_j / (eta_rl_j + c_ref)
    sigma_j = (1 - activation_j) sigma_max + activation_j sigma_shape_j.

Thus c_ref is the local resolution-limited count level where the RL-shape
information is half activated.
"""

from __future__ import annotations
import numpy as np
from ..input_checks import require_finite_array, require_float

def adaptive_sigma_from_rl(
    x_rl: np.ndarray,
    gamma_resolution_matrix: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    c_ref: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return the final sigma schedule, shape schedule, eta_rl, and metadata."""

    x_rl = require_finite_array(
        x_rl,
        "x_rl",
        ndim=1,
        nonnegative=True,
    )

    gamma_resolution_matrix = require_finite_array(
        gamma_resolution_matrix,
        "gamma_resolution_matrix",
        ndim=2,
        nonnegative=True,
    )

    if gamma_resolution_matrix.shape != (x_rl.size, x_rl.size):
        raise ValueError(
            "gamma_resolution_matrix must have shape "
            f"({x_rl.size}, {x_rl.size}), got {gamma_resolution_matrix.shape}."
        )

    sigma_min = require_float(
        sigma_min,
        "sigma_min",
        minimum=0.0,
        minimum_inclusive=False,
    )
    sigma_max = require_float(
        sigma_max,
        "sigma_max",
        minimum=sigma_min,
    )
    c_ref = require_float(
        c_ref,
        "c_ref",
        minimum=0.0,
        minimum_inclusive=False,
    )

    eta_rl = x_rl @ gamma_resolution_matrix
    eta_rl = require_finite_array(
        eta_rl,
        "eta_rl",
        ndim=1,
        nonnegative=True,
    )

    eta_total = float(np.sum(eta_rl))
    eta_mean = float(np.mean(eta_rl))

    if eta_total <= 0.0 or eta_mean <= 0.0:
        raise ValueError("eta_rl must have positive total and mean intensity.")

    shape_ratio = eta_rl / eta_mean

    sigma_shape = ( sigma_min + (sigma_max - sigma_min) / (1.0 + shape_ratio) )

    activation = eta_rl / (eta_rl + c_ref)

    sigma = ( (1.0 - activation) * sigma_max + activation * sigma_shape )

    sigma = require_finite_array(
        sigma,
        "sigma",
        ndim=1,
        nonnegative=False,
    )
    sigma_shape = require_finite_array(
        sigma_shape,
        "sigma_shape",
        ndim=1,
        nonnegative=False,
    )

    if np.any(sigma <= 0.0):
        raise ValueError("sigma must contain only positive values.")
    if np.any(sigma_shape <= 0.0):
        raise ValueError("sigma_shape must contain only positive values.")

    metadata = {
        "sigma_schedule": "eta_shape_local_count_activation",
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "c_ref": c_ref,
        "eta_total": eta_total,
        "eta_mean": eta_mean,
        "eta_min": float(np.min(eta_rl)),
        "eta_max": float(np.max(eta_rl)),
        "activation_min": float(np.min(activation)),
        "activation_mean": float(np.mean(activation)),
        "activation_max": float(np.max(activation)),
        "sigma_shape_min": float(np.min(sigma_shape)),
        "sigma_shape_mean": float(np.mean(sigma_shape)),
        "sigma_shape_max": float(np.max(sigma_shape)),
        "sigma_min_realized": float(np.min(sigma)),
        "sigma_mean_realized": float(np.mean(sigma)),
        "sigma_max_realized": float(np.max(sigma)),
    }

    return sigma, sigma_shape, eta_rl, metadata
