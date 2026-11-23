
"""
Calculation of the adaptive prior width from the RL reference.
"""

from __future__ import annotations

import numpy as np


def adaptive_sigma_from_rl(
    x_rl: np.ndarray,
    G_g: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    c_ref: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    eta_rl = x_rl @ G_g
    eta_mean = float(np.mean(eta_rl))

    if eta_mean <= 0.0:
        raise ValueError("eta_rl must have positive mean intensity.")

    shape_ratio = eta_rl / eta_mean

    sigma_shape = (sigma_min + (sigma_max - sigma_min) / (1.0 + shape_ratio))

    activation = eta_rl / (eta_rl + c_ref)

    sigma = ((1.0 - activation) * sigma_max + activation * sigma_shape)

    return sigma, sigma_shape, eta_rl

