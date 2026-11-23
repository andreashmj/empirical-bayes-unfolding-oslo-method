"""
Construction of the priors used for Bayesian unfolding.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from .richardson_lucy import rl_reference
from .sigma_schedule import adaptive_sigma_from_rl


def prepare_rl(
    params: dict[str, Any],
    R: np.ndarray,
    G_g: np.ndarray,
    n_obs: np.ndarray,
    background_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    
    x_rl, rl_details = rl_reference(
        n_obs=n_obs,
        R=R,
        G_g=G_g,
        background_reference=background_reference,
        rl_iterations=params["rl_iterations"]
    )

    x_rl = np.maximum(x_rl, params["rl_floor"])
    
    sigma, sigma_shape, eta_rl = adaptive_sigma_from_rl(
        x_rl=x_rl,
        G_g=G_g,
        sigma_min=params["sigma_min"],
        sigma_max=params["sigma_max"],
        c_ref=params["c_ref"]
    )
        
    return x_rl, eta_rl, sigma, sigma_shape, rl_details


def gamma_lognormal_rl(
    params: dict[str, Any],
    R: np.ndarray,
    G_g: np.ndarray,
    n_obs: np.ndarray,
    background_reference: np.ndarray,
) -> dict[str, Any]:

    x_rl, eta_rl, sigma, sigma_shape, rl_details = prepare_rl(params, R, G_g, n_obs, background_reference)

    return {
        "dist": "gamma_lognormal_mean",
        "params": {"alpha": params["alpha"], "mu_center": x_rl, "sigma": sigma},
        "x_rl": x_rl,
        "eta_rl": eta_rl,
        "sigma": sigma,
        "sigma_shape": sigma_shape,
        "rl_details": rl_details
    }


def gamma_lognormal_constant_center(
    params: dict[str, Any],
    R: np.ndarray,
    G_g: np.ndarray,
    n_obs: np.ndarray,
    background_reference: np.ndarray,
) -> dict[str, Any]:

    x_rl, eta_rl, sigma, sigma_shape, rl_details = prepare_rl(params, R, G_g, n_obs, background_reference)
    center = np.full_like(x_rl, np.mean(x_rl))

    return {
        "dist": "gamma_lognormal_mean",
        "params": {"alpha": params["alpha"], "mu_center": center, "sigma": sigma},
        "x_rl": x_rl,
        "eta_rl": eta_rl,
        "sigma": sigma,
        "sigma_shape": sigma_shape,
        "rl_details": rl_details
    }


def lognormal_matched_cv(
    params: dict[str, Any],
    R: np.ndarray,
    G_g: np.ndarray,
    n_obs: np.ndarray,
    background_reference: np.ndarray,
) -> dict[str, Any]:
    
    x_rl, eta_rl, sigma_gamma, sigma_shape, rl_details = prepare_rl(params, R, G_g, n_obs, background_reference)
    sigma = np.sqrt(sigma_gamma**2 + np.log(1.0 + 1.0 / params["alpha_reference"]))
    mu_log = np.log(x_rl) - 0.5 * sigma**2

    return {
        "dist": "lognormal",
        "params": {"mu_log": mu_log, "sigma": sigma},
        "x_rl": x_rl,
        "eta_rl": eta_rl,
        "sigma": sigma,
        "sigma_shape": sigma_shape,
        "rl_details": rl_details
    }


def gamma_lognormal_hyper(params: dict[str, Any]) -> dict[str, Any]:
   
    return {
        "dist": "gamma_lognormal_mean_hyper",
        "params": {
            key: params[key]
            for key in ("alpha", "theta_mu", "theta_sigma", "tau_sigma")
        }
    }


def make_prior(
    settings: dict[str, Any],
    R: np.ndarray,
    G_g: np.ndarray,
    n_obs: np.ndarray,
    background_reference: np.ndarray,
) -> dict[str, Any]:
    
    dist = settings["dist"]
    params = settings["params"]

    if dist == "gamma_lognormal_mean_rl":
        return gamma_lognormal_rl(params, R, G_g, n_obs, background_reference)

    if dist == "gamma_lognormal_mean_constant_center":
        return gamma_lognormal_constant_center(
            params, R, G_g, n_obs, background_reference
        )

    if dist == "lognormal_rl_matched_cv":
        return lognormal_matched_cv(params, R, G_g, n_obs, background_reference)

    if dist == "gamma_lognormal_mean_hyper":
        return gamma_lognormal_hyper(params)

    if dist in ("gamma_lognormal_mean", "lognormal"):
        return settings

    raise ValueError(f"Unknown prior distribution: {dist}")
