"""
Factory for prior construction.

The input is a prior profile from the prior YAML. The output is a resolved
prior configuration that can be passed directly to PyMCUnfolder.
The RL-based profiles use the OMpy right-hand convention,
    expected_on = x @ response_matrix + background_reference
    eta         = x @ gamma_resolution_matrix
where response_matrix = D @ G_g and gamma_resolution_matrix = G_g.
No clipped ON-OFF signal vector is used. If no background is included,
background_reference is the zero vector.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ..input_checks import require_finite_array, require_float, require_int
from .richardson_lucy import rl_reference
from .sigma_schedule import adaptive_sigma_from_rl


DEFAULT_RL_FLOOR = 1.0e-1
DEFAULT_SIGMA_MIN = 1.0
DEFAULT_SIGMA_MAX = 3.0
DEFAULT_C_REF = 100.0
DEFAULT_ALPHA = 1.0


class PriorBuilder:
    """Base class for prior-profile builders."""

    def build(
        self,
        response_matrix: np.ndarray,
        gamma_resolution_matrix: np.ndarray,
        n_on: np.ndarray,
        background_reference: np.ndarray,
        params: dict[str, Any],
    ) -> dict:
        raise NotImplementedError


_PRIOR_BUILDERS: dict[str, PriorBuilder] = {}


def register_prior(tag: str):
    """Register a prior builder under a YAML dist tag."""

    def wrapper(cls):
        _PRIOR_BUILDERS[tag] = cls()
        return cls

    return wrapper


def make_prior(
    profile: dict[str, Any],
    response_matrix: np.ndarray,
    gamma_resolution_matrix: np.ndarray,
    n_on: np.ndarray | None = None,
    background_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Resolve a prior profile into a PyMC-ready prior specification."""

    profile = copy.deepcopy(profile)

    if "dist" not in profile:
        raise KeyError("Prior profile must contain 'dist'.")

    dist_tag = str(profile["dist"]).strip().lower()
    params = profile.get("params", {}) or {}

    if not isinstance(params, dict):
        raise TypeError("Prior profile 'params' must be a mapping.")

    builder = _PRIOR_BUILDERS.get(dist_tag)

    if builder is None:
        if dist_tag in {"gamma_lognormal_mean", "lognormal"}:
            return profile

        raise ValueError(f"Unknown prior dist tag: {dist_tag!r}.")

    response_matrix = require_finite_array(
        response_matrix,
        "response_matrix",
        ndim=2,
        nonnegative=True,
    )
    gamma_resolution_matrix = require_finite_array(
        gamma_resolution_matrix,
        "gamma_resolution_matrix",
        ndim=2,
        nonnegative=True,
    )

    nparams, nobs = response_matrix.shape

    if gamma_resolution_matrix.shape != (nparams, nparams):
        raise ValueError(
            "gamma_resolution_matrix must have shape "
            f"({nparams}, {nparams}), got {gamma_resolution_matrix.shape}."
        )

    if n_on is None:
        raise ValueError(f"{dist_tag} requires n_on.")

    n_on = require_finite_array(
        n_on,
        "n_on",
        ndim=1,
        nonnegative=True,
    )

    if n_on.size != nobs:
        raise ValueError(f"n_on length must be {nobs}, got {n_on.size}.")

    if background_reference is None:
        background_reference = np.zeros_like(n_on, dtype=float)
    else:
        background_reference = require_finite_array(
            background_reference,
            "background_reference",
            ndim=1,
            nonnegative=True,
        )

    if background_reference.size != nobs:
        raise ValueError(
            "background_reference length must match n_on. "
            f"Got {background_reference.size}, expected {nobs}."
        )

    return builder.build(
        response_matrix=response_matrix,
        gamma_resolution_matrix=gamma_resolution_matrix,
        n_on=n_on,
        background_reference=background_reference,
        params=params,
    )


def _check_unknown_params(params: dict, allowed: set[str], dist_name: str) -> None:
    """Raise if a prior profile contains unsupported parameters."""

    unknown = sorted(set(params) - allowed)

    if unknown:
        raise ValueError(f"Unknown parameters for {dist_name}: {unknown}.")


def _positive_float(params: dict, name: str, default: float) -> float:
    """Read a finite positive floating-point parameter."""

    return require_float(
        params.get(name, default),
        name,
        minimum=0.0,
        minimum_inclusive=False,
    )


def _integer_or_auto(params: dict, name: str, default: str = "auto") -> int | str:
    """Read an integer parameter or the string 'auto'."""

    value = params.get(name, default)

    if isinstance(value, str):
        value = value.strip().lower()

        if value != "auto":
            raise ValueError(f"{name} must be an integer or 'auto'.")

        return "auto"

    return require_int(value, name, minimum=0)


def _eta_from_rl(
    x_rl: np.ndarray,
    gamma_resolution_matrix: np.ndarray,
) -> np.ndarray:
    """Return eta_rl = x_rl @ G_g."""

    eta_rl = x_rl @ gamma_resolution_matrix

    return require_finite_array(
        eta_rl,
        "eta_rl",
        ndim=1,
        nonnegative=True,
    )


def _rl_reference(
    n_on: np.ndarray,
    response_matrix: np.ndarray,
    gamma_resolution_matrix: np.ndarray,
    background_reference: np.ndarray,
    rl_iterations: int | str,
) -> tuple[np.ndarray, dict]:
    """Return the raw selected RL reference and metadata."""

    x_rl, rl_metadata = rl_reference(
        n_on=n_on,
        response_matrix=response_matrix,
        gamma_resolution_matrix=gamma_resolution_matrix,
        background_reference=background_reference,
        rl_iterations=rl_iterations,
    )

    x_rl = require_finite_array(
        x_rl,
        "x_rl",
        ndim=1,
        nonnegative=True,
    )

    return x_rl, rl_metadata


def _apply_rl_floor(
    x_rl: np.ndarray,
    rl_floor: float,
) -> tuple[np.ndarray, dict]:
    """Apply the RL stability floor used by RL-based empirical priors.

    The returned vector is the reference that enters the prior construction.
    It is used both as the mean-preserving prior center and as the input to the adaptive 
    sigma schedule. Keeping these two uses synchronized avoids asituation 
    where extremely small selected RL values are protected in the
    prior center but still treated as literal zero-scale inputs when computing
    the resolution-limited prior-width schedule.
    """

    x_rl = require_finite_array(
        x_rl,
        "x_rl",
        ndim=1,
        nonnegative=True,
    )
    rl_floor = require_float(
        rl_floor,
        "rl_floor",
        minimum=0.0,
        minimum_inclusive=False,
    )

    x_floored = np.maximum(x_rl, rl_floor)
    x_floored = require_finite_array(
        x_floored,
        "x_rl_floored",
        ndim=1,
        nonnegative=True,
    )

    floor_mask = x_rl < rl_floor
    metadata = {
        "rl_floor": float(rl_floor),
        "rl_floor_applied": bool(np.any(floor_mask)),
        "rl_floor_fraction": float(np.mean(floor_mask)),
        "rl_raw_min": float(np.min(x_rl)),
        "rl_raw_max": float(np.max(x_rl)),
        "rl_floored_min": float(np.min(x_floored)),
        "rl_floored_max": float(np.max(x_floored)),
    }

    return x_floored, metadata


@register_prior("gamma_lognormal_mean_rl")
class GammaLogNormalMeanRL(PriorBuilder):
    """Main RL-centered Gamma--Lognormal mean prior."""

    def build(
        self,
        response_matrix: np.ndarray,
        gamma_resolution_matrix: np.ndarray,
        n_on: np.ndarray,
        background_reference: np.ndarray,
        params: dict[str, Any],
    ) -> dict:
        allowed = {
            "alpha",
            "sigma_min",
            "sigma_max",
            "c_ref",
            "rl_iterations",
            "rl_floor",
        }
        _check_unknown_params(params, allowed, "gamma_lognormal_mean_rl")

        alpha = _positive_float(params, "alpha", DEFAULT_ALPHA)

        sigma_min = _positive_float(params, "sigma_min", DEFAULT_SIGMA_MIN)
        sigma_max = _positive_float(params, "sigma_max", DEFAULT_SIGMA_MAX)
        c_ref = _positive_float(params, "c_ref", DEFAULT_C_REF)

        rl_iterations = _integer_or_auto(params, "rl_iterations", default="auto")
        rl_floor = _positive_float(params, "rl_floor", DEFAULT_RL_FLOOR)
        x_rl, rl_metadata = _rl_reference(
            n_on=n_on,
            response_matrix=response_matrix,
            gamma_resolution_matrix=gamma_resolution_matrix,
            background_reference=background_reference,
            rl_iterations=rl_iterations,
        )

        x_rl_floored, floor_metadata = _apply_rl_floor(x_rl, rl_floor)

        sigma, sigma_shape, eta_rl, sigma_metadata = adaptive_sigma_from_rl(
            x_rl=x_rl_floored,
            gamma_resolution_matrix=gamma_resolution_matrix,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            c_ref=c_ref,
        )

        return {
            "dist": "gamma_lognormal_mean",
            "params": {
                "alpha": float(alpha),
                "mu_center": x_rl_floored.tolist(),
                "sigma": sigma.tolist(),
            },
            "x_rl": x_rl_floored.tolist(),
            "eta_rl": eta_rl.tolist(),
            "sigma": sigma.tolist(),
            "sigma_shape": sigma_shape.tolist(),
            "meta": {
                "source": "rl",
                "rl_background_model": "fixed_reference_in_on_likelihood",
                **rl_metadata,
                **floor_metadata,
                **sigma_metadata,
            },
        }


@register_prior("gamma_lognormal_mean_constant_center")
class GammaLogNormalMeanConstantCenter(PriorBuilder):
    """Robustness prior with a constant prior center.

    The stability-floored RL reference is still computed for diagnostics
    and for the adaptive sigma schedule, but the prior center is replaced by the
    mean floored RL scale over the active Eg window.
    """
    
    def build(
        self,
        response_matrix: np.ndarray,
        gamma_resolution_matrix: np.ndarray,
        n_on: np.ndarray,
        background_reference: np.ndarray,
        params: dict[str, Any],
    ) -> dict:
        allowed = {
            "alpha",
            "sigma_min",
            "sigma_max",
            "c_ref",
            "rl_iterations",
            "rl_floor",
        }
        _check_unknown_params(params, allowed, "gamma_lognormal_mean_constant_center")

        alpha = _positive_float(params, "alpha", DEFAULT_ALPHA)

        sigma_min = _positive_float(params, "sigma_min", DEFAULT_SIGMA_MIN)
        sigma_max = _positive_float(params, "sigma_max", DEFAULT_SIGMA_MAX)
        c_ref = _positive_float(params, "c_ref", DEFAULT_C_REF)

        rl_iterations = _integer_or_auto(params, "rl_iterations", default="auto")
        rl_floor = _positive_float(params, "rl_floor", DEFAULT_RL_FLOOR)
        x_rl, rl_metadata = _rl_reference(
            n_on=n_on,
            response_matrix=response_matrix,
            gamma_resolution_matrix=gamma_resolution_matrix,
            background_reference=background_reference,
            rl_iterations=rl_iterations,
        )

        x_center_rl, floor_metadata = _apply_rl_floor(x_rl, rl_floor)
        center_value = float(np.mean(x_center_rl))
        x_center = np.full_like(x_rl, center_value, dtype=float)

        sigma, sigma_shape, eta_rl, sigma_metadata = adaptive_sigma_from_rl(
            x_rl=x_center_rl,
            gamma_resolution_matrix=gamma_resolution_matrix,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            c_ref=c_ref,
        )

        return {
            "dist": "gamma_lognormal_mean",
            "params": {
                "alpha": float(alpha),
                "mu_center": x_center.tolist(),
                "sigma": sigma.tolist(),
            },
            "x_rl": x_center_rl.tolist(),
            "eta_rl": eta_rl.tolist(),
            "sigma": sigma.tolist(),
            "sigma_shape": sigma_shape.tolist(),
            "meta": {
                "source": "constant_center_from_rl_mean",
                "center_value": center_value,
                "rl_background_model": "fixed_reference_in_on_likelihood",
                **rl_metadata,
                **floor_metadata,
                **sigma_metadata,
            },
        }


@register_prior("gamma_lognormal_mean_hyper")
class GammaLogNormalMeanHyper(PriorBuilder):
    """Fully Bayesian hyperprior version.

    The RL reference is computed only for diagnostics and result-file
    consistency. It is not used to center the hyperprior.
    """

    def build(
        self,
        response_matrix: np.ndarray,
        gamma_resolution_matrix: np.ndarray,
        n_on: np.ndarray,
        background_reference: np.ndarray,
        params: dict[str, Any],
    ) -> dict:
        allowed = {
            "alpha",
            "theta_mu",
            "theta_sigma",
            "tau_sigma",
            "rl_iterations",
        }
        _check_unknown_params(params, allowed, "gamma_lognormal_mean_hyper")

        alpha = _positive_float(params, "alpha", 1.0)
        theta_mu = require_float(params.get("theta_mu", 0.0), "theta_mu")
        theta_sigma = _positive_float(params, "theta_sigma", 5.0)
        tau_sigma = _positive_float(params, "tau_sigma", 2.5)

        rl_iterations = _integer_or_auto(params, "rl_iterations", default="auto")
        x_rl, rl_metadata = _rl_reference(
            n_on=n_on,
            response_matrix=response_matrix,
            gamma_resolution_matrix=gamma_resolution_matrix,
            background_reference=background_reference,
            rl_iterations=rl_iterations,
        )

        eta_rl = _eta_from_rl(x_rl, gamma_resolution_matrix)

        return {
            "dist": "gamma_lognormal_mean_hyper",
            "params": {
                "alpha": float(alpha),
                "theta_mu": float(theta_mu),
                "theta_sigma": float(theta_sigma),
                "tau_sigma": float(tau_sigma),
            },
            "x_rl": x_rl.tolist(),
            "eta_rl": eta_rl.tolist(),
            "meta": {
                "source": "hyperprior",
                "rl_used_for": "diagnostics",
                "rl_background_model": "fixed_reference_in_on_likelihood",
                **rl_metadata,
            },
        }


@register_prior("lognormal_rl_matched_cv")
class LogNormalRLMatchedCV(PriorBuilder):
    """Lognormal robustness prior matched to the main prior's marginal CV."""

    def build(
        self,
        response_matrix: np.ndarray,
        gamma_resolution_matrix: np.ndarray,
        n_on: np.ndarray,
        background_reference: np.ndarray,
        params: dict[str, Any],
    ) -> dict:
        allowed = {
            "alpha_reference",
            "sigma_min",
            "sigma_max",
            "c_ref",
            "rl_iterations",
            "rl_floor",
        }
        _check_unknown_params(params, allowed, "lognormal_rl_matched_cv")

        alpha_reference = _positive_float(params, "alpha_reference", DEFAULT_ALPHA)

        sigma_min = _positive_float(params, "sigma_min", DEFAULT_SIGMA_MIN)
        sigma_max = _positive_float(params, "sigma_max", DEFAULT_SIGMA_MAX)
        c_ref = _positive_float(params, "c_ref", DEFAULT_C_REF)

        rl_iterations = _integer_or_auto(params, "rl_iterations", default="auto")
        rl_floor = _positive_float(params, "rl_floor", DEFAULT_RL_FLOOR)
        x_rl, rl_metadata = _rl_reference(
            n_on=n_on,
            response_matrix=response_matrix,
            gamma_resolution_matrix=gamma_resolution_matrix,
            background_reference=background_reference,
            rl_iterations=rl_iterations,
        )

        x_center, floor_metadata = _apply_rl_floor(x_rl, rl_floor)

        sigma_gamma, sigma_shape, eta_rl, sigma_metadata = adaptive_sigma_from_rl(
            x_rl=x_center,
            gamma_resolution_matrix=gamma_resolution_matrix,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            c_ref=c_ref,
        )

        sigma_lognormal = np.sqrt(
            sigma_gamma**2 + np.log(1.0 + 1.0 / alpha_reference)
        )

        mu_log = np.log(x_center) - 0.5 * sigma_lognormal**2

        return {
            "dist": "lognormal",
            "params": {
                "mu_log": mu_log.tolist(),
                "sigma": sigma_lognormal.tolist(),
            },
            "x_rl": x_center.tolist(),
            "eta_rl": eta_rl.tolist(),
            "sigma": sigma_lognormal.tolist(),
            "sigma_gamma_reference": sigma_gamma.tolist(),
            "sigma_shape": sigma_shape.tolist(),
            "meta": {
                "source": "rl_matched_cv",
                "alpha_reference": float(alpha_reference),
                "rl_background_model": "fixed_reference_in_on_likelihood",
                **rl_metadata,
                **floor_metadata,
                **sigma_metadata,
            },
        }
