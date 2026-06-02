"""
PyMC unfolding model for one excitation-energy row.

The implementation follows the OMpy right-hand convention used throughout the
code base,
    eta = x @ G_g
    nu  = x @ D @ G_g
where x is the emitted spectrum, eta is the resolution-limited emitted spectrum,
and nu is the expected detected signal.
Returned draw arrays have shape
    (chain, draw, Eg)
"""

from __future__ import annotations

import time
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions import transforms as tr
from pytensor import config as pytensor_config

from .input_checks import (
    require_bool,
    require_finite_array,
    require_float,
    require_int,
)


FLOATX = "float64"
pytensor_config.floatX = FLOATX


def _extract_chain_draw_eg(data_array, name: str) -> np.ndarray:
    """Extract an xarray DataArray as an array with shape (chain, draw, Eg)."""

    dims = list(data_array.dims)

    if "chain" in dims and "draw" in dims:
        order = ["chain", "draw"] + [
            dim for dim in dims if dim not in ("chain", "draw")
        ]
        array = np.asarray(data_array.transpose(*order).values, dtype=np.float64)

    elif "draw" in dims:
        order = ["draw"] + [dim for dim in dims if dim != "draw"]
        array = np.asarray(data_array.transpose(*order).values, dtype=np.float64)
        array = array[None, ...]

    else:
        raise ValueError(
            f"{name}: expected chain/draw or draw dimension, got dims={dims}."
        )

    if array.ndim != 3:
        raise ValueError(
            f"{name}: expected extracted shape (chain, draw, Eg), got {array.shape}."
        )

    return array


def _finite_vector(values, nparams: int, name: str) -> np.ndarray:
    """Return a finite vector with the expected length."""

    vector = require_finite_array(values, name).reshape(-1)

    if vector.size != nparams:
        raise ValueError(f"{name}: expected length {nparams}, got {vector.size}.")

    return vector.astype(np.float64, copy=False)


def _positive_vector(values, nparams: int, name: str) -> np.ndarray:
    """Return a finite positive vector with the expected length."""

    vector = _finite_vector(values, nparams, name)

    if np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")

    return vector


def _nonnegative_vector(values, nparams: int, name: str) -> np.ndarray:
    """Return a finite non-negative vector with the expected length."""

    vector = _finite_vector(values, nparams, name)

    if np.any(vector < 0.0):
        raise ValueError(f"{name} must contain only non-negative values.")

    return vector


def _finite_matrix(values, name: str) -> np.ndarray:
    """Return a finite two-dimensional matrix."""

    return require_finite_array(values, name, ndim=2).astype(np.float64, copy=False)


class PyMCUnfolder:
    """PyMC implementation for one excitation-energy row."""

    def __init__(self, prior_config: dict, sample_config: dict, debug: bool = False):
        self.prior_config = prior_config
        self.sample_config = sample_config
        self.debug = bool(debug)

        self.transform_name = str(self.sample_config["transform"]).strip().lower()
        self.transform_kwargs = self._positive_transform_kwargs(self.transform_name)

        initialization_config = self.sample_config.get("initialization", {}) or {}
        self.rl_init = require_bool(
            initialization_config.get("rl_init", False),
            "initialization.rl_init",
        )

        self.initvals: dict[str, np.ndarray | float] = {}
        self.x = None

    @staticmethod
    def _positive_transform_kwargs(transform_name: str) -> dict:
        """Return PyMC keyword arguments for the positive log transform."""

        if transform_name != "log":
            raise ValueError("The implementation requires transform: log.")

        return {"default_transform": tr.log}

    def _init_kwargs(self, name: str, value) -> dict:
        """Return an initval keyword when RL initialization is enabled."""

        if not self.rl_init:
            return {}

        if isinstance(value, np.ndarray):
            init_value = np.asarray(value, dtype=np.float64)
        elif isinstance(value, (list, tuple)):
            init_value = np.asarray(value, dtype=np.float64)
        else:
            init_value = float(value)

        self.initvals[name] = init_value
        return {"initval": init_value}

    def _sampler_initvals(self) -> dict | None:
        """Return sampler initial values, or None when RL initialization is disabled."""

        if not self.rl_init or not self.initvals:
            return None

        out: dict[str, np.ndarray | float] = {}

        for name, value in self.initvals.items():
            if isinstance(value, np.ndarray):
                out[name] = np.array(value, dtype=np.float64, copy=True)
            else:
                out[name] = float(value)

        return out

    def _build_prior(self, nparams: int):
        """Build the emitted-spectrum prior in the active Eg window."""

        self.initvals = {}

        dist = str(self.prior_config["dist"]).strip().lower()
        params = self.prior_config.get("params", {}) or {}

        if dist == "gamma_lognormal_mean":
            self.x = self._build_gamma_lognormal_mean_prior(params, nparams)

        elif dist == "gamma_lognormal_mean_hyper":
            self.x = self._build_gamma_lognormal_mean_hyper_prior(params, nparams)

        elif dist == "lognormal":
            self.x = self._build_lognormal_prior(params, nparams)

        else:
            raise ValueError(
                "Resolved prior dist must be one of: "
                "gamma_lognormal_mean, gamma_lognormal_mean_hyper, lognormal."
            )

        return self.x

    def _build_gamma_lognormal_mean_prior(self, params: dict, nparams: int):
        """Build the Gamma-Lognormal mean prior."""

        alpha = require_float(
            params["alpha"],
            "prior.alpha",
            minimum=0.0,
            minimum_inclusive=False,
        )

        mu_center = _positive_vector(
            params["mu_center"],
            nparams,
            "prior.mu_center",
        )
        sigma = _positive_vector(
            params["sigma"],
            nparams,
            "prior.sigma",
        )

        # The latent mean m_j is log-normal. The shift -sigma_j^2/2 makes
        # E[m_j] equal to mu_center_j before the Gamma layer is applied.
        log_m_center = np.log(mu_center) - 0.5 * sigma**2

        # Start at m_j = mu_center_j when RL initialization is enabled.
        # Since log(m_j) = log_m_center_j + sigma_j z_mj, this corresponds to
        # z_mj = sigma_j / 2.
        z_m_init = 0.5 * sigma

        z_m = pm.Normal(
            "z_m",
            mu=0.0,
            sigma=1.0,
            shape=nparams,
            **self._init_kwargs("z_m", z_m_init),
        )

        log_m = pm.Deterministic(
            "log_m",
            pt.constant(log_m_center, dtype=FLOATX)
            + pt.constant(sigma, dtype=FLOATX) * z_m,
        )

        mean_local = pm.Deterministic("m", pt.exp(log_m))
        beta_x = alpha / mean_local

        x_kwargs = {}
        x_kwargs.update(self._init_kwargs("x", mu_center))
        x_kwargs.update(self.transform_kwargs)

        return pm.Gamma(
            "x",
            alpha=alpha,
            beta=beta_x,
            shape=nparams,
            **x_kwargs,
        )

    def _build_gamma_lognormal_mean_hyper_prior(self, params: dict, nparams: int):
        """Build the fully hyperparameterized Gamma-Lognormal mean prior."""

        alpha = require_float(
            params["alpha"],
            "prior.alpha",
            minimum=0.0,
            minimum_inclusive=False,
        )
        theta_mu = require_float(params["theta_mu"], "prior.theta_mu")
        theta_sigma = require_float(
            params["theta_sigma"],
            "prior.theta_sigma",
            minimum=0.0,
            minimum_inclusive=False,
        )
        tau_sigma = require_float(
            params["tau_sigma"],
            "prior.tau_sigma",
            minimum=0.0,
            minimum_inclusive=False,
        )

        theta = pm.Normal(
            "theta",
            mu=theta_mu,
            sigma=theta_sigma,
            **self._init_kwargs("theta", theta_mu),
        )

        tau_kwargs = {}
        tau_kwargs.update(self._init_kwargs("tau", tau_sigma))
        tau_kwargs.update(self.transform_kwargs)

        tau = pm.HalfNormal(
            "tau",
            sigma=tau_sigma,
            **tau_kwargs,
        )

        z = pm.Normal(
            "z",
            mu=0.0,
            sigma=1.0,
            shape=nparams,
            **self._init_kwargs("z", np.zeros(nparams, dtype=np.float64)),
        )

        log_m = pm.Deterministic("log_m", theta + tau * z)
        mean_local = pm.Deterministic("m", pt.exp(log_m))
        beta_x = alpha / mean_local

        x_init = np.full(nparams, np.exp(theta_mu), dtype=np.float64)

        x_kwargs = {}
        x_kwargs.update(self._init_kwargs("x", x_init))
        x_kwargs.update(self.transform_kwargs)

        return pm.Gamma(
            "x",
            alpha=alpha,
            beta=beta_x,
            shape=nparams,
            **x_kwargs,
        )

    def _build_lognormal_prior(self, params: dict, nparams: int):
        """Build a non-centered Lognormal prior for the emitted spectrum."""

        mu_log = _finite_vector(params["mu_log"], nparams, "prior.mu_log")
        sigma = _positive_vector(params["sigma"], nparams, "prior.sigma")

        z_x = pm.Normal(
            "z_x",
            mu=0.0,
            sigma=1.0,
            shape=nparams,
            **self._init_kwargs("z_x", np.zeros(nparams, dtype=np.float64)),
        )

        log_x = pm.Deterministic(
            "log_x",
            pt.constant(mu_log, dtype=FLOATX)
            + pt.constant(sigma, dtype=FLOATX) * z_x,
        )

        return pm.Deterministic("x", pt.exp(log_x))

    def _background_from_config(
        self,
        n_off_obs: np.ndarray | None,
        nparams: int,
        mode: str,
    ):
        """Build the configured background component for prior or posterior mode."""

        background_config = self.sample_config["background_model"]
        kind = str(background_config["kind"]).strip().lower()

        background_alpha = require_float(
            background_config["a0"],
            "background_model.a0",
            minimum=0.0,
            minimum_inclusive=False,
        )

        if kind == "latent_gamma":
            background_expected, background_b0, n_off_checked = (
                self._latent_gamma_background_prior(
                    n_off_obs=n_off_obs,
                    nparams=nparams,
                    background_alpha=background_alpha,
                )
            )

            if mode == "posterior":
                self._add_off_count_likelihood(
                    background_expected=background_expected,
                    n_off_obs=n_off_checked,
                )

            return background_expected, background_b0

        if kind == "fixed_postmean":
            return self._fixed_postmean_background(
                n_off_obs=n_off_obs,
                nparams=nparams,
                background_alpha=background_alpha,
            )

        raise ValueError(
            "background_model.kind must be either 'latent_gamma' or "
            "'fixed_postmean'."
        )

    @staticmethod
    def _background_b0_from_off_counts(
        n_off_obs: np.ndarray,
        background_alpha: float,
    ) -> float:
        """Return Gamma prior rate b0 from the OFF-count mean."""

        mean_off = float(np.mean(n_off_obs))

        if mean_off <= 0.0:
            raise ValueError(
                "Mean OFF count must be positive for the background model."
            )

        return background_alpha / mean_off

    def _latent_gamma_background_prior(
        self,
        n_off_obs: np.ndarray | None,
        nparams: int,
        background_alpha: float,
    ):
        """Build the empirical Gamma prior for the latent background expectation."""

        if n_off_obs is None:
            raise ValueError("background_model.kind='latent_gamma' requires n_off_obs.")

        n_off_obs = _nonnegative_vector(n_off_obs, nparams, "n_off_obs")
        background_b0 = self._background_b0_from_off_counts(
            n_off_obs,
            background_alpha,
        )

        background_init = (background_alpha + n_off_obs) / (background_b0 + 1.0)

        background_kwargs = {}
        background_kwargs.update(self._init_kwargs("bg_exp", background_init))
        background_kwargs.update(self.transform_kwargs)

        background_expected = pm.Gamma(
            "bg_exp",
            alpha=background_alpha,
            beta=background_b0,
            shape=nparams,
            **background_kwargs,
        )

        return background_expected, background_b0, n_off_obs

    @staticmethod
    def _add_off_count_likelihood(background_expected, n_off_obs: np.ndarray) -> None:
        """Condition the posterior background model on the observed OFF counts."""

        pm.Poisson("n_off_obs", mu=background_expected, observed=n_off_obs)

    def _fixed_postmean_background(
        self,
        n_off_obs: np.ndarray | None,
        nparams: int,
        background_alpha: float,
    ):
        """Fix the background to the Gamma-Poisson posterior mean from OFF counts."""

        if n_off_obs is None:
            raise ValueError("background_model.kind='fixed_postmean' requires n_off_obs.")

        n_off_obs = _nonnegative_vector(n_off_obs, nparams, "n_off_obs")
        background_b0 = self._background_b0_from_off_counts(
            n_off_obs,
            background_alpha,
        )

        background_fixed = (background_alpha + n_off_obs) / (background_b0 + 1.0)

        background_expected = pm.Deterministic(
            "bg_exp",
            pt.constant(background_fixed, dtype=FLOATX),
        )

        return background_expected, background_b0


    @staticmethod
    def _wanted_var_names(include_background: bool) -> list[str]:
        """Return variables that must be extracted from the InferenceData object."""

        names = ["x"]

        if include_background:
            names.append("bg_exp")

        return names

    def _sample_prior(self, model: pm.Model, var_names: list[str]) -> az.InferenceData:
        """Draw prior samples of the requested latent variables."""

        prior_config = self.sample_config["prior"]

        draws = require_int(prior_config["draws"], "prior.draws", minimum=1)
        random_seed = require_int(
            prior_config["random_seed"],
            "prior.random_seed",
            minimum=0,
        )

        idata = pm.sample_prior_predictive(
            draws=draws,
            random_seed=random_seed,
            var_names=var_names,
            model=model,
            return_inferencedata=True,
        )

        idata.attrs["run_mode"] = "prior"
        idata.attrs["prior_random_seed"] = random_seed
        idata.attrs["prior_sampling_method"] = "sample_prior_predictive"
        idata.attrs["var_names"] = ",".join(var_names)

        return idata

    def _sample_posterior(
        self,
        model: pm.Model,
        var_names: list[str],
    ) -> az.InferenceData:
        """Sample from the posterior using the configured NUTS backend."""

        nuts_config = self.sample_config["nuts"]
        nuts_sampler = str(nuts_config.get("nuts_sampler", "pymc")).strip().lower()

        draws = require_int(nuts_config["draws"], "nuts.draws", minimum=1)
        tune = require_int(nuts_config["tune"], "nuts.tune", minimum=0)
        chains = require_int(nuts_config["chains"], "nuts.chains", minimum=1)
        target_accept = require_float(
            nuts_config["target_accept"],
            "nuts.target_accept",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
        max_treedepth = require_int(
            nuts_config["max_treedepth"],
            "nuts.max_treedepth",
            minimum=1,
        )
        random_seed = require_int(
            nuts_config["random_seed"],
            "nuts.random_seed",
            minimum=0,
        )

        if self.debug:
            print(
                f"[posterior] sampler={nuts_sampler} draws={draws} tune={tune} "
                f"chains={chains} target_accept={target_accept} "
                f"max_treedepth={max_treedepth}"
            )

        start_time = time.perf_counter()

        if nuts_sampler == "pymc":
            idata = self._sample_pymc_nuts(
                model=model,
                var_names=var_names,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                max_treedepth=max_treedepth,
                random_seed=random_seed,
                nuts_config=nuts_config,
            )

        elif nuts_sampler in {"numpyro", "blackjax"}:
            idata = self._sample_jax_nuts(
                nuts_sampler=nuts_sampler,
                model=model,
                var_names=var_names,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                max_treedepth=max_treedepth,
                random_seed=random_seed,
                nuts_config=nuts_config,
            )

        elif nuts_sampler == "nutpie":
            idata = self._sample_nutpie_nuts(
                model=model,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                max_treedepth=max_treedepth,
                random_seed=random_seed,
                nuts_config=nuts_config,
            )

        else:
            raise ValueError(
                "nuts.nuts_sampler must be one of: pymc, numpyro, blackjax, nutpie."
            )

        sampling_time_s = float(time.perf_counter() - start_time)

        idata.attrs["run_mode"] = "posterior"
        idata.attrs["sampler_backend"] = nuts_sampler
        idata.attrs["draws"] = draws
        idata.attrs["tune"] = tune
        idata.attrs["chains"] = chains
        idata.attrs["target_accept"] = target_accept
        idata.attrs["max_treedepth"] = max_treedepth
        idata.attrs["nuts_random_seed"] = random_seed
        idata.attrs["sampling_time_s"] = sampling_time_s
        idata.attrs["var_names"] = ",".join(var_names)
        idata.attrs["rl_init"] = "true" if self.rl_init else "false"

        return idata

    def _sample_pymc_nuts(
        self,
        model: pm.Model,
        var_names: list[str],
        draws: int,
        tune: int,
        chains: int,
        target_accept: float,
        max_treedepth: int,
        random_seed: int,
        nuts_config: dict,
    ) -> az.InferenceData:
        """Sample with the standard PyMC NUTS backend."""

        pymc_config = nuts_config.get("pymc", {}) or {}

        cores = require_int(pymc_config.get("cores", chains), "nuts.pymc.cores", minimum=1)
        init = str(pymc_config.get("init", "jitter+adapt_diag"))
        dense_mass = require_bool(
            pymc_config.get("dense_mass", False),
            "nuts.pymc.dense_mass",
        )
        progressbar = require_bool(
            pymc_config.get("progressbar", True),
            "nuts.pymc.progressbar",
        )

        if dense_mass and init == "jitter+adapt_diag":
            init = "jitter+adapt_full"

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            init=init,
            initvals=self._sampler_initvals(),
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            random_seed=random_seed,
            progressbar=progressbar,
            compute_convergence_checks=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
            var_names=var_names,
            model=model,
        )

        idata.attrs["cores"] = cores
        idata.attrs["init"] = init
        idata.attrs["dense_mass"] = "true" if dense_mass else "false"

        return idata

    def _sample_jax_nuts(
        self,
        nuts_sampler: str,
        model: pm.Model,
        var_names: list[str],
        draws: int,
        tune: int,
        chains: int,
        target_accept: float,
        max_treedepth: int,
        random_seed: int,
        nuts_config: dict,
    ) -> az.InferenceData:
        """Sample with the NumPyro or BlackJAX NUTS backend."""

        jax_config = nuts_config.get("jax", {}) or {}

        platform = str(jax_config.get("platform", "auto")).strip().lower()
        chain_method = str(jax_config.get("chain_method", "parallel")).strip().lower()
        x64 = require_bool(jax_config.get("x64", True), "nuts.jax.x64")
        progressbar = require_bool(
            jax_config.get("progressbar", True),
            "nuts.jax.progressbar",
        )
        jitter = require_bool(jax_config.get("jitter", True), "nuts.jax.jitter")
        nuts_kwargs = dict(jax_config.get("nuts_kwargs", {}) or {})

        import jax

        jax.config.update("jax_enable_x64", x64)

        if platform not in {"", "auto"}:
            jax.config.update("jax_platform_name", platform)

        import pymc.sampling.jax as pmjax

        if nuts_sampler == "numpyro":
            if "max_tree_depth" in nuts_kwargs:
                raise ValueError(
                    "Set nuts.max_treedepth, not "
                    "nuts.jax.nuts_kwargs.max_tree_depth."
                )

            nuts_kwargs["max_tree_depth"] = max_treedepth

            idata = pmjax.sample_numpyro_nuts(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                initvals=self._sampler_initvals(),
                jitter=jitter,
                model=model,
                var_names=var_names,
                chain_method=chain_method,
                progressbar=progressbar,
                compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": False},
                nuts_kwargs=nuts_kwargs,
            )

        else:
            if "max_num_doublings" in nuts_kwargs:
                raise ValueError(
                    "Set nuts.max_treedepth, not "
                    "nuts.jax.nuts_kwargs.max_num_doublings."
                )

            nuts_kwargs["max_num_doublings"] = max_treedepth

            idata = pmjax.sample_blackjax_nuts(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                initvals=self._sampler_initvals(),
                jitter=jitter,
                model=model,
                var_names=var_names,
                chain_method=chain_method,
                progressbar=progressbar,
                compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": False},
                nuts_kwargs=nuts_kwargs,
            )

        idata.attrs["jax_platform"] = platform
        idata.attrs["jax_chain_method"] = chain_method
        idata.attrs["jax_x64"] = "true" if x64 else "false"
        idata.attrs["jax_jitter"] = "true" if jitter else "false"

        return idata

    def _sample_nutpie_nuts(
        self,
        model: pm.Model,
        draws: int,
        tune: int,
        chains: int,
        target_accept: float,
        max_treedepth: int,
        random_seed: int,
        nuts_config: dict,
    ) -> az.InferenceData:
        """Sample with the Nutpie NUTS backend."""

        nutpie_config = nuts_config.get("nutpie", {}) or {}

        compile_backend = str(
            nutpie_config.get("compile_backend", "numba")
        ).strip().lower()
        cores = require_int(
            nutpie_config.get("cores", chains),
            "nuts.nutpie.cores",
            minimum=1,
        )
        save_warmup = require_bool(
            nutpie_config.get("save_warmup", False),
            "nuts.nutpie.save_warmup",
        )
        store_divergences = require_bool(
            nutpie_config.get("store_divergences", False),
            "nuts.nutpie.store_divergences",
        )
        progress_bar = require_bool(
            nutpie_config.get("progressbar", True),
            "nuts.nutpie.progressbar",
        )
        use_initial_points = require_bool(
            nutpie_config.get("use_initial_points", False),
            "nuts.nutpie.use_initial_points",
        )

        import nutpie

        initial_points = self._sampler_initvals() if use_initial_points else None

        if compile_backend == "numba":
            compiled_model = nutpie.compile_pymc_model(
                model,
                backend="numba",
                initial_points=initial_points,
            )
        elif compile_backend == "jax":
            compiled_model = nutpie.compile_pymc_model(
                model,
                backend="jax",
                gradient_backend="jax",
                initial_points=initial_points,
            )
        else:
            raise ValueError("nuts.nutpie.compile_backend must be 'numba' or 'jax'.")

        sample_kwargs = dict(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            maxdepth=max_treedepth,
            seed=random_seed,
            save_warmup=save_warmup,
            store_divergences=store_divergences,
            progress_bar=progress_bar,
        )

        if "low_rank_modified_mass_matrix" in nutpie_config:
            sample_kwargs["low_rank_modified_mass_matrix"] = require_bool(
                nutpie_config["low_rank_modified_mass_matrix"],
                "nuts.nutpie.low_rank_modified_mass_matrix",
            )

        if "transform_adapt" in nutpie_config:
            sample_kwargs["transform_adapt"] = require_bool(
                nutpie_config["transform_adapt"],
                "nuts.nutpie.transform_adapt",
            )

        idata = nutpie.sample(compiled_model, **sample_kwargs)

        idata.attrs["nutpie_compile_backend"] = compile_backend
        idata.attrs["cores"] = cores
        idata.attrs["nutpie_use_initial_points"] = (
            "true" if use_initial_points else "false"
        )

        return idata

    def unfold(
        self,
        n_obs: np.ndarray,
        n_off_obs: np.ndarray | None,
        D: np.ndarray,
        G_g: np.ndarray,
        mode: str,
        include_background: bool,
        output_trace_path: str | Path | None,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Run prior or posterior sampling for one excitation-energy row."""

        n_obs = require_finite_array(
            n_obs,
            "n_obs",
            ndim=1,
            nonnegative=True,
        ).reshape(-1)

        D = _finite_matrix(D, "D")
        G_g = _finite_matrix(G_g, "G_g")

        response_matrix = D @ G_g

        if response_matrix.ndim != 2 or response_matrix.shape[0] != response_matrix.shape[1]:
            raise ValueError(f"D @ G_g must be square, got {response_matrix.shape}.")

        nparams = response_matrix.shape[0]

        if n_obs.size != nparams:
            raise ValueError(f"n_obs length must be {nparams}, got {n_obs.size}.")

        if n_off_obs is not None:
            n_off_obs = _nonnegative_vector(n_off_obs, nparams, "n_off_obs")

        include_background = require_bool(include_background, "include_background")

        mode = str(mode).lower().strip()
        if mode not in {"prior", "posterior"}:
            raise ValueError("mode must be either 'prior' or 'posterior'.")

        with pm.Model() as model:
            x = self._build_prior(nparams=nparams)

            if self.debug or debug:
                print(
                    f"[model] prior={self.prior_config['dist']} "
                    f"transform={self.transform_name} "
                    f"rl_init={self.rl_init}"
                )

            response_constant = pt.constant(response_matrix, dtype=FLOATX)
            nu = pm.Deterministic("nu", pt.dot(x, response_constant))

            expected_counts = nu
            background_b0 = None

            if include_background:
                background_expected, background_b0 = self._background_from_config(
                    n_off_obs=n_off_obs,
                    nparams=nparams,
                    mode=mode,
                )
                expected_counts = nu + background_expected

            if mode == "posterior":
                pm.Poisson("n_obs", mu=expected_counts, observed=n_obs)

            var_names = self._wanted_var_names(include_background)

            if mode == "prior":
                idata = self._sample_prior(model, var_names)
                draws_group = idata.prior
            else:
                idata = self._sample_posterior(model, var_names)
                draws_group = idata.posterior

            idata.attrs["mode"] = mode
            idata.attrs["likelihood"] = "poisson"
            idata.attrs["include_background"] = (
                "true" if include_background else "false"
            )
            idata.attrs["background_model"] = (
                str(self.sample_config["background_model"]["kind"])
                if include_background
                else "none"
            )
            idata.attrs["bg_b0"] = (
                float(background_b0) if background_b0 is not None else "none"
            )

            if output_trace_path is not None:
                output_trace_path = Path(output_trace_path)
                output_trace_path.parent.mkdir(parents=True, exist_ok=True)
                az.to_netcdf(idata, output_trace_path)

            x_draws = _extract_chain_draw_eg(draws_group["x"], "x")

            background_draws = None
            if include_background:
                background_draws = _extract_chain_draw_eg(
                    draws_group["bg_exp"],
                    "bg_exp",
                )

            return x_draws, background_draws
