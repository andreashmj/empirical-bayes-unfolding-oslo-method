"""
Unfolding model in PYMC for one excitation-energy row.
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


FLOATX = "float64"
pytensor_config.floatX = FLOATX


def _extract_chain_draw_eg(data_array, name: str) -> np.ndarray:

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
        raise ValueError(f"{name}: expected chain/draw or draw dimension, got dims={dims}.")

    if array.ndim != 3:
        raise ValueError(f"{name}: expected extracted shape (chain, draw, Eg), got {array.shape}.")

    return array



class PyMCUnfolder:
    """
    Main class for running the unfolding on one Ex bin with PyMC.
    
    """

    def __init__(self, prior_config: dict, sample_config: dict, debug: bool = False):
        self.prior_config = prior_config
        self.sample_config = sample_config
        self.debug = bool(debug)

        self.transform_name = str(self.sample_config["transform"]).strip().lower()

        if self.transform_name != "log":
            raise ValueError("The implementation requires transform: log.")

        self.transform_kwargs = {"default_transform": tr.log}
    

        initialization_config = self.sample_config.get("initialization", {}) or {}
        self.rl_init = initialization_config.get("rl_init", False)

        self.initvals: dict[str, np.ndarray | float] = {}
        

    def _init_kwargs(self, name: str, value) -> dict:
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
        self.initvals = {}

        dist = self.prior_config["dist"]
        params = self.prior_config["params"]

        if dist == "gamma_lognormal_mean":
            return self._build_gamma_lognormal_mean_prior(params, nparams)

        if dist == "gamma_lognormal_mean_hyper":
            return self._build_gamma_lognormal_mean_hyper_prior(params, nparams)

        if dist == "lognormal":
            return self._build_lognormal_prior(params, nparams)

        raise ValueError(f"Unknown prior distribution: {dist}")

    def _build_gamma_lognormal_mean_prior(self, params: dict, nparams: int):

        mu_center = np.asarray(params["mu_center"], dtype=np.float64)
        sigma = np.asarray(params["sigma"], dtype=np.float64)
        alpha = float(params["alpha"])
        
        log_m_center = np.log(mu_center) - 0.5 * sigma**2

        
        z_m_init = 0.5 * sigma

        z_m = pm.Normal(
            "z_m",
            mu=0.0,
            sigma=1.0,
            shape=nparams,
            **self._init_kwargs("z_m", z_m_init),
        )

        log_m = pm.Deterministic("log_m", pt.constant(log_m_center, dtype=FLOATX) + pt.constant(sigma, dtype=FLOATX) * z_m)

        mean_local = pm.Deterministic("m", pt.exp(log_m))
    
        
        u_kwargs = {}
        u_kwargs.update(self._init_kwargs("u_x", np.ones(nparams, dtype=np.float64)))
        u_kwargs.update(self.transform_kwargs)

        u_x = pm.Gamma(
            "u_x",
            alpha=alpha,
            beta=alpha,
            shape=nparams,
            **u_kwargs,
        )

        return pm.Deterministic("x", mean_local * u_x)

    def _build_gamma_lognormal_mean_hyper_prior(self, params: dict, nparams: int):

        alpha = float(params["alpha"])
        theta_mu = float(params["theta_mu"])
        theta_sigma = float(params["theta_sigma"])
        tau_sigma = float(params["tau_sigma"])
        theta = pm.Normal(
            "theta",
            mu=theta_mu,
            sigma=theta_sigma,
            **self._init_kwargs("theta", theta_mu)
        )

        tau_kwargs = {}
        tau_kwargs.update(self._init_kwargs("tau", tau_sigma))
        tau_kwargs.update(self.transform_kwargs)

        tau = pm.HalfNormal(
            "tau",
            sigma=tau_sigma,
            **tau_kwargs
        )

        z = pm.Normal(
            "z",
            mu=0.0,
            sigma=1.0,
            shape=nparams,
            **self._init_kwargs("z", np.zeros(nparams, dtype=np.float64))
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
            **x_kwargs
        )

    def _build_lognormal_prior(self, params: dict, nparams: int):

        mu_log = params["mu_log"]
        sigma = params["sigma"]

        z_x = pm.Normal(
            "z_x",
            mu=0.0,
            sigma=1.0,
            shape=nparams,
            **self._init_kwargs("z_x", np.zeros(nparams, dtype=np.float64)) 
            )

        log_x = pm.Deterministic("log_x", pt.constant(mu_log, dtype=FLOATX) + pt.constant(sigma, dtype=FLOATX) * z_x)

        return pm.Deterministic("x", pt.exp(log_x))

    

    def _build_background(
        self,
        n_off_obs: np.ndarray,
        nparams: int,
        mode: str,
    ):
        config = self.sample_config["background_model"]
        kind = config["kind"]
        a0 = config["a0"]

        mean_off = float(np.mean(n_off_obs))
        if mean_off <= 0.0:
            raise ValueError("Mean OFF count must be positive.")

        b0 = a0 / mean_off

        if kind == "latent_gamma":
            bg_init = (a0 + n_off_obs) / (b0 + 1.0)

            bg_kwargs = {}
            bg_kwargs.update(self._init_kwargs("bg_exp", bg_init))
            bg_kwargs.update(self.transform_kwargs)

            background = pm.Gamma(
                "bg_exp",
                alpha=a0,
                beta=b0,
                shape=nparams,
                **bg_kwargs,
            )

            if mode == "posterior":
                pm.Poisson(
                    "n_off_obs",
                    mu=background,
                    observed=n_off_obs,
                )

        elif kind == "fixed_postmean":
            background = pm.Deterministic("bg_exp", pt.constant((a0 + n_off_obs) / (b0 + 1.0), dtype=FLOATX))

        else:
            raise ValueError(f"Unknown background model: {kind}")

        return background, b0




    def _sample_prior(self, model: pm.Model, var_names: list[str]) -> az.InferenceData:

        prior_config = self.sample_config["prior"]

        draws = int(prior_config["draws"])
        random_seed = int(prior_config["random_seed"])

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

        nuts_config = self.sample_config["nuts"]
        nuts_sampler = str(nuts_config.get("nuts_sampler", "pymc")).strip().lower()

        draws = int(nuts_config["draws"])
        tune = int(nuts_config["tune"])
        chains = int(nuts_config["chains"])
        target_accept = float(nuts_config["target_accept"])
        max_treedepth = int(nuts_config["max_treedepth"])
        random_seed = int(nuts_config["random_seed"])

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
            raise ValueError("nuts.nuts_sampler must be one of: pymc, numpyro, blackjax, nutpie.")

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

        pymc_config = nuts_config.get("pymc", {}) or {}

        cores = pymc_config.get("cores", chains)
        init = pymc_config.get("init", "jitter+adapt_diag")
        dense_mass = pymc_config.get("dense_mass", False)
        progressbar = pymc_config.get("progressbar", True)

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
    

        jax_config = nuts_config.get("jax", {}) or {}

        platform = str(jax_config.get("platform", "auto")).strip().lower()
        chain_method = str(jax_config.get("chain_method", "parallel")).strip().lower()
        x64 = jax_config.get("x64", True)
        progressbar = jax_config.get("progressbar", True)
        jitter = jax_config.get("jitter", True)
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

        nutpie_config = nuts_config.get("nutpie", {}) or {}
        compile_backend = str(nutpie_config.get("compile_backend", "numba")).strip().lower()
        cores = int(nutpie_config.get("cores", chains))
            
        save_warmup = nutpie_config.get("save_warmup", False)
    
        store_divergences = nutpie_config.get("store_divergences", False)
        progress_bar = nutpie_config.get("progressbar", True)
        use_initial_points = nutpie_config.get("use_initial_points", False)

        import nutpie

        initial_points = self._sampler_initvals() if use_initial_points else None

        if compile_backend == "numba":
            compiled_model = nutpie.compile_pymc_model(model, backend="numba", initial_points=initial_points)
        elif compile_backend == "jax":
            compiled_model = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax", initial_points=initial_points)
        else:
            raise ValueError("nuts.nutpie.compile_backend must be numba or jax.")

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
            sample_kwargs["low_rank_modified_mass_matrix"] = nutpie_config["low_rank_modified_mass_matrix"]

        if "transform_adapt" in nutpie_config:
            sample_kwargs["transform_adapt"] = nutpie_config["transform_adapt"]

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
        R: np.ndarray,
        mode: str,
        include_background: bool,
        output_trace_path: str | Path | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Run prior or posterior sampling for one excitation-energy bin.
        
        """

        nparams = R.shape[0]
        
        if R.shape != (n_obs.size, n_obs.size):
            raise ValueError("R and n_obs have incompatible dimensions.")

        mode = str(mode).lower().strip()
        if mode not in {"prior", "posterior"}:
            raise ValueError("mode must be either prior or posterior.")

        with pm.Model() as model:
            x = self._build_prior(nparams)

            if self.debug or debug:
                print(
                    f"prior={self.prior_config['dist']} "
                    f"transform={self.transform_name} "
                    f"rl_init={self.rl_init}"
                )

            nu = pm.Deterministic("nu", pt.dot(x, pt.constant(R, dtype=FLOATX)))

            expected_counts = nu
            background_b0 = None

            if include_background:
                background, background_b0 = self._build_background(n_off_obs=n_off_obs, nparams=nparams, mode=mode)
                expected_counts = nu + background

            if mode == "posterior":
                pm.Poisson("n_obs", mu=expected_counts, observed=n_obs)

            var_names = ["x", "bg_exp"] if include_background else ["x"]

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
            idata.attrs["background_model"] = (str(self.sample_config["background_model"]["kind"]) if include_background else "none")
            idata.attrs["bg_b0"] = (float(background_b0) if background_b0 is not None else "none")

            if output_trace_path is not None:
                output_trace_path = Path(output_trace_path)
                output_trace_path.parent.mkdir(parents=True, exist_ok=True)
                az.to_netcdf(idata, output_trace_path)

            x_draws = _extract_chain_draw_eg(draws_group["x"], "x")

            background_draws = None
            if include_background:
                background_draws = _extract_chain_draw_eg(draws_group["bg_exp"], "bg_exp")

            return x_draws, background_draws
