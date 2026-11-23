"""
Functions for loading the run, prior and sampler configs.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml
from ompy.units import from_unit

from .input_checks import require_bool, require_float, require_int


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        raise ValueError(f"Empty YAML file: {path}")
    if not isinstance(config, dict):
        raise TypeError(
            f"Expected YAML mapping in {path}, got {type(config).__name__}.")
    return config




def load_run_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)

    if config.get("ex_energies") is not None:
        config["ex_energies"] = [
            float(from_unit(value, "keV")) for value in config["ex_energies"]
        ]

    for key in ("ex_start_energy", "ex_end_energy", "fwhm_ex_kev", "eg_margin_kev"):
        if config.get(key) is not None:
            config[key] = float(from_unit(config[key], "keV"))

    return config


def dict_to_namespace(obj: Any) -> Any:

    if isinstance(obj, dict):
        return argparse.Namespace(
            **{key: dict_to_namespace(value) for key, value in obj.items()} )
    if isinstance(obj, list):
        return [dict_to_namespace(value) for value in obj]

    return obj



def _required_value(config: dict[str, Any], key: str, name: str) -> Any:
    """Return a required mapping value."""
    if key not in config:
        raise KeyError(f"Config must contain '{name}'.")
    return config[key]



def load_prior_config(path):
    config = load_yaml(path)

    if "dist" not in config:
        raise ValueError("Prior configuration requires dist.")

    return config


def load_sampler_config(path):
    config = load_yaml(path)
    _validate_sampler_config(config)
    return config

    

def _validate_sampler_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError("Sampler config must be a mapping.")

    _validate_prior_sampler_config(_required_value(config, "prior", "prior"))
    _validate_initialization_config(config.get("initialization", {}))
    _validate_nuts_config(_required_value(config, "nuts", "nuts"))


def _validate_prior_sampler_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError("Sampler config section prior must be a mapping.")

    require_int(_required_value(config, "draws", "prior.draws"), "prior.draws", minimum=1)
    require_int(_required_value(config, "random_seed", "prior.random_seed"), "prior.random_seed", minimum=0)


def _validate_initialization_config(config: dict[str, Any] | None) -> None:

    if config is None:
        return
    if not isinstance(config, dict):
        raise TypeError("Sampler config section initialization must be a mapping.")
    if "rl_init" in config:
        require_bool(config["rl_init"], "initialization.rl_init")

def _validate_nuts_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError("Sampler config section nuts must be a mapping.")

    nuts_sampler = str(config.get("nuts_sampler", "pymc")).strip().lower()

    if nuts_sampler not in {"pymc", "numpyro", "blackjax", "nutpie"}:
        raise ValueError(
            "nuts.nuts_sampler must be one of: pymc, numpyro, blackjax, nutpie." )

    require_int(_required_value(config, "draws", "nuts.draws"), "nuts.draws", minimum=1)
    require_int(_required_value(config, "tune", "nuts.tune"), "nuts.tune", minimum=0)
    require_int(_required_value(config, "chains", "nuts.chains"), "nuts.chains", minimum=1)
    require_int(_required_value(config, "max_treedepth", "nuts.max_treedepth"), "nuts.max_treedepth", minimum=1)
    require_int(_required_value(config, "random_seed", "nuts.random_seed"),"nuts.random_seed", minimum=0)

    require_float(_required_value(config, "target_accept", "nuts.target_accept"), "nuts.target_accept",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False)

    if nuts_sampler == "pymc" and "pymc" in config:
        _validate_pymc_backend_config(config["pymc"])

    if nuts_sampler in {"numpyro", "blackjax"} and "jax" in config:
        _validate_jax_config(config["jax"])

    if nuts_sampler == "nutpie" and "nutpie" in config:
        _validate_nutpie_config(config["nutpie"])
        
        
    ranges = []

    for rule in config.get("target_accept_by_ex", []):
        lower, upper = map(float, rule["range"])

        if lower >= upper:
            raise ValueError("Invalid target_accept Ex range.")

        require_float(
            rule["value"],
            "target_accept_by_ex.value",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )

        ranges.append((lower, upper))

    ranges.sort()

    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError("Overlapping target_accept Ex ranges.")


def _validate_pymc_backend_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError("Sampler config section nuts.pymc must be a mapping.")
    if "cores" in config:
        require_int(config["cores"], "nuts.pymc.cores", minimum=1)
    if "progressbar" in config:
        require_bool(config["progressbar"], "nuts.pymc.progressbar")
    if "dense_mass" in config:
        require_bool(config["dense_mass"], "nuts.pymc.dense_mass")
    if "init" in config and not isinstance(config["init"], str):
        raise TypeError("nuts.pymc.init must be a string.")


def _validate_jax_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError("Sampler config section nuts.jax must be a mapping.")
    if "platform" in config and not isinstance(config["platform"], str):
        raise TypeError("nuts.jax.platform must be a string.")
    if "chain_method" in config and not isinstance(config["chain_method"], str):
        raise TypeError("nuts.jax.chain_method must be a string.")
    if "x64" in config:
        require_bool(config["x64"], "nuts.jax.x64")
    if "progressbar" in config:
        require_bool(config["progressbar"], "nuts.jax.progressbar")
    if "jitter" in config:
        require_bool(config["jitter"], "nuts.jax.jitter")
    if "nuts_kwargs" in config and not isinstance(config["nuts_kwargs"], dict):
        raise TypeError("nuts.jax.nuts_kwargs must be a mapping.")


def _validate_nutpie_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError("Sampler config section nuts.nutpie must be a mapping.")
    if "compile_backend" in config:
        compile_backend = str(config["compile_backend"]).strip().lower()
        if compile_backend not in {"numba", "jax"}:
            raise ValueError("nuts.nutpie.compile_backend must be numba or jax.")
    if "cores" in config:
        require_int(config["cores"], "nuts.nutpie.cores", minimum=1)
    if "save_warmup" in config:
        require_bool(config["save_warmup"], "nuts.nutpie.save_warmup")
    if "store_divergences" in config:
        require_bool(config["store_divergences"], "nuts.nutpie.store_divergences")
    if "progressbar" in config:
        require_bool(config["progressbar"], "nuts.nutpie.progressbar")
    if "use_initial_points" in config:
        require_bool(config["use_initial_points"], "nuts.nutpie.use_initial_points")
    if "low_rank_modified_mass_matrix" in config:
        require_bool(
            config["low_rank_modified_mass_matrix"],
            "nuts.nutpie.low_rank_modified_mass_matrix")
    if "transform_adapt" in config:
        require_bool(config["transform_adapt"], "nuts.nutpie.transform_adapt")
