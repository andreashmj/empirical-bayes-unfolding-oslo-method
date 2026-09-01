"""
Functions for loading the YAML configuration files into the unfolding code base. This include the run, prior and sampler configs.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from .input_checks import require_bool, require_float, require_int


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
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
    """Load a run configuration file."""
    return load_yaml(path)


def dict_to_namespace(obj: Any) -> Any:
    """Convert dictionaries to argparse.Namespace objects."""

    if isinstance(obj, dict):
        return argparse.Namespace(
            **{key: dict_to_namespace(value) for key, value in obj.items()} )
    if isinstance(obj, list):
        return [dict_to_namespace(value) for value in obj]

    return obj


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return an update of base without modifying the inputs."""

    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _required_value(config: dict[str, Any], key: str, name: str) -> Any:
    """Return a required mapping value."""
    if key not in config:
        raise KeyError(f"Config must contain '{name}'.")
    return config[key]


class ProfileMapper:
    """Resolve a named profile for one excitation energy. A mapping rule can either list exact excitation energies,
        energies: [2500, 4000]
    or define an excitation-energy range,
        range: [3000, 6000]
    Exact-energy rules have higher priority than range rules. 
    """

    def __init__(self, profiles: dict[str, Any], mapping: list[dict[str, Any]] ):
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("Profile config must contain a non-empty profiles mapping.")

        if not isinstance(mapping, list) or not mapping:
            raise ValueError("Profile config must contain a non-empty mapping list.")

        self.profiles = copy.deepcopy(profiles)
        self._ranges: list[tuple[float, float, str]] = []
        self._energies: dict[float, str] = {}

        for rule in mapping:
            if not isinstance(rule, dict):
                raise TypeError("Each profile mapping rule must be a mapping.")
            profile_name = str(_required_value(rule, "profile", "profile"))
            if profile_name not in self.profiles:
                raise KeyError(f"Profile {profile_name!r} is not defined in profiles.")

            has_range = "range" in rule
            has_energies = "energies" in rule

            if has_range == has_energies:
                raise ValueError(
                    "Each profile mapping rule must contain exactly one of range or energies.")

            if has_range:
                self._add_range(rule["range"], profile_name)
            else:
                self._add_energies(rule["energies"], profile_name)

    def _add_range(self, values: Any, profile_name: str) -> None:
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError("Profile mapping range must contain two values.")

        lower = require_float(values[0], "profile range lower")
        upper = require_float(values[1], "profile range upper")

        if lower >= upper:
            raise ValueError("Profile mapping range lower value must be < upper value.")
        self._ranges.append((lower, upper, profile_name))

    def _add_energies(self, values: Any, profile_name: str) -> None:
        if not isinstance(values, (list, tuple)):
            raise TypeError("Profile mapping energies must be a list.")

        for value in values:
            energy = require_float(value, "profile mapping energy")

            if energy in self._energies:
                raise ValueError(f"Duplicate profile mapping for Ex={energy}.")

            self._energies[energy] = profile_name

    def profile_for(self, ex_energy: float) -> dict[str, Any]:
        """Return a copy of the profile for one excitation energy."""

        ex_energy = require_float(ex_energy, "ex_energy")

        if ex_energy in self._energies:
            return copy.deepcopy(self.profiles[self._energies[ex_energy]])

        for lower, upper, profile_name in self._ranges:
            if lower <= ex_energy < upper:
                return copy.deepcopy(self.profiles[profile_name])

        raise KeyError(f"No profile matches Ex={ex_energy}.")


class SamplerMapper:
    """
    Build the sampler configuration for one excitation-energy bin. The sampler YAML has a complete configuration and can have profile overrides. For a requested Ex value, the matching profile is merged into the base configuration and the final result is checked.
    """

    def __init__(self, sampler_config: dict[str, Any]):
        sampler_config = copy.deepcopy(sampler_config)

        profiles = sampler_config.pop("profiles", None)
        mapping = sampler_config.pop("mapping", None)

        self.base_config = sampler_config
        self.profile_mapper = _optional_profile_mapper(profiles, mapping)

        _validate_sampler_config(self.base_config)

    def config_for(self, ex_energy: float) -> dict[str, Any]:
        """Return the sampler configuration for one excitation-energy bin."""
        config = copy.deepcopy(self.base_config)

        if self.profile_mapper is not None:
            profile = self.profile_mapper.profile_for(ex_energy)
            _validate_sampler_profile(profile)
            config = _deep_update(config, profile)

        _validate_sampler_config(config)
        return config

def _optional_profile_mapper(
    profiles: Any,
    mapping: Any,
) -> ProfileMapper | None:
    if profiles is None and mapping is None:
        return None

    if profiles is None or mapping is None:
        raise ValueError(
            "Sampler config must contain both profiles and mapping or neither")
    return ProfileMapper(profiles, mapping)


def load_prior_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a prior configuration file."""
    config = load_yaml(path)

    profiles = _required_value(config, "profiles", "profiles")
    mapping = _required_value(config, "mapping", "mapping")
    ProfileMapper(profiles, mapping)

    return config


def load_sampler_config(path: str | Path) -> dict[str, Any]:
    """Load a sampler configuration file."""
    config = load_yaml(path)
    if ("profiles" in config) != ("mapping" in config):
        raise ValueError(
            "Sampler config must contain both profiles and mapping, or neither.")
    return config
    
def _validate_sampler_profile(profile: dict[str, Any]) -> None:
    """Validate one Ex-dependent sampler-profile override."""
    if not isinstance(profile, dict):
        raise TypeError("Sampler profile must be a mapping.")

    allowed_top_level = {"nuts"}
    unknown_keys = sorted(set(profile) - allowed_top_level)

    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(
            "Sampler profiles may only override NUTS tuning settings. "
            f"Invalid top-level key(s): {joined}."
        )

    nuts_profile = profile.get("nuts", {})

    if not isinstance(nuts_profile, dict):
        raise TypeError("Sampler profile section nuts must be a mapping.")

    if "nuts_sampler" in nuts_profile:
        raise ValueError("Sampler profiles must not override nuts.nuts_sampler.")

def _validate_sampler_config(config: dict[str, Any]) -> None:
    """Validate a complete sampler configuration."""
    if not isinstance(config, dict):
        raise TypeError("Sampler config must be a mapping.")

    _validate_prior_sampler_config(_required_value(config, "prior", "prior"))
    _validate_initialization_config(config.get("initialization", {}))
    _validate_nuts_config(_required_value(config, "nuts", "nuts"))


def _validate_prior_sampler_config(config: dict[str, Any]) -> None:
    """Validate the prior-draw part of the sampler configuration."""
    if not isinstance(config, dict):
        raise TypeError("Sampler config section prior must be a mapping.")

    require_int(_required_value(config, "draws", "prior.draws"), "prior.draws", minimum=1)
    require_int(
        _required_value(config, "random_seed", "prior.random_seed"),
        "prior.random_seed",
        minimum=0,
    )


def _validate_initialization_config(config: dict[str, Any] | None) -> None:
    """Validate optional sampler initialization settings."""

    if config is None:
        return
    if not isinstance(config, dict):
        raise TypeError("Sampler config section initialization must be a mapping.")
    if "rl_init" in config:
        require_bool(config["rl_init"], "initialization.rl_init")

def _validate_nuts_config(config: dict[str, Any]) -> None:
    """Validate the NUTS part of the sampler configuration."""

    if not isinstance(config, dict):
        raise TypeError("Sampler config section nuts must be a mapping.")

    nuts_sampler = str(config.get("nuts_sampler", "pymc")).strip().lower()

    if nuts_sampler not in {"pymc", "numpyro", "blackjax", "nutpie"}:
        raise ValueError(
            "nuts.nuts_sampler must be one of: pymc, numpyro, blackjax, nutpie." )

    require_int(_required_value(config, "draws", "nuts.draws"), "nuts.draws", minimum=1)
    require_int(_required_value(config, "tune", "nuts.tune"), "nuts.tune", minimum=0)
    require_int(_required_value(config, "chains", "nuts.chains"), "nuts.chains", minimum=1)
    require_int(
        _required_value(config, "max_treedepth", "nuts.max_treedepth"),
        "nuts.max_treedepth",
        minimum=1)
    require_int(
        _required_value(config, "random_seed", "nuts.random_seed"),
        "nuts.random_seed",
        minimum=0)

    require_float(
        _required_value(config, "target_accept", "nuts.target_accept"),
        "nuts.target_accept",
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


def _validate_pymc_backend_config(config: dict[str, Any]) -> None:
    """Validate PyMC backend settings."""

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
    """Validate JAX backend settings."""

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
    """Validate Nutpie backend settings."""

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
