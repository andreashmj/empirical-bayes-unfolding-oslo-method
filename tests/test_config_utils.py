"""Tests for the configuration YAML files."""

import copy
import pytest
from src.config_utils import ProfileMapper, SamplerMapper

def base_sampler_config():
    return {
        "prior": {
            "draws": 10,
            "random_seed": 1,
        },
        "initialization": {
            "rl_init": True,
        },
        "nuts": {
            "nuts_sampler": "pymc",
            "draws": 20,
            "tune": 10,
            "chains": 2,
            "target_accept": 0.90,
            "max_treedepth": 10,
            "random_seed": 2,
            "pymc": {
                "cores": 1,
                "progressbar": False,
                "dense_mass": False,
                "init": "jitter+adapt_diag",
            },
        },
    }

def test_prior_mapper_range_and_exact_energy_mapping_are_copied():
    profiles = {
        "low": {"dist": "low_dist", "params": {"alpha": 1.0}},
        "exact": {"dist": "exact_dist", "params": {"alpha": 2.0}},
    }
    mapping = [
        {"range": [0.0, 3000.0], "profile": "low"},
        {"energies": [4000.0], "profile": "exact"},
    ]

    mapper = ProfileMapper(profiles, mapping)

    low = mapper.profile_for(2500.0)
    exact = mapper.profile_for(4000.0)

    assert low["dist"] == "low_dist"
    assert exact["dist"] == "exact_dist"

    low["params"]["alpha"] = 99.0
    assert mapper.profile_for(2500.0)["params"]["alpha"] == pytest.approx(1.0)

    with pytest.raises(KeyError):
        mapper.profile_for(9500.0)


def test_prior_mapper_rejects_bad_rules():
    profiles = {"a": {"dist": "x"}}

    with pytest.raises(ValueError):
        ProfileMapper(profiles, [{"profile": "a", "range": [2.0, 1.0]}])

    with pytest.raises(ValueError):
        ProfileMapper(profiles, [{"profile": "a", "range": [0.0, 1.0], "energies": [1.0]}])

    with pytest.raises(KeyError):
        ProfileMapper(profiles, [{"profile": "missing", "range": [0.0, 1.0]}])


def test_sampler_mapper_deep_profile_override():
    config = base_sampler_config()
    config["profiles"] = {
        "low": {
            "nuts": {
                "target_accept": 0.99,
                "pymc": {
                    "cores": 2,
                },
            },
        },
    }
    config["mapping"] = [
        {"range": [0.0, 3000.0], "profile": "low"},
    ]

    mapper = SamplerMapper(config)
    resolved = mapper.config_for(2500.0)

    assert resolved["nuts"]["target_accept"] == pytest.approx(0.99)
    assert resolved["nuts"]["pymc"]["cores"] == 2
    assert resolved["nuts"]["pymc"]["progressbar"] is False
    assert resolved["prior"]["draws"] == 10

    original = copy.deepcopy(config)
    resolved["nuts"]["pymc"]["cores"] = 99
    assert mapper.config_for(2500.0)["nuts"]["pymc"]["cores"] == 2
    assert config == original


def test_sampler_mapper_rejects_prior_or_backend_override():
    config = base_sampler_config()
    config["profiles"] = {"bad": {"prior": {"draws": 20}}}
    config["mapping"] = [{"range": [0.0, 1.0e9], "profile": "bad"}]

    mapper = SamplerMapper(config)

    with pytest.raises(ValueError):
        mapper.config_for(2500.0)

    config = base_sampler_config()
    config["profiles"] = {"bad": {"nuts": {"nuts_sampler": "numpyro"}}}
    config["mapping"] = [{"range": [0.0, 1.0e9], "profile": "bad"}]

    mapper = SamplerMapper(config)

    with pytest.raises(ValueError):
        mapper.config_for(2500.0)
