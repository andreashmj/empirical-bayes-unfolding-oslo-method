"""Tests for using the RL stability floor consistently in prior construction."""

from __future__ import annotations
import numpy as np
import pytest
from src.prior.factory import make_prior

RAW_RL = np.array([1.0e-9, 10.0, 1000.0], dtype=float)
RL_FLOOR = 0.1

def _fake_rl_reference(*args, **kwargs):
    return RAW_RL.copy(), {"rl_iterations_selected": 7}

@pytest.fixture(autouse=True)
def patch_rl_reference(monkeypatch):
    monkeypatch.setattr("src.prior.factory._rl_reference", _fake_rl_reference)


def _build(dist: str):
    response = np.eye(RAW_RL.size)
    gamma_resolution = np.eye(RAW_RL.size)
    n_on = np.ones(RAW_RL.size)
    background = np.zeros(RAW_RL.size)

    params = {
        "sigma_min": 1.0,
        "sigma_max": 3.0,
        "c_ref": 100.0,
        "rl_floor": RL_FLOOR,
        "rl_iterations": "auto",
    }

    if dist == "lognormal_rl_matched_cv":
        params["alpha_reference"] = 1.0
    else:
        params["alpha"] = 1.0

    return make_prior(
        {
            "dist": dist,
            "params": params,
        },
        response_matrix=response,
        gamma_resolution_matrix=gamma_resolution,
        n_on=n_on,
        background_reference=background,
    )

def test_gamma_lognormal_rl_uses_floored_reference_for_center_and_sigma_schedule():
    prior = _build("gamma_lognormal_mean_rl")

    expected_floored = np.maximum(RAW_RL, RL_FLOOR)

    np.testing.assert_allclose(prior["params"]["mu_center"], expected_floored)
    np.testing.assert_allclose(prior["x_rl"], expected_floored)
    np.testing.assert_allclose(prior["eta_rl"], expected_floored)

    assert prior["meta"]["rl_floor_applied"] is True
    assert prior["meta"]["rl_floor_fraction"] == pytest.approx(1.0 / 3.0)
    assert prior["meta"]["eta_min"] == pytest.approx(RL_FLOOR)


def test_constant_center_uses_floored_reference_for_mean_center_and_sigma_schedule():
    prior = _build("gamma_lognormal_mean_constant_center")

    expected_floored = np.maximum(RAW_RL, RL_FLOOR)
    expected_center = np.full_like(expected_floored, np.mean(expected_floored))

    np.testing.assert_allclose(prior["params"]["mu_center"], expected_center)
    np.testing.assert_allclose(prior["x_rl"], expected_floored)
    np.testing.assert_allclose(prior["eta_rl"], expected_floored)

    assert prior["meta"]["center_value"] == pytest.approx(np.mean(expected_floored))
    assert prior["meta"]["eta_min"] == pytest.approx(RL_FLOOR)


def test_matched_lognormal_uses_floored_reference_for_center_and_sigma_schedule():
    prior = _build("lognormal_rl_matched_cv")

    expected_floored = np.maximum(RAW_RL, RL_FLOOR)

    np.testing.assert_allclose(prior["x_rl"], expected_floored)
    np.testing.assert_allclose(prior["eta_rl"], expected_floored)

    # For the matched lognormal prior, mu_log is chosen so that the prior mean
    # equals the floored RL reference after accounting for lognormal variance.
    mu_log = np.asarray(prior["params"]["mu_log"], dtype=float)
    sigma = np.asarray(prior["params"]["sigma"], dtype=float)
    reconstructed_mean = np.exp(mu_log + 0.5 * sigma**2)

    np.testing.assert_allclose(reconstructed_mean, expected_floored)
    assert prior["meta"]["eta_min"] == pytest.approx(RL_FLOOR)
