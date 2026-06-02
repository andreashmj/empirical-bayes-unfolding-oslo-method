"""Tests for the functions in the prior factory script."""

import numpy as np
import pytest
from src.prior.factory import make_prior


def test_gamma_lognormal_rl_default_floor_gives_positive_center_and_width():
    profile = {
        "dist": "gamma_lognormal_mean_rl",
        "params": {
            "alpha": 1.0,
            "sigma_min": 1.0,
            "sigma_max": 3.0,
            "c_ref": 100.0,
            "rl_iterations": 1,
        },
    }

    n_on = np.array([5.0, 10.0, 0.0])
    response = np.eye(3)

    prior = make_prior(
        profile=profile,
        response_matrix=response,
        gamma_resolution_matrix=np.eye(3),
        n_on=n_on,
        background_reference=np.zeros(3),
    )

    params = prior["params"]
    mu_center = np.asarray(params["mu_center"], dtype=float)
    sigma = np.asarray(params["sigma"], dtype=float)

    assert prior["dist"] == "gamma_lognormal_mean"
    assert params["alpha"] == pytest.approx(1.0)

    # With identity response and one RL iteration, the third bin has zero RL
    # intensity and is lifted by the stability floor.
    assert mu_center[0] == pytest.approx(5.0)
    assert mu_center[1] == pytest.approx(10.0)
    assert mu_center[2] == pytest.approx(1.0e-1)

    assert np.all(mu_center > 0.0)
    assert sigma.shape == n_on.shape
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)


def test_constant_center_prior_uses_mean_of_floored_rl_reference():
    profile = {
        "dist": "gamma_lognormal_mean_constant_center",
        "params": {
            "alpha": 1.0,
            "sigma_min": 1.0,
            "sigma_max": 3.0,
            "c_ref": 100.0,
            "rl_iterations": 1,
            "rl_floor": 0.1,
        },
    }

    n_on = np.array([5.0, 10.0, 0.0])
    response = np.eye(3)

    prior = make_prior(
        profile=profile,
        response_matrix=response,
        gamma_resolution_matrix=np.eye(3),
        n_on=n_on,
        background_reference=np.zeros(3),
    )

    params = prior["params"]
    mu_center = np.asarray(params["mu_center"], dtype=float)
    sigma = np.asarray(params["sigma"], dtype=float)

    expected_center = np.mean(np.array([5.0, 10.0, 0.1]))

    np.testing.assert_allclose(mu_center, np.full(3, expected_center))
    assert np.all(mu_center > 0.0)
    assert sigma.shape == n_on.shape
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)
