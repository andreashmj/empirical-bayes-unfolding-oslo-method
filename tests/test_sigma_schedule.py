"""Tests for the adaptive prior-width schedule."""

import numpy as np
import pytest
from src.prior.sigma_schedule import adaptive_sigma_from_rl


def test_adaptive_sigma_matches_formula_for_identity_resolution():
    x_rl = np.array([1.0, 10.0, 100.0])
    G_g = np.eye(3)

    sigma, sigma_shape, eta_rl, meta = adaptive_sigma_from_rl(
        x_rl=x_rl,
        gamma_resolution_matrix=G_g,
        sigma_min=1.0,
        sigma_max=3.0,
        c_ref=100.0,
    )

    eta_mean = np.mean(x_rl)
    expected_shape = 1.0 + (3.0 - 1.0) / (1.0 + x_rl / eta_mean)
    expected_activation = x_rl / (x_rl + 100.0)
    expected_sigma = (1.0 - expected_activation) * 3.0 + expected_activation * expected_shape

    np.testing.assert_allclose(eta_rl, x_rl)
    np.testing.assert_allclose(sigma_shape, expected_shape)
    np.testing.assert_allclose(sigma, expected_sigma)

    assert np.all(sigma > 0.0)
    assert np.all(sigma <= 3.0)
    assert sigma[-1] < sigma[0]

    assert meta["sigma_schedule"] == "eta_shape_local_count_activation"
    assert meta["c_ref"] == pytest.approx(100.0)
    assert meta["eta_mean"] == pytest.approx(eta_mean)


def test_adaptive_sigma_rejects_bad_shapes_and_zero_eta():
    with pytest.raises(ValueError):
        adaptive_sigma_from_rl(
            x_rl=np.array([1.0, 2.0]),
            gamma_resolution_matrix=np.eye(3),
            sigma_min=1.0,
            sigma_max=3.0,
            c_ref=100.0,
        )

    with pytest.raises(ValueError):
        adaptive_sigma_from_rl(
            x_rl=np.zeros(3),
            gamma_resolution_matrix=np.eye(3),
            sigma_min=1.0,
            sigma_max=3.0,
            c_ref=100.0,
        )


def test_uniform_width_case_is_allowed():
    x_rl = np.array([5.0, 10.0, 20.0])
    sigma, sigma_shape, _, _ = adaptive_sigma_from_rl(
        x_rl=x_rl,
        gamma_resolution_matrix=np.eye(3),
        sigma_min=1.0,
        sigma_max=1.0,
        c_ref=100.0,
    )

    np.testing.assert_allclose(sigma, np.ones(3))
    np.testing.assert_allclose(sigma_shape, np.ones(3))
