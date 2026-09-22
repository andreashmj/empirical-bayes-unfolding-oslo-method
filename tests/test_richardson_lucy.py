""" Tests for the the Richardson-Lucy iteration method."""

import numpy as np
import pytest

from src.prior.richardson_lucy import (
    eta_relative_change,
    richardson_lucy_iterates,
    rl_reference,
)


def test_richardson_lucy_identity_response_one_iteration_matches_counts():
    n_on = np.array([5.0, 10.0, 0.0])
    response = np.eye(3)

    iterates = richardson_lucy_iterates(
        n_on=n_on, response_matrix=response,
        background_reference=None,
        n_iter=1)

    assert iterates.shape == (2, 3)
    np.testing.assert_allclose(iterates[0], np.full(3, 5.0))
    np.testing.assert_allclose(iterates[1], n_on, rtol=1.0e-10, atol=1.0e-10)


def test_richardson_lucy_with_background_stays_nonnegative():
    n_on = np.array([6.0, 11.0, 1.0])
    background = np.ones(3)
    response = np.eye(3)

    iterates = richardson_lucy_iterates(
        n_on=n_on,
        response_matrix=response,
        background_reference=background,
        n_iter=3
    )

    assert iterates.shape == (4, 3)
    assert np.all(np.isfinite(iterates))
    assert np.all(iterates >= 0.0)


def test_eta_change_known_values():
    eta_iterates = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [4.0, 1.0],
        ]
    )

    change = eta_relative_change(eta_iterates, window=1)

    assert np.isnan(change[0])
    assert change[1] == pytest.approx(1.0)
    assert change[2] == pytest.approx(2.0)


def test_rl_reference_fixed_iteration_metadata():
    n_on = np.array([5.0, 10.0, 0.0])
    response = np.eye(3)

    x_rl, meta = rl_reference(
        n_on=n_on,
        response_matrix=response,
        gamma_resolution_matrix=np.eye(3),
        background_reference=None,
        rl_iterations=1
    )

    np.testing.assert_allclose(x_rl, n_on, rtol=1.0e-10, atol=1.0e-10)
    assert meta["rl_iteration_rule"] == "fixed"
    assert meta["rl_iteration_status"] == "fixed"
    assert meta["rl_iteration"] == 1
