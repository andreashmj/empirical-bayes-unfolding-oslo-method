"""Tests for helper functions used in the run_unfolding script."""

import argparse
import numpy as np
import pytest

from src.run_unfolding import (
    _background_b0_from_off_counts,
    _background_postmean_from_off_counts,
    _stack_draws,
    _stack_operators,
    _stack_vectors,
)


def test_stack_draws_stores_ex_eg_chain_draw_with_nan_padding():
    chain_draw_eg = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)

    out = _stack_draws(
        arrays=[chain_draw_eg],
        ex_values=np.array([2500.0]),
        eg_axis=np.arange(5, dtype=float),
        active_eg_lengths=[4],
        name="x",
    )

    assert out.dims == ("Ex", "Eg", "chain", "draw")
    assert out.shape == (1, 5, 2, 3)

    expected = np.moveaxis(chain_draw_eg, source=2, destination=0)
    np.testing.assert_allclose(out.values[0, :4, :, :], expected)
    assert np.all(np.isnan(out.values[0, 4, :, :]))


def test_stack_vectors_and_operators_pad_inactive_region():
    vector = np.array([1.0, 2.0, 3.0])
    vector_out = _stack_vectors(
        vectors=[vector],
        ex_values=np.array([4000.0]),
        eg_axis=np.arange(5, dtype=float),
        active_eg_lengths=[3],
        name="n_obs",
    )

    np.testing.assert_allclose(vector_out.values[0, :3], vector)
    assert np.all(np.isnan(vector_out.values[0, 3:]))

    operator = np.arange(9, dtype=float).reshape(3, 3)
    operator_out = _stack_operators(
        operators=[operator],
        ex_values=np.array([4000.0]),
        eg_axis=np.arange(5, dtype=float),
        active_eg_lengths=[3],
        name="D",
    )

    np.testing.assert_allclose(operator_out.values[0, :3, :3], operator)
    assert np.all(np.isnan(operator_out.values[0, 3:, :]))
    assert np.all(np.isnan(operator_out.values[0, :, 3:]))


def test_background_gamma_rate_and_postmean():
    n_off = np.array([2.0, 4.0, 6.0])
    a0 = 1.0

    b0 = _background_b0_from_off_counts(n_off, a0)
    assert b0 == pytest.approx(1.0 / 4.0)

    postmean = _background_postmean_from_off_counts(n_off, a0, b0)
    np.testing.assert_allclose(postmean, (a0 + n_off) / (b0 + 1.0))

    with pytest.raises(ValueError):
        _background_b0_from_off_counts(np.zeros(3), a0)
