"""Tests for helpers used in the PyMC unfolding script."""


import numpy as np
import pytest
import xarray as xr

from src.pymc_unfolding import (
    _extract_chain_draw_eg,
    _finite_matrix,
    _nonnegative_vector,
    _positive_vector,
)


def test_extract_chain_draw_eg_reorders_dimensions():
    values = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4)
    data_array = xr.DataArray(values, dims=("draw", "chain", "Eg"))

    out = _extract_chain_draw_eg(data_array, "x")

    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(out, np.transpose(values, (1, 0, 2)))


def test_extract_chain_draw_eg_adds_chain_for_draw_only_array():
    values = np.arange(3 * 4, dtype=float).reshape(3, 4)
    data_array = xr.DataArray(values, dims=("draw", "Eg"))

    out = _extract_chain_draw_eg(data_array, "x")

    assert out.shape == (1, 3, 4)
    np.testing.assert_allclose(out[0], values)


def test_pymc_vector_helpers_validate_shape_and_sign():
    np.testing.assert_allclose(_positive_vector([1.0, 2.0], 2, "v"), [1.0, 2.0])
    np.testing.assert_allclose(_nonnegative_vector([0.0, 2.0], 2, "v"), [0.0, 2.0])

    with pytest.raises(ValueError):
        _positive_vector([0.0, 1.0], 2, "v")

    with pytest.raises(ValueError):
        _nonnegative_vector([-1.0, 1.0], 2, "v")

    with pytest.raises(ValueError):
        _positive_vector([1.0], 2, "v")


def test_finite_matrix_rejects_non_matrix():
    matrix = _finite_matrix([[1.0, 2.0], [3.0, 4.0]], "M")
    assert matrix.shape == (2, 2)

    with pytest.raises(ValueError):
        _finite_matrix([1.0, 2.0], "M")
