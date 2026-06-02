"""Test that OMpy specialize_like returns the normalized active response.

The Richardson-Lucy reference update used in this project does not include an
explicit sensitivity denominator. That is valid only if the composite active
response used by RL is already normalized.

In the project code, run_unfolding.py does:
    n_cut = n_matrix.iloc[ex_index, active_eg_slice]
    D_cut, G_g_cut = response.specialize_like(n_cut)
    response_matrix = D_cut.values @ G_g_cut.values
and then RL uses the right-hand convention:
    expected_on = estimate @ response_matrix + background_reference
Therefore the response-normalization contract is row normalization:
    response_matrix.sum(axis=1) == 1
for every emitted-energy bin in the active domain.
"""

from __future__ import annotations

import numpy as np
import pytest

om = pytest.importorskip("ompy")
pytest.importorskip("ompy.response")


def _small_active_domain_vector():
    """Return an OMpy Vector like n_cut in run_unfolding.py.

    Important: we construct a two-row Matrix and then take one row. OMpy's
    Matrix constructor needs at least two X coordinates to infer the X-axis
    spacing, so a one-row Matrix is not a valid fixture.
    """
    eg_axis = np.arange(120.0, 2040.0, 120.0, dtype=float)

    # Must have length >= 2. A one-element X axis crashes inside OMpy before
    # specialize_like is ever tested.
    ex_axis = np.array([3000.0, 3120.0], dtype=float)

    values = np.ones((ex_axis.size, eg_axis.size), dtype=float)

    matrix = (
        om.Matrix(values=values, X=ex_axis, Y=eg_axis)
        .set_xalias("Ex")
        .set_yalias("Eg")
    )

    return matrix.iloc[0, :]


def test_specialize_like_returns_row_normalized_composite_response():
    n_cut = _small_active_domain_vector()

    try:
        response = (
            om.response.Response.from_db("OSCAR2020")
            .normalize_sigma("1330keV", "30keV")
        )
    except Exception as exc:
        pytest.skip(f"OSCAR2020 response database is not available: {exc}")

    D_cut, G_g_cut = response.specialize_like(n_cut)

    D = np.asarray(D_cut.values, dtype=float)
    G_g = np.asarray(G_g_cut.values, dtype=float)
    response_matrix = D @ G_g

    assert response_matrix.ndim == 2
    assert response_matrix.shape[0] == response_matrix.shape[1]

    row_sums = response_matrix.sum(axis=1)

    assert np.all(row_sums > 0.0)

    np.testing.assert_allclose(
        row_sums,
        np.ones_like(row_sums),
        rtol=5.0e-6,
        atol=5.0e-8,
        err_msg=(
            "The active composite response D @ G_g is not row-normalized. "
            f"row_sums.min()={row_sums.min():.8g}, "
            f"row_sums.max()={row_sums.max():.8g}, "
            f"n_bins={row_sums.size}."
        ),
    )
