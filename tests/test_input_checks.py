import numpy as np
import pytest

from src.input_checks import (
    require_bool,
    require_finite_array,
    require_float,
    require_int,
)


def test_require_bool_accepts_only_bool():
    assert require_bool(True, "flag") is True
    assert require_bool(np.bool_(False), "flag") is False

    with pytest.raises(ValueError):
        require_bool(1, "flag")

    with pytest.raises(ValueError):
        require_bool("true", "flag")


def test_require_int_rejects_bool_and_checks_minimum():
    assert require_int(3, "n") == 3
    assert require_int(np.int64(4), "n") == 4

    with pytest.raises(ValueError):
        require_int(True, "n")

    with pytest.raises(ValueError):
        require_int(0, "n", minimum=1)


def test_require_float_rejects_bool_nonfinite_and_bounds():
    assert require_float(1.5, "x") == pytest.approx(1.5)

    with pytest.raises(ValueError):
        require_float(False, "x")

    with pytest.raises(ValueError):
        require_float(np.inf, "x")

    with pytest.raises(ValueError):
        require_float(1.0, "x", minimum=1.0, minimum_inclusive=False)

    with pytest.raises(ValueError):
        require_float(2.0, "x", maximum=2.0, maximum_inclusive=False)


def test_require_finite_array_shape_and_nonnegative():
    out = require_finite_array([0.0, 1.0], "arr", ndim=1, nonnegative=True)
    np.testing.assert_allclose(out, [0.0, 1.0])

    with pytest.raises(ValueError):
        require_finite_array([[1.0]], "arr", ndim=1)

    with pytest.raises(ValueError):
        require_finite_array([1.0, np.nan], "arr")

    with pytest.raises(ValueError):
        require_finite_array([-1.0, 0.0], "arr", nonnegative=True)
