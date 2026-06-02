"""
Reader for unfolded result NetCDF files.

The stored draw convention is
    x[Ex, Eg, chain, draw].
This module returns row-wise vectors and draw arrays in a form used by the
analysis and figure scripts. Python variable names use lowercase names such as
ex, ex_actual, and eg_axis. NetCDF coordinate names remain Ex and Eg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def load_nc(path: str | Path) -> xr.Dataset:
    """Load a result NetCDF file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    return xr.open_dataset(path)


def require_mode(ds: xr.Dataset, expected_mode: str, path: str | Path) -> None:
    """Check that the result file has the expected run mode."""

    expected = str(expected_mode).strip().lower()
    actual = str(ds.attrs.get("mode", ds.attrs.get("run_mode", ""))).strip().lower()

    if actual and actual != expected:
        raise ValueError(f"{path}: expected mode={expected!r}, got {actual!r}.")


def actual_ex(ds: xr.Dataset, ex: float | None) -> float:
    """Return the actual Ex coordinate closest to the requested value."""

    ex_values = np.asarray(ds["Ex"].values, dtype=float)

    if ex is None:
        return ex_values[0].item()

    index = np.argmin(np.abs(ex_values - float(ex)))
    return ex_values[index].item()


def index_for_ex(ds: xr.Dataset, ex: float | None) -> int:
    """Return the Ex index closest to the requested value."""

    ex_values = np.asarray(ds["Ex"].values, dtype=float)
    ex_value = actual_ex(ds, ex)

    return np.argmin(np.abs(ex_values - ex_value)).item()


def requested_ex(ds: xr.Dataset, ex_actual: float) -> float | None:
    """Return the requested Ex value, if stored."""

    if "Ex_req" not in ds:
        return None

    value = ds["Ex_req"].sel(Ex=ex_actual, method="nearest").values.item()

    if not np.isfinite(value):
        return None

    return value


def active_eg_length(ds: xr.Dataset, ex_actual: float) -> int:
    """Return the active number of Eg bins for one Ex row."""

    if "Eg_len" in ds:
        return ds["Eg_len"].sel(Ex=ex_actual, method="nearest").values.item()

    values = np.asarray(
        ds["n_obs"].sel(Ex=ex_actual, method="nearest").values,
        dtype=float,
    )

    return np.sum(np.isfinite(values)).item()


def active_eg_grid(ds: xr.Dataset, active_eg_length: int) -> np.ndarray:
    """Return the active Eg grid."""

    return np.asarray(ds["Eg"].values[:active_eg_length], dtype=float)


def row_vector(
    ds: xr.Dataset,
    name: str,
    ex_actual: float,
    active_eg_length: int,
    fill_nan: float | None = 0.0,
) -> np.ndarray:
    """Return one per-Ex vector sliced to the active Eg range."""

    if name not in ds:
        raise KeyError(f"Dataset missing variable '{name}'.")

    values = np.asarray(
        ds[name]
        .sel(Ex=ex_actual, method="nearest")
        .isel(Eg=slice(0, active_eg_length))
        .values,
        dtype=float).reshape(-1)

    if fill_nan is not None:
        values = np.nan_to_num(values, nan=fill_nan)

    return values


def row_samples(
    ds: xr.Dataset,
    name: str,
    ex_actual: float,
    active_eg_length: int,
) -> xr.DataArray:
    """Return one per-Ex draw variable with dimensions sample and Eg."""

    if name not in ds:
        raise KeyError(f"Dataset missing variable '{name}'.")

    data_array = ds[name]

    if "Ex" in data_array.dims:
        data_array = data_array.sel(Ex=ex_actual, method="nearest")

    if "Eg" not in data_array.dims:
        raise ValueError(f"Variable '{name}' must have an Eg dimension.")

    data_array = data_array.isel(Eg=slice(0, active_eg_length))
    sample_dims = [dim for dim in data_array.dims if dim != "Eg"]

    if sample_dims:
        data_array = data_array.transpose(*sample_dims, "Eg")
        n_samples = int(np.prod([data_array.sizes[dim] for dim in sample_dims]))
        values = np.asarray(data_array.values, dtype=float).reshape(
            n_samples,
            active_eg_length,
        )
    else:
        values = np.asarray(data_array.values, dtype=float).reshape(
            1,
            active_eg_length,
        )

    return xr.DataArray(
        values,
        dims=("sample", "Eg"),
        coords={
            "sample": np.arange(values.shape[0]),
            "Eg": active_eg_grid(ds, active_eg_length),
        },
        name=name,
    )


def draw_cube(
    ds: xr.Dataset,
    name: str,
    ex_actual: float,
    active_eg_length: int,
) -> np.ndarray:
    """Return one draw variable with shape chain by draw by Eg."""

    if name not in ds:
        raise KeyError(f"Dataset missing variable '{name}'.")

    data_array = (
        ds[name]
        .sel(Ex=ex_actual, method="nearest")
        .isel(Eg=slice(0, active_eg_length))
        .transpose("chain", "draw", "Eg")
    )

    return np.asarray(data_array.values, dtype=float)


def operator_matrix(
    ds: xr.Dataset,
    name: str,
    ex_actual: float,
    active_eg_length: int,
) -> np.ndarray:
    """Return one per-Ex operator matrix with shape Eg by Eg."""

    if name not in ds:
        raise KeyError(f"Dataset missing operator '{name}'.")

    data_array = ds[name].sel(Ex=ex_actual, method="nearest")
    dim0, dim1 = data_array.dims

    matrix = np.asarray(
        data_array.isel(
            {
                dim0: slice(0, active_eg_length),
                dim1: slice(0, active_eg_length),
            }
        ).values,
        dtype=float,
    )

    return np.nan_to_num(matrix, nan=0.0)


def scalar_by_ex(
    ds: xr.Dataset,
    name: str,
    ex_actual: float,
    default=np.nan,
):
    """Return one scalar per-Ex variable, if stored."""

    if name not in ds:
        return default

    return ds[name].sel(Ex=ex_actual, method="nearest").values.item()


class UnfoldingResults:
    """Reader for one processed draws.nc file."""

    def __init__(self, path: str | Path, expected_mode: str):
        self.path = str(path)
        self.ds = load_nc(self.path)
        require_mode(self.ds, expected_mode, self.path)
        self.expected_mode = str(expected_mode).strip().lower()

    def ex_actual(self, ex: float | None) -> float:
        """Return actual Ex coordinate closest to the requested value."""

        return actual_ex(self.ds, ex)

    def ex_index(self, ex: float | None) -> int:
        """Return Ex index closest to the requested value."""

        return index_for_ex(self.ds, ex)

    def ex_requested(self, ex_actual: float) -> float | None:
        """Return requested Ex value, if stored."""

        return requested_ex(self.ds, ex_actual)

    def n_eg(self, ex_actual: float) -> int:
        """Return active Eg length for one Ex row."""

        return active_eg_length(self.ds, ex_actual)

    def eg(self, ex_actual: float) -> np.ndarray:
        """Return active Eg grid."""

        return active_eg_grid(self.ds, self.n_eg(ex_actual))

    def x(self, ex_actual: float) -> xr.DataArray:
        """Return x draws with dimensions sample and Eg."""

        return row_samples(self.ds, "x", ex_actual, self.n_eg(ex_actual))

    def x_cube(self, ex_actual: float) -> np.ndarray:
        """Return x draws with shape chain by draw by Eg."""

        return draw_cube(self.ds, "x", ex_actual, self.n_eg(ex_actual))

    def bg(self, ex_actual: float) -> xr.DataArray | None:
        """Return background expectation draws, if stored."""

        if "bg_exp" not in self.ds:
            return None

        return row_samples(self.ds, "bg_exp", ex_actual, self.n_eg(ex_actual))

    def eta(self, ex_actual: float) -> xr.DataArray:
        """Return eta = x @ G_g draws."""

        x_draws = self.x(ex_actual)
        _, g_gamma, _ = self.ops(ex_actual)

        values = np.asarray(x_draws.values, dtype=float) @ g_gamma

        return xr.DataArray(
            values,
            dims=("sample", "Eg"),
            coords=x_draws.coords,
            name="eta",
        )

    def eta_cube(self, ex_actual: float) -> np.ndarray:
        """Return eta = x @ G_g draws with shape chain by draw by Eg."""

        x_draws = self.x_cube(ex_actual)
        _, g_gamma, _ = self.ops(ex_actual)

        return np.einsum("cde,ef->cdf", x_draws, g_gamma)

    def nu(self, ex_actual: float) -> xr.DataArray:
        """Return nu = x @ D @ G_g draws."""

        x_draws = self.x(ex_actual)
        _, _, response_matrix = self.ops(ex_actual)

        values = np.asarray(x_draws.values, dtype=float) @ response_matrix

        return xr.DataArray(
            values,
            dims=("sample", "Eg"),
            coords=x_draws.coords,
            name="nu",
        )

    def total_expected(self, ex_actual: float) -> xr.DataArray:
        """Return total expected ON counts."""

        nu_draws = self.nu(ex_actual)
        bg_draws = self.bg(ex_actual)

        if bg_draws is None:
            return nu_draws

        values = (
            np.asarray(nu_draws.values, dtype=float)
            + np.asarray(bg_draws.values, dtype=float)
        )

        return xr.DataArray(
            values,
            dims=("sample", "Eg"),
            coords=nu_draws.coords,
            name="total_expected",
        )

    def x_true(self, ex_actual: float) -> np.ndarray:
        """Return emitted truth vector."""

        return row_vector(self.ds, "x_true", ex_actual, self.n_eg(ex_actual))

    def eta_true(self, ex_actual: float) -> np.ndarray:
        """Return eta_true = x_true @ G_g."""

        x_true = self.x_true(ex_actual)
        _, g_gamma, _ = self.ops(ex_actual)

        return x_true @ g_gamma

    def nu_true(self, ex_actual: float) -> np.ndarray:
        """Return nu_true = x_true @ D @ G_g."""

        x_true = self.x_true(ex_actual)
        _, _, response_matrix = self.ops(ex_actual)

        return x_true @ response_matrix

    def n_obs(self, ex_actual: float) -> np.ndarray:
        """Return observed ON counts."""

        return row_vector(self.ds, "n_obs", ex_actual, self.n_eg(ex_actual))

    def n_off_obs(self, ex_actual: float) -> np.ndarray | None:
        """Return observed OFF counts, if stored."""

        if "n_off_obs" not in self.ds:
            return None

        return row_vector(self.ds, "n_off_obs", ex_actual, self.n_eg(ex_actual))

    def bg_obs(self, ex_actual: float) -> np.ndarray | None:
        """Return observed OFF counts, if stored."""

        return self.n_off_obs(ex_actual)

    def x_rl(self, ex_actual: float) -> np.ndarray | None:
        """Return x-space RL reference, if stored."""

        if "x_rl" not in self.ds:
            return None

        return row_vector(self.ds, "x_rl", ex_actual, self.n_eg(ex_actual))

    def eta_rl(self, ex_actual: float) -> np.ndarray | None:
        """Return eta-space RL reference, if stored."""

        if "eta_rl" in self.ds:
            return row_vector(self.ds, "eta_rl", ex_actual, self.n_eg(ex_actual))

        x_rl = self.x_rl(ex_actual)
        if x_rl is None:
            return None

        _, g_gamma, _ = self.ops(ex_actual)
        return x_rl @ g_gamma

    def prior_sigma(self, ex_actual: float) -> np.ndarray | None:
        """Return final prior sigma schedule, if stored."""

        if "prior_sigma" not in self.ds:
            return None

        return row_vector(self.ds, "prior_sigma", ex_actual, self.n_eg(ex_actual))

    def prior_sigma_shape(self, ex_actual: float) -> np.ndarray | None:
        """Return eta-shape-only sigma schedule, if stored."""

        if "prior_sigma_shape" not in self.ds:
            return None

        return row_vector(self.ds, "prior_sigma_shape", ex_actual, self.n_eg(ex_actual))

    def rl_iteration(self, ex_actual: float) -> float:
        """Return RL iteration used for the reference, if stored."""

        return scalar_by_ex(self.ds, "rl_iteration", ex_actual)

    def rl_delta_eta(self, ex_actual: float) -> float:
        """Return eta-space RL change at the chosen iteration, if stored."""

        return scalar_by_ex(self.ds, "rl_delta_eta", ex_actual)

    def rl_noise_level(self, ex_actual: float) -> float:
        """Return eta-space RL noise level at the chosen iteration, if stored."""

        return scalar_by_ex(self.ds, "rl_noise_level", ex_actual)

    def rl_change_noise_ratio(self, ex_actual: float) -> float:
        """Return RL change-to-noise ratio at the chosen iteration, if stored."""

        return scalar_by_ex(self.ds, "rl_change_noise_ratio", ex_actual)

    def ops(self, ex_actual: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return D, G_g, and response matrix D @ G_g."""

        active_eg_length = self.n_eg(ex_actual)

        d_matrix = operator_matrix(self.ds, "D", ex_actual, active_eg_length)
        g_gamma = operator_matrix(self.ds, "G_g", ex_actual, active_eg_length)
        response_matrix = d_matrix @ g_gamma

        return d_matrix, g_gamma, response_matrix
