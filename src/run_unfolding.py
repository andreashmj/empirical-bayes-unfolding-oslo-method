"""
Run Bayesian unfolding for selected excitation-energy rows.

For each selected Ex row, this module builds the prior, resolves the sampler
configuration for that row, runs prior or posterior sampling, and stores the
resulting x-space draws in one NetCDF file.
Stored draw convention:
    x[Ex, Eg, chain, draw]
The implementation follows the OMpy right-hand convention used throughout the
code base:
    eta = x @ G_g
    nu  = x @ D @ G_g
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import ompy as om
import ompy.response
import xarray as xr

from .config_utils import (
    ProfileMapper,
    SamplerMapper,
    dict_to_namespace,
    load_prior_config,
    load_run_config,
    load_sampler_config,
)
from .input_checks import (
    require_bool,
    require_finite_array,
    require_float,
)
from .paths import repo_path
from .prior.factory import make_prior
from .pymc_unfolding import PyMCUnfolder
from .synthetic_data import SyntheticDataLoader


def _run_dir(config: argparse.Namespace) -> Path:
    output_dir = repo_path(config.output_dir)
    return output_dir / str(config.dataset_id) / str(config.run_id) / str(config.mode)


def _draws_path(config: argparse.Namespace) -> Path:
    return _run_dir(config) / "draws.nc"


def _trace_path(
    config: argparse.Namespace,
    ex_requested: float,
    n_selected: int,
) -> Path | None:
    if not require_bool(config.save_trace, "save_trace"):
        return None

    run_dir = _run_dir(config)

    if n_selected == 1:
        return run_dir / "trace.nc"

    return run_dir / "traces" / f"Ex{ex_requested:.0f}.nc"


def _as_chain_draw_eg(
    array: np.ndarray,
    active_eg_length: int,
    name: str,
) -> np.ndarray:
    """Return a draw array with shape (chain, draw, Eg)."""

    array = np.asarray(array, dtype=np.float64)

    if array.ndim != 3:
        raise ValueError(
            f"{name}: expected shape (chain, draw, Eg), got {array.shape}."
        )

    if array.shape[-1] < active_eg_length:
        raise ValueError(
            f"{name}: Eg axis is too short. "
            f"Expected at least {active_eg_length}, got {array.shape[-1]}."
        )

    return array[..., :active_eg_length]


def _stack_draws(
    arrays: list[np.ndarray],
    ex_values: np.ndarray,
    eg_axis: np.ndarray,
    active_eg_lengths: list[int],
    name: str,
) -> xr.DataArray:
    """Stack per-Ex draw arrays into one DataArray with dims (Ex, Eg, chain, draw)."""

    checked_arrays: list[np.ndarray] = []
    n_chains = None
    n_draws = None

    for ex_index, array in enumerate(arrays):
        active_eg_length = active_eg_lengths[ex_index]
        draw_array = _as_chain_draw_eg(
            array,
            active_eg_length,
            f"{name}[{ex_index}]",
        )

        current_chains, current_draws = draw_array.shape[:2]

        if n_chains is None:
            n_chains = current_chains
        elif current_chains != n_chains:
            raise ValueError(
                f"{name}: inconsistent chain count at Ex row {ex_index}: "
                f"{current_chains} vs {n_chains}."
            )

        if n_draws is None:
            n_draws = current_draws
        elif current_draws != n_draws:
            raise ValueError(
                f"{name}: inconsistent draw count at Ex row {ex_index}: "
                f"{current_draws} vs {n_draws}."
            )

        checked_arrays.append(draw_array)

    if n_chains is None or n_draws is None:
        raise ValueError(f"{name}: no arrays to stack.")

    data = np.full(
        (len(ex_values), len(eg_axis), n_chains, n_draws),
        np.nan,
        dtype=np.float64,
    )

    for ex_index, draw_array in enumerate(checked_arrays):
        active_eg_length = active_eg_lengths[ex_index]

        # Sampler output is (chain, draw, Eg). The stored convention is
        # (Ex, Eg, chain, draw), so each Ex block must be (Eg, chain, draw).
        eg_chain_draw_block = np.moveaxis(
            draw_array[:, :, :active_eg_length],
            source=2,
            destination=0,
        )

        data[ex_index, :active_eg_length, :, :] = eg_chain_draw_block

    return xr.DataArray(
        data,
        dims=("Ex", "Eg", "chain", "draw"),
        coords={
            "Ex": ex_values,
            "Eg": eg_axis,
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
        },
        name=name,
    )


def _stack_vectors(
    vectors: list[np.ndarray | None],
    ex_values: np.ndarray,
    eg_axis: np.ndarray,
    active_eg_lengths: list[int],
    name: str,
) -> xr.DataArray:
    """Stack per-Ex vectors into one DataArray with dims (Ex, Eg)."""

    data = np.full((len(ex_values), len(eg_axis)), np.nan, dtype=np.float64)

    for ex_index, vector in enumerate(vectors):
        if vector is None:
            continue

        active_eg_length = active_eg_lengths[ex_index]
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)

        if vector.size < active_eg_length:
            raise ValueError(
                f"{name}: vector at Ex row {ex_index} is too short. "
                f"Expected at least {active_eg_length}, got {vector.size}."
            )

        data[ex_index, :active_eg_length] = vector[:active_eg_length]

    return xr.DataArray(
        data,
        dims=("Ex", "Eg"),
        coords={"Ex": ex_values, "Eg": eg_axis},
        name=name,
    )


def _stack_operators(
    operators: list[np.ndarray],
    ex_values: np.ndarray,
    eg_axis: np.ndarray,
    active_eg_lengths: list[int],
    name: str,
) -> xr.DataArray:
    """Stack per-Ex operator matrices into one DataArray with dims (Ex, Eg_in, Eg_out)."""

    data = np.full(
        (len(ex_values), len(eg_axis), len(eg_axis)),
        np.nan,
        dtype=np.float64,
    )

    for ex_index, operator in enumerate(operators):
        active_eg_length = active_eg_lengths[ex_index]
        operator = np.asarray(operator, dtype=np.float64)

        if operator.ndim != 2:
            raise ValueError(f"{name}: expected 2D operator, got {operator.shape}.")

        if operator.shape[0] < active_eg_length or operator.shape[1] < active_eg_length:
            raise ValueError(
                f"{name}: operator at Ex row {ex_index} is too small. "
                f"Expected at least ({active_eg_length}, {active_eg_length}), "
                f"got {operator.shape}."
            )

        data[
            ex_index,
            :active_eg_length,
            :active_eg_length,
        ] = operator[:active_eg_length, :active_eg_length]

    return xr.DataArray(
        data,
        dims=("Ex", "Eg_in", "Eg_out"),
        coords={"Ex": ex_values, "Eg_in": eg_axis, "Eg_out": eg_axis},
        name=name,
    )


def _background_b0_from_off_counts(
    n_off_observed: np.ndarray,
    background_alpha: float,
) -> float:
    """Return the Gamma prior rate b0 used by the background model."""

    n_off_observed = require_finite_array(
        n_off_observed,
        "n_off_observed",
        ndim=1,
        nonnegative=True,
    )
    background_alpha = require_float(
        background_alpha,
        "background_alpha",
        minimum=0.0,
        minimum_inclusive=False,
    )

    mean_off = float(np.mean(n_off_observed))

    if mean_off <= 0.0:
        raise ValueError(
            "Cannot set the background Gamma rate because the mean OFF count "
            "is not positive."
        )

    return background_alpha / mean_off


def _background_postmean_from_off_counts(
    n_off_observed: np.ndarray,
    background_alpha: float,
    background_b0: float,
) -> np.ndarray:
    """Return the Gamma--Poisson posterior mean background from OFF counts."""

    n_off_observed = require_finite_array(
        n_off_observed,
        "n_off_observed",
        ndim=1,
        nonnegative=True,
    )
    background_alpha = require_float(
        background_alpha,
        "background_alpha",
        minimum=0.0,
        minimum_inclusive=False,
    )
    background_b0 = require_float(
        background_b0,
        "background_b0",
        minimum=0.0,
        minimum_inclusive=False,
    )

    return (background_alpha + n_off_observed) / (background_b0 + 1.0)


def _sampler_backend_from_config(sample_config: dict) -> str:
    """Return the configured NUTS backend name."""

    nuts_config = sample_config.get("nuts", {})
    return str(nuts_config.get("nuts_sampler", "pymc")).strip().lower()


class UnfoldManager:
    """Run unfolding for the Ex rows selected by the run configuration."""

    def __init__(self, data_loader: SyntheticDataLoader, config: argparse.Namespace) -> None:
        self.data_loader = data_loader
        self.config = config

        self.debug = require_bool(config.debug, "debug")
        self.include_background = require_bool(
            config.include_background,
            "include_background",
        )

        self.n_matrix = data_loader.n
        self.n_off_matrix = data_loader.n_off if self.include_background else None
        self.eg_axis = data_loader.Eg
        self.ex_axis = data_loader.Ex
        self.x_true_matrix = data_loader.x_true

        if self.include_background:
            self.background_model_kind = str(config.background_model.kind).strip()
            self.background_alpha = require_float(
                config.background_model.a0,
                "background_model.a0",
                minimum=0.0,
                minimum_inclusive=False,
            )

            if self.background_model_kind not in {"latent_gamma", "fixed_postmean"}:
                raise ValueError(
                    "background_model.kind must be either "
                    "'latent_gamma' or 'fixed_postmean'."
                )
        else:
            self.background_model_kind = "none"
            self.background_alpha = np.nan

        self.response_db = str(config.response_db)
        self.sigma_eg_gen = str(data_loader.resp_norm_sigma)
        self.sigma_eg_unfold = str(config.sigma_eg_unfold)

        if self.sigma_eg_unfold == self.sigma_eg_gen:
            self.response_unfold = data_loader.resp
        else:
            self.response_unfold = (
                om.response.Response.from_db(self.response_db)
                .normalize_sigma(data_loader.resp_norm_anchor, self.sigma_eg_unfold)
            )

        self.ex_requested = self._selected_ex_values(config)

        if not self.ex_requested:
            raise ValueError("No Ex rows were selected for unfolding.")

        self.selected_indices = [self.n_matrix.index_X(value) for value in self.ex_requested]

        self.prior_config_path = repo_path(config.prior_config)
        prior_config = load_prior_config(self.prior_config_path)
        self.prior_mapper = ProfileMapper(
            profiles=prior_config["profiles"],
            mapping=prior_config["mapping"],
        )

        self.sample_config_path = repo_path(config.sample_config)
        self.sample_config_template = load_sampler_config(self.sample_config_path)
        self.sampler_mapper = SamplerMapper(self.sample_config_template)
        self.sampler_backend = _sampler_backend_from_config(self.sample_config_template)

        self.x_draws_by_ex: list[np.ndarray] = []
        self.background_draws_by_ex: list[np.ndarray | None] = []

        self.x_true_by_ex: list[np.ndarray] = []
        self.x_rl_by_ex: list[np.ndarray | None] = []
        self.eta_rl_by_ex: list[np.ndarray | None] = []
        self.prior_sigma_by_ex: list[np.ndarray | None] = []
        self.prior_sigma_shape_by_ex: list[np.ndarray | None] = []

        self.rl_iteration_by_ex: list[float] = []
        self.rl_delta_eta_by_ex: list[float] = []
        self.rl_noise_level_by_ex: list[float] = []
        self.rl_change_noise_ratio_by_ex: list[float] = []

        self.n_observed_by_ex: list[np.ndarray] = []
        self.n_off_observed_by_ex: list[np.ndarray | None] = []
        self.background_b0_by_ex: list[float] = []

        self.D_by_ex: list[np.ndarray] = []
        self.G_g_by_ex: list[np.ndarray] = []

        self.ex_requested_values: list[float] = []
        self.ex_actual_values: list[float] = []
        self.active_eg_lengths: list[int] = []

        self.sampling_time_s_by_ex: list[float] = []
        self.target_accept_by_ex: list[float] = []
        self.max_treedepth_by_ex: list[int] = []
        self.nuts_random_seed_by_ex: list[int] = []

    def _selected_ex_values(self, config: argparse.Namespace) -> list[float]:
        """Return requested Ex values from the run configuration."""

        if getattr(config, "ex_energies", None) is not None:
            return [float(value) for value in config.ex_energies]

        if getattr(config, "ex_start_energy", None) is not None:
            ex_start = float(config.ex_start_energy)
            ex_end = float(config.ex_end_energy)

            if ex_start > ex_end:
                raise ValueError("ex_start_energy must be <= ex_end_energy.")

            ex_axis = np.asarray(self.ex_axis, dtype=float)
            mask = (ex_axis >= ex_start) & (ex_axis <= ex_end)

            return [float(value) for value in ex_axis[mask]]

        return [float(value) for value in np.asarray(self.ex_axis, dtype=float)]

    def run_unfolding(self) -> None:
        """Run unfolding and write the result NetCDF file."""

        if self.debug:
            print("\n=== Starting unfolding run ===")

        run_dir = _run_dir(self.config)
        draws_path = _draws_path(self.config)
        run_dir.mkdir(parents=True, exist_ok=True)

        mode = str(self.config.mode).lower().strip()

        if mode not in {"prior", "posterior"}:
            raise ValueError("mode must be either 'prior' or 'posterior'.")

        for row_index, ex_index in enumerate(self.selected_indices):
            self._run_one_ex_unfolding(row_index, ex_index, mode)

        self.finalize_and_save(draws_path)

    def _background_reference_for_rl(
        self,
        n_observed: np.ndarray,
        n_off_observed: np.ndarray | None,
        background_b0: float,
    ) -> np.ndarray:
        """Return the fixed background reference used by the RL prior reference."""

        if not self.include_background:
            return np.zeros_like(n_observed, dtype=np.float64)

        if n_off_observed is None:
            raise ValueError("Background-aware RL requires n_off_observed.")

        return _background_postmean_from_off_counts(
            n_off_observed,
            self.background_alpha,
            background_b0,
        )
    

    def _run_one_ex_unfolding(self, row_index: int, ex_index: int, mode: str) -> None:
        """Run unfolding for one selected Ex row."""

        ex_requested = float(self.ex_requested[row_index])
        ex_actual = float(self.ex_axis[ex_index])

        active_eg_slice = self.data_loader.eg_slice_for_index(ex_index)
        active_eg_length = active_eg_slice.stop

        self.ex_requested_values.append(ex_requested)
        self.ex_actual_values.append(ex_actual)
        self.active_eg_lengths.append(active_eg_length)

        n_cut = self.n_matrix.iloc[ex_index, active_eg_slice]
        n_observed = require_finite_array(
            n_cut.values,
            "n_observed",
            ndim=1,
            nonnegative=True,
        )
        self.n_observed_by_ex.append(n_observed)

        n_off_observed = None
        background_b0 = np.nan

        if self.include_background:
            n_off_observed = require_finite_array(
                self.n_off_matrix.iloc[ex_index, active_eg_slice].values,
                "n_off_observed",
                ndim=1,
                nonnegative=True,
            )
            background_b0 = _background_b0_from_off_counts(
                n_off_observed,
                self.background_alpha,
            )
            self.n_off_observed_by_ex.append(n_off_observed)
        else:
            self.n_off_observed_by_ex.append(None)

        self.background_b0_by_ex.append(float(background_b0))

        background_reference = self._background_reference_for_rl(
            n_observed=n_observed,
            n_off_observed=n_off_observed,
            background_b0=background_b0,
        )

        x_true = require_finite_array(
            self.x_true_matrix.iloc[ex_index, active_eg_slice].values,
            "x_true",
            ndim=1,
            nonnegative=True,
        )
        self.x_true_by_ex.append(x_true)

        D_cut, G_g_cut = self.response_unfold.specialize_like(n_cut)

        D = np.asarray(D_cut.values, dtype=np.float64)
        G_g = np.asarray(G_g_cut.values, dtype=np.float64)
        response_matrix = D @ G_g

        self.D_by_ex.append(D)
        self.G_g_by_ex.append(G_g)

        prior_profile = copy.deepcopy(self.prior_mapper.profile_for(ex_requested))

        prior_config = make_prior(
            prior_profile,
            response_matrix=response_matrix,
            gamma_resolution_matrix=G_g,
            n_on=n_observed,
            background_reference=background_reference,
        )
        
        self._store_prior_outputs(prior_config)

        sample_config = self.sampler_mapper.config_for(ex_requested)
        sample_config["transform"] = str(self.config.transform)
        sample_config["background_model"] = {
            "kind": self.background_model_kind,
            "a0": self.background_alpha,
        }

        self.target_accept_by_ex.append(float(sample_config["nuts"]["target_accept"]))
        self.max_treedepth_by_ex.append(int(sample_config["nuts"]["max_treedepth"]))
        self.nuts_random_seed_by_ex.append(int(sample_config["nuts"]["random_seed"]))

        if self.debug:
            print(
                f"\n[Ex_req {ex_requested:.0f} keV] -> Ex={ex_actual:.0f} keV | "
                f"Eg bins={active_eg_length} | "
                f"background={self.include_background} | mode={mode} | "
                f"backend={self.sampler_backend} | "
                f"target_accept={sample_config['nuts']['target_accept']}"
            )

            rl_iteration = self.rl_iteration_by_ex[-1]
            rl_change_noise_ratio = self.rl_change_noise_ratio_by_ex[-1]

            if np.isfinite(rl_iteration):
                print(
                    f"[prior] RL iteration={rl_iteration:.0f}, "
                    f"change/noise={rl_change_noise_ratio:.4g}"
                )
        trace_path = _trace_path(
            self.config,
            ex_requested,
            len(self.selected_indices),
        )

        start_time = time.perf_counter()

        unfolder = PyMCUnfolder(
            prior_config=prior_config,
            sample_config=sample_config,
            debug=self.debug,
        )

        x_draws, background_draws = unfolder.unfold(
            n_obs=n_observed,
            n_off_obs=n_off_observed,
            D=D,
            G_g=G_g,
            mode=mode,
            include_background=self.include_background,
            output_trace_path=trace_path,
            debug=self.debug,
        )

        sampling_time_s = time.perf_counter() - start_time
        self.sampling_time_s_by_ex.append(sampling_time_s)

        self.x_draws_by_ex.append(
            _as_chain_draw_eg(
                x_draws,
                active_eg_length,
                f"x_draws Ex={ex_requested:.0f}",
            )
        )

        if self.include_background and background_draws is not None:
            self.background_draws_by_ex.append(
                _as_chain_draw_eg(
                    background_draws,
                    active_eg_length,
                    f"background_draws Ex={ex_requested:.0f}",
                )
            )
        else:
            self.background_draws_by_ex.append(None)

        if self.debug:
            print(
                f"[timing] Ex_req {ex_requested:.0f} keV: "
                f"sampling_time_s={sampling_time_s:.1f}"
            )

    def _store_prior_outputs(self, prior_config: dict) -> None:
        """Store prior reference vectors and RL diagnostics."""

        x_rl = prior_config.get("x_rl", None)
        eta_rl = prior_config.get("eta_rl", None)
        prior_sigma = prior_config.get("sigma", None)
        prior_sigma_shape = prior_config.get("sigma_shape", None)

        self.x_rl_by_ex.append(None if x_rl is None else np.asarray(x_rl, dtype=np.float64))
        self.eta_rl_by_ex.append(None if eta_rl is None else np.asarray(eta_rl, dtype=np.float64))
        self.prior_sigma_by_ex.append(
            None if prior_sigma is None else np.asarray(prior_sigma, dtype=np.float64)
        )
        self.prior_sigma_shape_by_ex.append(
            None
            if prior_sigma_shape is None
            else np.asarray(prior_sigma_shape, dtype=np.float64)
        )

        metadata = prior_config.get("meta", {}) or {}

        rl_iteration = metadata.get("rl_iteration", None)
        rl_delta_eta = metadata.get("rl_delta_eta", None)
        rl_noise_level = metadata.get("rl_noise_level", None)
        rl_change_noise_ratio = metadata.get("rl_change_noise_ratio", None)

        self.rl_iteration_by_ex.append(
            np.nan if rl_iteration is None else float(rl_iteration)
        )
        self.rl_delta_eta_by_ex.append(
            np.nan if rl_delta_eta is None else float(rl_delta_eta)
        )
        self.rl_noise_level_by_ex.append(
            np.nan if rl_noise_level is None else float(rl_noise_level)
        )
        self.rl_change_noise_ratio_by_ex.append(
            np.nan if rl_change_noise_ratio is None else float(rl_change_noise_ratio)
        )

    def finalize_and_save(self, result_nc_path: str | Path) -> None:
        """Collect per-Ex results and write one NetCDF file."""

        result_nc_path = Path(result_nc_path)
        result_nc_path.parent.mkdir(parents=True, exist_ok=True)

        max_active_eg_length = max(self.active_eg_lengths)
        eg_axis = np.asarray(self.eg_axis[:max_active_eg_length], dtype=float)

        ex_actual_values = np.asarray(self.ex_actual_values, dtype=float)
        ex_requested_values = np.asarray(self.ex_requested_values, dtype=float)
        active_eg_lengths = np.asarray(self.active_eg_lengths, dtype=np.int64)

        dataset_variables = {
            "x": _stack_draws(
                self.x_draws_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "x",
            ),
            "x_true": _stack_vectors(
                self.x_true_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "x_true",
            ),
            "n_obs": _stack_vectors(
                self.n_observed_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "n_obs",
            ),
            "D": _stack_operators(
                self.D_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "D",
            ),
            "G_g": _stack_operators(
                self.G_g_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "G_g",
            ),
        }

        if self.include_background:
            background_draw_arrays: list[np.ndarray] = []

            for ex_index, background_draws in enumerate(self.background_draws_by_ex):
                if background_draws is None:
                    reference = _as_chain_draw_eg(
                        self.x_draws_by_ex[ex_index],
                        int(active_eg_lengths[ex_index]),
                        f"x[{ex_index}]",
                    )
                    background_draw_arrays.append(
                        np.full_like(reference, np.nan, dtype=np.float64)
                    )
                else:
                    background_draw_arrays.append(background_draws)

            dataset_variables["bg_exp"] = _stack_draws(
                background_draw_arrays,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "bg_exp",
            )

            dataset_variables["n_off_obs"] = _stack_vectors(
                self.n_off_observed_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "n_off_obs",
            )

        if any(value is not None for value in self.x_rl_by_ex):
            dataset_variables["x_rl"] = _stack_vectors(
                self.x_rl_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "x_rl",
            )

        if any(value is not None for value in self.eta_rl_by_ex):
            dataset_variables["eta_rl"] = _stack_vectors(
                self.eta_rl_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "eta_rl",
            )

        if any(value is not None for value in self.prior_sigma_by_ex):
            dataset_variables["prior_sigma"] = _stack_vectors(
                self.prior_sigma_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "prior_sigma",
            )

        if any(value is not None for value in self.prior_sigma_shape_by_ex):
            dataset_variables["prior_sigma_shape"] = _stack_vectors(
                self.prior_sigma_shape_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "prior_sigma_shape",
            )

        ds = xr.Dataset(dataset_variables).assign_coords(
            Ex=ex_actual_values,
            Eg=eg_axis,
        )

        ds["Ex_req"] = xr.DataArray(ex_requested_values, dims=("Ex",))
        ds["Eg_len"] = xr.DataArray(active_eg_lengths, dims=("Ex",))
        ds["sampling_time_s"] = xr.DataArray(
            np.asarray(self.sampling_time_s_by_ex, dtype=np.float64),
            dims=("Ex",),
        )
        ds["target_accept"] = xr.DataArray(
            np.asarray(self.target_accept_by_ex, dtype=np.float64),
            dims=("Ex",),
        )
        ds["max_treedepth"] = xr.DataArray(
            np.asarray(self.max_treedepth_by_ex, dtype=np.int64),
            dims=("Ex",),
        )
        ds["nuts_random_seed"] = xr.DataArray(
            np.asarray(self.nuts_random_seed_by_ex, dtype=np.int64),
            dims=("Ex",),
        )

        ds["rl_iteration"] = xr.DataArray(
            np.asarray(self.rl_iteration_by_ex, dtype=np.float64),
            dims=("Ex",),
        )
        ds["rl_delta_eta"] = xr.DataArray(
            np.asarray(self.rl_delta_eta_by_ex, dtype=np.float64),
            dims=("Ex",),
        )
        ds["rl_noise_level"] = xr.DataArray(
            np.asarray(self.rl_noise_level_by_ex, dtype=np.float64),
            dims=("Ex",),
        )
        ds["rl_change_noise_ratio"] = xr.DataArray(
            np.asarray(self.rl_change_noise_ratio_by_ex, dtype=np.float64),
            dims=("Ex",),
        )

        if self.include_background:
            ds["bg_b0"] = xr.DataArray(
                np.asarray(self.background_b0_by_ex, dtype=np.float64),
                dims=("Ex",),
            )

        self._add_dataset_metadata(ds)

        ds.to_netcdf(str(result_nc_path), engine="h5netcdf")
        print(f"Saved draws -> {result_nc_path}")

    def _add_dataset_metadata(self, ds: xr.Dataset) -> None:
        """Add run and data-generation metadata to the output dataset."""

        ds.attrs["run_id"] = str(self.config.run_id)
        ds.attrs["mode"] = str(self.config.mode)
        ds.attrs["include_background"] = "true" if self.include_background else "false"
        ds.attrs["background_model"] = self.background_model_kind
        ds.attrs["bg_a0"] = (
            float(self.background_alpha) if self.include_background else "none"
        )
        ds.attrs["likelihood"] = "poisson"
        ds.attrs["transform"] = str(self.config.transform)
        ds.attrs["sampler_backend"] = self.sampler_backend

        ds.attrs["response_db"] = self.response_db
        ds.attrs["sigma_eg_gen"] = self.sigma_eg_gen
        ds.attrs["sigma_eg_unfold"] = self.sigma_eg_unfold

        ds.attrs["prior_config"] = str(self.prior_config_path)
        ds.attrs["sample_config"] = str(self.sample_config_path)

        if "random_seed" in self.sample_config_template["prior"]:
            ds.attrs["prior_random_seed"] = int(
                self.sample_config_template["prior"]["random_seed"]
            )

        for key, value in self.data_loader.metadata.items():
            attr_key = f"loader_{key}"

            if value is None:
                ds.attrs[attr_key] = "none"
            elif isinstance(value, (bool, np.bool_)):
                ds.attrs[attr_key] = "true" if bool(value) else "false"
            elif isinstance(value, (tuple, list)):
                ds.attrs[attr_key] = ",".join(str(item) for item in value)
            elif isinstance(value, np.integer):
                ds.attrs[attr_key] = int(value)
            elif isinstance(value, np.floating):
                ds.attrs[attr_key] = float(value)
            else:
                ds.attrs[attr_key] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Bayesian unfolding. Chains are always stored separately."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML run config.")
    cli = parser.parse_args()

    cfg_dict = load_run_config(str(repo_path(cli.config)))
    config = dict_to_namespace(cfg_dict)

    loader = SyntheticDataLoader(
        mat_path=config.mat_path,
        response_db=config.response_db,
        rebin_factors=tuple(config.rebin_factors),
        mat_scale=config.mat_scale,
        lower_eg_cut=config.lower_eg_cut,
        include_background=config.include_background,
        rng_seed=config.rng_seed,
        sigma_eg=config.sigma_eg_gen,
        bg_fraction=config.bg_fraction,
        bg_flat_fraction=config.bg_flat_fraction,
        eg_tail_mass=config.eg_tail_mass,
        include_ex_smearing=config.include_ex_smearing,
        fwhm_ex_kev=config.fwhm_ex_kev,
    )

    manager = UnfoldManager(loader, config=config)
    manager.run_unfolding()


if __name__ == "__main__":
    main()
