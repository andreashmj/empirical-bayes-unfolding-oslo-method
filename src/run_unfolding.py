"""
Run empirical-Bayes unfolding method for the excitation-energies given by the run configuration file. 
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
    dict_to_namespace,
    load_prior_config,
    load_run_config,
    load_sampler_config
)
from .input_checks import (
    require_bool,
    require_finite_array,
    require_float
)
from .paths import bayes_run_dir, repo_path
from .prior.factory import make_prior
from .pymc_unfolding import PyMCUnfolder
from .synthetic_data import SyntheticDataLoader
from .experimental_data import ExperimentalDataLoader

# TODO: 
# - Structure the code so that unfolding for different Ex bins can be run in parallel
# - Adaptive bin width on the Eg axis


def _run_dir(config: argparse.Namespace) -> Path:
    return bayes_run_dir(vars(config))


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

    array = np.asarray(array, dtype=np.float64)

    if array.ndim != 3:
        raise ValueError(
            f"{name}: expected shape (chain, draw, Eg), got {array.shape}.")
    if array.shape[-1] < active_eg_length:
        raise ValueError(
            f"{name}: Eg axis is too short. "
            f"Expected at least {active_eg_length}, got {array.shape[-1]}." )

    return array[..., :active_eg_length]


def _stack_draws(
    arrays: list[np.ndarray],
    ex_values: np.ndarray,
    eg_axis: np.ndarray,
    active_eg_lengths: list[int],
    name: str
) -> xr.DataArray:


    checked_arrays: list[np.ndarray] = []
    n_chains = None
    n_draws = None

    for ex_index, array in enumerate(arrays):
        active_eg_length = active_eg_lengths[ex_index]
        draw_array = _as_chain_draw_eg(
            array,
            active_eg_length,
            f"{name}[{ex_index}]")

        current_chains, current_draws = draw_array.shape[:2]

        if n_chains is None:
            n_chains = current_chains
        elif current_chains != n_chains:
            raise ValueError(f"{name}: inconsistent chain count at Ex row {ex_index}: {current_chains} vs {n_chains}.")

        if n_draws is None:
            n_draws = current_draws
        elif current_draws != n_draws:
            raise ValueError(f"{name}: inconsistent draw count at Ex row {ex_index}: {current_draws} vs {n_draws}.")

        checked_arrays.append(draw_array)

    if n_chains is None or n_draws is None:
        raise ValueError(f"{name}: no arrays to stack.")

    data = np.full((len(ex_values), len(eg_axis), n_chains, n_draws), np.nan, dtype=np.float64)

    for ex_index, draw_array in enumerate(checked_arrays):
        active_eg_length = active_eg_lengths[ex_index]

        eg_chain_draw_block = np.moveaxis(
            draw_array[:, :, :active_eg_length],
            source=2,
            destination=0
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
        name=name
        )


def _stack_vectors(
    vectors: list[np.ndarray | None],
    ex_values: np.ndarray,
    eg_axis: np.ndarray,
    active_eg_lengths: list[int],
    name: str
) -> xr.DataArray:

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
        name=name)


def _stack_operators(
    operators: list[np.ndarray],
    ex_values: np.ndarray,
    eg_axis: np.ndarray,
    active_eg_lengths: list[int],
    name: str
) -> xr.DataArray:

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
        name=name
    )

def background_reference_from_off(
    n_off_obs: np.ndarray,
    a0: float,
) -> tuple[np.ndarray, float]:

    mean_off = float(np.mean(n_off_obs))

    if mean_off <= 0.0:
        raise ValueError("Mean OFF count must be positive.")

    b0 = a0 / mean_off
    background_reference = (a0 + n_off_obs) / (b0 + 1.0)

    return background_reference, b0
    
    
def target_accept_for_ex(
    ex: float,
    nuts_config: dict,
) -> float:

    for rule in nuts_config.get("target_accept_by_ex", []):
        lower, upper = map(float, rule["range"])

        if lower <= ex < upper:
            return float(rule["value"])

    return float(nuts_config["target_accept"])




class UnfoldManager:
    """
    Unfolding run for the Ex bins given by the run configuration.
    
    """

    def __init__(self, data_loader, config: argparse.Namespace) -> None:
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
        
        self.has_truth = bool(getattr(data_loader, "has_truth", True))

        self.reference_matrices = {}

        if self.has_truth:
            x_true = data_loader.x_true
            G_in = data_loader.G_in

            x_mixed = x_true if G_in is None else G_in @ x_true

            self.reference_matrices = {
                "x_true": x_true,
                "x_mixed_true": x_mixed,
                "eta_true": data_loader.eta,
                "nu_true": data_loader.nu,
            }

        self.reference_by_ex = {name: [] for name in self.reference_matrices }
        

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
                    "background_model.kind must be either latent_gamma or fixed_postmean." )
        else:
            self.background_model_kind = "none"
            self.background_alpha = np.nan

        self.response_db = str(config.response_db)
        self.fwhm_eg_unfold = str(config.fwhm_eg_unfold)

        if not self.has_truth or self.fwhm_eg_unfold == data_loader.fwhm_eg_gen:
            self.response_unfold = data_loader.resp
        else:
            self.response_unfold = (
                om.response.Response.from_db(self.response_db)
                .normalize_FWHM("1330keV", self.fwhm_eg_unfold)
            )

        self.ex_requested = self._selected_ex_values(config)

        if not self.ex_requested:
            raise ValueError("No Ex bins were chosen for unfolding.")

        self.selected_indices = [self.n_matrix.index_X(value) for value in self.ex_requested]

        self.prior_config_path = repo_path(config.prior_config)
        self.prior_settings = load_prior_config(self.prior_config_path)

        self.sample_config_path = repo_path(config.sample_config)
        self.sample_config_template = load_sampler_config(self.sample_config_path)

        self.sampler_backend = str(self.sample_config_template["nuts"].get("nuts_sampler", "pymc")).strip().lower()

        self.x_draws_by_ex: list[np.ndarray] = []
        self.background_draws_by_ex: list[np.ndarray | None] = []

        
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
        

    def _selected_ex_values(self, config: argparse.Namespace) -> list[float]:
    

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

    def run_unfolding(self, save: bool = True) -> None:

        if self.debug:
            print("\n----Starting unfolding run----")

        if save:
            _run_dir(self.config).mkdir(parents=True, exist_ok=True)

        mode = str(self.config.mode).lower().strip()

        if mode not in {"prior", "posterior"}:
            raise ValueError("mode must be either prior or posterior.")

        for row_index, ex_index in enumerate(self.selected_indices):
            self._run_one_ex_unfolding(row_index, ex_index, mode)

        if save:
            self.finalize_and_save(_draws_path(self.config))

    
    

    def _run_one_ex_unfolding(self, row_index: int, ex_index: int, mode: str) -> None:

        ex_requested = float(self.ex_requested[row_index])
        ex_actual = float(self.ex_axis[ex_index])
        
        ex_index_axis = self.n_matrix.X_index

        ex_left = float(ex_index_axis.left_edge(ex_index))
        ex_right = float(ex_index_axis.right_edge(ex_index))
        ex_center = float(ex_index_axis.mid(ex_index))

        active_eg_slice = self.data_loader.eg_slice_for_index(ex_index)
        active_eg_length = active_eg_slice.stop

        self.ex_requested_values.append(ex_requested)
        self.ex_actual_values.append(ex_actual)
        self.active_eg_lengths.append(active_eg_length)

        n_cut = self.n_matrix.iloc[ex_index, active_eg_slice]
        n_obs = require_finite_array(
            n_cut.values,
            "n_obs",
            ndim=1,
            nonnegative=True,
        )
        self.n_observed_by_ex.append(n_obs)

        n_off_obs = None
        background_b0 = np.nan

        if self.include_background:
            n_off_obs = np.asarray(
                self.n_off_matrix.iloc[ex_index, active_eg_slice].values,
                dtype=np.float64,
            )
            background_reference, background_b0 = background_reference_from_off(
                n_off_obs, self.background_alpha
            )
        else:
            background_reference = np.zeros_like(n_obs, dtype=np.float64)

        self.n_off_observed_by_ex.append(n_off_obs)
        self.background_b0_by_ex.append(background_b0)


        for name, matrix in self.reference_matrices.items():
            values = np.asarray(
                matrix.iloc[ex_index, active_eg_slice].values, dtype=np.float64
            )
            self.reference_by_ex[name].append(values)

        D_cut, G_g_cut = self.response_unfold.specialize_like(n_cut)

        D = np.asarray(D_cut.values, dtype=np.float64)
        G_g = np.asarray(G_g_cut.values, dtype=np.float64)
        R = D @ G_g

        self.D_by_ex.append(D)
        self.G_g_by_ex.append(G_g)

        prior_config = make_prior(
            settings=self.prior_settings,
            R=R,
            G_g=G_g,
            n_obs=n_obs,
            background_reference=background_reference,
        )
                
        self._store_prior_outputs(prior_config)

        sample_config = copy.deepcopy(self.sample_config_template)
        nuts_config = sample_config["nuts"]

        nuts_config["target_accept"] = target_accept_for_ex(ex_requested, nuts_config)
        nuts_config.pop("target_accept_by_ex", None)

        sample_config["transform"] = str(self.config.transform)
        sample_config["background_model"] = {
            "kind": self.background_model_kind,
            "a0": self.background_alpha,
        }

        self.target_accept_by_ex.append(float(sample_config["nuts"]["target_accept"]))
        
        if self.debug:
            print(
                f"\n[Ex_req {ex_requested:.0f} keV] "
                f"-> Ex bin=[{ex_left:.0f}, {ex_right:.0f}) keV "
                f"(center={ex_center:.0f} keV) | "
                f"Eg bins={active_eg_length} | "
                f"background={self.include_background} | mode={mode} | "
                f"backend={self.sampler_backend} | "
                f"target_accept={sample_config['nuts']['target_accept']}"
            )
        

            rl_iteration = self.rl_iteration_by_ex[-1]
            rl_change_noise_ratio = self.rl_change_noise_ratio_by_ex[-1]

            if np.isfinite(rl_iteration):
                print(
                    f"RL iteration={rl_iteration:.0f}, "
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
            n_obs=n_obs,
            n_off_obs=n_off_obs,
            R=R,
            mode=mode,
            include_background=self.include_background,
            output_trace_path=trace_path
        )

        sampling_time_s = time.perf_counter() - start_time
        self.sampling_time_s_by_ex.append(sampling_time_s)

        self.x_draws_by_ex.append(x_draws)

        if self.include_background and background_draws is not None:
            self.background_draws_by_ex.append(background_draws)
        else:
            self.background_draws_by_ex.append(None)

        if self.debug:
            print(
                f"Ex_req {ex_requested:.0f} keV: "
                f"sampling_time_s={sampling_time_s:.1f}"
            )

    def _store_prior_outputs(self, prior_config: dict) -> None:

        x_rl = prior_config.get("x_rl", None)
        eta_rl = prior_config.get("eta_rl", None)
        prior_sigma = prior_config.get("sigma", None)
        prior_sigma_shape = prior_config.get("sigma_shape", None)

        self.x_rl_by_ex.append(None if x_rl is None else np.asarray(x_rl, dtype=np.float64))
        self.eta_rl_by_ex.append(None if eta_rl is None else np.asarray(eta_rl, dtype=np.float64))
        self.prior_sigma_by_ex.append(None if prior_sigma is None else np.asarray(prior_sigma, dtype=np.float64))
        self.prior_sigma_shape_by_ex.append(
            None if prior_sigma_shape is None
            else np.asarray(prior_sigma_shape, dtype=np.float64)
        )

        rl_details = prior_config.get("rl_details", {})

        rl_iteration = rl_details.get("rl_iteration")
        rl_delta_eta = rl_details.get("rl_delta_eta")
        rl_noise_level = rl_details.get("rl_noise_level")
        rl_change_noise_ratio = rl_details.get("rl_change_noise_ratio")
        
        self.rl_iteration_by_ex.append(np.nan if rl_iteration is None else float(rl_iteration))
        self.rl_delta_eta_by_ex.append(np.nan if rl_delta_eta is None else float(rl_delta_eta))
        self.rl_noise_level_by_ex.append(np.nan if rl_noise_level is None else float(rl_noise_level))
        self.rl_change_noise_ratio_by_ex.append(np.nan if rl_change_noise_ratio is None else float(rl_change_noise_ratio))

    def finalize_and_save(self, result_nc_path: str | Path) -> None:

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
                "x"
            ),
            "n_obs": _stack_vectors(
                self.n_observed_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "n_obs"
            ),
            "D": _stack_operators(
                self.D_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "D"
            ),
            "G_g": _stack_operators(
                self.G_g_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "G_g"
            )
        }



        for name, vectors in self.reference_by_ex.items():
            dataset_variables[name] = _stack_vectors(
                vectors,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                name
            )

        if self.include_background:
            background_draw_arrays: list[np.ndarray] = []

            for ex_index, background_draws in enumerate(self.background_draws_by_ex):
                if background_draws is None:
                    reference = _as_chain_draw_eg(
                        self.x_draws_by_ex[ex_index],
                        int(active_eg_lengths[ex_index]),
                        f"x[{ex_index}]"
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
                "bg_exp"
            )

            dataset_variables["n_off_obs"] = _stack_vectors(
                self.n_off_observed_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "n_off_obs"
            )

        if any(value is not None for value in self.x_rl_by_ex):
            dataset_variables["x_rl"] = _stack_vectors(
                self.x_rl_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "x_rl"
            )

        if any(value is not None for value in self.eta_rl_by_ex):
            dataset_variables["eta_rl"] = _stack_vectors(
                self.eta_rl_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "eta_rl"
            )

        if any(value is not None for value in self.prior_sigma_by_ex):
            dataset_variables["prior_sigma"] = _stack_vectors(
                self.prior_sigma_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "prior_sigma"
            )

        if any(value is not None for value in self.prior_sigma_shape_by_ex):
            dataset_variables["prior_sigma_shape"] = _stack_vectors(
                self.prior_sigma_shape_by_ex,
                ex_actual_values,
                eg_axis,
                list(active_eg_lengths),
                "prior_sigma_shape"
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

        self._add_dataset_settings(ds)

        ds.to_netcdf(str(result_nc_path), engine="h5netcdf")
        print(f"Saved draws to path: {result_nc_path}")

    def _add_dataset_settings(self, ds: xr.Dataset) -> None:
        ds.attrs.update(
            dataset_id=str(self.config.dataset_id),
            run_id=str(self.config.run_id),
            mode=str(self.config.mode),
            sampler_backend=self.sampler_backend,
            prior_config=str(self.config.prior_config),
            sample_config=str(self.config.sample_config),
        )

        for key, value in self.data_loader.run_settings.items():
            if value is not None:
                ds.attrs[key] = (
                    str(value).lower() if isinstance(value, bool) else value
                )

        if self.include_background:
            ds.attrs["background_model"] = self.background_model_kind
            ds.attrs["bg_a0"] = self.background_alpha
            
        ds.attrs["target_accept"] = float(
            self.sample_config_template["nuts"]["target_accept"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian unfolding.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML run config.")
    cli = parser.parse_args()

    cfg_dict = load_run_config(str(repo_path(cli.config)))
    config = dict_to_namespace(cfg_dict)
    
    dataset_kind = str(getattr(config, "dataset_kind", "synthetic")).strip().lower()
    
    if dataset_kind == "experimental":
        loader = ExperimentalDataLoader(
            prompt_path=config.prompt_path,
            background_path=getattr(config, "background_path", None),
            response_db=config.response_db,
            rebin_factors=tuple(config.rebin_factors),
            lower_eg_cut=config.lower_eg_cut,
            include_background=config.include_background,
            fwhm_eg_unfold=config.fwhm_eg_unfold,
            eg_margin=getattr(config, "eg_margin", "200keV"),
        )
    elif dataset_kind == "synthetic":
        loader = SyntheticDataLoader.from_config(config)
    else:
        raise ValueError("dataset_kind must be synthetic or experimental.")

    manager = UnfoldManager(loader, config=config)
    manager.run_unfolding()

if __name__ == "__main__":
    main()
