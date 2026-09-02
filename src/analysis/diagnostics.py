"""
Sampling diagnostics for unfolded result files. This script computes R-hat and bulk ESS per Eg bin for x or eta draws.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from .results import UnfoldingResults


DiagVar = Literal["x", "eta"]


def diagnostic_draws(results: UnfoldingResults, ex_actual: float, var: DiagVar) -> np.ndarray:
    var = str(var).strip().lower()
    if var == "x":
        return results.x_cube(ex_actual)
    if var == "eta":
        return results.eta_cube(ex_actual)

    raise ValueError("var must be 'x' or 'eta'.")


def truth_vector(results: UnfoldingResults, ex_actual: float, var: DiagVar) -> np.ndarray:
    var = str(var).strip().lower()

    if var == "x":
        return results.x_true(ex_actual)

    if var == "eta":
        return results.eta_true(ex_actual)

    raise ValueError("var must be 'x' or 'eta'.")


def rhat_ess_by_eg(
    draws: np.ndarray,
    eg_axis: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute R-hat and bulk ESS.
    
    """

    idata = az.from_dict(
        posterior={name: draws},
        coords={"Eg": eg_axis},
        dims={name: ["Eg"]},
    )

    rhat = np.asarray(az.rhat(idata, var_names=[name])[name].values, dtype=float)
    ess = np.asarray(
        az.ess(idata, var_names=[name], method="bulk")[name].values,
        dtype=float,
    )

    return rhat.reshape(-1), ess.reshape(-1)

def rhat_ess_summary(draws: np.ndarray, eg_axis: np.ndarray, name: str) -> dict:
    rhat, ess = rhat_ess_by_eg(draws, eg_axis, name=name)

    rhat_index = np.nanargmax(rhat).item()
    ess_index = np.nanargmin(ess).item()

    return {
        "Rhat_max": np.nanmax(rhat).item(),
        "Rhat_max_Eg_keV": eg_axis[rhat_index].item(),
        "Rhat_max_index": rhat_index,
        "ESS_bulk_min": np.nanmin(ess).item(),
        "ESS_bulk_min_Eg_keV": eg_axis[ess_index].item(),
        "ESS_bulk_min_index": ess_index,
        "ESS_bulk_median": np.nanmedian(ess).item(),
        "ESS_bulk_mean": np.nanmean(ess).item(),
    }


def sample_stat_array(
    sample_stats: xr.Dataset,
    names: list[str],
) -> np.ndarray | None:
    for name in names:
        if name in sample_stats:
            return np.asarray(sample_stats[name].values)
    return None


def trace_stats(
    trace_nc: str | Path | None,
    max_treedepth: float | None = None,
) -> dict:

    empty = {
        "divergences": np.nan,
        "tree_depth_hit_count": np.nan,
        "tree_depth_hit_percent": np.nan,
    }

    if trace_nc is None:
        return empty

    trace_nc = Path(trace_nc)

    if not trace_nc.exists():
        return empty

    idata = az.from_netcdf(trace_nc)

    if not hasattr(idata, "sample_stats"):
        return empty

    sample_stats = idata.sample_stats

    diverging = sample_stat_array(sample_stats, ["diverging", "divergences"])
    if diverging is None:
        divergences = np.nan
    else:
        divergences = np.nansum(diverging).item()

    tree_depth = sample_stat_array(sample_stats, ["tree_depth", "treedepth", "depth"])

    if tree_depth is None or max_treedepth is None or not np.isfinite(max_treedepth):
        tree_depth_hit_count = np.nan
        tree_depth_hit_percent = np.nan
    else:
        tree_depth = np.asarray(tree_depth, dtype=float)
        finite = np.isfinite(tree_depth)

        if not np.any(finite):
            tree_depth_hit_count = np.nan
            tree_depth_hit_percent = np.nan
        else:
            tree_depth_hit_count = np.sum(tree_depth[finite] >= max_treedepth).item()
            tree_depth_hit_percent = 100.0 * tree_depth_hit_count / np.sum(finite).item()

    return {
        "divergences": divergences,
        "tree_depth_hit_count": tree_depth_hit_count,
        "tree_depth_hit_percent": tree_depth_hit_percent,
    }


def trace_path_for_ex(
    draws_nc: Path,
    ex_requested: float,
    n_ex: int,
) -> Path | None:
    run_dir = draws_nc.parent

    if n_ex == 1:
        trace_path = run_dir / "trace.nc"
    else:
        trace_path = run_dir / "traces" / f"Ex{ex_requested:.0f}.nc"

    if trace_path.exists():
        return trace_path

    return None


def per_ex_value(ds: xr.Dataset, name: str, ex_index: int, default=np.nan):

    if name in ds:
        return ds[name].isel(Ex=ex_index).values.item()

    if name in ds.attrs:
        return ds.attrs[name]

    return default


def summarize_ex(
    results: UnfoldingResults,
    ex_index: int,
    draws_nc: str | Path,
    var: DiagVar = "eta",
) -> dict:

    ds = results.ds
    draws_nc = Path(draws_nc)

    ex_actual = ds["Ex"].values[ex_index].item()
    eg_axis = results.eg(ex_actual)

    draws = diagnostic_draws(results, ex_actual, var=var)
    diagnostic_summary = rhat_ess_summary(draws, eg_axis, name=str(var))

    sampling_time_s = per_ex_value(ds, "sampling_time_s", ex_index)
    target_accept = per_ex_value(ds, "target_accept", ex_index)
    max_treedepth = per_ex_value(ds, "max_treedepth", ex_index)
    nuts_random_seed = per_ex_value(ds, "nuts_random_seed", ex_index, default=-1)

    if np.isfinite(sampling_time_s) and sampling_time_s > 0.0:
        ess_per_s = diagnostic_summary["ESS_bulk_min"] / sampling_time_s
    else:
        ess_per_s = np.nan

    ex_requested = results.ex_requested(ex_actual)
    if ex_requested is None:
        ex_requested = ex_actual

    trace_path = trace_path_for_ex(
        draws_nc=draws_nc,
        ex_requested=ex_requested,
        n_ex=ds.sizes["Ex"],
    )
    nuts_summary = trace_stats(trace_path, max_treedepth=max_treedepth)

    n_obs = results.n_obs(ex_actual)
    n_off_obs = results.n_off_obs(ex_actual)

    if n_off_obs is None:
        sum_n_off_obs = np.nan
    else:
        sum_n_off_obs = np.sum(n_off_obs).item()

    nonzero_n_obs = n_obs[n_obs > 0.0]
    if nonzero_n_obs.size:
        median_nonzero_n_obs = np.median(nonzero_n_obs).item()
    else:
        median_nonzero_n_obs = np.nan

    out = {
        "path": str(draws_nc),
        "trace_path": "none" if trace_path is None else str(trace_path),
        "mode": str(ds.attrs.get("mode", "")),
        "backend": str(ds.attrs.get("sampler_backend", "")),
        "Ex_req_keV": ex_requested,
        "Ex_keV": ex_actual,
        "N_Eg": results.n_eg(ex_actual),
        "var": str(var),
        "target_accept": target_accept,
        "max_treedepth": max_treedepth,
        "nuts_random_seed": nuts_random_seed,
        "sampling_time_s": sampling_time_s,
        "ESS_bulk_min_per_s": ess_per_s,
        "sum_n_obs": np.sum(n_obs).item(),
        "sum_n_off_obs": sum_n_off_obs,
        "max_n_obs": np.max(n_obs).item(),
        "median_nonzero_n_obs": median_nonzero_n_obs,
    }

    out.update(diagnostic_summary)
    out.update(nuts_summary)

    return out


def summarize_run(
    draws_nc: str | Path,
    var: DiagVar = "eta",
) -> pd.DataFrame:

    results = UnfoldingResults(draws_nc, expected_mode="posterior")

    rows = [
        summarize_ex(
            results=results,
            ex_index=ex_index,
            draws_nc=draws_nc,
            var=var,
        )
        for ex_index in range(results.ds.sizes["Ex"])
    ]

    return pd.DataFrame(rows)
