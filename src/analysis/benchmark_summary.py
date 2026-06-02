"""
Create raw and formatted sampler benchmark tables for the paper.

The script discovers posterior draws files under

    result_root/benchmarks/<statistics>/<run_id>/posterior/draws.nc

and writes exactly two CSV files by default:

    benchmark_raw_eta.csv
    benchmark_formatted_eta.csv

Only the paper-reported backends are included by default: PyMC, NumPyro, and
BlackJAX. Nutpie is skipped unless explicitly requested with --include-backends.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_BACKENDS = ("pymc", "numpyro", "blackjax")


def statistics_label(case_name: str) -> str:
    """Return display label from a benchmark case directory name."""

    case_name = str(case_name).strip()

    if case_name == "high_stat":
        return "High"
    if case_name == "low_stat":
        return "Low"

    return case_name.replace("_", " ").title()


def statistics_order(label: str) -> int:
    """Return display order for statistics cases."""

    label = str(label).strip().lower()

    if label == "high":
        return 0
    if label == "low":
        return 1

    return 99


def backend_from_run_id(run_id: str) -> str:
    """Infer backend name from a run ID such as pymc_rep01."""

    run_id = str(run_id).strip().lower()

    if "_rep" in run_id:
        return run_id.split("_rep", 1)[0]

    return run_id.split("_", 1)[0]


def backend_label(backend: str) -> str:
    """Return display label for a backend."""

    backend = str(backend).strip().lower()
    labels = {
        "pymc": "PyMC",
        "numpyro": "NumPyro",
        "blackjax": "BlackJAX",
        "nutpie": "Nutpie",
    }
    return labels.get(backend, backend)


def backend_order(backend: str) -> int:
    """Return display order for backends."""

    backend = str(backend).strip().lower()
    order = {
        "pymc": 0,
        "numpyro": 1,
        "blackjax": 2,
        "nutpie": 3,
    }
    return order.get(backend, 99)


def read_backend(draws_nc: Path, run_id: str) -> str:
    """Read backend from file metadata, using run ID as fallback."""

    with xr.open_dataset(draws_nc) as ds:
        backend = str(ds.attrs.get("sampler_backend", "")).strip().lower()
    if backend:
        return backend
    return backend_from_run_id(run_id)


def benchmark_root(result_root: str | Path) -> Path:
    """Return benchmark root directory."""

    result_root = Path(result_root)
    if result_root.name == "benchmarks":
        return result_root
    return result_root / "benchmarks"


def discover_draws(
    result_root: str | Path,
    include_backends: tuple[str, ...],
) -> list[dict]:
    """Find benchmark posterior draws files for included backends."""

    root = benchmark_root(result_root)
    if not root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {root}")
    files = sorted(root.glob("*_stat/*/posterior/draws.nc"))

    if not files:
        raise FileNotFoundError(f"No benchmark draws.nc files found under {root}")

    included = {str(name).strip().lower() for name in include_backends}
    out: list[dict] = []
    skipped: list[str] = []

    for draws_nc in files:
        case_name = draws_nc.parents[2].name
        run_id = draws_nc.parents[1].name
        backend = read_backend(draws_nc, run_id)

        if backend not in included:
            skipped.append(f"{case_name}/{run_id} ({backend})")
            continue

        out.append(
            {
                "statistics": statistics_label(case_name),
                "case": case_name,
                "run_id": run_id,
                "backend": backend,
                "draws_nc": draws_nc,
            }
        )

    if not out:
        raise FileNotFoundError(
            f"No benchmark draws.nc files for included backends {sorted(included)} under {root}"
        )

    if skipped:
        print("Skipped backends not included in table:", flush=True)
        for item in skipped:
            print(f"  {item}", flush=True)
    return out


def active_eg_length(ds: xr.Dataset, ex_index: int) -> int:
    """Return active number of Eg bins for one Ex row."""
    if "Eg_len" in ds:
        return int(ds["Eg_len"].isel(Ex=ex_index).values.item())

    if "n_obs" not in ds:
        return int(ds.sizes["Eg"])

    values = np.asarray(ds["n_obs"].isel(Ex=ex_index).values, dtype=float)
    return int(np.sum(np.isfinite(values)))


def active_eg_axis(ds: xr.Dataset, n_eg: int) -> np.ndarray:
    """Return active Eg grid."""
    return np.asarray(ds["Eg"].values[:n_eg], dtype=float)


def per_ex_value(ds: xr.Dataset, name: str, ex_index: int, default=np.nan):
    """Return scalar value from a per-Ex variable or dataset attribute."""
    if name in ds:
        return ds[name].isel(Ex=ex_index).values.item()

    if name in ds.attrs:
        return ds.attrs[name]
    return default


def row_vector(
    ds: xr.Dataset,
    name: str,
    ex_index: int,
    n_eg: int,
    fill_nan: float | None = 0.0,
) -> np.ndarray:
    """Return one per-Ex vector sliced to the active Eg range."""
    if name not in ds:
        raise KeyError(f"Dataset missing variable {name!r}.")

    values = np.asarray(
        ds[name].isel(Ex=ex_index, Eg=slice(0, n_eg)).values,
        dtype=float,
    ).reshape(-1)

    if fill_nan is not None:
        values = np.nan_to_num(values, nan=fill_nan)

    return values


def operator_matrix(ds: xr.Dataset, name: str, ex_index: int, n_eg: int) -> np.ndarray:
    """Return one per-Ex operator matrix sliced to the active Eg range."""
    if name not in ds:
        raise KeyError(f"Dataset missing operator {name!r}.")

    data_array = ds[name].isel(Ex=ex_index)
    dim0, dim1 = data_array.dims

    values = np.asarray(
        data_array.isel({dim0: slice(0, n_eg), dim1: slice(0, n_eg)}).values,
        dtype=float,
    )

    return np.nan_to_num(values, nan=0.0)


def diagnostic_draws(ds: xr.Dataset, ex_index: int, n_eg: int, var: str) -> np.ndarray:
    """Return draws with shape chain by draw by Eg for x or eta."""

    var = str(var).strip().lower()
    if "x" not in ds:
        raise KeyError("Dataset missing variable 'x'.")
    x_draws = np.asarray(
        ds["x"]
        .isel(Ex=ex_index, Eg=slice(0, n_eg))
        .transpose("chain", "draw", "Eg")
        .values,
        dtype=float,
    )

    if var == "x":
        return x_draws

    if var == "eta":
        g_gamma = operator_matrix(ds, "G_g", ex_index, n_eg)
        return np.einsum("cde,ef->cdf", x_draws, g_gamma)

    raise ValueError("var must be either 'x' or 'eta'.")


def rhat_ess_summary(draws: np.ndarray, eg_axis: np.ndarray, name: str) -> dict:
    """Summarize per-Eg R-hat and bulk ESS."""
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

    rhat = rhat.reshape(-1)
    ess = ess.reshape(-1)

    rhat_index = int(np.nanargmax(rhat))
    ess_index = int(np.nanargmin(ess))
    return {
        "Rhat_max": float(np.nanmax(rhat)),
        "Rhat_max_Eg_keV": float(eg_axis[rhat_index]),
        "Rhat_max_index": rhat_index,
        "ESS_bulk_min": float(np.nanmin(ess)),
        "ESS_bulk_min_Eg_keV": float(eg_axis[ess_index]),
        "ESS_bulk_min_index": ess_index,
        "ESS_bulk_median": float(np.nanmedian(ess)),
        "ESS_bulk_mean": float(np.nanmean(ess)),
    }


def trace_path_for_ex(draws_nc: Path, ex_requested: float, n_ex: int) -> Path | None:
    """Return expected trace path for one Ex row."""
    run_dir = draws_nc.parent
    if n_ex == 1:
        trace_path = run_dir / "trace.nc"
    else:
        trace_path = run_dir / "traces" / f"Ex{ex_requested:.0f}.nc"
    if trace_path.exists():
        return trace_path
    return None


def sample_stat_array(sample_stats: xr.Dataset, names: list[str]) -> np.ndarray | None:
    """Return the first matching sample-stat array."""
    for name in names:
        if name in sample_stats:
            return np.asarray(sample_stats[name].values)
    return None


def trace_stats_fast(trace_nc: Path | None, max_treedepth: float | None) -> dict:
    """Return divergence and tree-depth diagnostics from sample_stats only.

    This intentionally opens only the /sample_stats group instead of reading the
    complete ArviZ InferenceData file. Loading the whole trace file can be very
    slow for the large benchmark runs and can make the summary script appear to
    hang after the CSV rows have mostly been computed.
    """

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
    try:
        with xr.open_dataset(trace_nc, group="sample_stats") as sample_stats:
            diverging = sample_stat_array(sample_stats, ["diverging", "divergences"])
            if diverging is None:
                divergences = np.nan
            else:
                divergences = float(np.nansum(diverging))

            tree_depth = sample_stat_array(
                sample_stats,
                ["tree_depth", "treedepth", "depth"],
            )
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
                    tree_depth_hit_count = float(np.sum(tree_depth[finite] >= max_treedepth))
                    tree_depth_hit_percent = float(
                        100.0 * tree_depth_hit_count / np.sum(finite)
                    )
    except Exception as error:
        print(f"Warning: could not read sample_stats from {trace_nc}: {error}", flush=True)
        return empty

    return {
        "divergences": divergences,
        "tree_depth_hit_count": tree_depth_hit_count,
        "tree_depth_hit_percent": tree_depth_hit_percent,
    }


def summarize_ex(ds: xr.Dataset, draws_nc: Path, ex_index: int, var: str) -> dict:
    """Summarize one Ex row in one posterior result file."""

    ex_actual = float(ds["Ex"].values[ex_index].item())
    n_eg = active_eg_length(ds, ex_index)
    eg_axis = active_eg_axis(ds, n_eg)

    draws = diagnostic_draws(ds, ex_index=ex_index, n_eg=n_eg, var=var)
    diagnostic_summary = rhat_ess_summary(draws, eg_axis=eg_axis, name=str(var))

    sampling_time_s = per_ex_value(ds, "sampling_time_s", ex_index)
    target_accept = per_ex_value(ds, "target_accept", ex_index)
    max_treedepth = per_ex_value(ds, "max_treedepth", ex_index)
    nuts_random_seed = per_ex_value(ds, "nuts_random_seed", ex_index, default=-1)

    if "Ex_req" in ds:
        ex_requested = ds["Ex_req"].isel(Ex=ex_index).values.item()
        if not np.isfinite(ex_requested):
            ex_requested = ex_actual
    else:
        ex_requested = ex_actual

    if np.isfinite(sampling_time_s) and sampling_time_s > 0.0:
        ess_per_s = diagnostic_summary["ESS_bulk_min"] / sampling_time_s
    else:
        ess_per_s = np.nan

    trace_path = trace_path_for_ex(draws_nc, ex_requested=ex_requested, n_ex=ds.sizes["Ex"])
    nuts_summary = trace_stats_fast(trace_path, max_treedepth=max_treedepth)

    n_obs = row_vector(ds, "n_obs", ex_index, n_eg)

    if "n_off_obs" in ds:
        n_off_obs = row_vector(ds, "n_off_obs", ex_index, n_eg)
        sum_n_off_obs = float(np.sum(n_off_obs))
    else:
        sum_n_off_obs = np.nan

    nonzero_n_obs = n_obs[n_obs > 0.0]
    median_nonzero_n_obs = (
        float(np.median(nonzero_n_obs)) if nonzero_n_obs.size else np.nan
    )

    out = {
        "path": str(draws_nc),
        "trace_path": "none" if trace_path is None else str(trace_path),
        "mode": str(ds.attrs.get("mode", ds.attrs.get("run_mode", ""))),
        "backend": str(ds.attrs.get("sampler_backend", "")),
        "Ex_req_keV": float(ex_requested),
        "Ex_keV": ex_actual,
        "N_Eg": int(n_eg),
        "var": str(var),
        "target_accept": target_accept,
        "max_treedepth": max_treedepth,
        "nuts_random_seed": nuts_random_seed,
        "sampling_time_s": sampling_time_s,
        "ESS_bulk_min_per_s": ess_per_s,
        "sum_n_obs": float(np.sum(n_obs)),
        "sum_n_off_obs": sum_n_off_obs,
        "max_n_obs": float(np.max(n_obs)),
        "median_nonzero_n_obs": median_nonzero_n_obs,
    }

    out.update(diagnostic_summary)
    out.update(nuts_summary)
    return out


def summarize_draws_file(draws_nc: Path, var: str) -> pd.DataFrame:
    """Summarize all Ex rows in one posterior draws.nc file."""

    with xr.open_dataset(draws_nc) as ds:
        mode = str(ds.attrs.get("mode", ds.attrs.get("run_mode", ""))).strip().lower()
        if mode and mode != "posterior":
            raise ValueError(f"{draws_nc}: expected posterior mode, got {mode!r}.")

        rows = [
            summarize_ex(ds=ds, draws_nc=draws_nc, ex_index=ex_index, var=var)
            for ex_index in range(ds.sizes["Ex"])
        ]

    return pd.DataFrame(rows)


def raw_table(
    result_root: str | Path,
    var: str,
    include_backends: tuple[str, ...],
) -> pd.DataFrame:
    """Return raw per-run, per-Ex benchmark diagnostics."""

    items = discover_draws(result_root, include_backends=include_backends)
    rows: list[pd.DataFrame] = []

    for index, item in enumerate(items, start=1):
        print(
            f"[{index}/{len(items)}] summarizing "
            f"{item['case']}/{item['run_id']} ({item['backend']})",
            flush=True,
        )

        frame = summarize_draws_file(item["draws_nc"], var=var)
        frame.insert(0, "statistics", item["statistics"])
        frame.insert(1, "case", item["case"])
        frame.insert(2, "run_id", item["run_id"])

        if "backend" not in frame or not str(frame["backend"].iloc[0]).strip():
            frame["backend"] = item["backend"]

        frame["backend"] = frame["backend"].astype(str).str.lower()
        rows.append(frame)

    table = pd.concat(rows, ignore_index=True)

    table["_statistics_order"] = table["statistics"].map(statistics_order)
    table["_backend_order"] = table["backend"].map(backend_order)

    table = table.sort_values(
        ["_statistics_order", "Ex_req_keV", "_backend_order", "run_id"]
    )

    return table.drop(columns=["_statistics_order", "_backend_order"])


def aggregate_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated benchmark runs."""

    group_columns = [
        "statistics",
        "backend",
        "Ex_req_keV",
        "N_Eg",
        "target_accept",
    ]

    table = (
        raw.groupby(group_columns, dropna=False)
        .agg(
            runs=("sampling_time_s", "count"),
            time_mean_s=("sampling_time_s", "mean"),
            time_std_s=("sampling_time_s", "std"),
            divergences_max=("divergences", "max"),
            tree_depth_hit_percent_max=("tree_depth_hit_percent", "max"),
            Rhat_max=("Rhat_max", "max"),
            ESS_bulk_min_mean=("ESS_bulk_min", "mean"),
            ESS_bulk_min_per_s_mean=("ESS_bulk_min_per_s", "mean"),
        )
        .reset_index()
    )

    table["_statistics_order"] = table["statistics"].map(statistics_order)
    table["_backend_order"] = table["backend"].map(backend_order)

    table = table.sort_values(["_statistics_order", "Ex_req_keV", "_backend_order"])

    return table.drop(columns=["_statistics_order", "_backend_order"])


def format_target_accept(value: float) -> str:
    """Format target acceptance value."""

    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def format_mtd(value: float) -> str:
    """Format maximum tree-depth hit percentage."""

    if not np.isfinite(value):
        return ""

    if abs(value) < 0.05:
        return "0"

    return f"{value:.1f}"


def format_time(mean: float, std: float) -> str:
    """Format time as mean ± standard deviation, rounded to whole seconds."""

    if not np.isfinite(mean):
        return ""

    if not np.isfinite(std) or std == 0.0:
        return f"{mean:.0f}"

    return f"{mean:.0f} ± {std:.0f}"


def format_for_paper(table: pd.DataFrame) -> pd.DataFrame:
    """Format the benchmark table for paper use."""

    out = pd.DataFrame()

    out["Stats."] = table["statistics"].astype(str)
    out["Ex_keV"] = table["Ex_req_keV"].round().astype(int)
    out["N_bins"] = table["N_Eg"].astype(int)
    out["Sampler"] = table["backend"].map(backend_label)
    out["Target acc."] = table["target_accept"].map(format_target_accept)
    out["Div."] = table["divergences_max"].fillna(0).round().astype(int)
    out["MTD (%)"] = table["tree_depth_hit_percent_max"].map(format_mtd)
    out["Rhat_max"] = table["Rhat_max"].map(lambda value: f"{value:.3f}")
    out["ESS_min"] = table["ESS_bulk_min_mean"].round().astype(int)
    out["ESS_min/s"] = table["ESS_bulk_min_per_s_mean"].map(lambda value: f"{value:.1f}")
    out["Time (s)"] = [
        format_time(mean, std)
        for mean, std in zip(table["time_mean_s"], table["time_std_s"])
    ]

    return out


def parse_backends(values: list[str] | None) -> tuple[str, ...]:
    """Return backend filter from CLI values."""

    if values is None or len(values) == 0:
        return DEFAULT_BACKENDS

    return tuple(str(value).strip().lower() for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark posterior diagnostics."
    )
    parser.add_argument("--result-root",
        default="results",
        help="Result root containing the benchmarks directory.",
    )
    parser.add_argument("--out-dir",
        default="results/benchmarks",
        help="Directory for benchmark CSV output.",
    )
    parser.add_argument("--var",
        choices=["x", "eta"],
        default="eta",
        help="Variable used for R-hat and ESS diagnostics.",
    )
    parser.add_argument("--include-backends",
        nargs="+",
        default=None,
        help=(
            "Backends to include in the paper table. Defaults to "
            "pymc numpyro blackjax. Use this only if you intentionally want "
            "to include another backend."
        ),
    )
    args = parser.parse_args()

    include_backends = parse_backends(args.include_backends)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Included backends: {' '.join(include_backends)}", flush=True)
    print(f"Variable: {args.var}", flush=True)

    raw = raw_table(
        args.result_root,
        var=args.var,
        include_backends=include_backends,
    )
    formatted = format_for_paper(aggregate_table(raw))

    raw_path = out_dir / f"benchmark_raw_{args.var}.csv"
    formatted_path = out_dir / f"benchmark_formatted_{args.var}.csv"

    raw.to_csv(raw_path, index=False)
    formatted.to_csv(formatted_path, index=False)

    print(f"Saved raw benchmark table -> {raw_path}", flush=True)
    print(f"Saved formatted benchmark table -> {formatted_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
