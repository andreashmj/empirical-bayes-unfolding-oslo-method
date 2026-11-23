import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import xarray as xr
import ompy as om
from ompy.rebin import rebin_2D
from scipy.sparse import csr_matrix

if __package__:
    from .remove_co_120sn import remove_contamination
    from .normalize_120sn import normalize_samples, SN
else:
    from remove_co_120sn import remove_contamination
    from normalize_120sn import normalize_samples, SN


FIRSTGEN_ITERATIONS = 23
EX_MIN, EX_MAX, EG_MIN = 4500.0, 1000*SN, 1300.0
MATRIX_BIN, EXTRACTION_BIN, BATCH_SIZE = 64.0, 128.0, 16


def response_operators(ds):
    lengths = ds.Eg_len.values.astype(int) if "Eg_len" in ds else np.full(ds.sizes["Ex"], ds.sizes["Eg"])
    operators = []
    for j, length in enumerate(lengths):
        operators.append(csr_matrix(ds.G_g.isel(Ex=j).values[:length, :length]))
    return lengths, operators


def posterior_batches(ds, lengths, operators):
    # Preserve every saved (chain, draw) matrix; no resampling across Ex rows.
    for chain in range(ds.sizes["chain"]):
        for start in range(0, ds.sizes["draw"], BATCH_SIZE):
            stop = min(start+BATCH_SIZE, ds.sizes["draw"])
            raw = ds.x.isel(chain=chain, draw=slice(start, stop)).transpose("draw", "Ex", "Eg").values
            values = np.zeros(raw.shape, dtype=float)
            for j, (length, operator) in enumerate(zip(lengths, operators)):
                if length:
                    values[:, j, :length] = (operator.T @ raw[:, j, :length].T).T
            yield np.maximum(values, 0)


def common_grid(Ex):
    top = min(Ex[-1] + np.median(np.diff(Ex))/2, 10000.0)
    return np.arange(MATRIX_BIN/2, top-MATRIX_BIN/2+1e-8, MATRIX_BIN)


def rebin_unfolded(values, Ex, Eg, grid):
    result = np.empty((len(values), len(grid), len(grid)))
    for i, matrix in enumerate(values):
        matrix = rebin_2D(matrix, mids_in=Eg, mids_out=grid, axis=1)
        result[i] = rebin_2D(matrix, mids_in=Ex, mids_out=grid, axis=0)
    return np.maximum(result, 0)


def first_generation_method():
    method = om.FirstGeneration()
    method.num_iterations = FIRSTGEN_ITERATIONS
    method.multiplicity_estimation = "statistical"
    method.statistical_upper, method.statistical_lower = 430.0, 200.0
    method.statistical_ratio = 0.3
    method.Ex_entry_shift, method.Ex_entry_statistical = 200.0, 300.0
    method.use_slide = False
    return method


def primary_matrices(ds, path):
    Ex, Eg = ds.Ex.values.astype(float), ds.Eg.values.astype(float)
    if Ex.min() > MATRIX_BIN:
        raise ValueError("First generation requires the low-Ex spectra; the posterior omits them.")
    count = ds.sizes["chain"] * ds.sizes["draw"]
    if count < 2:
        raise ValueError("At least two posterior matrices are needed to estimate chi-square weights.")
    grid = common_grid(Ex)
    factor = int(EXTRACTION_BIN/MATRIX_BIN)
    n = len(grid) // factor
    fit_grid = grid[:n*factor].reshape(n, factor).mean(axis=1)
    primary = np.lib.format.open_memmap(path, mode="w+", dtype=float, shape=(count, n, n))
    mean, m2 = np.zeros((n, n)), np.zeros((n, n))
    lengths, operators = response_operators(ds)
    method, index = first_generation_method(), 0
    print(f"Processing all {count} matrices ({ds.sizes['chain']} chains x {ds.sizes['draw']} draws)", flush=True)
    for values in posterior_batches(ds, lengths, operators):
        values = remove_contamination(values, Ex, Eg)
        values = rebin_unfolded(values, Ex, Eg, grid)
        for matrix in values:
            fg = method(om.Matrix(values=matrix, Ex=grid, Eg=grid)).values
            coarse = fg[:n*factor, :n*factor].reshape(n, factor, n, factor).sum(axis=(1, 3))
            primary[index] = coarse
            delta = coarse - mean
            mean += delta/(index+1)
            m2 += delta*(coarse-mean)
            index += 1
            print(f"Contamination + first generation {index}/{count}", flush=True)
    primary.flush()
    return primary, fit_grid, np.sqrt(np.maximum(m2/(count-1), 0))


def extract(primary, grid, errors, chains, draws):
    cut = om.Action("matrix")
    cut.trapezoid(Ex_min=EX_MIN, Ex_max=EX_MAX, Eg_min=EG_MIN,
                  Eg_max=EX_MAX, inplace=True)
    spread = om.Matrix(values=errors, Ex=grid, Eg=grid)
    cut.act_on(spread)
    extractor = om.Extractor(trapezoid=cut, path=None)
    extractor.randomize_initial_values = False
    extractor.extend_fit_by_resolution = False
    extractor.method = "Powell"
    extractor.options = {"disp": False, "ftol": 1e-5, "xtol": 1e-4}
    nlds, gsfs = [], []
    for i, values in enumerate(primary):
        matrix = om.Matrix(values=np.array(values), Ex=grid, Eg=grid)
        cut.act_on(matrix)
        print(f"Weighted chi-square extraction {i+1}/{len(primary)}", flush=True)
        # The same initialization rule and error matrix for every individual sample.
        nld, gsf = extractor.decompose(matrix=matrix, std=spread, x0="CT-like")
        nlds.append(nld.values)
        gsfs.append(gsf.values)
    return xr.Dataset(
        {"nld_unnormalized": (("member", "E_nld"), np.array(nlds)),
         "gsf_unnormalized": (("member", "E_gsf"), np.array(gsfs)),
         "input_chain": ("member", np.repeat(chains, len(draws))),
         "input_draw": ("member", np.tile(draws, len(chains)))},
        coords={"member": np.arange(len(primary)), "E_nld": nld.E, "E_gsf": gsf.E},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("draws_nc", type=Path)
    parser.add_argument("--levels-kev", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/oslo_120sn"))
    args = parser.parse_args()
    levels = np.atleast_1d(np.loadtxt(args.levels_kev)) / 1000
    args.outdir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="oslo_120sn_") as work:
        with xr.open_dataset(args.draws_nc) as ds:
            primary, grid, errors = primary_matrices(ds, Path(work)/"primary.npy")
            chains, draws = ds.chain.values.copy(), ds.draw.values.copy()
        unnormalized = extract(primary, grid, errors, chains, draws)
        del primary
    normalized = normalize_samples(unnormalized, levels)
    output = args.outdir / "normalized_nld_gsf_samples.nc"
    normalized.to_netcdf(output, engine="h5netcdf")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
