"""
Compute the quantities for the resampling study
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from src.analysis.envelopes import adaptive_envelope


def band(samples, mass=0.95, trim=0.05):
    draws = xr.DataArray(samples, dims=("sample", "Eg"))
    _, _, lower, upper = adaptive_envelope(draws, mass=mass, trim=trim)
    return np.asarray(lower), np.asarray(upper)


def posterior_coverage(samples, lower, upper):
    return 100.0 * np.all((samples >= lower) & (samples <= upper), axis=1).mean()


def analyze(folder, expected=None, mass=0.95, trim=0.05):
    files = sorted(folder.glob("resample[0-9][0-9][0-9].nc"))

    if not files:
        raise FileNotFoundError(folder)

    if expected is not None and len(files) != expected:
        raise ValueError( f"Expected {expected} resamples, found {len(files)}")

    truth_coverage = []
    posterior_coverage_values = []
    binwise_coverage = []
    relative_width = []

    ex_values = []
    n_bins = []

    for path in files:
        with h5py.File(path) as f:
            eta = np.asarray(f["eta"], dtype=float)
            truth = np.asarray(f["eta_true"], dtype=float).reshape(-1)
            ex = float(f.attrs["ex_actual_kev"])

        if eta.ndim != 3 or eta.shape[0] != 4:
            raise ValueError(f"Invalid posterior shape in {path}")

        j = len(truth)

        if eta.shape[2] != j:
            raise ValueError(f"Truth/posterior mismatch in {path}")

        if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(truth)):
            raise ValueError(f"Non-finite values in {path}")

        ex_values.append(ex)
        n_bins.append(j)

        samples = eta.reshape(-1, j)
        lower, upper = band(samples, mass, trim)

        inside = (truth >= lower) & (truth <= upper)

        truth_coverage.append(float(inside.all()))
        binwise_coverage.append(100.0 * inside.mean())
        relative_width.append((upper - lower).sum() / truth.sum())


        a = eta[:2].reshape(-1, j)
        b = eta[2:].reshape(-1, j)

        la, ua = band(a, mass, trim)
        lb, ub = band(b, mass, trim)

        coverage_ab = posterior_coverage(b, la, ua)
        coverage_ba = posterior_coverage(a, lb, ub)

        posterior_coverage_values.append( 0.5 * (coverage_ab + coverage_ba))

        print(f"{path.name} done", flush=True)

    if len(set(ex_values)) != 1 or len(set(n_bins)) != 1:
        raise ValueError("Inconsistent Ex or Eg bins across resamples")

    summary = pd.DataFrame([{
        "Ex_MeV": ex_values[0] / 1000.0,
        "n_resamples": len(files),
        "n_bins": n_bins[0],
        "simult_truth_coverage_pct": 100.0 * np.mean(truth_coverage),
        "simult_posterior_coverage_pct": np.mean(posterior_coverage_values),
        "mean_binwise_truth_coverage_pct": np.mean(binwise_coverage),
        "mean_relative_total_band_width": np.mean(relative_width),
    }])

    output = folder / "resampling_summary.csv"
    summary.to_csv(output, index=False)

    print()
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expect", type=int, default=100)
    parser.add_argument("--mass", type=float, default=0.95)
    parser.add_argument("--trim", type=float, default=0.05)

    args = parser.parse_args()

    analyze(args.input, expected=args.expect, mass=args.mass, trim=args.trim)

