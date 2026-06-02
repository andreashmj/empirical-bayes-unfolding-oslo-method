"""
Inspect per-bin prior widths stored in a result NetCDF file.

The script reports prior_sigma, prior_sigma_shape, eta_rl, x_rl, and the
marginal prior coefficient of variation implied by the Gamma-lognormal layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..paths import repo_path


def actual_ex(ds: xr.Dataset, ex: float | None) -> float:
    """Return actual Ex coordinate closest to the requested value."""

    ex_values = np.asarray(ds["Ex"].values, dtype=float)

    if ex is None:
        return ex_values[0].item()

    index = np.argmin(np.abs(ex_values - float(ex)))
    return ex_values[index].item()


def active_eg_length(ds: xr.Dataset, ex_actual: float) -> int:
    """Return active Eg length for one Ex row."""

    if "Eg_len" in ds:
        return int(ds["Eg_len"].sel(Ex=ex_actual, method="nearest").values.item())

    values = np.asarray(ds["n_obs"].sel(Ex=ex_actual, method="nearest").values, dtype=float)
    return int(np.sum(np.isfinite(values)))


def row(ds: xr.Dataset, name: str, ex_actual: float, n_eg: int) -> np.ndarray:
    """Return one active per-Ex vector."""

    if name not in ds:
        raise KeyError(f"Dataset is missing '{name}'.")

    return np.asarray(
        ds[name]
        .sel(Ex=ex_actual, method="nearest")
        .isel(Eg=slice(0, n_eg))
        .values,
        dtype=float,
    ).reshape(-1)


def marginal_cv_from_sigma(sigma: np.ndarray, alpha: float) -> np.ndarray:
    """Return marginal CV of the Gamma-lognormal prior."""

    return np.sqrt(np.exp(sigma**2) * (1.0 + 1.0 / float(alpha)) - 1.0)


def prior_width_frame(
    nc_path: str | Path,
    ex: float | None,
    alpha: float,
) -> pd.DataFrame:
    """Return per-bin prior-width diagnostics for one Ex row."""

    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    ex_actual = actual_ex(ds, ex)
    n_eg = active_eg_length(ds, ex_actual)
    eg_axis = np.asarray(ds["Eg"].values[:n_eg], dtype=float)

    sigma = row(ds, "prior_sigma", ex_actual, n_eg)
    sigma_shape = row(ds, "prior_sigma_shape", ex_actual, n_eg)
    eta_rl = row(ds, "eta_rl", ex_actual, n_eg)
    x_rl = row(ds, "x_rl", ex_actual, n_eg)

    cv = marginal_cv_from_sigma(sigma, alpha=alpha)

    frame = pd.DataFrame(
        {
            "ex_keV": ex_actual,
            "eg_keV": eg_axis,
            "x_rl": x_rl,
            "eta_rl": eta_rl,
            "prior_sigma": sigma,
            "prior_sigma_shape": sigma_shape,
            "prior_marginal_cv": cv,
        }
    )

    return frame


def print_summary(frame: pd.DataFrame) -> None:
    """Print compact prior-width summary."""

    ex_actual = float(frame["ex_keV"].iloc[0])
    print(f"Ex actual: {ex_actual:.0f} keV")
    print(f"active Eg bins: {len(frame)}")
    print()

    for column in ["prior_sigma", "prior_sigma_shape", "prior_marginal_cv", "eta_rl", "x_rl"]:
        print(f"{column} quantiles:")

        values = frame[column].to_numpy(dtype=float)

        for q in [0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]:
            print(f"  q={q:4.2f}: {np.nanquantile(values, q):.4g}")

        print()

    print("Bins with largest prior_sigma:")
    largest = frame.sort_values("prior_sigma", ascending=False).head(12)

    for _, row_values in largest.iterrows():
        print(
            f"  Eg={row_values['eg_keV']:8.1f} keV  "
            f"sigma={row_values['prior_sigma']:6.3f}  "
            f"sigma_shape={row_values['prior_sigma_shape']:6.3f}  "
            f"CV={row_values['prior_marginal_cv']:10.3g}  "
            f"eta_rl={row_values['eta_rl']:10.3g}  "
            f"x_rl={row_values['x_rl']:10.3g}"
        )


def make_plot(frame: pd.DataFrame, out: str | Path) -> None:
    """Save a diagnostic plot of RL intensity and prior widths."""

    eg_axis = frame["eg_keV"].to_numpy(dtype=float)
    eta_rl = frame["eta_rl"].to_numpy(dtype=float)
    x_rl = frame["x_rl"].to_numpy(dtype=float)
    sigma = frame["prior_sigma"].to_numpy(dtype=float)
    sigma_shape = frame["prior_sigma_shape"].to_numpy(dtype=float)
    cv = frame["prior_marginal_cv"].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.0, 6.5),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(eg_axis, eta_rl, drawstyle="steps-mid", label=r"$\eta_{\rm RL}$")
    axes[0].plot(eg_axis, x_rl, drawstyle="steps-mid", label=r"$x_{\rm RL}$", alpha=0.65)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_ylabel("RL scale")
    axes[0].legend(frameon=False)

    axes[1].plot(eg_axis, sigma, drawstyle="steps-mid", label="prior_sigma")
    axes[1].plot(
        eg_axis,
        sigma_shape,
        drawstyle="steps-mid",
        label="prior_sigma_shape",
        alpha=0.75,
    )
    axes[1].set_ylabel("sigma")
    axes[1].legend(frameon=False)

    axes[2].plot(eg_axis, cv, drawstyle="steps-mid", label="marginal CV")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("marginal CV")
    axes[2].set_xlabel(r"$E_\gamma$ (keV)")
    axes[2].legend(frameon=False)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved prior-width plot -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect per-bin prior sigma values in a result NetCDF file."
    )
    parser.add_argument("nc", help="Path to prior or posterior draws.nc.")
    parser.add_argument("--ex", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-fig", default=None)

    args = parser.parse_args()

    nc_path = repo_path(args.nc)
    frame = prior_width_frame(
        nc_path=nc_path,
        ex=args.ex,
        alpha=float(args.alpha),
    )

    print_summary(frame)

    if args.out_csv is not None:
        out_csv = repo_path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_csv, index=False)
        print(f"Saved prior-width table -> {out_csv}")

    if args.out_fig is not None:
        make_plot(frame, repo_path(args.out_fig))


if __name__ == "__main__":
    main()
