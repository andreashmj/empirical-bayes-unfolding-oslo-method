"""Plot normalized 120Sn NLD and gamma-ray strength function."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def summarize(values: np.ndarray):
    """Return median and central 68% and 95% intervals."""
    values = np.asarray(values, dtype=float)

    # Logarithmic axes require positive values.
    values = np.where(values > 0.0, values, np.nan)

    return {
        "q025": np.nanquantile(values, 0.025, axis=0),
        "q16": np.nanquantile(values, 0.16, axis=0),
        "median": np.nanquantile(values, 0.50, axis=0),
        "q84": np.nanquantile(values, 0.84, axis=0),
        "q975": np.nanquantile(values, 0.975, axis=0),
    }


def valid_columns(energy: np.ndarray, summary: dict[str, np.ndarray]):
    mask = np.isfinite(energy)

    for values in summary.values():
        mask &= np.isfinite(values)
        mask &= values > 0.0

    return energy[mask], {
        name: values[mask] for name, values in summary.items()
    }


def plot_band(ax, energy, summary, color):
    ax.fill_between(
        energy,
        summary["q025"],
        summary["q975"],
        color=color,
        alpha=0.20,
        linewidth=0,
        label="95% interval",
    )
    ax.fill_between(
        energy,
        summary["q16"],
        summary["q84"],
        color=color,
        alpha=0.40,
        linewidth=0,
        label="68% interval",
    )
    ax.plot(
        energy,
        summary["median"],
        color=color,
        linewidth=1.6,
        label="Median",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_nc", type=Path)
    parser.add_argument("--out", type=Path, default=Path("nld_gsf_120Sn.pdf"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    with xr.open_dataset(args.samples_nc, engine="h5netcdf") as ds:
        nld = np.asarray(ds["nld"].values)
        gsf = np.asarray(ds["gsf"].values)
        E_nld = np.asarray(ds["E_nld"].values)
        E_gsf = np.asarray(ds["E_gsf"].values)

        print(f"Number of propagated samples: {ds.sizes['sample']}")
        if "unfolding_member" in ds:
            print(
                "Number of unfolding members:",
                np.unique(ds["unfolding_member"].values).size,
            )

    E_nld, nld_summary = valid_columns(E_nld, summarize(nld))
    E_gsf, gsf_summary = valid_columns(E_gsf, summarize(gsf))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 4.0),
        constrained_layout=True,
    )

    # Nuclear level density
    plot_band(axes[0], E_nld, nld_summary, color="tab:blue")

    # The rho(Sn) constraint reported for 120Sn.
    axes[0].errorbar(
        9.105,
        3.66e5,
        yerr=0.54e5,
        fmt="o",
        color="black",
        markersize=4,
        capsize=3,
        label=r"$\rho(S_n)$",
    )

    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$E_x$ (MeV)")
    axes[0].set_ylabel(
        r"Nuclear level density $\rho(E_x)$ (MeV$^{-1}$)"
    )
    axes[0].legend(frameon=False)

    # Gamma-ray strength function
    plot_band(axes[1], E_gsf, gsf_summary, color="tab:red")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$E_\gamma$ (MeV)")
    axes[1].set_ylabel(
        r"$\gamma$-ray strength function $f(E_\gamma)$ "
        r"(MeV$^{-3}$)"
    )
    axes[1].legend(frameon=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
