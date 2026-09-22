"""
Figure showing unfolding results on synthetic data that is generated with Ex-smearing included.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ...analysis.envelopes import global_rank_envelope
from ...analysis.results import UnfoldingResults
from ...paths import repo_path
from ..plot_utils import (
    add_panel_label,
    fill_step_band,
    save_or_show,
    set_symlog_y_with_room,
    steps,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size


MASS = 0.95
LINTHRESH = 5.0

BAYES_COLOR = COLORS["blue"]
BAYES_BAND = COLORS["sky"]
TRUTH_COLOR = COLORS["black"]

DEV_COLOR = COLORS["purple"]
DEV_BAND = COLORS["green"]


def draw_matrix(draws: xr.DataArray) -> np.ndarray:

    return np.asarray(draws.transpose("sample", "Eg").values, dtype=float)


def scaled_deviation_draws(
    draws: xr.DataArray,
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eg_axis: np.ndarray,
) -> xr.DataArray:



    half_width = np.maximum(0.5 * (upper - lower), 1.0e-12)
    matrix = draw_matrix(draws)
    scaled = (matrix - truth[None, :]) / half_width[None, :]

    return xr.DataArray(
        scaled,
        dims=("sample", "Eg"),
        coords={
            "sample": np.arange(scaled.shape[0]),
            "Eg": eg_axis,
        },
        name="scaled_deviation",
    )


def load_case(
    nc_path: str | Path,
    ex_requested: float,
) -> dict:

    results = UnfoldingResults(repo_path(nc_path), expected_mode="posterior")

    ex_actual = float(results.ex_actual(float(ex_requested)))
    eta_draws = results.eta(ex_actual)
    eta_truth = np.asarray(results.eta_true(ex_actual), dtype=float)

    eg_axis, mean, lower, upper = global_rank_envelope(
        eta_draws,
        mass=MASS,
    )

    eg_axis = np.asarray(eg_axis, dtype=float)
    mean = np.asarray(mean, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    scaled_draws = scaled_deviation_draws(
        draws=eta_draws,
        truth=eta_truth,
        lower=lower,
        upper=upper,
        eg_axis=eg_axis)

    _, scaled_mean, scaled_lower, scaled_upper = global_rank_envelope(
        scaled_draws,
        mass=MASS
    )

    return {
        "path": repo_path(nc_path),
        "Ex_requested": float(ex_requested),
        "Ex_actual": ex_actual,
        "Eg_keV": eg_axis,
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "truth": eta_truth,
        "scaled_mean": np.asarray(scaled_mean, dtype=float),
        "scaled_lower": np.asarray(scaled_lower, dtype=float),
        "scaled_upper": np.asarray(scaled_upper, dtype=float)
    }


def plot_eta_panel(
    ax: plt.Axes,
    case: dict
) -> None:

    eg_axis = np.asarray(case["Eg_keV"], dtype=float)

    mean = np.asarray(case["mean"], dtype=float)
    lower = np.asarray(case["lower"], dtype=float)
    upper = np.asarray(case["upper"], dtype=float)
    truth = np.asarray(case["truth"], dtype=float)

    fill_step_band(
        ax,
        eg_axis,
        lower,
        upper,
        facecolor=BAYES_BAND,
        edgecolor=BAYES_BAND,
        alpha=0.25,
        linewidth=0.35,
        zorder=2.0
    )

    steps(
        ax,
        eg_axis,
        mean,
        color=BAYES_COLOR,
        lw=0.95,
        zorder=3.0,
    )

    steps(
        ax,
        eg_axis,
        truth,
        color=TRUTH_COLOR,
        lw=0.85,
        ls="--",
        zorder=3.0
    )

    set_symlog_y_with_room(
        ax,
        arrays=[mean, lower, upper, truth],
        linthresh=LINTHRESH,
        top_factor=10,
        y_lower=0.0,
    )

    ax.set_xlim(eg_axis[0], eg_axis[-1])
    ax.set_ylabel(r"$\boldsymbol{\eta}$ (counts)")

    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)


def scaled_ylim(*arrays: np.ndarray) -> tuple[float, float]:

    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    values = values[np.isfinite(values)]

    if values.size == 0:
        return -2.0, 2.0

    limit = max(float(np.nanmax(np.abs(values))), 2.0)

    return -1.10 * limit, 1.10 * limit


def plot_scaled_panel(
    ax: plt.Axes,
    case: dict,
) -> None:

    eg_axis = np.asarray(case["Eg_keV"], dtype=float)
    mean = np.asarray(case["scaled_mean"], dtype=float)
    lower = np.asarray(case["scaled_lower"], dtype=float)
    upper = np.asarray(case["scaled_upper"], dtype=float)

    fill_step_band(
        ax,
        eg_axis,
        lower,
        upper,
        facecolor=DEV_BAND,
        edgecolor=DEV_BAND,
        alpha=0.25,
        linewidth=0.35,
        zorder=2.0
    )

    steps(
        ax,
        eg_axis,
        mean,
        color=DEV_COLOR,
        lw=0.90,
        zorder=3.0,
    )

    ax.axhline(0.0, color=TRUTH_COLOR, lw=0.55, alpha=0.70, zorder=1.0)
    ax.axhline(1.0, color=TRUTH_COLOR, lw=0.45, ls=":", alpha=0.45, zorder=1.0)
    ax.axhline(-1.0, color=TRUTH_COLOR, lw=0.45, ls=":", alpha=0.45, zorder=1.0)

    ax.set_ylim(*scaled_ylim(mean, lower, upper))
    ax.set_xlim(eg_axis[0], eg_axis[-1])

    ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax.set_ylabel(r"$S(E_\gamma)$")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)


def legend_handles() -> tuple[list, list]:

    bayes_handle = (Patch(
            facecolor=BAYES_BAND,
            edgecolor=BAYES_BAND,
            alpha=0.25,
            linewidth=0.35,
        ),
        Line2D([0],[0],
            color=BAYES_COLOR,
            lw=0.95,
        ),
    )

    truth_handle = Line2D(
        [0], [0],
        color=TRUTH_COLOR,
        lw=0.85,
        ls="--",
    )

    handles = [bayes_handle, truth_handle]
    labels = [
        r"$\boldsymbol{\eta}_{\mathrm{Bayes}}$: mean + 95\% rank-env.",
        r"$\boldsymbol{\eta}_{\mathrm{true}}=(\mathbf{G}_{\mathrm{in}}\bm{\mu}_{\mathrm{true}})\mathbf{G}_{\gamma}$",
    ]

    return handles, labels


def plot_ex_smearing_posterior_comparison(
    left_nc: str | Path,
    right_nc: str | Path,
    ex: float = 4000.0,
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create the figure.
    """

    left = load_case(left_nc, ex_requested=ex)
    right = load_case(right_nc, ex_requested=ex)

    size = figure_size("text", height_to_width=0.66)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(2, 2,
        left=0.09,
        right=0.98,
        bottom=0.10,
        top=0.84,
        hspace=0.18,
        wspace=0.18,
    )

    ax_eta_left = fig.add_subplot(grid[0, 0])
    ax_eta_right = fig.add_subplot(grid[0, 1], sharex=ax_eta_left)
    ax_scaled_left = fig.add_subplot(grid[1, 0], sharex=ax_eta_left)
    ax_scaled_right = fig.add_subplot(grid[1, 1], sharex=ax_eta_right)

    plot_eta_panel(ax_eta_left, left)
    plot_eta_panel(ax_eta_right, right)

    plot_scaled_panel(ax_scaled_left, left)
    plot_scaled_panel(ax_scaled_right, right)

    ax_eta_left.tick_params(labelbottom=False)
    ax_eta_right.tick_params(labelbottom=False)
    ax_eta_right.set_ylabel("")
    ax_scaled_right.set_ylabel("")

    add_panel_label(ax_eta_left, "a", x=0.04, y=0.90, fontsize=9.5)
    add_panel_label(ax_eta_right, "b", x=0.04, y=0.90, fontsize=9.5)
    add_panel_label(ax_scaled_left, "c", x=0.04, y=0.90, fontsize=9.5)
    add_panel_label(ax_scaled_right, "d", x=0.04, y=0.90, fontsize=9.5)

    handles, labels = legend_handles()

    fig.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94), ncol=2,
        frameon=False,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=2.4,
        columnspacing=1.8,
        fontsize=8.5
    )

    save_or_show(
        fig,out,
        tag="ex-smearing-posterior-comparison",
        show=show,
    )

    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot posterior comparison for synthetic Ex-smearing runs."
    )

    parser.add_argument("--left-nc", required=True)
    parser.add_argument("--right-nc", required=True)
    parser.add_argument("--ex", type=float, default=4000.0)
    parser.add_argument(
        "--out",
        default="figures/results/ex_smearing_posterior_comparison.pdf",
    )
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    plot_ex_smearing_posterior_comparison(
        left_nc=args.left_nc,
        right_nc=args.right_nc,
        ex=args.ex,
        out=args.out,
        show=args.show
    )




if __name__ == "__main__":
    main()
