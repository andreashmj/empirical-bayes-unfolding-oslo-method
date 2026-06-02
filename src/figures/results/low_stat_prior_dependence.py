"""
Low-statistics prior-dependence figure.

This figure compares the baseline posterior with three alternative
posterior runs on the absolute resolution-limited count scale. It is intended
for the low-statistics Ex approx 9.5 MeV prior-dependence check, where scaled
deviations can visually exaggerate sub-count changes in the weak high-E_gamma
tail.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ...analysis.envelopes import global_rank_envelope
from ...analysis.results import UnfoldingResults
from ..plot_utils import (
    add_panel_label,
    add_top_legend,
    add_zero_line,
    fill_step_band,
    rgba,
    save_or_show,
    set_symlog_y_with_room,
    steps,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size

baseline_band = COLORS["sky"]
baseline_mean_color = COLORS["blue"]
alternative_band = COLORS["rose"]
alternative_mean_color = COLORS["wine"]
truth_color = COLORS["black"]

baseline_alpha = 0.24
alternative_alpha = 0.24
mean_linewidth = 0.62
truth_linewidth = 0.45
band_edge_width = 0.32


def variable_symbol(var: str) -> str:
    """Return the paper symbol for a stored variable name."""

    var = str(var).strip().lower()

    if var == "eta":
        return r"\boldsymbol{\eta}"

    if var == "x":
        return r"\boldsymbol{\mu}"

    raise ValueError("var must be either 'eta' or 'x'.")


def spectrum_draws_and_truth(
    results: UnfoldingResults,
    ex_actual: float,
    var: str,
):
    """Return draws, truth, and paper symbol for eta or x."""

    var = str(var).strip().lower()

    if var == "eta":
        return results.eta(ex_actual), results.eta_true(ex_actual), variable_symbol(var)

    if var == "x":
        return results.x(ex_actual), results.x_true(ex_actual), variable_symbol(var)

    raise ValueError("var must be either 'eta' or 'x'.")


def check_same_eg_axis(
    reference_eg: np.ndarray,
    candidate_eg: np.ndarray,
    label: str,
) -> None:
    """Check that two Eg grids agree exactly up to floating-point tolerance."""

    if reference_eg.shape != candidate_eg.shape:
        raise ValueError(f"Eg grid length mismatch for {label}.")

    if not np.allclose(reference_eg, candidate_eg, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Eg grid mismatch for {label}.")


def posterior_summary(
    results: UnfoldingResults,
    ex_actual: float,
    var: str,
    mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return Eg, posterior mean, envelope, truth, and variable symbol."""

    draws, truth_raw, symbol = spectrum_draws_and_truth(results, ex_actual, var)
    eg_axis, mean, lower, upper = global_rank_envelope(draws, mass=mass)
    truth = np.asarray(truth_raw, dtype=float)

    check_same_eg_axis(
        reference_eg=results.eg(ex_actual),
        candidate_eg=eg_axis,
        label="posterior envelope",
    )

    return eg_axis, mean, lower, upper, truth, symbol


def plot_absolute_comparison_panel(
    ax: plt.Axes,
    eg_axis: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_lower: np.ndarray,
    baseline_upper: np.ndarray,
    alternative_mean: np.ndarray,
    alternative_lower: np.ndarray,
    alternative_upper: np.ndarray,
    truth: np.ndarray,
    panel_label: str,
    symbol: str,
    y_scale_arrays: list[np.ndarray],
    symlog_linthresh: float,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    """Plot one baseline-vs-alternative posterior panel."""

    fill_step_band(
        ax,
        eg_axis,
        baseline_lower,
        baseline_upper,
        facecolor=baseline_band,
        edgecolor=baseline_band,
        alpha=baseline_alpha,
        linewidth=band_edge_width,
        zorder=1.7,
    )

    fill_step_band(
        ax,
        eg_axis,
        alternative_lower,
        alternative_upper,
        facecolor=alternative_band,
        edgecolor=alternative_band,
        alpha=alternative_alpha,
        linewidth=band_edge_width,
        zorder=2.0,
    )

    steps(
        ax,
        eg_axis,
        baseline_mean,
        color=baseline_mean_color,
        lw=mean_linewidth,
        zorder=3.0,
    )

    steps(
        ax,
        eg_axis,
        alternative_mean,
        color=alternative_mean_color,
        lw=mean_linewidth,
        zorder=3.1,
    )

    steps(
        ax,
        eg_axis,
        truth,
        color=truth_color,
        lw=truth_linewidth,
        ls="--",
        zorder=3.2,
    )

    set_symlog_y_with_room(
        ax,
        arrays=y_scale_arrays,
        linthresh=symlog_linthresh,
        top_factor=3.0,
        y_lower=-1.0,
    )
    add_zero_line(
        ax,
        color=truth_color,
        lw=0.42,
        ls=":",
        alpha=0.55,
        zorder=1.1,
    )

    if show_ylabel:
        ax.set_ylabel(rf"${symbol}$ (counts)")

    if show_xlabel:
        ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    else:
        ax.tick_params(labelbottom=False)

    add_panel_label(ax, label=panel_label)
    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)


def plot_low_stat_prior_dependence(
    baseline_post_nc: str | Path,
    alt_post_ncs: list[str | Path],
    ex: float | None = None,
    var: str = "eta",
    mass: float = 0.95,
    symlog_linthresh: float = 5.0,
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create the low-statistics prior-dependence comparison figure."""

    var = str(var).strip().lower()
    if var not in {"eta", "x"}:
        raise ValueError("var must be either 'eta' or 'x'.")

    if len(alt_post_ncs) != 3:
        raise ValueError(
            "Exactly three alternative posterior files are required: "
            "constant center, uniform sigma_j=1.0, and the combined stress test."
        )

    baseline_results = UnfoldingResults(baseline_post_nc, expected_mode="posterior")
    ex_actual = baseline_results.ex_actual(ex)

    (
        eg_axis,
        baseline_mean,
        baseline_lower,
        baseline_upper,
        truth,
        symbol,
    ) = posterior_summary(
        results=baseline_results,
        ex_actual=ex_actual,
        var=var,
        mass=mass,
    )

    alternative_summaries = []

    for path in alt_post_ncs:
        alternative_results = UnfoldingResults(path, expected_mode="posterior")

        if alternative_results.n_eg(ex_actual) != baseline_results.n_eg(ex_actual):
            raise ValueError(f"Eg_len mismatch between baseline and {path}.")

        (
            eg_alt,
            alternative_mean,
            alternative_lower,
            alternative_upper,
            alternative_truth,
            _,
        ) = posterior_summary(
            results=alternative_results,
            ex_actual=ex_actual,
            var=var,
            mass=mass,
        )

        check_same_eg_axis(eg_axis, eg_alt, label=str(path))

        if not np.allclose(truth, alternative_truth, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"Truth mismatch between baseline and {path}.")

        alternative_summaries.append(
            {
                "mean": alternative_mean,
                "lower": alternative_lower,
                "upper": alternative_upper,
            }
        )

    y_scale_arrays = [
        baseline_upper,
        baseline_mean,
        truth,
        *[summary["upper"] for summary in alternative_summaries],
        *[summary["mean"] for summary in alternative_summaries],
    ]

    mass_percent = round(100.0 * mass)

    size = figure_size("text", height_to_width=0.92)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        3,
        1,
        left=0.10,
        right=0.97,
        bottom=0.07,
        top=0.92,
        wspace=0.0,
        hspace=0.13,
    )

    axes: list[plt.Axes] = []
    shared_axis = None

    for panel_index in range(3):
        ax = fig.add_subplot(grid[panel_index, 0], sharex=shared_axis)

        if shared_axis is None:
            shared_axis = ax

        axes.append(ax)

    x_right = np.max(eg_axis).item()

    for panel_index, (ax, summary) in enumerate(zip(axes, alternative_summaries)):
        plot_absolute_comparison_panel(
            ax=ax,
            eg_axis=eg_axis,
            baseline_mean=baseline_mean,
            baseline_lower=baseline_lower,
            baseline_upper=baseline_upper,
            alternative_mean=summary["mean"],
            alternative_lower=summary["lower"],
            alternative_upper=summary["upper"],
            truth=truth,
            panel_label=chr(ord("a") + panel_index),
            symbol=symbol,
            y_scale_arrays=y_scale_arrays,
            symlog_linthresh=symlog_linthresh,
            show_xlabel=panel_index == 2,
            show_ylabel=True,
        )
        ax.set_xlim(-0.025 * x_right, x_right)

    baseline_handle = (
        Patch(
            facecolor=rgba(baseline_band, 0.24),
            edgecolor=baseline_band,
            linewidth=0.32,
        ),
        Line2D([0], [0], color=baseline_mean_color, lw=0.62),
    )
    alternative_handle = (
        Patch(
            facecolor=rgba(alternative_band, 0.24),
            edgecolor=alternative_band,
            linewidth=0.32,
        ),
        Line2D([0], [0], color=alternative_mean_color, lw=0.62),
    )
    truth_handle = Line2D([0], [0], color=truth_color, lw=0.45, ls="--")

    add_top_legend(
        fig,
        handles=[baseline_handle, alternative_handle, truth_handle],
        labels=[
            rf"Baseline post.: mean + {mass_percent}\% rank-env.",
            rf"Alternative post.: mean + {mass_percent}\% rank-env.",
            rf"${symbol}_{{\mathrm{{true}}}}$",
        ],
        ncol=3,
        y=0.985,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
        labelspacing=0.35,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        fontsize=8.8,
    )

    save_or_show(fig, out, tag="low-stat-prior-dependence", show=show)
    return fig
