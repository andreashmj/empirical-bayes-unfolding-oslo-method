"""
Prior-posterior comparison figure.

The figure overlays prior and posterior means with their global rank-envelope
bands, together with the corresponding truth spectrum.
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


def common_eg_axis(
    prior_results: UnfoldingResults,
    posterior_results: UnfoldingResults,
    ex_actual: float,
) -> np.ndarray:
    """Return common Eg axis after checking prior/posterior compatibility."""
    if prior_results.n_eg(ex_actual) != posterior_results.n_eg(ex_actual):
        raise ValueError("Eg length mismatch between prior and posterior result files.")

    eg_prior = prior_results.eg(ex_actual)
    eg_posterior = posterior_results.eg(ex_actual)

    if not np.allclose(eg_prior, eg_posterior, rtol=0.0, atol=1.0e-12):
        raise ValueError("Eg grid mismatch between prior and posterior result files.")

    return eg_posterior


def check_envelope_axis(
    reference_eg: np.ndarray,
    envelope_eg: np.ndarray,
    label: str,
) -> None:
    """Check that an envelope Eg grid matches the reference grid."""

    if not np.allclose(reference_eg, envelope_eg, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Envelope Eg grid mismatch for {label}.")

def plot_prior_posterior_comparison(
    prior_nc: str | Path,
    post_nc: str | Path,
    ex: float | None = None,
    var: str = "eta",
    mass: float = 0.95,
    symlog_linthresh: float = 5.0,
    out: str | Path | None = None,
    show: bool = True,
    layout: str = "column",
) -> plt.Figure:
    """Create the prior-posterior comparison figure."""

    layout = str(layout).strip().lower()
    if layout not in {"text", "column"}:
        raise ValueError("layout must be either 'text' or 'column'.")

    prior_results = UnfoldingResults(prior_nc, expected_mode="prior")
    posterior_results = UnfoldingResults(post_nc, expected_mode="posterior")

    ex_actual = posterior_results.ex_actual(ex)
    eg_axis = common_eg_axis(
        prior_results=prior_results,
        posterior_results=posterior_results,
        ex_actual=ex_actual,
    )
    prior_draws, _, symbol = spectrum_draws_and_truth(
        prior_results,
        ex_actual,
        var,
    )
    posterior_draws, truth_raw, _ = spectrum_draws_and_truth(
        posterior_results,
        ex_actual,
        var,
    )
    truth = np.asarray(truth_raw, dtype=float)

    eg_prior, mean_prior, lower_prior, upper_prior = global_rank_envelope(
        prior_draws,
        mass=mass,
    )
    eg_posterior, mean_posterior, lower_posterior, upper_posterior = (
        global_rank_envelope(
            posterior_draws,
            mass=mass,
        )
    )

    check_envelope_axis(eg_axis, eg_prior, label="prior")
    check_envelope_axis(eg_axis, eg_posterior, label="posterior")

    mass_percent = round(100.0 * mass)

    prior_band = COLORS["vermillion"]
    prior_mean = COLORS["orange"]
    posterior_band = COLORS["sky"]
    posterior_mean = COLORS["blue"]
    truth_color = COLORS["black"]

    prior_alpha = 0.22
    posterior_alpha = 0.28

    mean_linewidth = 0.65
    truth_linewidth = 0.45
    band_edge_width = 0.35

    # This is a one-column paper figure. Keep the size fixed even though the
    # shared CLI passes a generic layout argument.
    size = figure_size("column", height_to_width=0.90)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        1,
        1,
        left=0.17,
        right=0.97,
        bottom=0.18,
        top=0.79,
    )
    ax = fig.add_subplot(grid[0, 0])

    fill_step_band(
        ax,
        eg_axis,
        lower_prior,
        upper_prior,
        facecolor=prior_band,
        edgecolor=prior_band,
        alpha=prior_alpha,
        linewidth=band_edge_width,
        zorder=1.8,
    )

    fill_step_band(
        ax,
        eg_axis,
        lower_posterior,
        upper_posterior,
        facecolor=posterior_band,
        edgecolor=posterior_band,
        alpha=posterior_alpha,
        linewidth=band_edge_width,
        zorder=2.0,
    )

    steps(
        ax,
        eg_axis,
        mean_prior,
        color=prior_mean,
        lw=mean_linewidth,
        zorder=3.0,
    )

    steps(
        ax,
        eg_axis,
        mean_posterior,
        color=posterior_mean,
        lw=mean_linewidth,
        zorder=3.1,
    )

    truth_handle = steps(
        ax,
        eg_axis,
        truth,
        color=truth_color,
        lw=truth_linewidth,
        ls="--",
        zorder=3.2,
    )

    ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax.set_ylabel(rf"${symbol}$ (counts)")

    set_symlog_y_with_room(
        ax,
        arrays=[upper_prior, upper_posterior, truth],
        linthresh=symlog_linthresh,
        top_factor=1.25,
        y_lower=-1.0,
    )

    add_zero_line(
        ax,
        color=truth_color,
        lw=0.45,
        ls=":",
        alpha=0.55,
        zorder=1.1,
    )

    x_right = np.max(eg_axis).item()
    ax.set_xlim(-0.025 * x_right, x_right)

    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)

    prior_handle = (
        Patch(
            facecolor=rgba(prior_band, prior_alpha),
            edgecolor=prior_band,
            linewidth=band_edge_width,
        ),
        Line2D([0], [0], color=prior_mean, lw=mean_linewidth),
    )
    posterior_handle = (
        Patch(
            facecolor=rgba(posterior_band, posterior_alpha),
            edgecolor=posterior_band,
            linewidth=band_edge_width,
        ),
        Line2D([0], [0], color=posterior_mean, lw=mean_linewidth),
    )

    add_top_legend(
        fig,
        handles=[prior_handle, posterior_handle, truth_handle],
        labels=[
            rf"Prior: mean + {mass_percent}\% rank-env.",
            rf"Post.: mean + {mass_percent}\% rank-env.",
            rf"${symbol}_{{\mathrm{{true}}}}$",
        ],
        ncol=1,
        y=0.985,
        frameon=False,
        handlelength=1.7,
        columnspacing=0.9,
        labelspacing=0.32,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        fontsize=8.0,
    )

    save_or_show(fig, out, tag="prior-posterior-comparison", show=show)
    return fig
