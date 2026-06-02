"""
Prior spectrum figure.

The figure shows the prior mean, global rank-envelope bands, and the
corresponding truth spectrum for one excitation-energy row.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    steps,
    use_MeV_on_xaxis,
    set_symlog_y_with_room
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


def plot_prior_spectrum(
    prior_nc: str | Path,
    ex: float | None = None,
    var: str = "eta",
    mass: float = 0.95,
    inner_mass: float | None = 0.75,
    symlog_linthresh: float = 5.0,
    out: str | Path | None = None,
    show: bool = True,
    layout: str = "column",
) -> plt.Figure:
    """Create the prior spectrum figure."""

    results = UnfoldingResults(prior_nc, expected_mode="prior")

    ex_actual = results.ex_actual(ex)
    draws, truth_raw, symbol = spectrum_draws_and_truth(results, ex_actual, var)
    truth = np.asarray(truth_raw, dtype=float)

    eg_axis, mean_prior, lower, upper = global_rank_envelope(
        draws,
        mass=mass,
    )
    use_inner_band = inner_mass is not None and 0.0 < float(inner_mass) < float(mass)

    if use_inner_band:
        inner_mass_value = float(inner_mass)
        _, _, lower_inner, upper_inner = global_rank_envelope(
            draws,
            mass=inner_mass_value,
        )
    else:
        inner_mass_value = None
        lower_inner = None
        upper_inner = None

    mass_percent = round(100.0 * mass)
    inner_mass_percent = (
        round(100.0 * inner_mass_value)
        if inner_mass_value is not None
        else None
    )

    band_color = COLORS["vermillion"]
    inner_band_color = COLORS["green"]
    mean_color = COLORS["blue"]
    truth_color = COLORS["black"]
    
    band_alpha = 0.22
    inner_band_alpha = 0.20

    mean_linewidth = 0.65
    truth_linewidth = 0.45
    band_edge_width = 0.35

    size = figure_size("column", height_to_width=0.86)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        1,
        left=0.17,
        right=0.97,
        bottom=0.18,
        top=0.82,
    )
    ax = fig.add_subplot(grid[0, 0])

    fill_step_band(
        ax,
        eg_axis,
        lower,
        upper,
        facecolor=band_color,
        edgecolor=band_color,
        alpha=band_alpha,
        linewidth=band_edge_width,
        zorder=2.0,
    )
    if use_inner_band:
        fill_step_band(
            ax,
            eg_axis,
            lower_inner,
            upper_inner,
            facecolor=inner_band_color,
            edgecolor=inner_band_color,
            alpha=inner_band_alpha,
            linewidth=band_edge_width,
            zorder=2.1,
        )

    steps(
        ax,
        eg_axis,
        mean_prior,
        color=mean_color,
        lw=mean_linewidth,
        zorder=3.0,
    )
    
    truth_handle = steps(
        ax,
        eg_axis,
        truth,
        color=truth_color,
        lw=truth_linewidth,
        ls="--",
        zorder=3.1,
    )

    ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax.set_ylabel(rf"${symbol}$ (counts)")

    set_symlog_y_with_room(
        ax,
        arrays=[upper, truth],
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

    mean_handle = Line2D(
        [0],
        [0],
        color=mean_color,
        lw=mean_linewidth,
    )

    truth_legend_handle = Line2D(
        [0],
        [0],
        color=truth_color,
        lw=truth_linewidth,
        ls="--",
    )

    outer_band_handle = Patch(
        facecolor=rgba(band_color, band_alpha),
        edgecolor=band_color,
        linewidth=band_edge_width,
    )

    handles: list[object] = [
        mean_handle,
        truth_legend_handle,
    ]

    labels = [
        rf"${symbol}^{{\mathrm{{prior}}}}_{{\mathrm{{mean}}}}$",
        rf"${symbol}_{{\mathrm{{true}}}}$",
    ]

    if use_inner_band:
        inner_band_handle = Patch(
            facecolor=rgba(inner_band_color, inner_band_alpha),
            edgecolor=inner_band_color,
            linewidth=band_edge_width,
        )
        handles.append(inner_band_handle)
        labels.append(rf"{inner_mass_percent}\% rank-env.")

    handles.append(outer_band_handle)
    labels.append(rf"{mass_percent}\% rank-env.")

    add_top_legend(
        fig,
        handles=handles,
        labels=labels,
        ncol=2,
        y=0.985,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.0,
        labelspacing=0.32,
        fontsize=8.0,
    )

    save_or_show(fig, out, tag="prior-spectrum", show=show)
    return fig
