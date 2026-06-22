"""
Multi-ex posterior results figure.

The figure shows posterior spectra for several excitation-energy rows and the
corresponding scaled deviations relative to truth.

Each excitation-energy column has its own x axis and y axes.  Within a column,
the spectrum panel and scaled-deviation panel share the x axis.  The spectrum
panels use a linear count scale, not symlog.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ...analysis.envelopes import global_rank_envelope
from ...analysis.results import UnfoldingResults
from ..plot_utils import (
    add_panel_label,
    add_top_legend,
    fill_step_band,
    panel_id,
    rgba,
    save_or_show,
    set_linear_y_with_zero_room,
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
) -> tuple[xr.DataArray, np.ndarray, str]:
    """Return draws, truth, and paper symbol for eta or x."""

    var = str(var).strip().lower()

    if var == "eta":
        return results.eta(ex_actual), results.eta_true(ex_actual), variable_symbol(var)

    if var == "x":
        return results.x(ex_actual), results.x_true(ex_actual), variable_symbol(var)

    raise ValueError("var must be either 'eta' or 'x'.")


def scaled_deviation_draws(
    draws: xr.DataArray,
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eg_axis: np.ndarray,
) -> xr.DataArray:
    """Return scaled-deviation draws."""

    half_width = np.maximum(0.5 * (upper - lower), 1.0e-12)
    matrix = np.asarray(draws.transpose("sample", "Eg").values, dtype=float)
    truth = np.asarray(truth, dtype=float)
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


def make_panel_data(
    results: UnfoldingResults,
    ex_values: list[float],
    var: str,
    mass: float,
) -> tuple[list[dict], str]:
    """Build data dictionaries for all excitation-energy panels."""

    panels = []
    symbol = variable_symbol(var)

    for ex in ex_values:
        ex_actual = results.ex_actual(ex)
        ex_requested = results.ex_requested(ex_actual)

        draws, truth_raw, symbol = spectrum_draws_and_truth(results, ex_actual, var)
        truth = np.asarray(truth_raw, dtype=float)

        eg_axis, mean_post, lower_post, upper_post = global_rank_envelope(
            draws,
            mass=mass,
        )

        scaled_draws = scaled_deviation_draws(
            draws=draws,
            truth=truth,
            lower=lower_post,
            upper=upper_post,
            eg_axis=eg_axis,
        )

        eg_scaled, scaled_mean, scaled_lower, scaled_upper = global_rank_envelope(
            scaled_draws,
            mass=mass,
        )

        panels.append(
            {
                "ex_actual": ex_actual,
                "ex_requested": ex_requested,
                "eg_axis": np.asarray(eg_axis, dtype=float),
                "truth": truth,
                "mean_post": np.asarray(mean_post, dtype=float),
                "lower_post": np.asarray(lower_post, dtype=float),
                "upper_post": np.asarray(upper_post, dtype=float),
                "eg_scaled": np.asarray(eg_scaled, dtype=float),
                "scaled_mean": np.asarray(scaled_mean, dtype=float),
                "scaled_lower": np.asarray(scaled_lower, dtype=float),
                "scaled_upper": np.asarray(scaled_upper, dtype=float),
            }
        )

    return panels, symbol


def x_limits_for_axis(eg_axis: np.ndarray) -> tuple[float, float]:
    """Return x limits for one excitation-energy column."""

    eg_axis = np.asarray(eg_axis, dtype=float)
    finite = eg_axis[np.isfinite(eg_axis)]

    if finite.size == 0:
        return 0.0, 1.0

    x_right = np.max(finite).item()

    if not np.isfinite(x_right) or x_right <= 0.0:
        return 0.0, 1.0

    return -0.025 * x_right, x_right


def ticks_for_axis(eg_axis: np.ndarray) -> list[float]:
    """Return compact x ticks in keV for one excitation-energy column."""

    _, x_right = x_limits_for_axis(eg_axis)

    if x_right <= 0.0:
        return []

    if x_right <= 5000.0:
        step = 1000.0
    elif x_right <= 10000.0:
        step = 2000.0
    else:
        step = 3000.0

    ticks = np.arange(0.0, x_right + 0.5 * step, step)
    ticks = ticks[ticks <= x_right + 1.0e-9]

    return [tick.item() for tick in ticks]


def set_column_x_axis(
    axes: list[plt.Axes],
    eg_axis: np.ndarray,
) -> None:
    """Set x limits and ticks for one excitation-energy column."""

    x_limits = x_limits_for_axis(eg_axis)
    ticks = ticks_for_axis(eg_axis)

    for ax in axes:
        ax.set_xlim(*x_limits)

        if ticks:
            ax.set_xticks(ticks)

        use_MeV_on_xaxis(ax)


def scaled_deviation_y_limits_for_panel(
    panel: dict,
    minimum_abs: float = 2.0,
    padding_fraction: float = 0.05,
) -> tuple[float, float]:
    """Return y limits for one scaled-deviation panel."""

    values = [
        panel["scaled_mean"],
        panel["scaled_lower"],
        panel["scaled_upper"],
    ]

    finite_values = []

    for value in values:
        array = np.asarray(value, dtype=float).reshape(-1)
        array = array[np.isfinite(array)]

        if array.size:
            finite_values.append(array)

    if not finite_values:
        return -float(minimum_abs), float(minimum_abs)

    all_values = np.concatenate(finite_values)

    lower = min(np.min(all_values).item(), -float(minimum_abs), 0.0)
    upper = 1.18 * max(np.max(all_values).item(), float(minimum_abs), 0.0)

    span = upper - lower

    if not np.isfinite(span) or span <= 0.0:
        return -float(minimum_abs), float(minimum_abs)

    padding = float(padding_fraction) * span
    return lower - padding, upper + padding


def plot_spectrum_panel(
    ax: plt.Axes,
    panel: dict,
    symbol: str,
    label: str,
    show_ylabel: bool,
    post_band: str,
    post_mean: str,
    truth_color: str,
    alpha_band: float,
    mean_linewidth: float,
    truth_linewidth: float,
    edge_linewidth: float,
) -> None:
    """Plot one posterior spectrum panel on a linear count scale."""

    fill_step_band(
        ax,
        panel["eg_axis"],
        panel["lower_post"],
        panel["upper_post"],
        facecolor=post_band,
        edgecolor=post_band,
        alpha=alpha_band,
        linewidth=edge_linewidth,
        zorder=2.0,
    )

    steps(
        ax,
        panel["eg_axis"],
        panel["mean_post"],
        color=post_mean,
        lw=mean_linewidth,
        zorder=3.0,
    )

    steps(
        ax,
        panel["eg_axis"],
        panel["truth"],
        color=truth_color,
        lw=truth_linewidth,
        ls="--",
        zorder=3.1,
    )

    set_linear_y_with_zero_room(
        ax,
        arrays=[panel["upper_post"], panel["truth"]],
        top_factor=1.12,
        lower_fraction=0.035,
        draw_zero=True,
        zero_color=truth_color,
    )

    ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(rf"${symbol}$ (counts)")

    add_panel_label(ax, label=label)
    apply_axes_style(ax)


def plot_scaled_panel(
    ax: plt.Axes,
    panel: dict,
    label: str,
    show_ylabel: bool,
    dev_band: str,
    dev_mean: str,
    zero_color: str,
    alpha_dev: float,
    mean_linewidth: float,
    edge_linewidth: float,
    zero_linewidth: float,
) -> None:
    """Plot one scaled-deviation panel."""

    fill_step_band(
        ax,
        panel["eg_scaled"],
        panel["scaled_lower"],
        panel["scaled_upper"],
        facecolor=dev_band,
        edgecolor=dev_band,
        alpha=alpha_dev,
        linewidth=edge_linewidth,
        zorder=2.0,
    )

    steps(
        ax,
        panel["eg_scaled"],
        panel["scaled_mean"],
        color=dev_mean,
        lw=mean_linewidth,
        zorder=3.0,
    )

    ax.axhline(
        0.0,
        color=zero_color,
        ls=":",
        lw=zero_linewidth,
        alpha=0.85,
        zorder=1.5,
    )

    ax.set_ylim(
        *scaled_deviation_y_limits_for_panel(
            panel,
            minimum_abs=2.0,
            padding_fraction=0.05,
        )
    )

    ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")

    if show_ylabel:
        ax.set_ylabel(r"$S(E_\gamma)$")

    add_panel_label(ax, label=label)
    apply_axes_style(ax)


def plot_multi_ex_results(
    post_nc: str | Path,
    ex_values: list[float],
    var: str = "eta",
    mass: float = 0.95,
    out: str | Path | None = None,
    show: bool = True,
    layout: str = "text",
) -> plt.Figure:
    """Create the multi-ex posterior results figure."""

    results = UnfoldingResults(post_nc, expected_mode="posterior")

    panels, symbol = make_panel_data(
        results=results,
        ex_values=ex_values,
        var=var,
        mass=mass,
    )

    n_panels = len(panels)
    if n_panels < 1:
        raise ValueError("At least one ex value is required.")

    post_band = COLORS["sky"]
    post_mean = COLORS["orange"]
    dev_band = COLORS["green"]
    dev_mean = COLORS["purple"]
    truth_color = COLORS["black"]

    alpha_band = 0.28
    alpha_dev = 0.18

    mean_linewidth = 0.55
    truth_linewidth = 0.45
    edge_linewidth = 0.35
    zero_linewidth = 0.50

    mass_percent = round(100.0 * mass)

    # Multi-ex results are a text-width paper figure.  Each Ex value is one
    # independent column, so the x axis is shared only within that column.
    size = figure_size("text", height_to_width=0.70)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        2,
        n_panels,
        left=0.10,
        right=0.97,
        bottom=0.10,
        top=0.90,
        wspace=0.18,
        hspace=0.16,
    )

    top_axes: list[plt.Axes] = []
    bottom_axes: list[plt.Axes] = []

    for index, panel in enumerate(panels):
        ax_top = fig.add_subplot(grid[0, index])
        ax_bottom = fig.add_subplot(grid[1, index], sharex=ax_top)

        top_axes.append(ax_top)
        bottom_axes.append(ax_bottom)

        plot_spectrum_panel(
            ax=ax_top,
            panel=panel,
            symbol=symbol,
            label=panel_id(index),
            show_ylabel=index == 0,
            post_band=post_band,
            post_mean=post_mean,
            truth_color=truth_color,
            alpha_band=alpha_band,
            mean_linewidth=mean_linewidth,
            truth_linewidth=truth_linewidth,
            edge_linewidth=edge_linewidth,
        )

        plot_scaled_panel(
            ax=ax_bottom,
            panel=panel,
            label=panel_id(n_panels + index),
            show_ylabel=index == 0,
            dev_band=dev_band,
            dev_mean=dev_mean,
            zero_color=truth_color,
            alpha_dev=alpha_dev,
            mean_linewidth=mean_linewidth,
            edge_linewidth=edge_linewidth,
            zero_linewidth=zero_linewidth,
        )

        set_column_x_axis(
            axes=[ax_top, ax_bottom],
            eg_axis=panel["eg_axis"],
        )

    post_handle = (
        Patch(
            facecolor=rgba(post_band, alpha_band),
            edgecolor=post_band,
            linewidth=edge_linewidth,
        ),
        Line2D([0], [0], color=post_mean, lw=mean_linewidth),
    )
    truth_handle = Line2D(
        [0],
        [0],
        color=truth_color,
        lw=truth_linewidth,
        ls="--",
    )
    deviation_handle = (
        Patch(
            facecolor=rgba(dev_band, alpha_dev),
            edgecolor=dev_band,
            linewidth=edge_linewidth,
        ),
        Line2D([0], [0], color=dev_mean, lw=mean_linewidth),
    )

    add_top_legend(
        fig,
        handles=[post_handle, truth_handle, deviation_handle],
        labels=[
            rf"${symbol}^{{\mathrm{{post}}}}$: mean + {mass_percent}\% rank-env.",
            rf"${symbol}_{{\mathrm{{true}}}}$",
            rf"$S(E_\gamma)$: mean + {mass_percent}\% rank-env.",
        ],
        ncol=3,
        y=0.985,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        fontsize=8.8,
    )

    save_or_show(fig, out, tag="multi-ex-results", show=show)
    return fig
