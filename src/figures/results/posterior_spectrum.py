"""
Posterior spectrum figure.

The figure shows the posterior mean and global rank envelope for one
excitation-energy row, two zoom panels, and the scaled-deviation envelope.
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
    add_zoom_guides,
    fill_step_band,
    local_y_limits,
    rgba,
    save_or_show,
    set_linear_y_with_zero_room,
    set_local_y_axis,
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


def parse_zoom_windows(
    zoom: str | None,
    eg_axis: np.ndarray,
    truth: np.ndarray,
) -> list[tuple[float, float]]:
    """Return two zoom windows in keV."""

    if zoom is None:
        peak_index = np.nanargmax(truth).item()
        half_width = max(5, eg_axis.size // 30)

        start = max(0, peak_index - half_width)
        stop = min(eg_axis.size - 1, peak_index + half_width)
        middle = (start + stop) // 2

        return [
            (eg_axis[start].item(), eg_axis[middle].item()),
            (eg_axis[middle].item(), eg_axis[stop].item()),
        ]

    windows: list[tuple[float, float]] = []

    for token in zoom.split(","):
        lower_text, upper_text = token.split("-")
        lower = float(lower_text)
        upper = float(upper_text)

        if lower > upper:
            lower, upper = upper, lower

        windows.append((lower, upper))

    if len(windows) != 2:
        raise ValueError(
            "zoom must contain exactly two windows, for example '1000-1200,2700-3400'."
        )
    return windows


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


def plot_spectrum_band(
    ax: plt.Axes,
    eg_axis: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    band_color: str,
    mean_color: str,
    truth_color: str,
    band_alpha: float,
    mean_linewidth: float,
    truth_linewidth: float,
    band_edge_width: float,
) -> None:
    """Plot one spectrum band, mean line, and truth line."""

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

    steps(
        ax,
        eg_axis,
        mean,
        color=mean_color,
        lw=mean_linewidth,
        zorder=3.0,
    )

    steps(
        ax,
        eg_axis,
        truth,
        color=truth_color,
        lw=truth_linewidth,
        ls="--",
        zorder=3.1,
    )

def plot_posterior_spectrum(
    post_nc: str | Path,
    ex: float | None = None,
    var: str = "eta",
    mass: float = 0.95,
    zoom: str | None = None,
    symlog_linthresh: float | None = None,
    out: str | Path | None = None,
    show: bool = True,
    layout: str = "text",
) -> plt.Figure:
    """Create the posterior spectrum figure."""
    results = UnfoldingResults(post_nc, expected_mode="posterior")
    
    ex_actual = results.ex_actual(ex)
    draws, truth_raw, symbol = spectrum_draws_and_truth(results, ex_actual, var)
    eg_axis, mean_post, lower_post, upper_post = global_rank_envelope(
        draws,
        mass=mass,
    )
    truth = np.asarray(truth_raw, dtype=float)
    windows = parse_zoom_windows(
        zoom=zoom,
        eg_axis=eg_axis,
        truth=truth,
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

    post_band = COLORS["sky"]
    post_mean = COLORS["orange"]
    deviation_band = COLORS["green"]
    deviation_mean = COLORS["purple"]
    truth_color = COLORS["black"]

    band_alpha = 0.28
    deviation_alpha = 0.18

    mean_linewidth = 0.55
    truth_linewidth = 0.45
    band_edge_width = 0.35
    zero_linewidth = 0.55

    mass_percent = round(100.0 * mass)

    size = figure_size("text", height_to_width=0.92)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.08, 0.90, 0.72],
        width_ratios=[1.0, 1.0],
        left=0.10,
        right=0.97,
        bottom=0.075,
        top=0.935,
        wspace=0.20,
        hspace=0.16,
    )

    ax_overview = fig.add_subplot(grid[0, :])
    plot_spectrum_band(
        ax=ax_overview,
        eg_axis=eg_axis,
        mean=mean_post,
        lower=lower_post,
        upper=upper_post,
        truth=truth,
        band_color=post_band,
        mean_color=post_mean,
        truth_color=truth_color,
        band_alpha=band_alpha,
        mean_linewidth=mean_linewidth,
        truth_linewidth=truth_linewidth,
        band_edge_width=band_edge_width,
    )

    set_linear_y_with_zero_room(
        ax_overview,
        arrays=[upper_post, truth],
        top_factor=1.12,
        lower_fraction=0.035,
        draw_zero=True,
        zero_color=truth_color,
    )

    ax_overview.set_ylabel(rf"${symbol}$ (counts)")
    ax_overview.tick_params(labelbottom=False)

    x_right = np.max(eg_axis).item()
    ax_overview.set_xlim(-0.025 * x_right, x_right)

    apply_axes_style(ax_overview)
    use_MeV_on_xaxis(ax_overview)
    add_panel_label(ax_overview, label="a")

    zoom_axes: list[plt.Axes] = []

    for panel_index, (window, label) in enumerate(zip(windows, ["b", "c"])):
        ax_zoom = fig.add_subplot(grid[1, panel_index])
        zoom_axes.append(ax_zoom)

        lower, upper = window
        mask = (eg_axis >= lower) & (eg_axis <= upper)

        if not np.any(mask):
            ax_zoom.axis("off")
            continue
            
        plot_spectrum_band(
            ax=ax_zoom,
            eg_axis=eg_axis[mask],
            mean=mean_post[mask],
            lower=lower_post[mask],
            upper=upper_post[mask],
            truth=truth[mask],
            band_color=post_band,
            mean_color=post_mean,
            truth_color=truth_color,
            band_alpha=band_alpha,
            mean_linewidth=mean_linewidth,
            truth_linewidth=truth_linewidth,
            band_edge_width=band_edge_width,
        )

        y_lower, y_upper = local_y_limits(
            mask,
            [lower_post, upper_post, mean_post, truth],
        )
        ax_zoom.set_xlim(lower, upper)
        set_local_y_axis(ax_zoom, y_lower, y_upper, nbins=4)

        if panel_index == 0:
            ax_zoom.set_ylabel(rf"${symbol}$ (counts)")

        apply_axes_style(ax_zoom)
        use_MeV_on_xaxis(ax_zoom)
        add_panel_label(ax_zoom, label=label)

    add_zoom_guides(
        fig=fig,
        ax_overview=ax_overview,
        zoom_axes=zoom_axes,
        windows=windows,
        color=COLORS["gray"],
    )

    ax_deviation = fig.add_subplot(grid[2, :], sharex=ax_overview)

    fill_step_band(
        ax_deviation,
        eg_scaled,
        scaled_lower,
        scaled_upper,
        facecolor=deviation_band,
        edgecolor=deviation_band,
        alpha=deviation_alpha,
        linewidth=band_edge_width,
        zorder=2.0,
    )

    steps(
        ax_deviation,
        eg_scaled,
        scaled_mean,
        color=deviation_mean,
        lw=mean_linewidth,
        zorder=3.0,
    )

    ax_deviation.axhline(
        0.0,
        color=truth_color,
        ls=":",
        lw=zero_linewidth,
        alpha=0.85,
    )

    ax_deviation.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax_deviation.set_ylabel(r"$S(E_\gamma)$")

    apply_axes_style(ax_deviation)
    use_MeV_on_xaxis(ax_deviation)
    add_panel_label(ax_deviation, label="d")

    post_handle = (
        Patch(
            facecolor=rgba(post_band, band_alpha),
            edgecolor=post_band,
            linewidth=band_edge_width,
        ),
        Line2D([0], [0], color=post_mean, lw=mean_linewidth),
    )
    truth_handle = Line2D([0], [0], color=truth_color, lw=truth_linewidth, ls="--")
    deviation_handle = (
        Patch(
            facecolor=rgba(deviation_band, deviation_alpha),
            edgecolor=deviation_band,
            linewidth=band_edge_width,
        ),
        Line2D([0], [0], color=deviation_mean, lw=mean_linewidth),
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
        y=0.992,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        fontsize=8.8,
    )

    save_or_show(fig, out, tag="posterior-spectrum", show=show)
    return fig
