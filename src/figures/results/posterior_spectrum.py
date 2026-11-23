"""
Figure showing the posterior spectrum, including scaled deviation panel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from ..plot_utils import (
    steps,
    add_panel_label, add_top_legend, add_zero_line, add_zoom_guides,
    local_y_limits, save_or_show, set_linear_y_with_zero_room,
    set_local_y_axis, use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size
from ._common import band_handle, bin_edges, fill_bin_band, read_spectrum, scaled_envelope, envelope


def parse_zoom_windows(zoom):
    windows = []
    for interval in zoom.split(","):
        low, high = interval.split("-")
        windows.append((float(low), float(high)))
    first, second = windows
    return first, second


def plot_spectrum(ax, Eg, summary, truth):
    mean, lower, upper = summary
    fill_bin_band(ax, Eg, lower, upper, COLORS["sky"], 0.28)
    
    steps(ax, Eg, mean, color=COLORS["orange"], lw=0.55, zorder=3.0)
    steps(ax, Eg, truth, color=COLORS["black"], lw=0.45, ls="--", zorder=3.1)
    apply_axes_style(ax)
    use_MeV_on_xaxis(ax)


def plot_posterior_spectrum(post_nc, zoom, ex=None, var="eta", out=None, show=True):
    draws, truth, symbol = read_spectrum(post_nc, ex, var=var)
    Eg, mean, lower, upper = envelope(draws)
    
    scaled_mean, scaled_lower, scaled_upper = scaled_envelope(draws, truth, lower, upper)
    edges = bin_edges(Eg)
    windows = parse_zoom_windows(zoom)
    summary = mean, lower, upper

    size = figure_size("text", height_to_width=0.92)
    fig = plt.figure(figsize=size.as_tuple())
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.08, 0.90, 0.72],
        left=0.10, right=0.97, bottom=0.075, top=0.935,
        wspace=0.20, hspace=0.16,
    )
    
    full = fig.add_subplot(grid[0, :])
    plot_spectrum(full, Eg, summary, truth)
    set_linear_y_with_zero_room(full, [upper, truth], top_factor=1.12, lower_fraction=0.035)
    
    full.set_xlim(-0.025 * edges[-1], edges[-1])
    full.set_ylabel(rf"${symbol}$ (counts)")
    full.tick_params(labelbottom=False)
    add_panel_label(full, "a")

    zoom_axes = []
    for column, window in enumerate(windows):
        ax = fig.add_subplot(grid[1, column])
        zoom_axes.append(ax)
        plot_spectrum(ax, Eg, summary, truth)
        
        mask = (edges[:-1] < window[1]) & (edges[1:] > window[0])
        y_low, y_high = local_y_limits(mask, [lower, upper, mean, truth])
        ax.set_xlim(*window)
        
        set_local_y_axis(ax, y_low, 1.1*y_high, nbins=4)
        add_zero_line(ax, color=COLORS["black"])
        
        if column == 0:
            ax.set_ylabel(rf"${symbol}$ (counts)")
        add_panel_label(ax, "bc"[column])
        
    add_zoom_guides(fig, full, zoom_axes, windows, color=COLORS["gray"])

    deviation = fig.add_subplot(grid[2, :], sharex=full)
    fill_bin_band(deviation, Eg, scaled_lower, scaled_upper, COLORS["green"], 0.18)
    
    steps(deviation, Eg, scaled_mean, color=COLORS["purple"], lw=0.55, zorder=3.0)
    deviation.axhline(0.0, color=COLORS["black"], ls=":", lw=0.55, alpha=0.85)
    deviation.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    deviation.set_ylabel(r"$S(E_\gamma)$")
    deviation.set_ylim([-2.0, 3.0])
    
    apply_axes_style(deviation)
    use_MeV_on_xaxis(deviation)
    add_panel_label(deviation, "d")

    handles = [
        band_handle(COLORS["sky"], COLORS["orange"], 0.28),
        Line2D([0], [0], color=COLORS["black"], lw=0.45, ls="--"),
        band_handle(COLORS["green"], COLORS["purple"], 0.18),
    ]
    labels = [rf"${symbol}^{{\mathrm{{post}}}}$: mean + 95\% simult. band", rf"${symbol}_{{\mathrm{{true}}}}$", r"$S(E_\gamma)$: mean + 95\% simult. band"]
    add_top_legend(
        fig, handles, labels, ncol=3, y=0.992, fontsize=8.8,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    save_or_show(fig, out, tag="posterior-spectrum", show=show)
    return fig
