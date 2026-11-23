"""
Make figure that plots two different eta spectra and their scaled deviations with respect to the truth.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..plot_utils import steps, add_panel_label, add_top_legend, save_or_show, set_linear_y_with_zero_room, use_MeV_on_xaxis
from ..style import COLORS, apply_axes_style, figure_size
from ._common import band_handle, bin_edges, fill_bin_band, read_spectrum, envelope, scaled_envelope



def plot_multi_ex_results(post_nc, ex_values, out=None, show=True):
    first_ex, second_ex = ex_values
    size = figure_size("text", height_to_width=0.70)
    fig = plt.figure(figsize=size.as_tuple())
    grid = fig.add_gridspec(2, 2, left=0.10, right=0.97, bottom=0.10, top=0.90, wspace=0.18, hspace=0.16)

    for column, ex in enumerate((first_ex, second_ex)):
        draws, truth, _ = read_spectrum(post_nc, ex)
        Eg, mean, lower, upper = envelope(draws)
        scaled = scaled_envelope(draws, truth, lower, upper)
        edges = bin_edges(Eg)
        spectrum = fig.add_subplot(grid[0, column])
        deviation = fig.add_subplot(grid[1, column], sharex=spectrum)

        fill_bin_band(spectrum, Eg, lower, upper, COLORS["sky"], 0.28)
        steps(spectrum, Eg, mean, color=COLORS["orange"], lw=0.55, zorder=3.0)
        steps(spectrum, Eg, truth, color=COLORS["black"], lw=0.45, ls="--", zorder=3.1)
        
        set_linear_y_with_zero_room(spectrum, [upper, truth], top_factor=1.12, lower_fraction=0.035)
        spectrum.tick_params(labelbottom=False)
        add_panel_label(spectrum, "ab"[column])

        scaled_mean, scaled_lower, scaled_upper = scaled
        fill_bin_band(deviation, Eg, scaled_lower, scaled_upper, COLORS["green"], 0.18)
        steps(deviation, Eg, scaled_mean, color=COLORS["purple"], lw=0.55, zorder=3.0)
        
        deviation.axhline(0.0, color=COLORS["black"], ls=":", lw=0.50, alpha=0.85)
        values = np.concatenate(scaled)
        

        
        deviation.set_ylim(bottom=1.05*values.min(), top=1.35*values.max() )
        deviation.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
        add_panel_label(deviation, "cd"[column])

        if column == 0:
            spectrum.set_ylabel(r"$\boldsymbol{\eta}$ (counts)")
            deviation.set_ylabel(r"$S(E_\gamma)$")
        spectrum.set_xlim(-0.025 * edges[-1], edges[-1])
        spectrum.xaxis.set_major_locator(MaxNLocator(nbins=5))
        for ax in (spectrum, deviation):
            use_MeV_on_xaxis(ax)
            apply_axes_style(ax)

    handles = [
        band_handle(COLORS["sky"], COLORS["orange"], 0.28),
        Line2D([0], [0], color=COLORS["black"], lw=0.45, ls="--"),
        band_handle(COLORS["green"], COLORS["purple"], 0.18)
    ]
    labels = [r"$\boldsymbol{\eta}^{\mathrm{post}}$: mean + 95\% simult. band", r"$\boldsymbol{\eta}_{\mathrm{true}}$", r"$S(E_\gamma)$: mean + 95\% simult. band"]
    add_top_legend(
        fig, handles, labels, ncol=3, y=0.985, fontsize=8.8,
        handlelength=2.0, columnspacing=1.2,
        handler_map={tuple: HandlerTuple(ndivide=1)}
    )
    save_or_show(fig, out, tag="multi-ex-results", show=show)
    return fig
