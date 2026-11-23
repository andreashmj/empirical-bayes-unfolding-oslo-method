"""
Figure illustrating prior dependence in low-count regime.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from ..plot_utils import steps, add_panel_label, add_top_legend, add_zero_line, save_or_show, set_symlog_y_with_room, use_MeV_on_xaxis
from ..style import COLORS, apply_axes_style, figure_size
from ._common import band_handle, bin_edges, fill_bin_band, read_spectrum, envelope


def plot_low_stat_prior_dependence(
    baseline_post_nc, alt_post_ncs, ex=None, out=None, show=True,
):
    draws, truth, _ = read_spectrum(baseline_post_nc, ex)
    Eg, base_mean, base_lower, base_upper = envelope(draws)
    
    edges = bin_edges(Eg)
    alternatives = []
    
    for path in alt_post_ncs:
        draws, _, _ = read_spectrum(path, ex)
        alternatives.append(envelope(draws)[1:])
        
    first, second, third = alternatives
    scale = [base_upper, base_mean, truth]
    
    for mean, lower, upper in alternatives:
        scale.extend((mean, upper))

    base_band = COLORS["sky"]
    base_line = COLORS["blue"]
    alt_band = COLORS["rose"]
    alt_line = COLORS["wine"]
    black = COLORS["black"]
    
    size = figure_size("text", height_to_width=0.92)
    fig = plt.figure(figsize=size.as_tuple())
    grid = fig.add_gridspec(3, 1, left=0.10, right=0.97, bottom=0.07, top=0.92, hspace=0.13)
    
    for row, (mean, lower, upper) in enumerate((first, second, third)):
        ax = fig.add_subplot(grid[row, 0])
        fill_bin_band(ax, Eg, base_lower, base_upper, base_band, 0.24,
                      linewidth=0.32, zorder=1.7)
        fill_bin_band(ax, Eg, lower, upper, alt_band, 0.24, linewidth=0.32)
        
        steps(ax, Eg, base_mean, color=base_line, lw=0.62, zorder=3.0)
        steps(ax, Eg, mean, color=alt_line, lw=0.62, zorder=3.1)
        steps(ax, Eg, truth, color=black, lw=0.45, ls="--", zorder=3.2)
        
        set_symlog_y_with_room(ax, scale, linthresh=5.0, top_factor=3.0, y_lower=-1.0)
        add_zero_line(ax, color=black, lw=0.42, alpha=0.55, zorder=1.1)
        
        ax.set_xlim(-0.025 * edges[-1], edges[-1])
        ax.set_ylabel(r"$\boldsymbol{\eta}$ (counts)")
        
        if row == 2:
            ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
        else:
            ax.tick_params(labelbottom=False)
            
        add_panel_label(ax, "abc"[row])
        use_MeV_on_xaxis(ax)
        apply_axes_style(ax)

    handles = [
        band_handle(base_band, base_line, 0.24, lw=0.62, edgewidth=0.32),
        band_handle(alt_band, alt_line, 0.24, lw=0.62, edgewidth=0.32),
        Line2D([0], [0], color=black, lw=0.45, ls="--"),
    ]
    labels = [r"Baseline post.: mean + 95\% simult. band", r"Alternative post.: mean + 95\% simult. band", r"$\boldsymbol{\eta}_{\mathrm{true}}$"]
    add_top_legend(
        fig, handles, labels, ncol=3, y=0.985, fontsize=8.8,
        handlelength=2.0, columnspacing=1.4,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    save_or_show(fig, out, tag="low-stat-prior-dependence", show=show)
    return fig
