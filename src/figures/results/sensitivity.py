"""
Sensitivity plots for RL, alpha and sigma cases.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from ...analysis.results import UnfoldingResults
from ...paths import repo_path
from ..plot_utils import add_panel_label, add_top_legend, save_or_show, use_MeV_on_xaxis
from ..style import COLORS, apply_axes_style, figure_size
from ._common import bin_edges, comparison_envelopes, comparison_handles, plot_scaled_panel, scaled_limits



def plot_sensitivity(baseline_post_nc, alt_post_ncs, test, ex=None, out=None, show=True):
    reference, alternatives = comparison_envelopes(baseline_post_nc, alt_post_ncs, ex)
    edges = bin_edges(reference[0])
    band, line = {
        "rl": (COLORS["sky"], COLORS["orange"]),
        "alpha": (COLORS["wine"], COLORS["green"]),
        "sigma": (COLORS["vermillion"], COLORS["blue"]),
    }[test]

    if test == "sigma":
        first, second, third, fourth = alternatives
        panels = ((0, 0, first), (0, 1, second), (1, 0, third), (1, 1, fourth))
        size = figure_size("text", height_to_width=0.76)
        fig = plt.figure(figsize=size.as_tuple())
        grid = fig.add_gridspec(2, 2, left=0.08, right=0.97, bottom=0.08, top=0.90, wspace=0.12, hspace=0.16)
    else:
        first, second = alternatives
        panels = ((0, 0, first), (1, 0, second))
        size = figure_size("column", height_to_width=1.18)
        fig = plt.figure(figsize=size.as_tuple())
        grid = fig.add_gridspec(2, 1, left=0.17, right=0.97, bottom=0.10, top=0.82, hspace=0.20)
                               
                              
    limits = [scaled_limits(reference, alt) for alt in alternatives]
    common_ylim = (min(low for low, high in limits), max(high for low, high in limits))

    axes = []

    for index, (row, column, alternative) in enumerate(panels):
        ax = fig.add_subplot(grid[row, column], sharey=axes[0] if axes else None)
        axes.append(ax)
        plot_scaled_panel(ax, edges, reference, alternative, band, line)
        
        
        ax.axhline(0.0, color=COLORS["black"], lw=0.95, alpha=0.55, zorder=2.35)
        if column == 0:
            ax.set_ylabel(r"$S(E_\gamma)$")
        else:
            ax.tick_params(labelleft=False)
            
        if row == 1:
            ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
        else:
            ax.tick_params(labelbottom=False)
            
            
        add_panel_label(ax, "abcd"[index])
        
    
    
    axes[0].set_ylim(*common_ylim)
    handles = comparison_handles(band, line)
    
    legend_labels = [r"$S_{\mathrm{alt}}$: mean + 95\% simult. band", r"$S_{\mathrm{base}}$: mean + 95\% simult. band", r"reference levels"]
    if test == "sigma":
        add_top_legend(fig, handles, legend_labels, ncol=3, y=0.985, fontsize=9.0, handler_map={tuple: HandlerTuple(ndivide=1)})
    else:
        add_top_legend(fig, handles, legend_labels, ncol=1, y=0.965, fontsize=7.8,
                       handlelength=1.7, columnspacing=0.9, labelspacing=0.25,
                       handler_map={tuple: HandlerTuple(ndivide=1)})
    save_or_show(fig, out, tag="sensitivity", show=show)
    return fig
