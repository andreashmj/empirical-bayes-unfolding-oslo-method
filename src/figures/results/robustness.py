"""
Figure showing the robustness of the posterior to 4 different changes of the model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from ..plot_utils import add_panel_label, add_top_legend, save_or_show
from ..style import figure_size
from ._common import bin_edges, comparison_envelopes, comparison_handles, plot_scaled_panel, scaled_limits


def plot_robustness(baseline_post_nc, alt_post_ncs, ex=None, out=None, show=True):
    reference, alternatives = comparison_envelopes(baseline_post_nc, alt_post_ncs, ex)
    first, second, third, fourth = alternatives
    edges = bin_edges(reference[0])
    
    band = "#5AAE61"
    line = "#762A83"
    
    size = figure_size("text", height_to_width=0.76)
    fig = plt.figure(figsize=size.as_tuple())
    grid = fig.add_gridspec(2, 2, left=0.08, right=0.97, bottom=0.10, top=0.90, wspace=0.18, hspace=0.16)
    
    limits = [scaled_limits(reference, alt, padding_fraction=0.03) for alt in alternatives]
    common_ylim = (min(low for low, high in limits), max(high for low, high in limits))

    axes = []

    for index, alternative in enumerate(alternatives):
        row, column = divmod(index, 2)

        ax = fig.add_subplot(grid[row, column], sharey=axes[0] if axes else None)
        axes.append(ax)

        plot_scaled_panel(ax, edges, reference, alternative, band, line)

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
    handles[1][1].set_linewidth(0.60)
    
    legend_labels = [r"$S_{\mathrm{alt}}$: mean + 95\% simult. band", r"$S_{\mathrm{base}}$: mean + 95\% simult. band", r"reference levels"]
    add_top_legend(
        fig, handles, legend_labels, ncol=3, y=0.978, fontsize=9.0,
        handlelength=2.2, columnspacing=1.4,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    save_or_show(fig, out, tag="robustness", show=show)
    return fig
