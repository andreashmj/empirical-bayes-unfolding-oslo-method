"""
Figure comparing the prior and posterior in x or eta space.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D


from ..plot_utils import steps, add_top_legend, add_zero_line, save_or_show, set_symlog_y_with_room, use_MeV_on_xaxis
from ..style import COLORS, apply_axes_style, figure_size
from ._common import band_handle, bin_edges, fill_bin_band, read_spectrum, envelope


def plot_prior_posterior_comparison(prior_nc, post_nc, ex=None, var="eta", out=None, show=True):

    prior_draws, _, symbol = read_spectrum(prior_nc, ex, mode="prior", var=var)
    post_draws, truth, _ = read_spectrum(post_nc, ex, var=var)
    
    Eg, prior_mean, prior_lower, prior_upper = envelope(prior_draws)
    _, post_mean, post_lower, post_upper = envelope(post_draws)
    
    edges = bin_edges(Eg)

    prior_band = COLORS["vermillion"]
    prior_line = COLORS["orange"]
    post_band = COLORS["sky"]
    post_line = COLORS["blue"]
    black = COLORS["black"]

    size = figure_size("column", height_to_width=0.90)
    fig = plt.figure(figsize=size.as_tuple())
    
    grid = fig.add_gridspec(1, 1, left=0.17, right=0.97, bottom=0.18, top=0.79)
    ax = fig.add_subplot(grid[0, 0])
    
    fill_bin_band(ax, Eg, prior_lower, prior_upper, prior_band, 0.22, zorder=1.8)
    fill_bin_band(ax, Eg, post_lower, post_upper, post_band, 0.28)
    
    steps(ax, Eg, prior_mean, color=prior_line, lw=0.65, zorder=3.0)
    steps(ax, Eg, post_mean, color=post_line, lw=0.65, zorder=3.1)
    steps(ax, Eg, truth, color=black, lw=0.45, ls="--", zorder=3.2)
    
    
    set_symlog_y_with_room(ax, [prior_upper, post_upper, truth], linthresh=5.0, top_factor=1.25, y_lower=-1.0)
    
    
    add_zero_line(ax, color=black, lw=0.45, alpha=0.55, zorder=1.1)
    ax.set_xlim(-0.025 * edges[-1], edges[-1])
    ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax.set_ylabel(rf"${symbol}$ (counts)")
    
    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)

    handles = [
        band_handle(prior_band, prior_line, 0.22, lw=0.65),
        band_handle(post_band, post_line, 0.28, lw=0.65),
        Line2D([0], [0], color=black, lw=0.45, ls="--"),
    ]
    labels = [r"Prior: mean + 95\% simult. band", r"Post.: mean + 95\% simult. band", rf"${symbol}_{{\mathrm{{true}}}}$"]
    
    add_top_legend(
        fig, handles, labels, ncol=1, y=0.985, handlelength=1.7,
        columnspacing=0.9, labelspacing=0.32, fontsize=8.0,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    save_or_show(fig, out, tag="prior-posterior-comparison", show=show)
    return fig
