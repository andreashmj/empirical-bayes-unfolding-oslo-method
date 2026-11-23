"""
Figure with 9 panels showing the prior/posterior predictive checks for prompt, background and folded signal.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


from ...analysis.results import UnfoldingResults
from ...paths import repo_path
from ..plot_utils import steps, add_panel_label, add_top_legend, add_zero_line, save_or_show, set_symlog_y_with_room, use_MeV_on_xaxis
from ..style import COLORS, apply_axes_style, figure_size
from ._common import band_handle, bin_edges, fill_bin_band, envelope, scaled_envelope


def check_summaries(prior_nc, post_nc, ex):
    prior = UnfoldingResults(repo_path(prior_nc), expected_mode="prior")
    posterior = UnfoldingResults(repo_path(post_nc), expected_mode="posterior")
    
    try:
        ex_actual = posterior.ex_actual(ex)
        Eg = posterior.eg(ex_actual)
        references = [posterior.n_obs(ex_actual), posterior.n_off_obs(ex_actual), posterior.nu_true(ex_actual)]
        prior_rates = [prior.total_expected(ex_actual), prior.bg(ex_actual), prior.nu(ex_actual)]
        posterior_rates = [posterior.total_expected(ex_actual), posterior.bg(ex_actual), posterior.nu(ex_actual)]
    finally:
        prior.ds.close()
        posterior.ds.close()

    rng = np.random.default_rng(123)
    rows = []
    for index, reference in enumerate(references):
        prior_draws = prior_rates[index]
        posterior_draws = posterior_rates[index]
        if index < 2:
            
            prior_counts = rng.poisson(np.clip(prior_draws.values, 0.0, None)).astype(float)
            posterior_counts = rng.poisson(np.clip(posterior_draws.values, 0.0, None)).astype(float)
            prior_draws = prior_draws.copy(data=prior_counts)
            posterior_draws = posterior_draws.copy(data=posterior_counts)
            
        prior_summary = envelope(prior_draws)[1:]
        posterior_summary = envelope(posterior_draws)[1:]
        
        _, lower, upper = posterior_summary
        scaled = scaled_envelope(posterior_draws, reference, lower, upper)
        rows.append((prior_summary, posterior_summary, scaled, reference))
        
    return Eg, rows


def plot_model_checks(prior_nc, post_nc, ex=None, out=None, show=True):
    Eg, rows = check_summaries(prior_nc, post_nc, ex)
    edges = bin_edges(Eg)
    
    size = figure_size("text", height_to_width=0.90)
    fig = plt.figure(figsize=size.as_tuple())
    grid = fig.add_gridspec(1, 2, left=0.10, right=0.97, bottom=0.08, top=0.90, wspace=0.20, width_ratios=[2.0, 1.0])
    
    count_grid = grid[0].subgridspec(3, 2, wspace=0.15)
    scaled_grid = grid[1].subgridspec(3, 1)
    
    styles = ((COLORS["vermillion"], COLORS["blue"]), (COLORS["sky"], COLORS["orange"]))
    ylabels = (r"$\mathbf{n}$ (counts)", r"$\mathbf{n}_{\mathrm{off}}$ (counts)", r"$\boldsymbol{\nu}$ (counts)")
    
    
    scaled_axes = []
    scaled_bounds = []
    
    for row, (prior, posterior, scaled, reference) in enumerate(rows):
        prior_ax = fig.add_subplot(count_grid[row, 0])
        posterior_ax = fig.add_subplot(count_grid[row, 1], sharex=prior_ax, sharey=prior_ax)
        
        scaled_ax = fig.add_subplot(scaled_grid[row, 0], sharex=prior_ax, sharey=scaled_axes[0] if scaled_axes else None)
        scaled_axes.append(scaled_ax)

        for column, (ax, summary, style) in enumerate(zip((prior_ax, posterior_ax), (prior, posterior), styles)):
            mean, lower, upper = summary
            band, line = style
            fill_bin_band(ax, Eg, lower, upper, band, 0.28)
            steps(ax, Eg, mean, color=line, lw=0.55, zorder=3.0)
            steps(ax, Eg, reference, color=COLORS["black"], lw=0.40, ls="--", zorder=3.1)
            set_symlog_y_with_room(ax, [prior[2], posterior[2], reference], linthresh=5.0, top_factor=7.0, y_lower=-1.0)
            add_zero_line(ax, color=COLORS["black"], lw=0.45, alpha=0.55, zorder=1.1)
            add_panel_label(ax, "abcdefghi"[3 * row + column])

        prior_ax.set_ylabel(ylabels[row])
        posterior_ax.tick_params(labelleft=False)

        scaled_mean, scaled_lower, scaled_upper = scaled
        scaled_min = min(float(np.min(scaled_lower)), float(np.min(scaled_mean)), -2.0)
        scaled_max = max(float(np.max(scaled_upper)), float(np.max(scaled_mean)), 2.0)
        scaled_bounds.append((scaled_min, scaled_max))

        fill_bin_band(scaled_ax, Eg, scaled_lower, scaled_upper, COLORS["green"], 0.18)
        
        steps(scaled_ax, Eg, scaled_mean, color=COLORS["purple"], lw=0.30, zorder=3.0)
        scaled_ax.axhline(0.0, color=COLORS["black"], ls=":", lw=0.50, alpha=0.85)
        scaled_ax.set_ylabel(r"$S(E_\gamma)$")
        
        add_panel_label(scaled_ax, "cfi"[row])

        for ax in (prior_ax, posterior_ax, scaled_ax):
            if row == 2:
                ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
            else:
                ax.tick_params(labelbottom=False)
                
            ax.set_xlim(-0.025 * edges[-1], edges[-1])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            use_MeV_on_xaxis(ax)
            apply_axes_style(ax)

    common_low = min(low for low, high in scaled_bounds)
    common_high = max(high for low, high in scaled_bounds)
    padding = 0.1 * (common_high - common_low)

    for ax in scaled_axes:
        ax.set_ylim(common_low - padding, common_high + padding)

    handles = [
        band_handle(COLORS["vermillion"], COLORS["blue"], 0.28),
        band_handle(COLORS["sky"], COLORS["orange"], 0.28),
        Line2D([0], [0], color=COLORS["black"], lw=0.40, ls="--"),
        band_handle(COLORS["green"], COLORS["purple"], 0.18),
    ]
    labels = [r"Prior: mean + 95\% simult. band", r"Post.: mean + 95\% simult. band", "Ref.", r"$S(E_\gamma)$: mean + 95\% simult. band"]
    add_top_legend(fig, handles, labels, ncol=2, y=0.98, fontsize=9.0, handler_map={tuple: HandlerTuple(ndivide=1)})
    save_or_show(fig, out, tag="model-checks", show=show)
    return fig
