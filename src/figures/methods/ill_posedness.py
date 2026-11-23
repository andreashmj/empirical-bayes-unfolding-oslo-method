"""
Figure showing the same posterior draws in x-space and eta-space, alongside the corresponding truth.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ...analysis.results import UnfoldingResults
from ...paths import repo_path
from ..plot_utils import (
    add_panel_label, add_top_legend, add_zero_line, save_or_show,
    set_symlog_y_with_room, steps, use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size


def plot_ill_posedness(
    nc_path: str | Path, ex: float | None = None, max_draws: int = 10,
    seed: int = 123, out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:

    results = UnfoldingResults(repo_path(nc_path), expected_mode="posterior")
    try:
        ex_actual = results.ex_actual(ex)
        Eg = results.eg(ex_actual)
        
        bin_width = Eg[1] - Eg[0]
        edges = np.append(Eg, Eg[-1] + bin_width)
        
        x = results.x(ex_actual).transpose("sample", "Eg").values
        rng = np.random.default_rng(seed)
        draw_count = min(max_draws, len(x))
        
        indices = rng.choice(len(x), size=draw_count, replace=False)
        x = x[indices]
        
        _, G_g, _ = results.ops(ex_actual)
        row = results.ds.sel(Ex=ex_actual).isel(Eg=slice(0, len(Eg)))
        
        truths = (row["x_mixed_true"].values.copy(), row["eta_true"].values.copy())
        draws = (x, x @ G_g)
    finally:
        results.ds.close()

    fig = plt.figure(figsize=figure_size("column", height_to_width=1.05).as_tuple())
    grid = fig.add_gridspec(2, 1, left=0.15, right=0.97, bottom=0.12, top=0.83, hspace=0.08)
    
    axes = [fig.add_subplot(grid[0, 0])]
    axes.append(fig.add_subplot(grid[1, 0], sharex=axes[0]))
    
    colors = (COLORS["sky"], COLORS["green"])
    truth_colors = (COLORS["black"], COLORS["orange"])
    symbols = (r"\boldsymbol{x}", r"\boldsymbol{\eta}")
    
    truth_handles = []
    
    for i, (ax, samples, truth, color, tcolor, symbol) in enumerate(
        zip(axes, draws, truths, colors, truth_colors, symbols)
    ):
        for sample in samples:
            steps(ax, Eg, sample, color=color, lw=(0.20, 0.30)[i], alpha=(0.15, 0.35)[i], zorder=1)
            
        steps(ax, Eg, truth, color=tcolor, lw=0.45, ls="--", zorder=4)
        truth_handles.append(Line2D([0], [0], color=tcolor, lw=0.45, ls="--"))
        _, top = set_symlog_y_with_room(ax, [samples, truth], linthresh=5.0, top_factor=1.55, y_lower=-1.0)
        
        ticks = 10.0 ** np.arange(1, int(np.ceil(np.log10(top))) + 1)
        
        ax.set_yticks(np.r_[0.0, ticks[ticks <= top]])
        add_zero_line(ax, color=COLORS["black"], lw=0.45, alpha=0.55)
        ax.set_ylabel(rf"${symbol}$ (counts)")
        apply_axes_style(ax)
        use_MeV_on_xaxis(ax)
        
        add_panel_label(ax, "ab"[i], x=0.04, y=0.93, fontsize=9.0)
        
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    axes[1].set_xlim(-0.025 * edges[-1], edges[-1])
    
    handles = [Line2D([0], [0], color=color, lw=0.9, alpha=0.9) for color in colors]
    labels = [rf"${symbol}$ posterior draws" for symbol in symbols]
    labels += [rf"${symbol}_{{\mathrm{{true}}}}$" for symbol in symbols]
    
    add_top_legend(fig, handles + truth_handles, labels, ncol=2, y=0.985, fontsize=8.4)
    
    print(f"Plotted paired draws={len(x)}")
    save_or_show(fig, repo_path(out) if out is not None else None, tag="ill-posedness", show=show)
    return fig
