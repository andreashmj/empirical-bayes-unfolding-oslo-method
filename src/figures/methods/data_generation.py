"""
Figure illustrating the synthetic data generation. 
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

from ...config_utils import load_run_config
from ...paths import repo_path
from ...synthetic_data import SyntheticDataLoader
from ..plot_utils import add_panel_label, add_top_legend, add_zero_line, save_or_show, set_symlog_y_with_room, steps, use_MeV_on_xaxis, use_MeV_on_yaxis
from ..style import COLORS, apply_axes_style, figure_size


def load_synthetic(config: dict) -> SyntheticDataLoader:
    return SyntheticDataLoader(
        mat_path=repo_path(config["mat_path"]),
        response_db=config["response_db"],
        rebin_factors=tuple(config["rebin_factors"]),
        mat_scale=config["mat_scale"],
        lower_eg_cut=config["lower_eg_cut"],
        include_background=config["include_background"],
        bg_fraction=config["bg_fraction"],
        bg_flat_fraction=config["bg_flat_fraction"],
        eg_tail_mass=config["eg_tail_mass"],
        rng_seed=config["rng_seed"],
        fwhm_eg_gen=config["fwhm_eg_gen"],
        include_ex_smearing=config["include_ex_smearing"],
        fwhm_ex_gen=config["fwhm_ex_gen"],
    )


def plot_data_generation(
    config_path: str | Path,
    ex_values: tuple[float, float] = (2500.0, 8000.0),
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:

    config = load_run_config(repo_path(config_path))

    loader = load_synthetic(config)
    emitted, observed = loader.x_true, loader.n

    values = (np.asarray(emitted.values), np.asarray(observed.values))
    selected = [emitted.index_X(ex) for ex in ex_values]
    
    Ex = emitted.X_index.to_left()
    Eg = emitted.Y_index.to_left()
    
    ex_edges = np.append(Ex.bins, Ex.rightmost)
    eg_edges = np.append(Eg.bins, Eg.rightmost)

    lengths = loader.active_eg_lengths
    colors = (COLORS["blue"], COLORS["vermillion"])

    positive = np.concatenate([v[v > 0] for v in values])
    norm = LogNorm(1.0, max(float(np.percentile(positive, 99.5)), 1.01))

    fig = plt.figure(figsize=figure_size("text", height_to_width=0.90).as_tuple())
    grid = fig.add_gridspec(
        4, 2, height_ratios=[2.25, 0.20, 1.05, 1.05],
        left=0.115, right=0.975, bottom=0.080, top=0.925,
        wspace=0.12, hspace=0.22,
    )
    matrices = [fig.add_subplot(grid[0, 0])]
    matrices.append(fig.add_subplot(grid[0, 1], sharex=matrices[0], sharey=matrices[0]))

    for ax, matrix, label in zip(matrices, values, "ab"):
        image = ax.imshow(
            np.ma.masked_less_equal(matrix, 0), origin="lower", aspect="auto",
            extent=[eg_edges[0], eg_edges[-1], ex_edges[0], ex_edges[-1]],
            norm=norm, cmap="viridis", interpolation="nearest", rasterized=True,
        )
        
        for index, color in zip(selected, colors):
            ax.axhspan(ex_edges[index], ex_edges[index + 1], facecolor=color, alpha=0.30, edgecolor=color, linewidth=0.8, zorder=3)
            
        ax.tick_params(labelbottom=False)
        
        use_MeV_on_xaxis(ax)
        use_MeV_on_yaxis(ax)
        
        apply_axes_style(ax)
        add_panel_label(ax, label, x=0.035, y=0.965, fontsize=9.0,
                        box=True, box_alpha=0.72)

    matrices[0].set_ylabel(r"Excitation energy $E_x$ (MeV)")
    matrices[1].tick_params(labelleft=False)

    left, right = matrices[0].get_position(), matrices[1].get_position()
    width = 0.68 * (right.x1 - left.x0)
    cax = fig.add_axes([0.5 * (left.x0 + right.x1 - width), left.y0 - 0.04, width, 0.022])

    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal", extend="max")
    colorbar.ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    colorbar.ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    colorbar.set_label("Intensity (counts)", labelpad=2.0)
    colorbar.ax.tick_params(direction="in", length=3.0, width=0.8)
    colorbar.outline.set_linewidth(0.8)

    labels = (("c", "d"), ("e", "f"))
    ylabels = (r"$\boldsymbol{\mu}_{\mathrm{true}}$ (counts)", r"$\mathbf{n}$ (counts)")
    
    for row, (matrix, ylabel) in enumerate(zip(values, ylabels)):
        for column, (index, color) in enumerate(zip(selected, colors)):
            ax = fig.add_subplot(grid[row + 2, column], sharex=matrices[0])
            counts = matrix[index]
            steps(ax, Eg.bins, counts, color=color, lw=0.2, alpha=0.5, zorder=3)
            
            ax.axvline(eg_edges[int(lengths[index])], color=COLORS["black"], ls="--", lw=0.75, alpha=0.9, zorder=2.5)
            _, top = set_symlog_y_with_room(ax, [counts], linthresh=5.0, top_factor=30.0, y_lower=-1.0)

            ticks = 10.0 ** np.arange(1, int(np.ceil(np.log10(top))) + 1)
            ax.set_yticks(np.r_[0.0, ticks[ticks <= top]])
            add_zero_line(ax, color=COLORS["black"], lw=0.45, alpha=0.55)

            if column == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
            else:
                ax.tick_params(labelbottom=False)

            use_MeV_on_xaxis(ax)
            apply_axes_style(ax)

            add_panel_label(ax, labels[row][column], x=0.04, y=0.90, fontsize=9.0)

    matrices[0].set_xlim(-0.025 * eg_edges[-1], eg_edges[-1])

    handles = [Line2D([0], [0], color=color, lw=1.1) for color in colors]
    handles.append(Line2D([0], [0], color=COLORS["black"], lw=0.8, ls="--"))

    labels = [rf"$E_x \approx {ex / 1000:.1f}\,\mathrm{{MeV}}$" for ex in ex_values]
    add_top_legend(fig, handles, labels + [r"$E_\gamma^{\max}(E_x)$"],
                   ncol=3, y=0.985, handlelength=1.8, columnspacing=1.2, fontsize=8.6)

    for index in selected:
        print(f"Selected Ex bin (keV): {ex_edges[index]:.2f}-{ex_edges[index + 1]:.2f}")

    save_or_show(fig, repo_path(out) if out is not None else None, tag="data-generation", show=show)
    return fig
