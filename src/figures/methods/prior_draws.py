"""
Figure showing prior draws in eta-space with RL reference and truth.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ...analysis.results import UnfoldingResults
from ...paths import repo_path
from ..plot_utils import (
    add_top_legend,
    add_zero_line,
    save_or_show,
    set_symlog_y_with_room,
    steps,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size


def choose_draw_subset(draws: np.ndarray, max_draws: int, seed: int) -> np.ndarray:
    """
    Return a reproducible subset of draw curves.
    """
    draws = np.asarray(draws, dtype=float)
    if max_draws < 1:
        raise ValueError("max_draws must be positive.")
    n_draws = draws.shape[0]
    n_plot = min(max_draws, n_draws)

    if n_plot == n_draws:
        return draws

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_draws, size=n_plot, replace=False)

    return draws[indices, :]


def plot_prior_draws(
    nc_path: str | Path,
    ex: float | None = None,
    max_draws: int = 250,
    seed: int = 123,
    symlog_linthresh: float = 5.0,
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create the figure with prior draws in eta-space.
    """

    results = UnfoldingResults(repo_path(nc_path), expected_mode="prior")

    ex_actual = results.ex_actual(ex)
    eg_axis = results.eg(ex_actual)

    eta_true = np.asarray(results.eta_true(ex_actual), dtype=float)
    eta_rl = results.eta_rl(ex_actual)

    if eta_rl is None:
        raise ValueError("The result file does not contain an RL reference.")

    eta_rl = np.asarray(eta_rl, dtype=float)
    eta_draws = np.asarray(results.eta(ex_actual).values, dtype=float)
    eta_draws_plot = choose_draw_subset(
        draws=eta_draws,
        max_draws=max_draws,
        seed=seed,
    )

    size = figure_size("column", height_to_width=0.78)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        1,
        1,
        left=0.17,
        right=0.97,
        bottom=0.20,
        top=0.86,
    )
    ax = fig.add_subplot(grid[0, 0])

    draw_color = COLORS["purple"]
    rl_color = COLORS["blue"]
    truth_color = COLORS["black"]

    for draw_index in range(eta_draws_plot.shape[0]):
        steps(
            ax,
            eg_axis,
            eta_draws_plot[draw_index, :],
            color=draw_color,
            lw=0.18,
            alpha=0.18,
            zorder=1,
        )

    rl_handle = steps(
        ax,
        eg_axis,
        eta_rl,
        color=rl_color,
        lw=0.65,
        ls="-",
        alpha=0.95,
        zorder=3,
    )

    truth_handle = steps(
        ax,
        eg_axis,
        eta_true,
        color=truth_color,
        lw=0.45,
        ls="--",
        zorder=4,
    )

    ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax.set_ylabel(r"$\boldsymbol{\eta}$ (counts)")

    set_symlog_y_with_room(
        ax,
        arrays=[eta_draws_plot, eta_rl, eta_true],
        linthresh=symlog_linthresh,
        top_factor=1.35,
        y_lower=-1.0,
    )
    add_zero_line(
        ax,
        color=truth_color,
        lw=0.45,
        ls=":",
        alpha=0.55,
        zorder=1.1,
    )

    x_right = np.max(eg_axis).item()
    ax.set_xlim(-0.025 * x_right, x_right)

    apply_axes_style(ax)
    use_MeV_on_xaxis(ax)

    draw_proxy = Line2D([0], [0], color=draw_color, lw=0.9, alpha=0.9)

    add_top_legend(
        fig,
        handles=[draw_proxy, rl_handle, truth_handle],
        labels=[
            r"prior draws",
            r"$\boldsymbol{\eta}_{\mathrm{RL}}$",
            r"$\boldsymbol{\eta}_{\mathrm{true}}$",
        ],
        ncol=3,
        y=0.985,
        frameon=False,
        fontsize=8.4,
    )

    save_or_show(fig, out, tag="prior-draws", show=show)
    return fig
