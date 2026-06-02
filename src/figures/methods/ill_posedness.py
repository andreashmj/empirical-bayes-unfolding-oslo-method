"""
Ill-posedness figure.

The figure compares posterior draws in emitted space with the same draws after
mapping to the resolution-limited space eta = x @ G_g.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ...analysis.results import UnfoldingResults
from ...paths import repo_path
from ..plot_utils import (
    add_panel_labels,
    add_top_legend,
    add_zero_line,
    save_or_show,
    set_symlog_y_with_room,
    steps,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size


def choose_draw_subset(draws: np.ndarray, max_draws: int, seed: int) -> np.ndarray:
    """Return a reproducible subset of draws."""

    draws = np.asarray(draws, dtype=float)
    n_draws = draws.shape[0]
    n_plot = min(max(int(max_draws), 1), n_draws)

    if n_plot == n_draws:
        return draws

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_draws, size=n_plot, replace=False)

    return draws[indices, :]


def plot_draws(
    ax: plt.Axes,
    eg_axis: np.ndarray,
    draws: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    """Plot posterior draw curves."""

    for draw_index in range(draws.shape[0]):
        steps(
            ax,
            eg_axis,
            draws[draw_index, :],
            color=color,
            lw=linewidth,
            alpha=alpha,
            zorder=1,
        )


def set_symlog_axis_with_zero(
    ax: plt.Axes,
    arrays: list[np.ndarray],
    linthresh: float,
    top_factor: float,
    zero_color: str,
) -> None:
    """Set symlog limits and draw the dotted zero baseline."""

    set_symlog_y_with_room(
        ax,
        arrays=arrays,
        linthresh=linthresh,
        top_factor=top_factor,
        y_lower=-1.0,
    )
    add_zero_line(
        ax,
        color=zero_color,
        lw=0.45,
        ls=":",
        alpha=0.55,
        zorder=1.1,
    )


def plot_ill_posedness(
    nc_path: str | Path,
    ex: float | None = None,
    max_draws: int = 10,
    seed: int = 123,
    layout: str = "column",
    out: str | Path | None = None,
    symlog_linthresh: float = 5.0,
    show: bool = True,
) -> plt.Figure:
    """Create the emitted-space versus eta-space posterior-draw figure."""

    results = UnfoldingResults(repo_path(nc_path), expected_mode="posterior")

    ex_actual = results.ex_actual(ex)
    eg_axis = results.eg(ex_actual)

    x_true = np.asarray(results.x_true(ex_actual), dtype=float)
    eta_true = np.asarray(results.eta_true(ex_actual), dtype=float)

    x_draws = np.asarray(results.x(ex_actual).values, dtype=float)
    x_draws = choose_draw_subset(
        draws=x_draws,
        max_draws=max_draws,
        seed=seed,
    )

    _, gamma_resolution_matrix, _ = results.ops(ex_actual)
    eta_draws = x_draws @ gamma_resolution_matrix

    size = figure_size(layout, height_to_width=1.05)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        2,
        1,
        left=0.15 if str(layout).strip().lower() == "column" else 0.10,
        right=0.97,
        bottom=0.12,
        top=0.83,
        hspace=0.08,
    )

    ax_x = fig.add_subplot(grid[0, 0])
    ax_eta = fig.add_subplot(grid[1, 0], sharex=ax_x)

    x_draw_color = COLORS["sky"]
    eta_draw_color = COLORS["green"]
    x_truth_color = COLORS["black"]
    eta_truth_color = COLORS["orange"]

    plot_draws(
        ax_x,
        eg_axis=eg_axis,
        draws=x_draws,
        color=x_draw_color,
        linewidth=0.20,
        alpha=0.15,
    )

    x_true_handle = steps(
        ax_x,
        eg_axis,
        x_true,
        color=x_truth_color,
        lw=0.45,
        ls="--",
        zorder=3,
    )

    plot_draws(
        ax_eta,
        eg_axis=eg_axis,
        draws=eta_draws,
        color=eta_draw_color,
        linewidth=0.30,
        alpha=0.35,
    )

    eta_true_handle = steps(
        ax_eta,
        eg_axis,
        eta_true,
        color=eta_truth_color,
        lw=0.45,
        ls="--",
        zorder=4,
    )

    ax_x.set_ylabel(r"$\boldsymbol{\mu}$ (counts)")
    ax_eta.set_ylabel(r"$\boldsymbol{\eta}$ (counts)")
    ax_eta.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax_x.tick_params(labelbottom=False)

    set_symlog_axis_with_zero(
        ax_x,
        arrays=[x_draws, x_true],
        linthresh=symlog_linthresh,
        top_factor=1.55,
        zero_color=x_truth_color,
    )
    set_symlog_axis_with_zero(
        ax_eta,
        arrays=[eta_draws, eta_true],
        linthresh=symlog_linthresh,
        top_factor=1.55,
        zero_color=COLORS["black"],
    )

    x_right = np.max(eg_axis).item()
    ax_eta.set_xlim(-0.025 * x_right, x_right)

    for ax in (ax_x, ax_eta):
        apply_axes_style(ax)
        use_MeV_on_xaxis(ax)

    add_panel_labels(
        [ax_x, ax_eta],
        start=0,
        x=0.04,
        y=0.93,
        fontsize=9.0,
    )

    x_draw_proxy = Line2D([0], [0], color=x_draw_color, lw=0.9, alpha=0.9)
    eta_draw_proxy = Line2D([0], [0], color=eta_draw_color, lw=0.9, alpha=0.9)

    add_top_legend(
        fig,
        handles=[x_draw_proxy, eta_draw_proxy, x_true_handle, eta_true_handle],
        labels=[
            r"$\boldsymbol{\mu}$ posterior draws",
            r"$\boldsymbol{\eta}$ posterior draws",
            r"$\boldsymbol{\mu}_{\mathrm{true}}$",
            r"$\boldsymbol{\eta}_{\mathrm{true}}$",
        ],
        ncol=2,
        y=0.985,
        frameon=False,
        fontsize=8.4,
    )

    save_or_show(fig, out, tag="ill-posedness", show=show)
    return fig
