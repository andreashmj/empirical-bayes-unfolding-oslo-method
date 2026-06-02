"""
Synthetic-data generation overview figure.

The figure shows the emitted truth matrix, the observed ON-count matrix, and
selected one-dimensional gamma-energy slices from both matrices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

from ...paths import DATA_DIR, repo_path
from ...synthetic_data import SyntheticDataLoader
from ..plot_utils import (
    add_panel_label,
    add_top_legend,
    add_zero_line,
    save_or_show,
    set_symlog_y_with_room,
    steps,
    use_MeV_on_xaxis,
    use_MeV_on_yaxis,
)
from ..style import COLORS, apply_axes_style, figure_size


def _bin_edges(centers: np.ndarray) -> np.ndarray:
    """Return bin edges from bin centers."""

    centers = np.asarray(centers, dtype=float).reshape(-1)

    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=float)

    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first_edge = centers[0] - (midpoints[0] - centers[0])
    last_edge = centers[-1] + (centers[-1] - midpoints[-1])

    return np.concatenate([[first_edge], midpoints, [last_edge]])


def _chosen_ex_indices(ex_axis: np.ndarray, ex_values: tuple[float, float]) -> list[int]:
    """Return matrix-row indices for two requested Ex values."""

    if len(ex_values) != 2:
        raise ValueError("data_generation uses exactly two Ex slices.")

    indices = []

    for ex_value in ex_values:
        index = np.argmin(np.abs(ex_axis - float(ex_value))).item()

        if index not in indices:
            indices.append(index)

    if len(indices) != 2:
        raise ValueError("The two requested Ex values map to the same matrix row.")

    return indices


def _positive_norm(values_a: np.ndarray, values_b: np.ndarray) -> LogNorm:
    """Return common logarithmic normalization for the two matrix panels."""

    positive_values = np.concatenate([values_a.ravel(), values_b.ravel()])
    positive_values = positive_values[
        np.isfinite(positive_values) & (positive_values > 0.0)
    ]

    if positive_values.size == 0:
        vmax = 1.0
    else:
        vmax = max(np.percentile(positive_values, 99.5).item(), 1.0)

    return LogNorm(vmin=1.0, vmax=vmax)


def _masked_positive(values: np.ndarray) -> np.ma.MaskedArray:
    """Mask non-positive or non-finite matrix entries for log plotting."""

    return np.ma.masked_where((~np.isfinite(values)) | (values <= 0.0), values)


def _set_count_axis(
    ax: plt.Axes,
    arrays: list[np.ndarray],
    linthresh: float,
    top_factor: float = 2.0,
    y_lower: float = -1.0,
) -> None:
    """Apply a per-panel symlog count axis with large label headroom.

    The y limit is set from only the arrays plotted in this panel. This avoids
    sharing y scales between the two selected Ex cases while still using the
    common helper used elsewhere in the paper figures.
    """

    _, y_top = set_symlog_y_with_room(
        ax,
        arrays=arrays,
        linthresh=linthresh,
        top_factor=top_factor,
        y_lower=y_lower,
    )

    add_zero_line(
        ax,
        color=COLORS["black"],
        lw=0.45,
        ls=":",
        alpha=0.55,
        zorder=1.1,
    )

    #_set_compact_symlog_ticks(ax, y_top=y_top)


def _set_compact_symlog_ticks(ax: plt.Axes, y_top: float) -> None:
    """Use count ticks that avoid crowding 0 and 10^0 on symlog axes."""

    if not np.isfinite(y_top) or y_top <= 0.0:
        return

    ticks = [0.0]
    max_exponent = int(np.ceil(np.log10(max(y_top, 10.0))))

    for exponent in range(1, max_exponent + 1):
        tick = 10.0**exponent
        if tick <= 1.05 * y_top:
            ticks.append(tick)

    ax.set_yticks(ticks)


def _add_selected_ex_guides(
    axes: tuple[plt.Axes, plt.Axes],
    ex_edges: np.ndarray,
    ex_axis: np.ndarray,
    selected_indices: list[int],
    colors: list[str],
) -> None:
    """Mark selected Ex rows in the matrix panels."""

    for color, index in zip(colors, selected_indices):
        ex_low = ex_edges[index]
        ex_high = ex_edges[index + 1]
        ex_center = ex_axis[index]

        for ax in axes:
            ax.axhspan(
                ex_low,
                ex_high,
                facecolor=color,
                alpha=0.18,
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
            ax.axhline(
                ex_center,
                color=color,
                lw=0.9,
                ls="-",
                zorder=4,
            )


def _center_colorbar_under_matrices(
    cax: plt.Axes,
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    width_ratio: float = 0.68,
    y_offset: float = 0.052,
    height: float = 0.022,
) -> None:
    """Shorten and center the colorbar below the matrix panels."""

    left_position = ax_left.get_position()
    right_position = ax_right.get_position()

    matrix_x0 = left_position.x0
    matrix_x1 = right_position.x1
    matrix_width = matrix_x1 - matrix_x0

    colorbar_width = matrix_width * float(width_ratio)
    colorbar_x0 = matrix_x0 + 0.5 * (matrix_width - colorbar_width)

    cax.set_position(
        [
            colorbar_x0,
            left_position.y0 - float(y_offset),
            colorbar_width,
            float(height),
        ]
    )


def _plot_slice(
    ax: plt.Axes,
    eg_axis: np.ndarray,
    values: np.ndarray,
    color: str,
    active_eg_length: int,
) -> None:
    """Plot one selected Ex slice and mark its active Eg boundary."""

    steps(
        ax,
        eg_axis,
        values,
        color=color,
        lw=0.25,
        alpha=0.75,
        zorder=3,
    )

    eg_max = eg_axis[int(active_eg_length) - 1]
    ax.axvline(
        eg_max,
        color=COLORS["black"],
        ls="--",
        lw=0.75,
        alpha=0.9,
        zorder=2.5,
    )


def plot_data_generation(
    mat_path: str | Path | None = None,
    ex_values: tuple[float, float] = (2500.0, 8000.0),
    response_db: str = "OSCAR2020",
    rebin_factors: tuple[int, int] = (10, 2),
    lower_eg_cut: str = "60keV",
    sigma_eg: str = "30keV",
    include_background: bool = True,
    bg_fraction: float = 0.15,
    bg_flat_fraction: float = 0.10,
    mat_scale: float = 1.0,
    eg_tail_mass: float = 1.0e-6,
    rng_seed: int = 100,
    include_ex_smearing: bool = False,
    fwhm_ex_kev: float = 150.0,
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create the synthetic-data generation overview figure."""

    if mat_path is None:
        mat_path = DATA_DIR / "ExEg_1e8.npz"

    loader = SyntheticDataLoader(
        mat_path=repo_path(mat_path),
        response_db=response_db,
        rebin_factors=rebin_factors,
        mat_scale=mat_scale,
        lower_eg_cut=lower_eg_cut,
        include_background=include_background,
        bg_fraction=bg_fraction,
        bg_flat_fraction=bg_flat_fraction,
        eg_tail_mass=eg_tail_mass,
        rng_seed=rng_seed,
        sigma_eg=sigma_eg,
        include_ex_smearing=include_ex_smearing,
        fwhm_ex_kev=fwhm_ex_kev,
    )

    x_true_matrix = loader.x_true
    n_obs_matrix = loader.n

    x_true = np.asarray(x_true_matrix.values, dtype=float)
    n_obs = np.asarray(n_obs_matrix.values, dtype=float)

    ex_axis = np.asarray(x_true_matrix.Ex, dtype=float)
    eg_axis = np.asarray(x_true_matrix.Eg, dtype=float)

    selected_indices = _chosen_ex_indices(ex_axis, ex_values)
    active_eg_lengths = np.asarray(loader.active_eg_lengths, dtype=int)

    ex_edges = _bin_edges(ex_axis)
    eg_edges = _bin_edges(eg_axis)

    colors = [COLORS["blue"], COLORS["vermillion"]]

    size = figure_size("text", height_to_width=0.90)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[2.25, 0.20, 1.05, 1.05],
        left=0.115,
        right=0.975,
        bottom=0.080,
        top=0.925,
        wspace=0.12,
        hspace=0.22,
    )

    ax_x_matrix = fig.add_subplot(grid[0, 0])
    ax_n_matrix = fig.add_subplot(grid[0, 1], sharex=ax_x_matrix, sharey=ax_x_matrix)
    cax = fig.add_subplot(grid[1, :])

    norm = _positive_norm(x_true, n_obs)

    image_x = ax_x_matrix.imshow(
        _masked_positive(x_true),
        origin="lower",
        aspect="auto",
        extent=[eg_edges[0], eg_edges[-1], ex_edges[0], ex_edges[-1]],
        norm=norm,
        cmap="viridis",
        interpolation="nearest",
        rasterized=True,
    )

    ax_n_matrix.imshow(
        _masked_positive(n_obs),
        origin="lower",
        aspect="auto",
        extent=[eg_edges[0], eg_edges[-1], ex_edges[0], ex_edges[-1]],
        norm=norm,
        cmap="viridis",
        interpolation="nearest",
        rasterized=True,
    )

    ax_x_matrix.set_ylabel(r"Excitation energy $E_x$ (MeV)")
    ax_x_matrix.tick_params(labelbottom=False)
    ax_n_matrix.tick_params(labelbottom=False, labelleft=False)

    use_MeV_on_xaxis(ax_x_matrix)
    use_MeV_on_yaxis(ax_x_matrix)
    use_MeV_on_xaxis(ax_n_matrix)
    use_MeV_on_yaxis(ax_n_matrix)

    apply_axes_style(ax_x_matrix)
    apply_axes_style(ax_n_matrix)

    colorbar = fig.colorbar(image_x, cax=cax, orientation="horizontal", extend="max")
    colorbar.ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    colorbar.ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    colorbar.set_label(r"Intensity (counts)", labelpad=2.0)
    colorbar.ax.tick_params(direction="in", length=3.0, width=0.8)
    colorbar.outline.set_linewidth(0.8)

    _center_colorbar_under_matrices(
        cax=cax,
        ax_left=ax_x_matrix,
        ax_right=ax_n_matrix,
        width_ratio=0.68,
        y_offset=0.04,
        height=0.022,
    )

    _add_selected_ex_guides(
        axes=(ax_x_matrix, ax_n_matrix),
        ex_edges=ex_edges,
        ex_axis=ex_axis,
        selected_indices=selected_indices,
        colors=colors,
    )

    ax_mu_left = fig.add_subplot(grid[2, 0], sharex=ax_x_matrix)
    ax_mu_right = fig.add_subplot(grid[2, 1], sharex=ax_x_matrix)
    ax_n_left = fig.add_subplot(grid[3, 0], sharex=ax_x_matrix)
    ax_n_right = fig.add_subplot(grid[3, 1], sharex=ax_x_matrix)

    mu_axes = [ax_mu_left, ax_mu_right]
    n_axes = [ax_n_left, ax_n_right]

    for column, (index, color) in enumerate(zip(selected_indices, colors)):
        mu_values = np.asarray(x_true[index, :], dtype=float)
        n_values = np.asarray(n_obs[index, :], dtype=float)

        _plot_slice(
            ax=mu_axes[column],
            eg_axis=eg_axis,
            values=mu_values,
            color=color,
            active_eg_length=active_eg_lengths[index],
        )
        _plot_slice(
            ax=n_axes[column],
            eg_axis=eg_axis,
            values=n_values,
            color=color,
            active_eg_length=active_eg_lengths[index],
        )

        _set_count_axis(
            mu_axes[column],
            arrays=[mu_values],
            linthresh=5.0,
            top_factor=30.0,
            y_lower=-1.0,
        )
        _set_count_axis(
            n_axes[column],
            arrays=[n_values],
            linthresh=5.0,
            top_factor=30.0,
            y_lower=-1.0,
        )

    ax_mu_left.set_ylabel(r"$\boldsymbol{\mu}_{\mathrm{true}}$ (counts)")
    ax_n_left.set_ylabel(r"$\mathbf{n}$ (counts)")

    ax_mu_left.tick_params(labelbottom=False)
    ax_mu_right.tick_params(labelbottom=False)
    ax_n_left.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax_n_right.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")

    for ax in mu_axes + n_axes:
        use_MeV_on_xaxis(ax)
        apply_axes_style(ax)

    x_right = eg_edges[-1]
    x_left = -0.025 * x_right

    for ax in [ax_x_matrix, ax_n_matrix] + mu_axes + n_axes:
        ax.set_xlim(x_left, x_right)

    legend_handles = []
    legend_labels = []

    for color, index in zip(colors, selected_indices):
        legend_handles.append(Line2D([0], [0], color=color, lw=1.1))
        legend_labels.append(
            rf"$E_x \approx {ex_axis[index] / 1000.0:.1f}\,\mathrm{{MeV}}$"
        )

    legend_handles.append(Line2D([0], [0], color=COLORS["black"], lw=0.8, ls="--"))
    legend_labels.append(r"$E_{\gamma}^{\max}(E_x)$")

    add_top_legend(
        fig,
        handles=legend_handles,
        labels=legend_labels,
        ncol=3,
        y=0.985,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
        fontsize=8.6,
    )

    # Matrix panels. Use a light box so the labels remain visible on the
    # logarithmic colormaps.
    for label, ax in zip("ab", [ax_x_matrix, ax_n_matrix]):
        add_panel_label(
            ax,
            label=label,
            x=0.035,
            y=0.965,
            fontsize=9.0,
            box=True,
            box_alpha=0.72,
        )

    # One-dimensional slice panels.
    for label, ax in zip("cdef", [ax_mu_left, ax_mu_right, ax_n_left, ax_n_right]):
        add_panel_label(
            ax,
            label=label,
            x=0.04,
            y=0.90,
            fontsize=9.0,
            box=False,
        )

    save_or_show(fig, out, tag="data-generation", show=show)
    return fig
