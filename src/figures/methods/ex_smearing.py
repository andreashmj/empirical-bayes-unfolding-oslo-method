"""
Excitation-energy smearing bias figure. The plotted quantity is the detected-signal bias induced by Ex smearing: Delta V = (G_in - I) x_true (D G_g)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator

from ...paths import repo_path
from ...synthetic_data import SyntheticDataLoader
from ..plot_utils import (
    add_panel_label,
    save_or_show,
    use_MeV_on_xaxis,
    use_MeV_on_yaxis,
)
from ..style import apply_axes_style, figure_size


def median_spacing(values: np.ndarray) -> float:
    """Return the median absolute spacing of a one-dimensional grid."""
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size < 2:
        return np.nan

    spacing = np.diff(values)
    spacing = spacing[np.isfinite(spacing)]

    if spacing.size == 0:
        return np.nan
    return np.median(np.abs(spacing)).item()


def ex_rebin_factor(base_ex_axis: np.ndarray, fwhm_ex_kev: float, width_factor: float) -> int:
    """Return Ex rebin factor corresponding to a fraction of the Ex FWHM."""
    d_ex = median_spacing(base_ex_axis)

    if not np.isfinite(d_ex) or d_ex <= 0.0:
        d_ex = 1.0
    return max(1, round(float(width_factor) * float(fwhm_ex_kev) / d_ex))


def build_loader(
    mat_path: str | Path,
    response_db: str,
    ex_factor: int,
    lower_eg_cut: str,
    sigma_eg: str,
    fwhm_ex_kev: float,
) -> SyntheticDataLoader:
    """Build one synthetic loader with Ex smearing enabled."""

    return SyntheticDataLoader(
        mat_path=repo_path(mat_path),
        response_db=response_db,
        rebin_factors=(ex_factor, 1),
        mat_scale=1.0,
        lower_eg_cut=lower_eg_cut,
        include_background=False,
        rng_seed=42,
        sigma_eg=sigma_eg,
        include_ex_smearing=True,
        fwhm_ex_kev=fwhm_ex_kev,
    )


def ex_smearing_bias(loader: SyntheticDataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return Delta V, Ex grid, and Eg grid.
    """

    x_true = np.asarray(loader.x_true.values, dtype=float)
    D = np.asarray(loader.D.values, dtype=float)
    G_g = np.asarray(loader.G_g.values, dtype=float)
    response_matrix = D @ G_g

    G_in = loader.G_in
    if G_in is None:
        raise RuntimeError("Ex-smearing bias requires include_ex_smearing=True.")

    G_in = np.asarray(G_in.values, dtype=float)
    identity = np.eye(G_in.shape[0], dtype=float)

    bias = ((G_in - identity) @ x_true) @ response_matrix

    ex_axis = np.asarray(loader.Ex, dtype=float)
    eg_axis = np.asarray(loader.Eg, dtype=float)

    return bias, ex_axis, eg_axis


def bias_norm(bias_left: np.ndarray, bias_right: np.ndarray) -> tuple[SymLogNorm, float]:

    absolute_values = np.abs(np.concatenate([bias_left.ravel(), bias_right.ravel()]))
    absolute_values = absolute_values[np.isfinite(absolute_values)]

    if absolute_values.size == 0:
        vmax = 1.0
    else:
        vmax = np.nanpercentile(absolute_values, 99.5).item()

    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    linthresh = max(1.0, 0.01 * vmax)

    norm = SymLogNorm(
        linthresh=linthresh,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )

    return norm, linthresh


def add_border(ax: plt.Axes, eg_axis: np.ndarray, ex_axis: np.ndarray) -> None:
    x0 = eg_axis[0].item()
    x1 = eg_axis[-1].item()
    y0 = ex_axis[0].item()
    y1 = ex_axis[-1].item()

    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.9, ec="#4a4a4a",
        )
    )




def plot_ex_smearing(
    mat_path: str | Path = "data/ExEg_1e8.npz",
    response_db: str = "OSCAR2020",
    lower_eg_cut: str = "60keV",
    sigma_eg: str = "30keV",
    fwhm_ex_kev: float = 150.0,
    width_factor_left: float = 1.0,
    width_factor_right: float = 1.5,
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create the Ex-smearing bias heatmap figure.
    """

    base_loader = build_loader(
        mat_path=mat_path,
        response_db=response_db,
        ex_factor=1,
        lower_eg_cut=lower_eg_cut,
        sigma_eg=sigma_eg,
        fwhm_ex_kev=fwhm_ex_kev,
    )

    base_ex_axis = np.asarray(base_loader.Ex, dtype=float)

    left_factor = ex_rebin_factor(
        base_ex_axis=base_ex_axis,
        fwhm_ex_kev=fwhm_ex_kev,
        width_factor=width_factor_left,
    )
    right_factor = ex_rebin_factor(
        base_ex_axis=base_ex_axis,
        fwhm_ex_kev=fwhm_ex_kev,
        width_factor=width_factor_right,
    )

    left_loader = build_loader(
        mat_path=mat_path,
        response_db=response_db,
        ex_factor=left_factor,
        lower_eg_cut=lower_eg_cut,
        sigma_eg=sigma_eg,
        fwhm_ex_kev=fwhm_ex_kev,
    )
    right_loader = build_loader(
        mat_path=mat_path,
        response_db=response_db,
        ex_factor=right_factor,
        lower_eg_cut=lower_eg_cut,
        sigma_eg=sigma_eg,
        fwhm_ex_kev=fwhm_ex_kev,
    )

    bias_left, ex_left, eg_axis = ex_smearing_bias(left_loader)
    bias_right, ex_right, _ = ex_smearing_bias(right_loader)

    norm, linthresh = bias_norm(bias_left, bias_right)

    size = figure_size("text", height_to_width=0.50)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.80, 0.08],
        left=0.10,
        right=0.97,
        bottom=0.08,
        top=0.97,
        wspace=0.12,
        hspace=0.40,
    )

    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1], sharex=ax_left, sharey=ax_left)
    cax = fig.add_subplot(grid[1, :])

    for ax in (ax_left, ax_right):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    image_left = ax_left.imshow(
        bias_left,
        origin="lower",
        aspect="auto",
        extent=[eg_axis[0], eg_axis[-1], ex_left[0], ex_left[-1]],
        cmap="coolwarm",
        norm=norm,
        rasterized=True,
    )

    image_right = ax_right.imshow(
        bias_right,
        origin="lower",
        aspect="auto",
        extent=[eg_axis[0], eg_axis[-1], ex_right[0], ex_right[-1]],
        cmap="coolwarm",
        norm=norm,
        rasterized=True,
    )

    ax_right.tick_params(labelleft=False)

    for ax in (ax_left, ax_right):
        ax.set_xlim(0.0, eg_axis[-1].item())
        use_MeV_on_xaxis(ax)

    use_MeV_on_yaxis(ax_left)

    ax_left.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
    ax_left.set_yticks([2000, 4000, 6000, 8000, 10000])

    add_border(ax_left, eg_axis, ex_left)
    add_border(ax_right, eg_axis, ex_right)

    ax_left.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax_right.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax_left.set_ylabel(r"Excitation energy $E_x$ (MeV)")

    add_panel_label(ax_left, "a", x=0.05, y=0.90, fontsize=10.0)
    add_panel_label(ax_right, "b", x=0.05, y=0.90, fontsize=10.0)

    apply_axes_style(ax_left)
    apply_axes_style(ax_right)

    colorbar = fig.colorbar(image_right, cax=cax, orientation="horizontal", extend="both")
    colorbar.set_label(r"$\Delta\mathbf{V}$ (counts)")
    colorbar.ax.xaxis.set_minor_locator(
        SymmetricalLogLocator(
            base=10,
            linthresh=linthresh,
            subs=np.arange(2, 10),
        )
    )
    colorbar.ax.tick_params(direction="in", length=3.2, width=0.7)
    colorbar.outline.set_linewidth(0.8)

    left_position = ax_left.get_position()
    right_position = ax_right.get_position()

    colorbar_width_ratio = 0.75
    matrix_width = right_position.x1 - left_position.x0
    colorbar_width = matrix_width * colorbar_width_ratio
    colorbar_x0 = left_position.x0 + 0.5 * (matrix_width - colorbar_width)

    cax.set_position(
        [
            colorbar_x0,
            left_position.y0 - 0.17,
            colorbar_width,
            0.030,
        ]
    )

    save_or_show(fig, out, tag="ex-smearing", show=show)
    return fig
