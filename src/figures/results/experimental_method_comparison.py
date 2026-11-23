"""
Make figure that compares the different unfolding methods for experimental data.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import ompy as om
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from ...analysis.results import UnfoldingResults
from ...config_utils import load_run_config
from ...paths import repo_path
from ..plot_utils import (
    steps,
    add_panel_label, add_zero_line, add_zoom_guides, local_y_limits,
    fill_step_band, save_or_show, set_linear_y_with_zero_room, set_local_y_axis,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style
from ._common import envelope, rmle_bootstrap_envelope


styles = [(COLORS["blue"], COLORS["sky"], 0.25), (COLORS["vermillion"], COLORS["orange"], 0.18)]



def read_spectra(bayes_nc, rmle_nc, standard_nc, ex):
    results = UnfoldingResults(repo_path(bayes_nc), "posterior")
    try:
        if ex is None:
            ex = float(results.ds["Ex_req"].values[0])

        ex_actual = results.ex_actual(ex)
        Eg = results.eg(ex_actual)

        bayes = envelope(results.eta(ex_actual))[1:]
    finally:
        results.ds.close()

    with xr.open_dataset(repo_path(rmle_nc), engine="h5netcdf") as ds:
        row = ds.swap_dims({"Ex": "Ex_req"}).sel(Ex_req=ex)
        row = row.isel(Eg=slice(0, int(row["Eg_len"].item())))

        np.testing.assert_allclose(row["Eg"].values, Eg, rtol=0, atol=1e-6)

        rmle = rmle_bootstrap_envelope(row, Eg)

    with xr.open_dataset(repo_path(standard_nc), engine="h5netcdf") as ds:
        row = ds.swap_dims({"Ex": "Ex_req"}).sel(Ex_req=ex)

        np.testing.assert_allclose(row["Eg"].values[:len(Eg)], Eg, rtol=0, atol=1e-6)

        iterative = np.asarray(row["eta_hat"].values[:len(Eg)], dtype=float)

    return Eg, ex_actual, bayes, rmle, iterative



def draw_spectra(ax, Eg, bayes, rmle, iterative):
    handles = []
    for summary, (line_color, band_color, alpha) in zip((bayes, rmle), styles):
        mean, lower, upper = summary
        band = fill_step_band(ax, Eg, lower, upper, band_color,
                              edgecolor=band_color, alpha=alpha,
                              linewidth=0.35, zorder=2, rasterized=True)
        line = steps(ax, Eg, mean, color=line_color, lw=0.85, zorder=3)
        handles.append((band, line))
    line = steps(ax, Eg, iterative,
                     color=COLORS["green"], lw=1.0, ls=":", zorder=3)
    handles.append(line)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)
    return handles


def load_matrix(path, config):

    matrix = om.Matrix.from_path(str(repo_path(path)))
    matrix = matrix.set_xalias("Ex").set_yalias("Eg")
    matrix = matrix.loc[:, config["lower_eg_cut"]:]
    
    for axis, factor in zip(("Ex", "Eg"), config["rebin_factors"]):
        if factor > 1:
            matrix = matrix.rebin(axis, factor=factor)
    return matrix


def draw_matrix(ax, matrix, ex, norm):
    Ex = matrix.X_index.to_left()
    Eg = matrix.Y_index.to_left()
    
    extent = np.array([Eg.leftmost, Eg.rightmost, Ex.leftmost, Ex.rightmost]) / 1000
    values = np.ma.masked_less_equal(matrix.values, 0)
    image = ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest", extent=extent, norm=norm, cmap="viridis", rasterized=True)
    
    width = extent[1] - extent[0]
    image.set_clip_path(Rectangle((extent[0], 0), width, 12, transform=ax.transData))
    index = matrix.index_X(ex)
    selected = Rectangle((extent[0], Ex.bins[index] / 1000), width, Ex.step(index) / 1000,
                         facecolor="gray", edgecolor="none", alpha=0.60, zorder=5)
                         
    ax.add_patch(selected)
    ax.set_xlim(-0.025 * extent[1], extent[1])
    ax.set_ylim(-0.035 * 12, 12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_yticks(np.arange(0, 12.1, 2))
    add_zero_line(ax)
    ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    apply_axes_style(ax)
    return image


def plot_experimental_method_comparison(config_path, bayes_nc, rmle_nc, standard_nc, zoom1, zoom2, zoom3,ex=None, out=None, show=True):

    config = load_run_config(repo_path(config_path))
    Eg, ex_actual, bayes, rmle, iterative = read_spectra(bayes_nc, rmle_nc, standard_nc, ex)
    windows = [zoom1, zoom2, zoom3]
    
    bin_width = Eg[1] - Eg[0]
    edges = np.append(Eg, Eg[-1] + bin_width)
    
    matrices = [load_matrix(config[key], config) for key in ("prompt_path", "background_path")]
    values = np.concatenate([matrix.values.ravel() for matrix in matrices])
    norm = LogNorm(1, np.percentile(values[values > 0], 99.5))

    fig = plt.figure(figsize=(7.1, 8.0))
    grid = fig.add_gridspec(4, 6, left=0.105, right=0.985, bottom=0.075, top=0.94, height_ratios=[1.2, 0.055, 1.0, 1.0], hspace=0.33, wspace=0.95)
    prompt = fig.add_subplot(grid[0, :3])
    background = fig.add_subplot(grid[0, 3:], sharey=prompt)
    
    for ax, matrix, label in zip((prompt, background), matrices, "ab"):
        image = draw_matrix(ax, matrix, ex_actual, norm)
        add_panel_label(ax, label, box=True)
        
    prompt.set_ylabel(r"Excitation energy $E_x$ (MeV)")
    background.tick_params(labelleft=False)
    
    colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[1, 1:5]),
                            orientation="horizontal", extend="max")
    colorbar.set_label("counts", labelpad=2)
    colorbar.ax.tick_params(direction="in", length=3, width=0.8)
    colorbar.outline.set_linewidth(0.8)

    overview = fig.add_subplot(grid[2, :])
    handles = draw_spectra(overview, Eg, bayes, rmle, iterative)
    arrays = [*bayes, *rmle, iterative]
    low, high = set_linear_y_with_zero_room(overview, arrays)
    minimum = min(values.min() for values in arrays)
    
    overview.set_ylim(min(low, minimum - 0.035 * high), high)
    overview.set_xlim(-0.025 * edges[-1], edges[-1])
    overview.xaxis.set_major_locator(MaxNLocator(nbins=7))
    overview.set_ylabel(r"$\boldsymbol{\eta}$ (counts)")
    
    add_panel_label(overview, "c")

    zoom_axes = []
    
    for i, (window, label) in enumerate(zip(windows, "def")):
        ax = fig.add_subplot(grid[3, 2 * i:2 * i + 2])
        draw_spectra(ax, Eg, bayes, rmle, iterative)
        mask = (edges[:-1] < window[1]) & (edges[1:] > window[0])
        
        low, high = local_y_limits(mask, arrays, padding_fraction=0.10)
        ax.set_xlim(*window)
        set_local_y_axis(ax, low, high, nbins=4)
        
        add_zero_line(ax)
        add_panel_label(ax, label)
        
        zoom_axes.append(ax)
        
    zoom_axes[0].set_ylabel(r"$\boldsymbol{\eta}$ (counts)")
    fig.text(0.545, 0.024, r"Photon energy $E_\gamma$ (MeV)", ha="center")
    add_zoom_guides(fig, overview, zoom_axes, windows, color=COLORS["gray"])
    
    fig.legend(handles, ["Bayes", "RMLE", "Iterative"],
               handler_map={tuple: HandlerTuple(ndivide=1)}, loc="upper center",
               bbox_to_anchor=(0.55, 0.995), ncol=3, frameon=False, fontsize=8.6,
               handlelength=2.2, columnspacing=1.4)
               
    print(f"Selected Ex = {ex_actual:.2f} keV")
    path = repo_path(out) if out is not None else None
    save_or_show(fig, path, tag="experimental-method-comparison", show=show)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Prompt/background matrix preparation.")
    parser.add_argument("--bayes-nc", required=True)
    parser.add_argument("--rmle-nc", required=True)
    parser.add_argument("--standard-nc", required=True)
    parser.add_argument("--ex", type=float, help="Requested Ex in keV")
    parser.add_argument("--zoom1", nargs=2, type=float, required=True)
    parser.add_argument("--zoom2", nargs=2, type=float, required=True)
    parser.add_argument("--zoom3", nargs=2, type=float, required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    plot_experimental_method_comparison(
        config_path=args.config,
        bayes_nc=args.bayes_nc,
        rmle_nc=args.rmle_nc,
        standard_nc=args.standard_nc,
        zoom1=args.zoom1,
        zoom2=args.zoom2,
        zoom3=args.zoom3,
        ex=args.ex,
        out=args.out,
    )


if __name__ == "__main__":
    main()
