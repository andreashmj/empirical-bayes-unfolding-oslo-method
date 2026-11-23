"""
Make figure that compares the Bayes and RMLE results on synthetic data.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import sys
    
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ...paths import repo_path
from ..plot_utils import (
    steps,
    add_panel_label, add_zero_line, add_zoom_guides, local_y_limits,
    save_or_show, set_linear_y_with_zero_room, set_local_y_axis, use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size
from ._common import band_handle, bin_edges, fill_bin_band, read_spectrum, envelope, scaled_envelope, rmle_bootstrap_envelope
from .cli import main as run_cli


styles = ((COLORS["sky"], COLORS["blue"], 0.25), (COLORS["orange"], COLORS["vermillion"], 0.18))


def plot_spectra(ax, Eg, summaries, truth):
    for summary, (band, line, alpha) in zip(summaries, styles):
        fill_bin_band(ax, Eg, summary[1], summary[2], band, alpha)
        
    for summary, (_, line, _) in zip(summaries, styles):
        steps(ax, Eg, summary[0], color=line, lw=0.50, zorder=3.2)
        
    steps(ax, Eg, truth, color=COLORS["black"],
              lw=0.60, ls="--", zorder=3.3)
    apply_axes_style(ax)
    use_MeV_on_xaxis(ax)


def plot_bayes_frequentist_comparison(
    bayes_nc, rmle_nc, ex, zoom1, zoom2, zoom3, out=None, show=True,
):
    draws, truth, _ = read_spectrum(bayes_nc, ex)
    Eg, mean, lower, upper = envelope(draws)
    bayes = mean, lower, upper
    bayes_scaled = scaled_envelope(draws, truth, lower, upper)

    with xr.open_dataset(repo_path(rmle_nc), engine="h5netcdf") as ds:
        row = ds.swap_dims({"Ex": "Ex_req"}).sel(Ex_req=ex)
        row = row.isel(Eg=slice(0, int(row["Eg_len"].item())))

        np.testing.assert_allclose(row["Eg"].values, Eg, rtol=0, atol=1e-6)

        rmle = rmle_bootstrap_envelope(row, Eg)

        rmle_draws = xr.DataArray(
            row["eta_boot"].transpose("bootstrap", "Eg").values.copy(),
            dims=("sample", "Eg"),
            coords={"Eg": Eg},
        )

    rmle_scaled = scaled_envelope(rmle_draws, truth, rmle[1], rmle[2])


    summaries = (bayes, rmle)
    deviations = (bayes_scaled, rmle_scaled)
    edges = bin_edges(Eg)
    windows = (zoom1, zoom2, zoom3)
    arrays = [values for summary in summaries for values in summary] + [truth]
    scale = np.concatenate([values for summary in deviations for values in summary])
    

    size = figure_size("text", height_to_width=1.34)
    fig = plt.figure(figsize=size.as_tuple())
    grid = fig.add_gridspec(
        4, 3, height_ratios=[1.10, 0.92, 0.62, 0.62],
        left=0.10, right=0.98, bottom=0.06, top=0.94, wspace=0.24, hspace=0.16,
    )
    full = fig.add_subplot(grid[0, :])
    plot_spectra(full, Eg, summaries, truth)
    set_linear_y_with_zero_room(full, [bayes[2], rmle[2], truth],
                                top_factor=1.15, lower_fraction=0.035)
    full.set_xlim(-0.025 * edges[-1], edges[-1])
    full.set_ylabel(r"$\boldsymbol{\eta}$ (counts)")
    full.tick_params(labelbottom=False)
    add_panel_label(full, "a")

    zoom_axes = []
    for column, window in enumerate(windows):
        ax = fig.add_subplot(grid[1, column])
        zoom_axes.append(ax)
        plot_spectra(ax, Eg, summaries, truth)
        
        mask = (edges[:-1] < window[1]) & (edges[1:] > window[0])
        y_low, y_high = local_y_limits(mask, arrays)
        
        ax.set_xlim(*window)
        set_local_y_axis(ax, y_low, y_high, nbins=4)
        add_zero_line(ax, color=COLORS["black"], lw=0.55, alpha=0.55, zorder=1.1)
        
        if column == 0:
            ax.set_ylabel(r"$\boldsymbol{\eta}$ (counts)")
        add_panel_label(ax, "bcd"[column])
        
    add_zoom_guides(fig, full, zoom_axes, windows, color=COLORS["gray"])

    for index, (summary, method, style) in enumerate(zip(deviations, ("Bayes", "RMLE"), styles)):
        ax = fig.add_subplot(grid[index + 2, :], sharex=full)
        center, low, high = summary
        
        band, line, alpha = style
        fill_bin_band(ax, Eg, low, high, band, alpha)
        
        steps(ax, Eg, center, color=line, lw=0.75, zorder=3.0)
        ax.axhline(0.0, color=COLORS["black"], lw=0.55, ls=":", alpha=0.85, zorder=1.5)
        
        scale_summaries = deviations[:2]
        values = np.concatenate([v for s in scale_summaries for v in s])
        
        bottom = min(float(values.min()), -2.0)
        top = max(float(values.max()), 2.0)
        padding = 0.3 * (top - bottom)
        
        ax.set_ylim(1.05*bottom, top + padding)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylabel(rf"$S_{{\mathrm{{{method}}}}}(E_\gamma)$")
        
        if index == 1:
            ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
        else:
            ax.tick_params(labelbottom=False)
            
        apply_axes_style(ax)
        use_MeV_on_xaxis(ax)
        add_panel_label(ax, "efg"[index])
        

    handles = [band_handle(band, line, alpha, lw=0.75) for band, line, alpha in styles]
    handles.append(Line2D([0], [0], color=COLORS["black"], lw=0.60, ls="--"))
    labels = [r"Bayes: mean + 95\% simult. band", r"RMLE: mean + 95\% simult. bootstrap band", r"$\boldsymbol{\eta}_{\mathrm{true}}$"]
    fig.legend(
        handles, labels, handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3,
        frameon=False, labelspacing=0.35, handlelength=2.2, fontsize=9.0,
    )
    save_or_show(fig, repo_path(out) if out is not None else None, tag="bayes-rmle-comparison", show=show)
    return fig


def main():
    run_cli(["bayes-frequentist", *sys.argv[1:]])


if __name__ == "__main__":
    main()
