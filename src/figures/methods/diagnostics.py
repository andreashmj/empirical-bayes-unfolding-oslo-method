"""
Figure showing rank-normalized R-hat and bulk ESS per bin
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...analysis.diagnostics import diagnostic_draws, rhat_ess_by_eg
from ...analysis.results import UnfoldingResults
from ...paths import repo_path
from ..plot_utils import add_panel_label, save_or_show, use_MeV_on_xaxis
from ..style import COLORS, apply_axes_style, figure_size


def plot_diagnostics(
    nc_path: str | Path, ex: float | None = None, var: str = "eta",
    out: str | Path | None = None, show: bool = True,
) -> plt.Figure:
    results = UnfoldingResults(repo_path(nc_path), expected_mode="posterior")
    try:
        ex_actual = results.ex_actual(ex)
        Eg = results.eg(ex_actual)
        draws = diagnostic_draws(results, ex_actual, var=var)
        rhat, ess = rhat_ess_by_eg(draws, Eg, name=var)
    finally:
        results.ds.close()

    symbol = {"x": "x", "eta": r"\eta"}[var]
    fig, axes = plt.subplots(
        1, 2, figsize=figure_size("text", height_to_width=0.42).as_tuple(),
        constrained_layout=True, sharex=True,
    )
    labels = (rf"$\widehat{{R}}({symbol}_j)$", rf"$\mathrm{{ESS}}_{{\mathrm{{bulk}}}}({symbol}_j)$")
    for ax, values, color, ylabel, letter in zip(
        axes, (rhat, ess), (COLORS["blue"], COLORS["rose"]), labels, "ab"
    ):
        finite = np.isfinite(values)
        ax.plot(Eg[finite], values[finite], ".", ms=1.5, color=color)
        ax.set(xlabel=r"Photon energy $E_\gamma$ (MeV)", ylabel=ylabel)
        apply_axes_style(ax)
        use_MeV_on_xaxis(ax)
        add_panel_label(ax, letter)
    axes[1].set_ylim(bottom=0.0)
    print(f"Saved Ex={ex_actual:.2f} keV; diagnostics for {var}")
    save_or_show(fig, repo_path(out) if out is not None else None, tag="diagnostics", show=show)
    return fig
