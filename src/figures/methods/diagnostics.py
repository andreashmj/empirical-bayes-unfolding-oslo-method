"""
Sampling-diagnostics figure. The figure shows per-bin rank-normalized split R-hat and bulk ESS for either
the emitted spectrum x or the resolution-limited spectrum eta.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...analysis.diagnostics import (
    diagnostic_draws,
    rhat_ess_by_eg,
    truth_vector,
)
from ...analysis.results import UnfoldingResults
from ...paths import repo_path
from ..plot_utils import (
    add_panel_labels,
    save_or_show,
    steps,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size


def latex_variable(var: str) -> str:
    var = str(var).strip().lower()

    if var == "x":
        return r"\mu"

    if var == "eta":
        return r"\eta"

    raise ValueError("var must be 'x' or 'eta'.")



def plot_diagnostics(
    draws_nc: str | Path,
    ex: float | None = None,
    var: str = "eta",
    show_truth: bool = True,
    layout: str = "text",
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create the per-Eg R-hat and bulk-ESS diagnostic figure.
    
    """

    var = str(var).strip().lower()
    variable_symbol = latex_variable(var)

    results = UnfoldingResults(repo_path(draws_nc), expected_mode="posterior")

    ex_actual = results.ex_actual(ex)
    eg_axis = results.eg(ex_actual)

    draws = diagnostic_draws(results, ex_actual, var=var)
    rhat, ess = rhat_ess_by_eg(draws, eg_axis, name=var)

    if show_truth:
        y_true = truth_vector(results, ex_actual, var=var)
    else:
        y_true = None

    size = figure_size(layout, height_to_width=0.42)

    fig, (ax_rhat, ax_ess) = plt.subplots(
        1,
        2,
        figsize=size.as_tuple(),
        constrained_layout=True,
        sharex=True,
    )

    finite_rhat = np.isfinite(rhat)
    ax_rhat.plot(
        eg_axis[finite_rhat],
        rhat[finite_rhat],
        marker=".",
        linestyle="None",
        ms=1.5,
        color=COLORS["blue"],
    )

    ax_rhat.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax_rhat.set_ylabel(rf"$\widehat{{R}}({variable_symbol}_j)$")
    apply_axes_style(ax_rhat)
    use_MeV_on_xaxis(ax_rhat)

    finite_ess = np.isfinite(ess)
    ax_ess.plot(
        eg_axis[finite_ess],
        ess[finite_ess],
        marker=".",
        linestyle="None",
        ms=1.5,
        color=COLORS["rose"],
    )

    ax_ess.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax_ess.set_ylabel(rf"$\mathrm{{ESS}}_{{\mathrm{{bulk}}}}({variable_symbol}_j)$")
    ax_ess.set_ylim(bottom=0.0)
    apply_axes_style(ax_ess)
    use_MeV_on_xaxis(ax_ess)


    add_panel_labels([ax_rhat, ax_ess], box=False)

    save_or_show(fig, out, tag="diagnostics", show=show)
    return fig
