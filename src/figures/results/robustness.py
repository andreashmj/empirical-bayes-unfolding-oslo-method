"""
Posterior robustness figure.

The figure compares a small set of robustness posterior runs with a baseline
posterior using
    S(E_gamma) = (alternative - baseline) / w_base,
where w_base is the half-width of the baseline posterior rank envelope.
A baseline-vs-baseline permutation band is shown as a null reference.
Each panel gets its own y-axis limits. The x-axis is shared across panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ...analysis.envelopes import global_rank_envelope
from ...analysis.results import UnfoldingResults
from ..plot_utils import (
    add_panel_label,
    add_top_legend,
    fill_step_band,
    rgba,
    save_or_show,
    steps,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size


def spectrum_draws(
    results: UnfoldingResults,
    ex_actual: float,
    var: str,
) -> xr.DataArray:
    """Return posterior draws for eta or x."""
    var = str(var).strip().lower()

    if var == "eta":
        return results.eta(ex_actual)

    if var == "x":
        return results.x(ex_actual)
    raise ValueError("var must be either 'eta' or 'x'.")


def draw_matrix(draws: xr.DataArray) -> np.ndarray:
    """Return a draw DataArray as a sample-by-Eg matrix."""
    return np.asarray(draws.transpose("sample", "Eg").values, dtype=float)


def rank_envelope_from_matrix(
    matrix: np.ndarray,
    eg_axis: np.ndarray,
    mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return global rank envelope from an array with shape sample by Eg."""

    matrix = np.asarray(matrix, dtype=float)
    eg_axis = np.asarray(eg_axis, dtype=float)

    data_array = xr.DataArray(
        matrix,
        dims=("sample", "Eg"),
        coords={
            "sample": np.arange(matrix.shape[0]),
            "Eg": eg_axis,
        },
    )
    return global_rank_envelope(data_array, mass=mass)


def paired_matrices(
    baseline_matrix: np.ndarray,
    alternative_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline and alternative matrices with the same sample count."""

    n_baseline = baseline_matrix.shape[0]
    n_alternative = alternative_matrix.shape[0]
    n_samples = min(n_baseline, n_alternative)

    if n_samples < 2:
        raise ValueError("At least two posterior samples are required.")

    if n_baseline == n_samples:
        baseline_indices = np.arange(n_samples)
    else:
        baseline_indices = rng.choice(n_baseline, size=n_samples, replace=False)

    if n_alternative == n_samples:
        alternative_indices = np.arange(n_samples)
    else:
        alternative_indices = rng.choice(n_alternative, size=n_samples, replace=False)

    return baseline_matrix[baseline_indices, :], alternative_matrix[alternative_indices, :]


def null_scaled_deviation(
    baseline_matrix: np.ndarray,
    baseline_half_width: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return baseline-vs-baseline null scaled deviations."""

    n_samples = baseline_matrix.shape[0]

    if n_samples < 2:
        raise ValueError("At least two baseline samples are required.")

    permutation = rng.permutation(n_samples)

    if np.all(permutation == np.arange(n_samples)):
        permutation = np.roll(permutation, 1)

    return (baseline_matrix - baseline_matrix[permutation, :]) / baseline_half_width[None, :]


def scaled_difference(
    baseline_matrix: np.ndarray,
    alternative_matrix: np.ndarray,
    baseline_half_width: np.ndarray,
) -> np.ndarray:
    """Return alternative-vs-baseline scaled deviations."""
    return (alternative_matrix - baseline_matrix) / baseline_half_width[None, :]

def y_limits_for_scaled_panel(
    null_envelope: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    alternative_envelope: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    reference_levels: list[float],
    padding_fraction: float = 0.05,
) -> tuple[float, float]:
    """Return y limits for one scaled-deviation panel.

    The null envelope is included in every panel because it is the reference
    distribution. Only the current alternative envelope is included, so a large
    robustness effect in one panel does not compress the other panels.
    """
    values = [
        null_envelope[1],
        null_envelope[2],
        null_envelope[3],
        alternative_envelope[1],
        alternative_envelope[2],
        alternative_envelope[3],
    ]

    finite_values = []

    for value in values:
        array = np.asarray(value, dtype=float).reshape(-1)
        array = array[np.isfinite(array)]

        if array.size:
            finite_values.append(array)

    if finite_values:
        all_values = np.concatenate(finite_values)
        lower = np.min(all_values).item()
        upper = np.max(all_values).item()
    else:
        lower, upper = -1.0, 1.0

    level_max = max(reference_levels)
    lower = min(lower, -level_max, 0.0)
    upper = max(upper, level_max, 0.0)

    span = upper - lower

    if not np.isfinite(span) or span <= 0.0:
        lower, upper = -1.0, 1.0
        span = 2.0

    padding = float(padding_fraction) * span

    return lower - padding, upper + padding


def panel_grid(
    n_panels: int,
    layout: str,
) -> tuple[int, int, str, float, dict]:
    """Return rows, columns, size layout, height ratio, and GridSpec kwargs."""

    layout = str(layout).strip().lower()
    if layout not in {"text", "column"}:
        raise ValueError("layout must be either 'text' or 'column'.")

    if n_panels == 1:
        return (
            1,
            1,
            "column" if layout == "column" else "text",
            0.48,
            {
                "left": 0.10 if layout == "text" else 0.17,
                "right": 0.97,
                "bottom": 0.15,
                "top": 0.82,
                "wspace": 0.0,
                "hspace": 0.0,
            },
        )

    if n_panels == 2 and layout == "column":
        return (
            2,
            1,
            "column",
            1.18,
            {
                "left": 0.17,
                "right": 0.97,
                "bottom": 0.13,
                "top": 0.84,
                "wspace": 0.0,
                "hspace": 0.16,
            },
        )

    n_columns = 2
    n_rows = int(np.ceil(n_panels / n_columns))

    return (
        n_rows,
        n_columns,
        "text",
        max(0.45, 0.38 * n_rows),
        {
            "left": 0.08,
            "right": 0.97,
            "bottom": 0.10,
            "top": 0.90,
            "wspace": 0.18,
            "hspace": 0.16,
        },
    )


def plot_scaled_panel(
    ax: plt.Axes,
    eg_null: np.ndarray,
    mean_null: np.ndarray,
    lower_null: np.ndarray,
    upper_null: np.ndarray,
    eg_alt: np.ndarray,
    mean_alt: np.ndarray,
    lower_alt: np.ndarray,
    upper_alt: np.ndarray,
    label: str,
    baseline_band_color: str,
    baseline_edge_color: str,
    alternative_band_color: str,
    alternative_mean_color: str,
    reference_color: str,
    reference_levels: list[float],
    show_xlabel: bool,
    show_ylabel: bool,
    y_limits: tuple[float, float],
) -> None:
    """Plot one robustness panel."""

    fill_step_band(
        ax,
        eg_null,
        lower_null,
        upper_null,
        facecolor=baseline_band_color,
        edgecolor=baseline_edge_color,
        alpha=0.33,
        linewidth=0.30,
        zorder=1.5,
    )

    steps(
        ax,
        eg_null,
        mean_null,
        color=baseline_edge_color,
        lw=0.80,
        zorder=3.0,
    )

    fill_step_band(
        ax,
        eg_alt,
        lower_alt,
        upper_alt,
        facecolor=alternative_band_color,
        edgecolor=alternative_band_color,
        alpha=0.24,
        linewidth=0.30,
        zorder=2.0,
    )

    steps(
        ax,
        eg_alt,
        mean_alt,
        color=alternative_mean_color,
        lw=0.80,
        zorder=3.0,
    )

    for level in reference_levels:
        ax.axhline(
            level,
            color=reference_color,
            lw=0.55,
            ls="--",
            alpha=0.22,
            zorder=1.0,
        )
        ax.axhline(
            -level,
            color=reference_color,
            lw=0.55,
            ls="--",
            alpha=0.22,
            zorder=1.0,
        )

    ax.set_ylim(*y_limits)

    if show_ylabel:
        ax.set_ylabel(r"$S(E_\gamma)$")

    if show_xlabel:
        ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    else:
        ax.tick_params(labelbottom=False)

    add_panel_label(ax, label=label)
    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)


def panel_descriptions(
    labels: list[str] | None,
    n_panels: int,
) -> list[str]:
    """Return panel descriptions for stdout/caption writing."""
    if labels is None:
        return [f"Robustness case {index + 1}" for index in range(n_panels)]

    if len(labels) != n_panels:
        raise ValueError("labels must have the same length as alt_post_ncs.")

    return list(labels)


def print_panel_mapping(labels: list[str]) -> None:
    """Print panel descriptions to stdout."""
    print("[robustness] panel mapping:")

    for index, label in enumerate(labels):
        panel_label = chr(ord("a") + index)
        print(f"  ({panel_label}) {label}")


def plot_robustness(
    baseline_post_nc: str | Path,
    alt_post_ncs: list[str | Path],
    labels: list[str] | None = None,
    ex: float | None = None,
    var: str = "eta",
    mass: float = 0.95,
    out: str | Path | None = None,
    show: bool = True,
    layout: str = "text",
    seed: int | None = 123,
) -> plt.Figure:
    """Create the posterior robustness figure."""

    if len(alt_post_ncs) == 0:
        raise ValueError("At least one alternative posterior file is required.")

    var = str(var).strip().lower()
    if var not in {"eta", "x"}:
        raise ValueError("var must be either 'eta' or 'x'.")

    n_panels = len(alt_post_ncs)
    descriptions = panel_descriptions(labels, n_panels=n_panels)

    rng = np.random.default_rng(seed)

    baseline_results = UnfoldingResults(baseline_post_nc, expected_mode="posterior")
    ex_actual = baseline_results.ex_actual(ex)
    eg_axis = baseline_results.eg(ex_actual)

    baseline_draws = spectrum_draws(baseline_results, ex_actual, var)
    _, _, baseline_lower, baseline_upper = global_rank_envelope(
        baseline_draws,
        mass=mass,
    )

    baseline_half_width = np.maximum(
        0.5 * (baseline_upper - baseline_lower),
        1.0e-12,
    )

    baseline_matrix = draw_matrix(baseline_draws)

    null_matrix = null_scaled_deviation(
        baseline_matrix=baseline_matrix,
        baseline_half_width=baseline_half_width,
        rng=rng,
    )
    null_envelope = rank_envelope_from_matrix(
        null_matrix,
        eg_axis,
        mass=mass,
    )

    alternative_envelopes = []

    for path in alt_post_ncs:
        alternative_results = UnfoldingResults(path, expected_mode="posterior")

        if alternative_results.n_eg(ex_actual) != baseline_results.n_eg(ex_actual):
            raise ValueError(f"Eg_len mismatch between baseline and {path}.")

        if not np.allclose(
            alternative_results.eg(ex_actual),
            eg_axis,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"Eg grid mismatch between baseline and {path}.")

        alternative_draws = spectrum_draws(alternative_results, ex_actual, var)
        alternative_matrix = draw_matrix(alternative_draws)

        baseline_paired, alternative_paired = paired_matrices(
            baseline_matrix=baseline_matrix,
            alternative_matrix=alternative_matrix,
            rng=rng,
        )

        alternative_scaled = scaled_difference(
            baseline_matrix=baseline_paired,
            alternative_matrix=alternative_paired,
            baseline_half_width=baseline_half_width,
        )

        alternative_envelopes.append(
            rank_envelope_from_matrix(
                alternative_scaled,
                eg_axis,
                mass=mass,
            )
        )

    reference_levels = [0.5, 1.0, 1.5, 2.0]

    baseline_band_color = "#9E9E9E"
    baseline_edge_color = "#6E6E6E"
    alternative_band_color = "#5AAE61"
    alternative_mean_color = "#762A83"
    reference_color = COLORS["black"]

    n_rows, n_columns, size_layout, height_to_width, grid_kwargs = panel_grid(
        n_panels=n_panels,
        layout=layout,
    )
    size = figure_size(size_layout, height_to_width=height_to_width)

    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        n_rows,
        n_columns,
        **grid_kwargs,
    )

    axes: list[plt.Axes] = []
    shared_axis = None

    for panel_index in range(n_rows * n_columns):
        row = panel_index // n_columns
        column = panel_index % n_columns

        ax = fig.add_subplot(
            grid[row, column],
            sharex=shared_axis,
        )

        if shared_axis is None:
            shared_axis = ax

        axes.append(ax)

    for panel_index, ax in enumerate(axes):
        if panel_index >= n_panels:
            ax.axis("off")
            continue

        row = panel_index // n_columns
        column = panel_index % n_columns
        show_xlabel = row == n_rows - 1
        show_ylabel = column == 0
        label = chr(ord("a") + panel_index)

        envelope = alternative_envelopes[panel_index]
        panel_y_limits = y_limits_for_scaled_panel(
            null_envelope=null_envelope,
            alternative_envelope=envelope,
            reference_levels=reference_levels,
        )

        plot_scaled_panel(
            ax=ax,
            eg_null=null_envelope[0],
            mean_null=null_envelope[1],
            lower_null=null_envelope[2],
            upper_null=null_envelope[3],
            eg_alt=envelope[0],
            mean_alt=envelope[1],
            lower_alt=envelope[2],
            upper_alt=envelope[3],
            label=label,
            baseline_band_color=baseline_band_color,
            baseline_edge_color=baseline_edge_color,
            alternative_band_color=alternative_band_color,
            alternative_mean_color=alternative_mean_color,
            reference_color=reference_color,
            reference_levels=reference_levels,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            y_limits=panel_y_limits,
        )

    mass_percent = round(100.0 * mass)

    alternative_handle = (
        Patch(
            facecolor=rgba(alternative_band_color, 0.24),
            edgecolor=alternative_band_color,
            linewidth=0.30,
        ),
        Line2D([0], [0], color=alternative_mean_color, lw=0.80),
    )

    baseline_handle = (
        Patch(
            facecolor=rgba(baseline_band_color, 0.33),
            edgecolor=baseline_edge_color,
            linewidth=0.30,
        ),
        Line2D([0], [0], color=baseline_edge_color, lw=0.60),
    )

    reference_handle = Line2D(
        [0],
        [0],
        color=reference_color,
        lw=0.55,
        ls="--",
        alpha=0.55,
    )

    if size_layout == "column":
        legend_ncol = 1
        legend_fontsize = 7.8
        legend_y = 0.987
        handlelength = 1.7
        columnspacing = 0.9
    else:
        legend_ncol = 3
        legend_fontsize = 9.0
        legend_y = 0.978
        handlelength = 2.2
        columnspacing = 1.4
        
    add_top_legend(
        fig,
        handles=[alternative_handle, baseline_handle, reference_handle],
        labels=[
            rf"$S_{{\mathrm{{alt}}}}$: mean + {mass_percent}\% rank-env.",
            rf"$S_{{\mathrm{{base}}}}$: mean + {mass_percent}\% rank-env.",
            r"reference levels",
        ],
        ncol=legend_ncol,
        y=legend_y,
        frameon=False,
        handlelength=handlelength,
        columnspacing=columnspacing,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        fontsize=legend_fontsize,
    )

    print_panel_mapping(descriptions)

    save_or_show(fig, out, tag="robustness", show=show)
    return fig
