"""
Figure comparing Bayes, RMLE and iterative unfolding for 28Si data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import ompy as om
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

from ...analysis.envelopes import global_rank_envelope
from ...analysis.results import UnfoldingResults
from ...config_utils import load_run_config
from ...paths import repo_path
from ..plot_utils import (
    add_panel_label,
    fill_step_band,
    save_or_show,
    steps,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style


LINTHRESH = 5.0
MATRIX_EX_MIN_KEV = 0.0
MATRIX_EX_MAX_KEV = 12000.0

BAYES_COLOR = COLORS.get("blue", "tab:blue")
BAYES_BAND = COLORS.get("sky", "lightskyblue")
RMLE_COLOR = COLORS.get("vermillion", "tab:orange")
RMLE_BAND = COLORS.get("orange", "moccasin")
ITER_COLOR = COLORS.get("green", "tab:green")



def cfg(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config else default


def requested_ex_values(config: dict[str, Any]) -> list[float]:
    values = config["ex_energies"]
    if len(values) != 2:
        raise ValueError("The 28Si comparison figure expects exactly two Ex values.")
    return [float(values[0]), float(values[1])]


def load_matrix(config: dict[str, Any], key: str) -> om.Matrix:
    matrix = om.Matrix.from_path(str(repo_path(config[key])))
    matrix = matrix.set_xalias("Ex").set_yalias("Eg")
    matrix = matrix.loc[:, str(cfg(config, "lower_eg_cut", "200keV")) :]

    rebin_ex, rebin_eg = tuple(cfg(config, "rebin_factors", [4, 1]))

    if int(rebin_ex) > 1:
        matrix = matrix.rebin("Ex", factor=int(rebin_ex))
    if int(rebin_eg) > 1:
        matrix = matrix.rebin("Eg", factor=int(rebin_eg))

    return matrix


def bayes_path(config: dict[str, Any]) -> Path:
    return (
        repo_path(cfg(config, "output_dir", "results"))
        / str(config["dataset_id"])
        / str(cfg(config, "run_id", "bayesian_pymc"))
        / str(cfg(config, "mode", "posterior"))/ "draws.nc"
    )


def rmle_path(config: dict[str, Any], ex: float) -> Path:
    return (
        repo_path(cfg(config, "output_dir", "results"))
        / str(config["dataset_id"])
        / "rmle"/ f"Ex{int(round(ex))}"/ "summary.npz"
    )


def iterative_path(config: dict[str, Any], ex: float) -> Path:
    return (
        repo_path(cfg(config, "output_dir", "results"))
        / str(config["dataset_id"])
        / "ompy_iterative"/ f"Ex{int(round(ex))}"/ "summary.npz"
    )


def axis_step(axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float)
    if axis.size < 2:
        return 1.0
    return float(np.nanmedian(np.diff(axis)))


def nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(axis, dtype=float) - float(value))))


def image_extent(ex_axis: np.ndarray, eg_axis: np.ndarray) -> tuple[float, float, float, float]:
    d_ex = axis_step(ex_axis)
    d_eg = axis_step(eg_axis)

    return (
        (eg_axis[0] - 0.5 * d_eg) / 1000.0,
        (eg_axis[-1] + 0.5 * d_eg) / 1000.0,
        (ex_axis[0] - 0.5 * d_ex) / 1000.0,
        (ex_axis[-1] + 0.5 * d_ex) / 1000.0,
    )


def shared_matrix_norm(prompt: om.Matrix, background: om.Matrix) -> LogNorm:
    values = np.concatenate(
        [
            np.asarray(prompt.values, dtype=float).ravel(),
            np.asarray(background.values, dtype=float).ravel(),
        ]
    )
    values = values[np.isfinite(values) & (values > 0.0)]

    vmax = max(float(np.nanpercentile(values, 99.5)), 1.0) if values.size else 1.0
    return LogNorm(vmin=1.0, vmax=vmax)


def masked_positive(values: np.ndarray) -> np.ma.MaskedArray:
    values = np.asarray(values, dtype=float)
    return np.ma.masked_where((~np.isfinite(values)) | (values <= 0.0), values)


def plot_matrix_panel(
    ax: plt.Axes,
    matrix: om.Matrix,
    selected_ex: list[float],
    norm: LogNorm,
) -> Any:
    ex_axis = np.asarray(matrix.Ex, dtype=float)
    eg_axis = np.asarray(matrix.Eg, dtype=float)

    image = ax.imshow(
        masked_positive(matrix.values),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=image_extent(ex_axis, eg_axis),
        norm=norm,
        cmap="viridis",
        rasterized=True,
    )

    d_ex = axis_step(ex_axis)

    for ex in selected_ex:
        index = nearest_index(ex_axis, ex)

        low = (ex_axis[index] - 0.5 * d_ex) / 1000.0
        high = (ex_axis[index] + 0.5 * d_ex) / 1000.0

        ax.axhspan(
            low,
            high,
            facecolor="gray",
            edgecolor="none",
            alpha=0.60,
            zorder=5,
        )

    ax.set_xlim(eg_axis[0] / 1000.0, eg_axis[-1] / 1000.0)
    ax.set_ylim(MATRIX_EX_MIN_KEV / 1000.0, MATRIX_EX_MAX_KEV / 1000.0)
    ax.set_ylabel(r"Excitation energy $E_x$ (MeV)")

    apply_axes_style(ax)
    return image


def bayesian_eta_summary(path: Path, ex_requested: float) -> dict[str, np.ndarray | float]:
    results = UnfoldingResults(path, expected_mode="posterior")
    ex_actual = float(results.ex_actual(ex_requested))
    eta_draws = results.eta(ex_actual)
    eg, mean, lower, upper = global_rank_envelope(eta_draws, mass=0.95)

    return {
        "Ex_requested": float(ex_requested),
        "Ex_actual": ex_actual,
        "Eg_keV": np.asarray(eg, dtype=float),
        "mean": np.asarray(mean, dtype=float),
        "lower": np.asarray(lower, dtype=float),
        "upper": np.asarray(upper, dtype=float),
    }





def read_rmle(path: Path, reference_eg: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        eg = np.asarray(data["Eg_keV"], dtype=float)
        mean = np.asarray(data["ci_eta_center"], dtype=float)
        lower = np.asarray(data["ci_eta_lo"], dtype=float)
        upper = np.asarray(data["ci_eta_hi"], dtype=float)

    return {
        "Eg_keV": eg,
        "mean": mean,
        "lower": lower,
        "upper": upper,
    }


def read_iterative(path: Path, reference_eg: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        eg = np.asarray(data["Eg_keV"], dtype=float)
        eta = np.asarray(data["eta_hat"], dtype=float)

    return {
        "Eg_keV": eg,
        "eta": eta,
    }



def plot_slice_panel(
    ax: plt.Axes,
    bayes: dict[str, np.ndarray | float],
    rmle: dict[str, np.ndarray],
    iterative: dict[str, np.ndarray],
) -> None:
    eg = np.asarray(bayes["Eg_keV"], dtype=float)
    
    bayes_mean = np.asarray(bayes["mean"], dtype=float)
    bayes_lower = np.asarray(bayes["lower"], dtype=float)
    bayes_upper = np.asarray(bayes["upper"], dtype=float)

    rmle_mean = np.asarray(rmle["mean"], dtype=float)
    rmle_lower = np.asarray(rmle["lower"], dtype=float)
    rmle_upper = np.asarray(rmle["upper"], dtype=float)

    iterative_eta = np.asarray(iterative["eta"], dtype=float)

    fill_step_band(
        ax,
        eg,
        bayes_lower,
        bayes_upper,
        facecolor=BAYES_BAND,
        edgecolor=BAYES_BAND,
        alpha=0.25,
        linewidth=0.35,
        zorder=2,
    )
    steps(ax, eg, bayes_mean, color=BAYES_COLOR, lw=0.95, zorder=3)

    fill_step_band(
        ax,
        eg,
        rmle_lower,
        rmle_upper,
        facecolor=RMLE_BAND,
        edgecolor=RMLE_BAND,
        alpha=0.18,
        linewidth=0.35,
        zorder=2,
    )
    steps(ax, eg, rmle_mean, color=RMLE_COLOR, lw=0.95, zorder=3)

    steps(ax, eg, iterative_eta, color=ITER_COLOR, lw=1.15, ls=":", zorder=3)
    

    ax.set_xlim(eg[0], eg[-1])
    ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax.set_ylabel(r"$\boldsymbol{\eta}$ (counts)")

    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)


def method_legend_handles() -> list:
    bayes_handle = (
        Patch(
            facecolor=BAYES_BAND,
            edgecolor=BAYES_BAND,
            alpha=0.25,
            linewidth=0.35,
        ),
        Line2D([0], [0], color=BAYES_COLOR, lw=0.95),
    )

    rmle_handle = (
        Patch(
            facecolor=RMLE_BAND,
            edgecolor=RMLE_BAND,
            alpha=0.18,
            linewidth=0.35,
        ),
        Line2D([0], [0], color=RMLE_COLOR, lw=0.95),
    )

    iterative_handle = Line2D([0], [0], color=ITER_COLOR, lw=1.15, ls=":")

    return [bayes_handle, rmle_handle, iterative_handle]



def plot_experimental_method_comparison(
    config_path: str | Path,
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    config = load_run_config(repo_path(config_path))
    ex_requested = requested_ex_values(config)

    prompt = load_matrix(config, "prompt_path")
    background = load_matrix(config, "background_path")

    bayes = [
        bayesian_eta_summary(bayes_path(config), ex_requested[0]),
        bayesian_eta_summary(bayes_path(config), ex_requested[1]),
    ]

    rmle = [
        read_rmle(rmle_path(config, ex_requested[0]), np.asarray(bayes[0]["Eg_keV"], dtype=float)),
        read_rmle(rmle_path(config, ex_requested[1]), np.asarray(bayes[1]["Eg_keV"], dtype=float)),
    ]

    iterative = [
        read_iterative(iterative_path(config, ex_requested[0]), np.asarray(bayes[0]["Eg_keV"], dtype=float)),
        read_iterative(iterative_path(config, ex_requested[1]), np.asarray(bayes[1]["Eg_keV"], dtype=float)),
    ]

    selected_ex = [
        float(bayes[0]["Ex_actual"]),
        float(bayes[1]["Ex_actual"]),
    ]

    fig = plt.figure(figsize=(7.1, 6.05))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.97,
        bottom=0.08,
        top=0.90,
        hspace=0.40,
        wspace=0.18,
    )

    ax_prompt = fig.add_subplot(grid[0, 0])
    ax_background = fig.add_subplot(grid[0, 1], sharey=ax_prompt)
    ax_slice_0 = fig.add_subplot(grid[1, 0])
    ax_slice_1 = fig.add_subplot(grid[1, 1])

    norm = shared_matrix_norm(prompt, background)

    image = plot_matrix_panel(ax_prompt, prompt, selected_ex, norm)
    plot_matrix_panel(ax_background, background, selected_ex, norm)

    ax_background.tick_params(labelleft=False)
    ax_background.set_ylabel("")

    plot_slice_panel(ax_slice_0, bayes[0], rmle[0], iterative[0])
    plot_slice_panel(ax_slice_1, bayes[1], rmle[1], iterative[1])

    ax_slice_1.set_ylabel("")

    add_panel_label(ax_prompt, "a", box=True)
    add_panel_label(ax_background, "b", box=True)
    add_panel_label(ax_slice_0, "c")
    add_panel_label(ax_slice_1, "d")

    cax = fig.add_axes([0.2, 0.2, 0.2, 0.02])
    colorbar = fig.colorbar(
        image,
        cax=cax,
        orientation="horizontal",
        extend="max",
    )
    colorbar.ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    colorbar.ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    colorbar.set_label("counts", labelpad=2.0)
    colorbar.ax.tick_params(direction="in", length=3.0, width=0.8)
    colorbar.outline.set_linewidth(0.8)
    
    
    left_position = ax_prompt.get_position()
    right_position = ax_background.get_position()

    matrix_x0 = left_position.x0
    matrix_x1 = right_position.x1
    matrix_width = matrix_x1 - matrix_x0

    colorbar_width = matrix_width * 0.65
    colorbar_x0 = matrix_x0 + 0.5 * (matrix_width - colorbar_width)
    
    
    cax.set_position([
            colorbar_x0,
            left_position.y0 - 0.06,
            colorbar_width,
            0.020]
    )

    fig.legend(
        handles=method_legend_handles(),
        labels=["Bayes", "RMLE", "Iterative method"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=2.2,
        columnspacing=1.4,
        fontsize=8.6,
    )

    save_or_show(
        fig,
        repo_path(out) if out is not None else None,
        tag="experimental-method-comparison",
        show=show,
    )

    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the 28Si experimental method-comparison figure."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--out",
        default="figures/results/experimental_28Si_method_comparison.pdf",
    )
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    plot_experimental_method_comparison(
        config_path=args.config,
        out=args.out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
