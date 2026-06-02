"""
Bayesian-versus-RMLE comparison figure.

The figure compares one Bayesian posterior result with one OMpy RMLE summary
for the same excitation-energy row. The RMLE summary is expected to contain the
point estimate and bootstrap confidence band written by
``src.frequentist.run_rmle_on_synthetic``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ...analysis.envelopes import global_rank_envelope
from ...analysis.results import UnfoldingResults
from ...paths import repo_path
from ..plot_utils import (
    add_panel_label,
    add_zoom_guides,
    fill_step_band,
    local_y_limits,
    rgba,
    save_or_show,
    set_linear_y_with_zero_room,
    set_local_y_axis,
    steps,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size


def variable_symbol(var: str) -> str:
    """Return the paper symbol for a stored variable name."""

    var = str(var).strip().lower()
    if var == "eta":
        return r"\boldsymbol{\eta}"
    if var == "x":
        return r"\boldsymbol{\mu}"
    if var == "nu":
        return r"\boldsymbol{\nu}"
    raise ValueError("var must be eta, x, or nu.")


def frequentist_key(var: str) -> str:
    """Return the prefix used in RMLE summary files."""

    var = str(var).strip().lower()
    if var == "x":
        return "mu"
    if var in {"eta", "nu"}:
        return var
    raise ValueError("var must be eta, x, or nu.")


def bayesian_draws_and_truth(
    results: UnfoldingResults,
    ex_actual: float,
    var: str,
) -> tuple[xr.DataArray, np.ndarray, str]:
    """Return Bayesian draws, truth vector, and paper symbol."""

    var = str(var).strip().lower()
    if var == "eta":
        return results.eta(ex_actual), results.eta_true(ex_actual), variable_symbol(var)
    if var == "x":
        return results.x(ex_actual), results.x_true(ex_actual), variable_symbol(var)
    if var == "nu":
        return results.nu(ex_actual), results.nu_true(ex_actual), variable_symbol(var)
    raise ValueError("var must be eta, x, or nu.")


def draw_matrix(draws: xr.DataArray) -> np.ndarray:
    """Return a draw DataArray as a sample-by-Eg matrix."""

    return np.asarray(draws.transpose("sample", "Eg").values, dtype=float)


def npz_get(npz: np.lib.npyio.NpzFile, keys: Sequence[str]) -> np.ndarray | None:
    """Return the first matching npz array, with case-insensitive fallback."""

    names = set(npz.files)
    for key in keys:
        if key in names:
            return npz[key]

    lower_name_map = {name.lower(): name for name in npz.files}
    for key in keys:
        lower_key = key.lower()
        if lower_key in lower_name_map:
            return npz[lower_name_map[lower_key]]

    return None


def find_ex_dir(frequentist_root: Path, ex: float) -> Path:
    """Find the RMLE output directory for one excitation-energy row."""

    ex_int = round(float(ex))
    candidates = [
        frequentist_root / f"Ex{ex_int}",
        frequentist_root / f"Ex{ex_int}keV",
        frequentist_root / f"Ex{ex_int}_keV",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    matches = [path for path in frequentist_root.glob(f"Ex{ex_int}*") if path.is_dir()]
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find a frequentist Ex directory under {frequentist_root} "
        f"for Ex={ex_int} keV."
    )


def check_same_grid(target_eg: np.ndarray, source_eg: np.ndarray, summary_path: Path) -> None:
    """Require the Bayesian and RMLE results to use the same Eg grid."""

    if target_eg.shape != source_eg.shape:
        raise ValueError(
            f"Eg-grid length mismatch for {summary_path}: "
            f"Bayesian has {target_eg.size}, RMLE has {source_eg.size}."
        )

    if not np.allclose(target_eg, source_eg, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"Eg-grid mismatch between Bayesian result and {summary_path}.")


def read_frequentist_summary(
    frequentist_root: str | Path,
    ex: float,
    var: str,
    target_eg: np.ndarray,
) -> dict:
    """Read one RMLE summary file and check grid compatibility."""

    key = frequentist_key(var)
    ex_dir = find_ex_dir(Path(frequentist_root), ex)
    summary_path = ex_dir / "summary.npz"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing frequentist summary file: {summary_path}")

    with np.load(summary_path, allow_pickle=False) as summary:
        source_eg = npz_get(summary, ["Eg_keV", "eg_keV", "Eg", "eg"])
        if source_eg is None:
            raise KeyError(f"{summary_path}: missing Eg axis.")

        source_eg = np.asarray(source_eg, dtype=float).reshape(-1)
        check_same_grid(np.asarray(target_eg, dtype=float), source_eg, summary_path)

        estimate = npz_get(summary, [f"{key}_hat"])
        center = npz_get(summary, [f"ci_{key}_center"])
        lower = npz_get(summary, [f"ci_{key}_lo"])
        upper = npz_get(summary, [f"ci_{key}_hi"])

    if center is None:
        if estimate is None:
            raise KeyError(f"{summary_path}: missing {key}_hat and ci_{key}_center.")
        center = estimate

    if lower is None or upper is None:
        raise KeyError(
            f"{summary_path}: missing confidence band for {key}. "
            f"Expected ci_{key}_lo and ci_{key}_hi."
        )

    return {
        "summary_path": summary_path,
        "Eg": source_eg,
        "center": np.asarray(center, dtype=float).reshape(-1),
        "lower": np.asarray(lower, dtype=float).reshape(-1),
        "upper": np.asarray(upper, dtype=float).reshape(-1),
    }


def rank_envelope_from_matrix(
    matrix: np.ndarray,
    eg_axis: np.ndarray,
    mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return global rank envelope from a sample-by-Eg matrix."""

    matrix = np.asarray(matrix, dtype=float)
    data_array = xr.DataArray(
        matrix,
        dims=("sample", "Eg"),
        coords={"sample": np.arange(matrix.shape[0]), "Eg": eg_axis},
    )
    return global_rank_envelope(data_array, mass=mass)


def default_zoom_windows(eg_axis: np.ndarray) -> list[tuple[float, float]]:
    """Return low-, mid-, and high-energy zoom windows."""

    lower = float(np.nanmin(eg_axis))
    upper = float(np.nanmax(eg_axis))
    span = max(upper - lower, 1.0)
    return [
        (lower, lower + span / 3.0),
        (lower + span / 3.0, lower + 2.0 * span / 3.0),
        (lower + 2.0 * span / 3.0, upper),
    ]


def parse_zoom_windows(
    zoom1: Sequence[float] | None,
    zoom2: Sequence[float] | None,
    zoom3: Sequence[float] | None,
    eg_axis: np.ndarray,
) -> list[tuple[float, float]]:
    """Return three zoom windows in keV."""

    windows = []
    for input_window, default_window in zip(
        [zoom1, zoom2, zoom3],
        default_zoom_windows(eg_axis),
    ):
        if input_window is None:
            windows.append(default_window)
            continue

        lower, upper = float(input_window[0]), float(input_window[1])
        if lower > upper:
            lower, upper = upper, lower
        windows.append((lower, upper))

    return windows


def mask_in_window(eg_axis: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Return an Eg-axis mask for one zoom window."""

    lower, upper = window
    return (eg_axis >= lower) & (eg_axis <= upper)


def scaled_deviation_matrix(
    matrix: np.ndarray,
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return scaled-deviation draws using an envelope half-width."""

    half_width = np.maximum(0.5 * (upper - lower), 1.0e-12)
    return (matrix - truth[None, :]) / half_width[None, :]


def scaled_deviation_envelope_from_band(
    center: np.ndarray,
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scaled-deviation center and band from a precomputed band."""

    half_width = np.maximum(0.5 * (upper - lower), 1.0e-12)
    return (
        (center - truth) / half_width,
        (lower - truth) / half_width,
        (upper - truth) / half_width,
    )


def scaled_y_limits(
    arrays: list[np.ndarray],
    minimum_abs: float = 2.0,
    padding_fraction: float = 0.05,
) -> tuple[float, float]:
    """Return common y-limits for scaled-deviation panels."""

    finite = []
    for values in arrays:
        array = np.asarray(values, dtype=float).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size:
            finite.append(array)

    if finite:
        values = np.concatenate(finite)
        lower = min(float(np.min(values)), -float(minimum_abs), 0.0)
        upper = max(float(np.max(values)), float(minimum_abs), 0.0)
    else:
        lower, upper = -float(minimum_abs), float(minimum_abs)

    span = upper - lower
    if not np.isfinite(span) or span <= 0.0:
        return -float(minimum_abs), float(minimum_abs)

    padding = float(padding_fraction) * span
    return lower - padding, upper + padding


def add_zoom_zero_line_if_visible(
    ax: plt.Axes,
    color: str,
    linewidth: float,
    lower_fraction: float = 0.035,
) -> None:
    """Draw a zero line only when the local axis already reaches zero."""

    y_lower, y_upper = ax.get_ylim()
    if not np.isfinite(y_lower) or not np.isfinite(y_upper) or y_upper <= 0.0:
        return
    if y_lower > 0.0:
        return
    if y_lower == 0.0:
        ax.set_ylim(-float(lower_fraction) * y_upper, y_upper)

    ax.axhline(0.0, color=color, lw=float(linewidth), ls=":", alpha=0.55, zorder=1.1)


def plot_spectrum_comparison(
    ax: plt.Axes,
    eg_axis: np.ndarray,
    bayes_mean: np.ndarray,
    bayes_lower: np.ndarray,
    bayes_upper: np.ndarray,
    freq_center: np.ndarray,
    freq_lower: np.ndarray,
    freq_upper: np.ndarray,
    truth: np.ndarray,
    colors: dict,
    linewidths: dict,
    alphas: dict,
) -> None:
    """Plot Bayesian band, RMLE band, central lines, and truth."""

    fill_step_band(
        ax,
        eg_axis,
        freq_lower,
        freq_upper,
        facecolor=colors["freq_band"],
        edgecolor=colors["freq_band"],
        alpha=alphas["freq"],
        linewidth=linewidths["edge"],
        zorder=1.8,
    )
    fill_step_band(
        ax,
        eg_axis,
        bayes_lower,
        bayes_upper,
        facecolor=colors["bayes_band"],
        edgecolor=colors["bayes_band"],
        alpha=alphas["bayes"],
        linewidth=linewidths["edge"],
        zorder=2.0,
    )
    steps(ax, eg_axis, bayes_mean, color=colors["bayes_line"], lw=linewidths["line"], zorder=3.2)
    steps(ax, eg_axis, freq_center, color=colors["freq_line"], lw=linewidths["line"], zorder=3.1)
    steps(ax, eg_axis, truth, color=colors["truth"], lw=linewidths["truth"], ls="--", zorder=3.3)


def plot_scaled_deviation(
    ax: plt.Axes,
    eg_axis: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color_band: str,
    color_line: str,
    color_zero: str,
    alpha_band: float,
    linewidth_line: float,
    linewidth_edge: float,
    linewidth_zero: float,
) -> None:
    """Plot one scaled-deviation band."""

    fill_step_band(
        ax,
        eg_axis,
        lower,
        upper,
        facecolor=color_band,
        edgecolor=color_band,
        alpha=alpha_band,
        linewidth=linewidth_edge,
        zorder=2.0,
    )
    steps(ax, eg_axis, mean, color=color_line, lw=linewidth_line, zorder=3.0)
    ax.axhline(0.0, color=color_zero, lw=linewidth_zero, ls=":", alpha=0.85, zorder=1.5)


def plot_bayes_frequentist_comparison(
    bayes_nc: str | Path,
    freq_dir: str | Path,
    ex: float,
    var: str = "eta",
    mass: float = 0.95,
    zoom1: Sequence[float] | None = None,
    zoom2: Sequence[float] | None = None,
    zoom3: Sequence[float] | None = None,
    layout: str = "text",
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create the Bayesian-versus-RMLE comparison figure."""

    var = str(var).strip().lower()
    if var not in {"eta", "x", "nu"}:
        raise ValueError("var must be eta, x, or nu.")

    layout = str(layout).strip().lower()
    if layout not in {"text", "column"}:
        raise ValueError("layout must be either 'text' or 'column'.")

    bayes_results = UnfoldingResults(bayes_nc, expected_mode="posterior")
    ex_actual = bayes_results.ex_actual(ex)
    bayes_draws, truth_raw, symbol = bayesian_draws_and_truth(bayes_results, ex_actual, var)

    truth = np.asarray(truth_raw, dtype=float)
    eg_axis = np.asarray(bayes_draws["Eg"].values, dtype=float)

    eg_bayes, bayes_mean, bayes_lower, bayes_upper = global_rank_envelope(
        bayes_draws,
        mass=mass,
    )
    bayes_matrix = draw_matrix(bayes_draws)
    bayes_scaled_matrix = scaled_deviation_matrix(
        bayes_matrix,
        truth,
        bayes_lower,
        bayes_upper,
    )
    eg_bayes_scaled, bayes_scaled_mean, bayes_scaled_lower, bayes_scaled_upper = (
        rank_envelope_from_matrix(bayes_scaled_matrix, eg_axis, mass=mass)
    )

    freq = read_frequentist_summary(
        frequentist_root=freq_dir,
        ex=ex,
        var=var,
        target_eg=eg_axis,
    )
    freq_scaled_mean, freq_scaled_lower, freq_scaled_upper = scaled_deviation_envelope_from_band(
        freq["center"],
        truth,
        freq["lower"],
        freq["upper"],
    )

    windows = parse_zoom_windows(zoom1=zoom1, zoom2=zoom2, zoom3=zoom3, eg_axis=eg_axis)
    masks = [mask_in_window(eg_axis, window) for window in windows]
    zoom_limits = [
        local_y_limits(
            mask,
            [
                bayes_lower,
                bayes_upper,
                bayes_mean,
                freq["lower"],
                freq["upper"],
                freq["center"],
                truth,
            ],
        )
        for mask in masks
    ]

    scaled_ylim = scaled_y_limits(
        [
            bayes_scaled_mean,
            bayes_scaled_lower,
            bayes_scaled_upper,
            freq_scaled_mean,
            freq_scaled_lower,
            freq_scaled_upper,
        ],
        minimum_abs=2.0,
        padding_fraction=0.05,
    )

    colors = {
        "bayes_band": COLORS["sky"],
        "bayes_line": COLORS["blue"],
        "freq_band": COLORS["orange"],
        "freq_line": COLORS["vermillion"],
        "bayes_dev_band": COLORS["green"],
        "bayes_dev_line": COLORS["green"],
        "freq_dev_band": COLORS["purple"],
        "freq_dev_line": COLORS["purple"],
        "truth": COLORS["black"],
    }
    alphas = {"bayes": 0.25, "freq": 0.18, "dev": 0.20}
    linewidths = {"line": 0.75, "truth": 0.60, "edge": 0.35, "zero": 0.55}
    mass_percent = round(100.0 * mass)

    size = figure_size("text", height_to_width=1.34)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.10, 0.92, 0.82, 0.82],
        left=0.10,
        right=0.98,
        bottom=0.06,
        top=0.92,
        wspace=0.24,
        hspace=0.16,
    )

    ax_full = fig.add_subplot(grid[0, :])
    plot_spectrum_comparison(
        ax=ax_full,
        eg_axis=eg_bayes,
        bayes_mean=bayes_mean,
        bayes_lower=bayes_lower,
        bayes_upper=bayes_upper,
        freq_center=freq["center"],
        freq_lower=freq["lower"],
        freq_upper=freq["upper"],
        truth=truth,
        colors=colors,
        linewidths=linewidths,
        alphas=alphas,
    )
    set_linear_y_with_zero_room(
        ax_full,
        arrays=[bayes_upper, freq["upper"], truth],
        top_factor=1.15,
        lower_fraction=0.035,
        draw_zero=True,
        zero_color=colors["truth"],
    )
    x_right = float(np.nanmax(eg_axis))
    ax_full.set_xlim(-0.025 * x_right, x_right)
    ax_full.set_ylabel(rf"${symbol}$ (counts)")
    ax_full.tick_params(labelbottom=False)
    apply_axes_style(ax_full)
    use_MeV_on_xaxis(ax_full)
    add_panel_label(ax_full, label="a")

    zoom_axes: list[plt.Axes] = []
    for panel_index, (window, mask, label, y_limits) in enumerate(
        zip(windows, masks, ["b", "c", "d"], zoom_limits)
    ):
        ax_zoom = fig.add_subplot(grid[1, panel_index])
        zoom_axes.append(ax_zoom)
        lower, upper = window

        if not np.any(mask):
            ax_zoom.axis("off")
            continue

        plot_spectrum_comparison(
            ax=ax_zoom,
            eg_axis=eg_axis[mask],
            bayes_mean=bayes_mean[mask],
            bayes_lower=bayes_lower[mask],
            bayes_upper=bayes_upper[mask],
            freq_center=freq["center"][mask],
            freq_lower=freq["lower"][mask],
            freq_upper=freq["upper"][mask],
            truth=truth[mask],
            colors=colors,
            linewidths=linewidths,
            alphas=alphas,
        )
        ax_zoom.set_xlim(lower, upper)
        set_local_y_axis(ax_zoom, y_limits[0], y_limits[1], nbins=4)
        add_zoom_zero_line_if_visible(
            ax_zoom,
            color=colors["truth"],
            linewidth=linewidths["zero"],
            lower_fraction=0.035,
        )
        if panel_index == 0:
            ax_zoom.set_ylabel(rf"${symbol}$ (counts)")
        apply_axes_style(ax_zoom)
        use_MeV_on_xaxis(ax_zoom)
        add_panel_label(ax_zoom, label=label)

    add_zoom_guides(
        fig=fig,
        ax_overview=ax_full,
        zoom_axes=zoom_axes,
        windows=windows,
        color=COLORS["gray"],
    )

    ax_bayes_scaled = fig.add_subplot(grid[2, :], sharex=ax_full)
    plot_scaled_deviation(
        ax=ax_bayes_scaled,
        eg_axis=eg_bayes_scaled,
        mean=bayes_scaled_mean,
        lower=bayes_scaled_lower,
        upper=bayes_scaled_upper,
        color_band=colors["bayes_dev_band"],
        color_line=colors["bayes_dev_line"],
        color_zero=colors["truth"],
        alpha_band=alphas["dev"],
        linewidth_line=linewidths["line"],
        linewidth_edge=linewidths["edge"],
        linewidth_zero=linewidths["zero"],
    )
    ax_bayes_scaled.set_ylim(*scaled_ylim)
    ax_bayes_scaled.set_ylabel(r"$S_{\mathrm{Bayes}}(E_\gamma)$")
    ax_bayes_scaled.tick_params(labelbottom=False)
    apply_axes_style(ax_bayes_scaled)
    use_MeV_on_xaxis(ax_bayes_scaled)
    add_panel_label(ax_bayes_scaled, label="e")

    ax_freq_scaled = fig.add_subplot(grid[3, :], sharex=ax_full)
    plot_scaled_deviation(
        ax=ax_freq_scaled,
        eg_axis=freq["Eg"],
        mean=freq_scaled_mean,
        lower=freq_scaled_lower,
        upper=freq_scaled_upper,
        color_band=colors["freq_dev_band"],
        color_line=colors["freq_dev_line"],
        color_zero=colors["truth"],
        alpha_band=alphas["dev"],
        linewidth_line=linewidths["line"],
        linewidth_edge=linewidths["edge"],
        linewidth_zero=linewidths["zero"],
    )
    ax_freq_scaled.set_ylim(*scaled_ylim)
    ax_freq_scaled.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    ax_freq_scaled.set_ylabel(r"$S_{\mathrm{RMLE}}(E_\gamma)$")
    apply_axes_style(ax_freq_scaled)
    use_MeV_on_xaxis(ax_freq_scaled)
    add_panel_label(ax_freq_scaled, label="f")

    bayes_spec_handle = (
        Patch(facecolor=rgba(colors["bayes_band"], alphas["bayes"]), edgecolor=colors["bayes_band"], linewidth=linewidths["edge"]),
        Line2D([0], [0], color=colors["bayes_line"], lw=linewidths["line"]),
    )
    freq_spec_handle = (
        Patch(facecolor=rgba(colors["freq_band"], alphas["freq"]), edgecolor=colors["freq_band"], linewidth=linewidths["edge"]),
        Line2D([0], [0], color=colors["freq_line"], lw=linewidths["line"]),
    )
    truth_handle = Line2D([0], [0], color=colors["truth"], lw=linewidths["truth"], ls="--")
    bayes_dev_handle = (
        Patch(facecolor=rgba(colors["bayes_dev_band"], alphas["dev"]), edgecolor=colors["bayes_dev_band"], linewidth=linewidths["edge"]),
        Line2D([0], [0], color=colors["bayes_dev_line"], lw=linewidths["line"]),
    )
    freq_dev_handle = (
        Patch(facecolor=rgba(colors["freq_dev_band"], alphas["dev"]), edgecolor=colors["freq_dev_band"], linewidth=linewidths["edge"]),
        Line2D([0], [0], color=colors["freq_dev_line"], lw=linewidths["line"]),
    )

    fig.legend(
        handles=[bayes_spec_handle, freq_spec_handle, truth_handle, bayes_dev_handle, freq_dev_handle],
        labels=[
            rf"${symbol}_{{\mathrm{{Bayes}}}}$: mean + {mass_percent}\% rank-env.",
            rf"${symbol}_{{\mathrm{{RMLE}}}}$: mean + {mass_percent}\% simult. band",
            rf"${symbol}_{{\mathrm{{true}}}}$",
            rf"$S_{{\mathrm{{Bayes}}}}(E_\gamma)$: mean + {mass_percent}\% rank-env.",
            rf"$S_{{\mathrm{{RMLE}}}}(E_\gamma)$: mean + {mass_percent}\% simult. band",
        ],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=2.2,
        fontsize=9.0,
    )

    save_or_show(fig, out, tag="bayes-frequentist-comparison", show=show)
    return fig


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone command-line parser."""

    parser = argparse.ArgumentParser(
        description="Create a Bayesian-versus-RMLE comparison figure."
    )
    parser.add_argument("--bayes-nc", required=True, help="Bayesian posterior draws.nc file.")
    parser.add_argument("--freq-dir", required=True, help="RMLE output root containing Ex*/summary.npz.")
    parser.add_argument("--ex", type=float, required=True, help="Excitation energy in keV.")
    parser.add_argument("--var", default="eta", choices=["eta", "x", "nu"])
    parser.add_argument("--mass", type=float, default=0.95)
    parser.add_argument("--zoom1", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))
    parser.add_argument("--zoom2", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))
    parser.add_argument("--zoom3", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))
    parser.add_argument("--layout", default="text", choices=["text", "column"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the standalone command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    plot_bayes_frequentist_comparison(
        bayes_nc=repo_path(args.bayes_nc),
        freq_dir=repo_path(args.freq_dir),
        ex=args.ex,
        var=args.var,
        mass=args.mass,
        zoom1=args.zoom1,
        zoom2=args.zoom2,
        zoom3=args.zoom3,
        layout=args.layout,
        out=repo_path(args.out) if args.out else None,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
