"""
Small plotting helpers used by figure scripts.
"""

from __future__ import annotations

from pathlib import Path
import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .style import PANEL_LABEL_SIZE


def rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    """RGBA tuple from a color and alpha."""

    red, green, blue, _ = to_rgba(color)
    return (red, green, blue, float(alpha))


def latex_ex_label(ex_requested: float | None, ex_actual: float) -> str:
    ex_value = float(ex_requested) if ex_requested is not None else float(ex_actual)
    return rf"$E_x \approx {ex_value:.0f}\,\mathrm{{keV}}$"


def steps(
    ax: plt.Axes,
    x,
    y,
    color="C0",
    lw: float = 0.85,
    alpha: float = 1.0,
    ls: str = "-",
    label: str | None = None,
    zorder: float = 3.0,
    rasterized: bool = False,
) -> Line2D:
    """
    Plot a step-mid curve and return the line object.
    
    """

    line, = ax.plot(
        x,
        y,
        drawstyle="steps-mid",
        color=color,
        lw=lw,
        alpha=alpha,
        ls=ls,
        label=label,
        zorder=zorder,
        rasterized=rasterized,
    )

    return line


def fill_step_band(
    ax: plt.Axes,
    x,
    lower,
    upper,
    facecolor,
    edgecolor=None,
    alpha: float = 0.25,
    linewidth: float = 0.35,
    zorder: float = 2.0,
    rasterized: bool = True,
):
    """
    Draw a step-mid uncertainty band.
    """

    if edgecolor is None:
        edgecolor = facecolor

    return ax.fill_between(
        x,
        lower,
        upper,
        step="mid",
        facecolor=rgba(facecolor, alpha),
        edgecolor=edgecolor,
        linewidth=linewidth,
        rasterized=rasterized,
        zorder=zorder,
    )


def add_top_legend(
    fig: plt.Figure,
    handles,
    labels,
    ncol: int = 2,
    y: float = 0.985,
    frameon: bool = False,
    handlelength: float = 2.2,
    columnspacing: float = 1.4,
    labelspacing: float = 0.35,
    fontsize: float | None = None,
    handler_map=None,
):
    """
    Add a single legend centered above the figure panels.
    """

    return fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=frameon,
        handlelength=handlelength,
        columnspacing=columnspacing,
        labelspacing=labelspacing,
        fontsize=fontsize,
        handler_map=handler_map,
    )


def save_or_show(
    fig: plt.Figure,
    out: str | Path | None,
    tag: str = "figure",
    show: bool = True,
) -> None:

    if out is not None:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
        print(f"[{tag}] wrote {output_path}")
        plt.close(fig)
        return

    if show:
        plt.show()


def panel_id(index: int) -> str:
    """
    Panel label letters
    """

    if index < 0:
        raise ValueError("panel index must be non-negative.")

    letters = string.ascii_lowercase
    label = ""
    value = int(index)

    while True:
        label = letters[value % 26] + label
        value = value // 26 - 1

        if value < 0:
            break

    return label


def panel_label_text(label: str) -> str:

    label = str(label).strip()

    if label.startswith("(") and label.endswith(")"):
        label = label[1:-1]

    return rf"\textbf{{({label})}}"


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = 0.03,
    y: float = 0.95,
    fontsize: float = PANEL_LABEL_SIZE,
    box: bool = False,
    box_alpha: float = 0.85,
) -> None:

    bbox = None

    if box:
        bbox = {
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": box_alpha,
        }

    ax.text(
        x,
        y,
        panel_label_text(label),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox=bbox,
        zorder=20,
    )


def add_panel_labels(
    axes,
    start: int = 0,
    x: float = 0.03,
    y: float = 0.95,
    fontsize: float = PANEL_LABEL_SIZE,
    box: bool = False,
    box_axes=None,
) -> None:

    boxed_axes = set(box_axes) if box_axes is not None else None

    for offset, ax in enumerate(axes):
        label = panel_id(start + offset)

        if boxed_axes is None:
            use_box = box
        else:
            use_box = ax in boxed_axes

        add_panel_label(
            ax,
            label,
            x=x,
            y=y,
            fontsize=fontsize,
            box=use_box,
        )


def _kev_to_mev_label(value: float, decimals: int) -> str:
    label = f"{value / 1000.0:.{decimals}f}"
    return label.rstrip("0").rstrip(".")


def use_MeV_on_xaxis(ax: plt.Axes, decimals: int = 2) -> plt.Axes:
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, position: _kev_to_mev_label(value, decimals))
    )
    return ax


def use_MeV_on_yaxis(ax: plt.Axes, decimals: int = 2) -> plt.Axes:
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, position: _kev_to_mev_label(value, decimals))
    )
    return ax


def hide_shared_x_ticklabels(axes) -> None:
    for ax in axes:
        ax.tick_params(labelbottom=False)


def hide_shared_y_ticklabels(axes) -> None:
    for ax in axes:
        ax.tick_params(labelleft=False)


def y_limit_top(arrays: list[np.ndarray], factor: float = 1.12) -> float:
    finite_values = []

    for values in arrays:
        array = np.asarray(values, dtype=float).reshape(-1)
        array = array[np.isfinite(array)]

        if array.size:
            finite_values.append(array)

    if not finite_values:
        return float(factor)

    maximum = np.max(np.concatenate(finite_values)).item()
    return float(factor) * max(maximum, 1.0)


def y_limits_with_zero_room(
    y_top: float,
    lower_fraction: float = 0.035,
) -> tuple[float, float]:
    y_top = max(float(y_top), 1.0)
    return -float(lower_fraction) * y_top, y_top


def set_linear_y_with_zero_room(
    ax: plt.Axes,
    arrays: list[np.ndarray],
    top_factor: float = 1.12,
    lower_fraction: float = 0.035,
    draw_zero: bool = True,
    zero_color: str = "black",
) -> tuple[float, float]:
    y_top = y_limit_top(arrays, factor=top_factor)
    limits = y_limits_with_zero_room(y_top, lower_fraction=lower_fraction)
    ax.set_ylim(*limits)

    if draw_zero:
        add_zero_line(ax, color=zero_color)

    return limits


def set_symlog_y_with_room(
    ax: plt.Axes,
    arrays: list[np.ndarray],
    linthresh: float = 5.0,
    top_factor: float = 1.25,
    y_lower: float = -1.0,
) -> tuple[float, float]:
    y_top = y_limit_top(arrays, factor=top_factor)
    ax.set_yscale("symlog", linthresh=float(linthresh))
    ax.set_ylim(float(y_lower), y_top)

    return float(y_lower), y_top


def local_y_limits(
    mask: np.ndarray,
    arrays: list[np.ndarray],
    padding_fraction: float = 0.08,
) -> tuple[float, float]:
    finite_values = []

    for values in arrays:
        array = np.asarray(values, dtype=float)[mask]
        array = array[np.isfinite(array)]

        if array.size:
            finite_values.append(array)

    if not finite_values:
        return 0.0, 1.0

    values = np.concatenate(finite_values)
    y_min = np.min(values).item()
    y_max = np.max(values).item()

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return 0.0, 1.0

    if y_max <= y_min:
        width = max(abs(y_min) * 0.05, 1.0)
        return y_min - width, y_max + width

    padding = float(padding_fraction) * (y_max - y_min)
    return y_min - padding, y_max + padding


def set_local_y_axis(
    ax: plt.Axes,
    y_lower: float,
    y_upper: float,
    nbins: int = 4,
    lower_tick_margin: float = 0.18,
    upper_tick_margin: float = 0.10,
) -> None:
    y_lower = float(y_lower)
    y_upper = float(y_upper)

    if not np.isfinite(y_lower) or not np.isfinite(y_upper) or y_upper <= y_lower:
        ax.set_ylim(0.0, 1.0)
        return

    locator = MaxNLocator(nbins=int(nbins))
    ticks = locator.tick_values(y_lower, y_upper)
    visible_ticks = ticks[(ticks >= y_lower) & (ticks <= y_upper)]

    if visible_ticks.size >= 2:
        tick_step = np.median(np.diff(visible_ticks)).item()

        if np.isfinite(tick_step) and tick_step > 0.0:
            y_lower = min(y_lower, visible_ticks[0] - lower_tick_margin * tick_step)
            y_upper = max(y_upper, visible_ticks[-1] + upper_tick_margin * tick_step)

    ax.set_ylim(y_lower, y_upper)
    ax.yaxis.set_major_locator(locator)


def add_zero_line(
    ax: plt.Axes,
    color: str = "black",
    lw: float = 0.45,
    ls: str = ":",
    alpha: float = 0.70,
    zorder: float = 1.2,
):
    y_lower, y_upper = ax.get_ylim()

    if not (y_lower < 0.0 < y_upper):
        return None

    return ax.axhline(
        0.0,
        color=color,
        lw=float(lw),
        ls=ls,
        alpha=float(alpha),
        zorder=float(zorder),
    )


def add_zoom_guides(
    fig: plt.Figure,
    ax_overview: plt.Axes,
    zoom_axes: list[plt.Axes],
    windows: list[tuple[float, float]],
    color: str = "0.45",
    shade_alpha: float = 0.055,
    edge_linewidth: float = 0.50,
    connector_linewidth: float = 0.42,
    connector_alpha: float = 0.60,
) -> None:
    y_bottom = ax_overview.get_ylim()[0]

    for ax_zoom, window in zip(zoom_axes, windows):
        lower, upper = window

        ax_overview.axvspan(
            lower,
            upper,
            facecolor=rgba(color, shade_alpha),
            edgecolor=color,
            linewidth=edge_linewidth,
            zorder=1.0,
        )

        left_connection = ConnectionPatch(
            xyA=(lower, y_bottom),
            coordsA="data",
            axesA=ax_overview,
            xyB=(0.0, 1.0),
            coordsB="axes fraction",
            axesB=ax_zoom,
            color=color,
            linewidth=connector_linewidth,
            alpha=connector_alpha,
            clip_on=False,
            zorder=10,
        )

        right_connection = ConnectionPatch(
            xyA=(upper, y_bottom),
            coordsA="data",
            axesA=ax_overview,
            xyB=(1.0, 1.0),
            coordsB="axes fraction",
            axesB=ax_zoom,
            color=color,
            linewidth=connector_linewidth,
            alpha=connector_alpha,
            clip_on=False,
            zorder=10,
        )

        fig.add_artist(left_connection)
        fig.add_artist(right_connection)
