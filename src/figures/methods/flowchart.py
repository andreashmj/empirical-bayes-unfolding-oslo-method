"""
Bayesian unfolding flowchart figure.

Left: Forward model (synthetic data generation)
Right: Statistical inference (Bayesian)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath

from ..plot_utils import save_or_show
from ..style import TEXTWIDTH_IN


def _box_style(edgecolor: str, linewidth: float = 1.4, padding: float = 0.65) -> dict:
    """Return the rounded-box style used for text nodes."""

    return {
        "boxstyle": f"round,pad={padding}",
        "facecolor": "white",
        "edgecolor": edgecolor,
        "linewidth": linewidth,
    }


def _add_panel_background(
    ax: plt.Axes,
    left: float,
    bottom: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Add one rounded panel background."""

    ax.add_patch(
        patches.FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0.4",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
            zorder=0,
        )
    )


def _add_text_box(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    fontsize: float,
    style: dict,
):
    """Add one centered flowchart text box."""

    return ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        bbox=style,
    )


def _patch_box(ax: plt.Axes, text_artist) -> tuple[float, float, float, float, float]:
    """Return a text box extent in data coordinates."""
    renderer = ax.figure.canvas.get_renderer()
    inverse = ax.transData.inverted()

    patch = text_artist.get_bbox_patch()
    bounds = patch.get_window_extent(renderer=renderer)

    (x0, y0), (x1, y1) = inverse.transform( [[bounds.x0, bounds.y0], [bounds.x1, bounds.y1]] )
    return x0, y0, x1, y1, y1 - y0


def _layout_column(
    ax: plt.Axes,
    x_center: float,
    boxes: list,
    panel_bottom: float,
    panel_height: float,
) -> None:
    """Pack one vertical column of text boxes inside a panel."""

    heights = [_patch_box(ax, text_box)[4] for text_box in boxes]

    y_top = panel_bottom + panel_height - 0.20
    y_bottom = panel_bottom + 0.20
    available_height = y_top - y_bottom

    total_height = sum(heights)
    if len(boxes) > 1:
        gap = (available_height - total_height) / (len(boxes) - 1)
    else:
        gap = 0.0

    gap = max(0.22, min(0.90, gap))

    current_top = y_top

    for text_box, height in zip(boxes, heights):
        y_center = current_top - 0.5 * height
        text_box.set_position((x_center, y_center))
        current_top = y_center - 0.5 * height - gap


def _align_bottoms(ax: plt.Axes, reference_box, moved_boxes: list, moved_bottom_box) -> None:
    """Shift one column so the bottom boxes align."""

    reference_bottom = _patch_box(ax, reference_box)[1]
    moved_bottom = _patch_box(ax, moved_bottom_box)[1]
    delta = moved_bottom - reference_bottom

    if abs(delta) <= 1.0e-6:
        return

    for text_box in moved_boxes:
        x, y = text_box.get_position()
        text_box.set_position((x, y - delta))

    ax.figure.canvas.draw()


def _connect_down(ax: plt.Axes, source_box, target_box, x_center: float) -> None:
    """Draw a vertical arrow between two stacked boxes."""

    pad = 0.02

    _, source_bottom, _, _, _ = _patch_box(ax, source_box)
    _, _, _, target_top, _ = _patch_box(ax, target_box)

    start = (x_center, source_bottom - pad)
    end = (x_center, target_top + pad)

    if start[1] <= end[1]:
        return

    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=12.5,
            lw=1.35,
            color="#000000",
            linestyle="solid",
            zorder=5,
        )
    )


def _connect_column_boxes(
    ax: plt.Axes,
    boxes: list,
    x_center: float,
) -> None:
    """Draw vertical arrows through one column of boxes."""

    for source_box, target_box in zip(boxes[:-1], boxes[1:]):
        _connect_down(ax, source_box, target_box, x_center)


def _connect_observed_to_likelihood(
    ax: plt.Axes,
    observed_box,
    likelihood_box,
    left_panel_left: float,
    column_width: float,
    gutter: float,
) -> None:
    """Draw the dashed S-link from observed data to the likelihood box."""

    gutter_center = left_panel_left + column_width + 0.5 * gutter
    boundary_x = gutter_center - 0.15

    obs_x0, obs_y0, obs_x1, obs_y1, _ = _patch_box(ax, observed_box)
    like_x0, like_y0, like_x1, like_y1, _ = _patch_box(ax, likelihood_box)

    start = (obs_x1, 0.5 * (obs_y0 + obs_y1))
    end = (like_x0 + 0.12, 0.5 * (like_y0 + like_y1))

    symmetry_center = (boundary_x, 0.5 * (start[1] + end[1]))
    x_span = end[0] - start[0]
    handle_length = 0.45 * x_span

    control_1 = (start[0] + handle_length, start[1])
    control_2 = (
        2.0 * symmetry_center[0] - control_1[0],
        2.0 * symmetry_center[1] - control_1[1],
    )

    dashed_shift_left = 0.12
    head_length = 0.10
    dashed_end = (end[0] - head_length - dashed_shift_left, end[1])

    path = MplPath(
        [start, control_1, control_2, dashed_end],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
        ],
    )

    ax.add_patch(
        patches.PathPatch(
            path,
            fill=False,
            lw=1.35,
            linestyle=(0, (2.6, 2.0)),
            capstyle="butt",
            joinstyle="round",
            edgecolor="#000000",
            zorder=6,
        )
    )

    ax.add_patch(
        patches.FancyArrowPatch(
            dashed_end,
            end,
            arrowstyle="->",
            mutation_scale=12.5,
            lw=1.35,
            color="#000000",
            linestyle="solid",
            zorder=7,
            clip_on=False,
        )
    )


def _add_titles(
    ax: plt.Axes,
    left_column_center: float,
    right_column_center: float,
) -> None:
    """Add main and column titles."""

    ax.text(
        5.0,
        7.6,
        r"\textbf{Bayesian unfolding framework (signal + background)}",
        ha="center",
        va="center",
        fontsize=16,
    )

    ax.text(
        left_column_center,
        7.15,
        r"\textbf{Forward model (synthetic data generation)}",
        ha="center",
        va="center",
        fontsize=12,
    )

    ax.text(
        right_column_center,
        7.15,
        r"\textbf{Statistical inference (Bayesian)}",
        ha="center",
        va="center",
        fontsize=12,
    )


def _add_flowchart_boxes(
    ax: plt.Axes,
    left_column_center: float,
    right_column_center: float,
) -> tuple[list, list]:
    """Add all flowchart boxes and return left and right columns."""

    padding = 0.65
    fontsize_large = 13
    fontsize_regular = 12

    truth_style = _box_style("#0072B2", linewidth=1.4, padding=padding)
    prior_style = _box_style("#6F42C1", linewidth=1.4, padding=padding)
    likelihood_style = _box_style("#E69F00", linewidth=1.4, padding=padding)
    observed_style = _box_style("#D55E00", linewidth=1.4, padding=padding)
    posterior_style = _box_style("#009E73", linewidth=1.4, padding=padding)
    process_style = _box_style("#6C757D", linewidth=1.2, padding=padding)

    left_y0 = 6.05
    right_y0 = 6.05

    true_spectrum = _add_text_box(
        ax,
        left_column_center,
        left_y0,
        r"True spectrum" "\n" r"$\boldsymbol{\mu}_{\mathrm{true}}$",
        fontsize_large,
        truth_style,
    )

    apply_operators = _add_text_box(
        ax,
        left_column_center,
        left_y0 - 1.0,
        r"Apply operators" "\n"
        r"$\boldsymbol{\eta}_{\mathrm{true}}"
        r" = \mathbf{G}_{\gamma}\,\boldsymbol{\mu}_{\mathrm{true}},\quad$"
        r"$\boldsymbol{\nu}_{\mathrm{true}}"
        r" = \mathbf{G}_{\gamma}\,\mathbf{D}\,\boldsymbol{\mu}_{\mathrm{true}}$",
        fontsize_regular,
        process_style,
    )

    background_rule = _add_text_box(
        ax,
        left_column_center,
        left_y0 - 2.0,
        r"Background (signal-shaped + uniform)" "\n"
        r"$\mathbf{b}_{\mathrm{true}}"
        r"=\mathrm{BG}\!\left(\boldsymbol{\nu}_{\mathrm{true}};"
        r"\rho,p\right)$",
        fontsize_regular,
        process_style,
    )

    background_measurement = _add_text_box(
        ax,
        left_column_center,
        left_y0 - 3.0,
        r"Background measurement" "\n"
        r"$\mathbf{n}_{\mathrm{off}}\sim\mathrm{Poisson}"
        r"(\mathbf{b}_{\mathrm{true}})$",
        fontsize_regular,
        process_style,
    )

    total_counts = _add_text_box(
        ax,
        left_column_center,
        left_y0 - 4.0,
        r"Total counts sampling" "\n"
        r"$\mathbf{n}\sim\mathrm{Poisson}"
        r"(\boldsymbol{\nu}_{\mathrm{true}}+\mathbf{b}_{\mathrm{true}})$",
        fontsize_regular,
        process_style,
    )

    observed_data = _add_text_box(
        ax,
        left_column_center,
        left_y0 - 5.0,
        r"Observed data" "\n" r"$(\mathbf{n},\ \mathbf{n}_{\mathrm{off}})$",
        fontsize_large,
        observed_style,
    )

    priors = _add_text_box(
        ax,
        right_column_center,
        right_y0,
        r"Priors" "\n" r"$\pi(\boldsymbol{\mu})\,,\;\pi(\mathbf{b})$",
        fontsize_large,
        prior_style,
    )

    likelihood = _add_text_box(
        ax,
        right_column_center,
        right_y0 - 1.0,
        r"Likelihood" "\n"
        r"$\mathbf{n}_{\mathrm{off}}\mid\mathbf{b}"
        r"\sim\mathrm{Poisson}(\mathbf{b}),\quad$" "\n"
        r"$\mathbf{n}\mid\boldsymbol{\mu},\mathbf{b}"
        r"\sim\mathrm{Poisson}(\mathbf{G}_{\gamma}\, \mathbf{D}\, \boldsymbol{\mu}"
        r"+\mathbf{b})$",
        fontsize_regular,
        likelihood_style,
    )

    posterior = _add_text_box(
        ax,
        right_column_center,
        right_y0 - 2.0,
        r"Posterior (NUTS)" "\n"
        r"$\pi(\boldsymbol{\mu},\mathbf{b}\mid"
        r"\mathbf{n},\mathbf{n}_{\mathrm{off}})"
        r"\propto\ \mathcal{L}(\mathbf{n},\mathbf{n}_{\mathrm{off}}\mid"
        r"\boldsymbol{\mu},\mathbf{b})\ \pi(\boldsymbol{\mu})\,\pi(\mathbf{b})$",
        fontsize_regular,
        posterior_style,
    )

    posterior_samples = _add_text_box(
        ax,
        right_column_center,
        right_y0 - 3.0,
        r"Posterior samples"
        "\n" r"$\boldsymbol{\mu}^{(1)},\ \boldsymbol{\mu}^{(2)},\ \ldots$",
        fontsize_regular,
        posterior_style,
    )

    transform = _add_text_box(
        ax,
        right_column_center,
        right_y0 - 4.0,
        r"Transform" "\n"
        r"$\boldsymbol{\eta} = \mathbf{G}_{\gamma}\, \boldsymbol{\mu}$",
        fontsize_regular,
        process_style,
    )

    eta_posterior = _add_text_box(
        ax,
        right_column_center,
        right_y0 - 5.0,
        r"Posterior on $\,\boldsymbol{\eta}$" "\n"
        r"$\pi(\boldsymbol{\eta}\mid\mathbf{n},\mathbf{n}_{\mathrm{off}})$",
        fontsize_large,
        posterior_style,
    )

    left_boxes = [
        true_spectrum,
        apply_operators,
        background_rule,
        background_measurement,
        total_counts,
        observed_data,
    ]

    right_boxes = [
        priors,
        likelihood,
        posterior,
        posterior_samples,
        transform,
        eta_posterior,
    ]

    return left_boxes, right_boxes


def plot_flowchart(out: str | Path | None = None, show: bool = True) -> plt.Figure:
    """Create the Bayesian unfolding flowchart."""

    fig = plt.figure(figsize=(1.25 * TEXTWIDTH_IN, TEXTWIDTH_IN))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 8.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    left_margin = 0.60
    right_margin = 0.60
    gutter = 0.50
    panel_bottom = 0.90
    panel_height = 6.00

    total_width = 10.0 - left_margin - right_margin - gutter
    column_width = total_width / 2.0

    left_panel_left = left_margin
    right_panel_left = left_margin + column_width + gutter

    left_column_center = left_panel_left + 0.5 * column_width
    right_column_center = right_panel_left + 0.5 * column_width

    _add_panel_background(
        ax,
        left_panel_left,
        panel_bottom,
        column_width,
        panel_height,
        facecolor="#E2F0FF",
        edgecolor="#D5DEE8",
    )

    _add_panel_background(
        ax,
        right_panel_left,
        panel_bottom,
        column_width,
        panel_height,
        facecolor="#E4FCEB",
        edgecolor="#D5DEE8",
    )

    _add_titles(
        ax,
        left_column_center=left_column_center,
        right_column_center=right_column_center,
    )

    left_boxes, right_boxes = _add_flowchart_boxes(
        ax,
        left_column_center=left_column_center,
        right_column_center=right_column_center,
    )

    fig.canvas.draw()

    _layout_column(
        ax,
        x_center=left_column_center,
        boxes=left_boxes,
        panel_bottom=panel_bottom,
        panel_height=panel_height,
    )

    _layout_column(
        ax,
        x_center=right_column_center,
        boxes=right_boxes,
        panel_bottom=panel_bottom,
        panel_height=panel_height,
    )

    fig.canvas.draw()

    _align_bottoms(
        ax,
        reference_box=left_boxes[-1],
        moved_boxes=right_boxes,
        moved_bottom_box=right_boxes[-1],
    )

    _connect_column_boxes(
        ax,
        boxes=left_boxes,
        x_center=left_column_center,
    )

    _connect_column_boxes(
        ax,
        boxes=right_boxes,
        x_center=right_column_center,
    )

    _connect_observed_to_likelihood(
        ax,
        observed_box=left_boxes[-1],
        likelihood_box=right_boxes[1],
        left_panel_left=left_panel_left,
        column_width=column_width,
        gutter=gutter,
    )

    save_or_show(fig, out, tag="flowchart", show=show)
    return fig


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Create the Bayesian unfolding flowchart.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)

    args = parser.parse_args()

    plot_flowchart(
        out=args.out,
        show=bool(args.show),
    )


if __name__ == "__main__":
    _cli()
