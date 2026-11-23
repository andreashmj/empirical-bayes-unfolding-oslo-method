"""
Figure showing the workflow of the Bayesian method and data generation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

from ...paths import repo_path
from ..plot_utils import save_or_show
from ..style import TEXTWIDTH_IN


left_face = "#D2E8FC"
right_face = "#D6EFDD"
box_edge = "black"
box_edge_width = 0.70


height_to_width = 0.72


def box_bounds(ax, text):
   
    patch = text.get_bbox_patch()
    renderer = ax.figure.canvas.get_renderer()
    pixel_bounds = patch.get_window_extent(renderer)
    drawing_coordinates = ax.transData.inverted()
    return pixel_bounds.transformed(drawing_coordinates)


def draw_column(ax, x, labels):
  
    boxes = []

    for label in labels:
        box = ax.text(x, 4.0, label, ha="center", va="center", fontsize=9.5, linespacing=1.3,
            bbox={ "boxstyle": "round,pad=0.40,rounding_size=0.06", "facecolor": "white", "edgecolor": "black", "linewidth": 0.70},
        )
        boxes.append(box)

    
    ax.figure.canvas.draw()
    heights = []

    for box in boxes:
        bounds = box_bounds(ax, box)
        heights.append(bounds.height)

    column_height = 6.85
    total_box_height = sum(heights)
    number_of_gaps = len(boxes) - 1
    gap = (column_height - total_box_height) / number_of_gaps
    top = 7.15

    for box, height in zip(boxes, heights):
        center = top - height / 2.0
        box.set_position((x, center))
        top -= height + gap

 
    ax.figure.canvas.draw()

    for index in range(len(boxes) - 1):
        source = box_bounds(ax, boxes[index])
        target = box_bounds(ax, boxes[index + 1])
        start = (x, source.y0)
        end = (x, target.y1)

        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=9,
            linewidth=0.75,
            color="0.25",
            shrinkA=3,
            shrinkB=3,
            zorder=3,
        )
        ax.add_patch(arrow)

    return boxes


def plot_flowchart(
    out: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    fig = plt.figure(
        figsize=(TEXTWIDTH_IN, HEIGHT_TO_WIDTH * TEXTWIDTH_IN),
        facecolor="white",
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    backgrounds = [
        (0.00, 0.10, 5.00, 7.18, "#D2E8FC", 2.5, "Synthetic data generation"),
        (5.00, 0.10, 5.00, 7.18, "#D6EFDD", 7.5, "Bayesian inference"),
    ]

    for x0, y0, width, height, facecolor, title_x, title in backgrounds:
        background = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor=facecolor,
            edgecolor="none",
            linewidth=0.0,
            zorder=0,
        )
        ax.add_patch(background)

        ax.text(title_x, 7.65, rf"\textbf{{{title}}}", ha="center", va="center", fontsize=12)


    
    generation_labels = [
        r"$\textbf{Emitted truth matrix}$" "\n"
        r"$\mathbf{M}_{\mathrm{true}}$",

        r"$\textbf{Excitation-energy mixing}$" ": select bin $k$" "\n"
        r"$\mathbf{x}_{\mathrm{true}}=(\mathbf{M}_{\mathrm{true}}"
        r"\mathbf{G}_{\mathrm{in}})_{:,k}$",

        r"$\textbf{$\gamma$-energy response}$" "\n"
        r"$\boldsymbol{\eta}_{\mathrm{true}}=\mathbf{G}_{\gamma}"
        r"\mathbf{x}_{\mathrm{true}}$" "\n"
        r"$\boldsymbol{\nu}_{\mathrm{true}}=\mathbf{G}_{\gamma}\mathbf{D}"
        r"\mathbf{x}_{\mathrm{true}}$",

        r"$\textbf{Background}$" " (signal-shaped + uniform)\n"
        r"$\mathbf{b}_{\mathrm{true}}=\mathrm{BG}"
        r"(\boldsymbol{\nu}_{\mathrm{true}};\rho,q)$",

        r"$\textbf{Independent Poisson sampling}$" "\n"
        r"$\mathbf{n}_{\mathrm{off}}\sim\mathrm{Poisson}"
        r"(\mathbf{b}_{\mathrm{true}})$" "\n"
        r"$\mathbf{n}\sim\mathrm{Poisson}"
        r"(\boldsymbol{\nu}_{\mathrm{true}}+\mathbf{b}_{\mathrm{true}})$",

        r"$\textbf{Observed data}$" "\n"
        r"$(\mathbf{n},\mathbf{n}_{\mathrm{off}})$",
    ]

    inference_labels = [
        r"$\textbf{Empirical-Bayes priors}$" "\n"
        r"Reference from $(\mathbf{n},\mathbf{n}_{\mathrm{off}})$" "\n"
        r"$p(\mathbf{x}),\;p(\mathbf{b})$",

        r"$\textbf{Likelihood}$" "\n"
        r"$\mathbf{n}_{\mathrm{off}}\mid\mathbf{b}"
        r"\sim\mathrm{Poisson}(\mathbf{b})$" "\n"
        r"$\mathbf{n}\mid\mathbf{x},\mathbf{b}\sim\mathrm{Poisson}("
        r"\mathbf{G}_{\gamma}\mathbf{D}\mathbf{x}+\mathbf{b})$",

        r"$\textbf{Posterior}$" "\n"
        r"$p(\mathbf{x},\mathbf{b}\mid\mathbf{n},"
        r"\mathbf{n}_{\mathrm{off}})$" 
        r"$\propto\mathcal{L}(\mathbf{x},\mathbf{b})\,"
        r"p(\mathbf{x})\,p(\mathbf{b})$",

        r"$\textbf{Posterior samples (NUTS)}$" "\n"
        r"$\mathbf{x}^{(1)},\;\mathbf{x}^{(2)},\;\ldots$",

        r"$\textbf{Transform}$" "\n"
        r"$\boldsymbol{\eta}=\mathbf{G}_{\gamma}\mathbf{x}$",

        r"$\textbf{Posterior on}$" r" $\boldsymbol{\eta}$" "\n"
        r"$p(\boldsymbol{\eta}\mid\mathbf{n},\mathbf{n}_{\mathrm{off}})$",
    ]

    left_boxes = draw_column(ax, 2.5, generation_labels)
    right_boxes = draw_column(ax, 7.5, inference_labels)


    observed = box_bounds(ax, left_boxes[-1])
    likelihood = box_bounds(ax, right_boxes[1])
    start_y = (observed.y0 + observed.y1) / 2.0
    end_y = (likelihood.y0 + likelihood.y1) / 2.0

    points = [(observed.x1, start_y),  (5.0, start_y), (5.0, end_y), (likelihood.x0, end_y) ]
    path_codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4 ]
    path = MplPath(points, path_codes)

    arrow = FancyArrowPatch(path=path, arrowstyle="->", mutation_scale=9, linewidth=0.75, color="0.35", linestyle=(0, (3, 2)), zorder=3)
    ax.add_patch(arrow)

    if out is not None:
        out = repo_path(out)

    save_or_show(fig, out, tag="flowchart", show=show)
    return fig

