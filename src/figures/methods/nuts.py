"""
NUTS sampler schematic.

The figure illustrates one NUTS tree expansion with forward and backward
leapfrog points, candidate nodes, one selected sample, endpoint momenta, and
the No-U-Turn span.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from ...paths import repo_path
from ..plot_utils import save_or_show
from ..style import COLORS, apply_axes_style, figure_size

SCALE_X = 2.0
SCALE_Y = 1.0
BANANA_CURVATURE = 0.25

class AnnotationHandler(HandlerLine2D):
    """Legend handler for annotation-arrow proxies."""
    def __init__(self, mutation_scale: float) -> None:
        self.mutation_scale = float(mutation_scale)
        super().__init__()

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        x_data, _ = self.get_xdata(
            legend,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
        )
        y_data = ((height - ydescent) / 2.0) * np.ones_like(x_data, dtype=float)

        arrow = FancyArrowPatch(
            posA=(x_data[0], y_data[0]),
            posB=(x_data[-1], y_data[-1]),
            mutation_scale=self.mutation_scale,
            **orig_handle.arrowprops,
        )
        arrow.set_transform(trans)
        return (arrow,)


def target_potential(position: np.ndarray) -> float:
    """Return the potential energy for the toy banana-shaped target."""
    x_value, y_value = position

    scaled_x = x_value / SCALE_X
    scaled_y = (y_value + BANANA_CURVATURE * (x_value**2 - SCALE_X**2)) / SCALE_Y
    return 0.5 * (scaled_x**2 + scaled_y**2)


def grad_log_density(position: np.ndarray) -> np.ndarray:
    """Return the gradient of the log density for the toy target."""
    x_value, y_value = position
    scaled_y = (y_value + BANANA_CURVATURE * (x_value**2 - SCALE_X**2)) / SCALE_Y

    d_u_dx = (x_value / SCALE_X**2 + (scaled_y / SCALE_Y) * (2.0 * BANANA_CURVATURE * x_value / SCALE_Y))
    d_u_dy = scaled_y / SCALE_Y
    return np.array([-d_u_dx, -d_u_dy], dtype=float)


def leapfrog(
    position: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Take one leapfrog step."""
    momentum_half = momentum + 0.5 * step_size * grad_log_density(position)
    position_new = position + step_size * momentum_half
    momentum_new = momentum_half + 0.5 * step_size * grad_log_density(position_new)

    return position_new, momentum_new


def log_joint_density(position: np.ndarray, momentum: np.ndarray) -> float:
    """Return log joint density up to an additive constant."""
    return -target_potential(position) - 0.5 * np.dot(momentum, momentum).item()

def draw_momentum_arrow(
    ax: plt.Axes,
    center: np.ndarray,
    vector: np.ndarray,
    length: float,
    linewidth: float,
    mutation_scale: float,
    color: str,
    zorder: float,
) -> None:
    """Draw one momentum arrow."""

    unit_vector = np.asarray(vector, dtype=float)
    unit_vector = unit_vector / np.linalg.norm(unit_vector)

    tail = np.asarray(center, dtype=float)
    head = tail + float(length) * unit_vector

    ax.annotate(
        "",
        xy=head,
        xytext=tail,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": float(linewidth),
            "color": color,
            "mutation_scale": float(mutation_scale),
        },
        zorder=float(zorder),
    )


def draw_angle_arc(
    ax: plt.Axes,
    center: np.ndarray,
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    radius: float,
    linewidth: float,
    color: str,
    zorder: float,
) -> None:
    """Draw an angle arc between two vectors."""

    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    vector_a = vector_a / np.linalg.norm(vector_a)
    vector_b = vector_b / np.linalg.norm(vector_b)

    angle_a = np.arctan2(vector_a[1], vector_a[0])
    angle_b = np.arctan2(vector_b[1], vector_b[0])

    delta_angle = (angle_b - angle_a + np.pi) % (2.0 * np.pi) - np.pi
    angles = np.linspace(angle_a, angle_a + delta_angle, 80)

    x_values = center[0] + float(radius) * np.cos(angles)
    y_values = center[1] + float(radius) * np.sin(angles)

    ax.plot(
        x_values,
        y_values,
        color=color,
        linewidth=float(linewidth),
        zorder=float(zorder),
    )


def build_trajectory() -> dict[str, np.ndarray]:
    """Build the symmetric forward and backward toy NUTS trajectory."""

    start_position = np.array([0.0, -1.0], dtype=float)
    start_momentum = np.array([0.8, 0.45], dtype=float)

    step_size = 0.18
    n_steps = 7

    forward_positions = [start_position]
    forward_momenta = [start_momentum]
    backward_positions = [start_position]
    backward_momenta = [start_momentum]

    for _ in range(n_steps):
        position_forward, momentum_forward = leapfrog(
            forward_positions[-1],
            forward_momenta[-1],
            step_size,
        )
        forward_positions.append(position_forward)
        forward_momenta.append(momentum_forward)

        position_backward, momentum_backward = leapfrog(
            backward_positions[-1],
            backward_momenta[-1],
            -step_size,
        )
        backward_positions.append(position_backward)
        backward_momenta.append(momentum_backward)

    forward_positions = np.asarray(forward_positions, dtype=float)
    forward_momenta = np.asarray(forward_momenta, dtype=float)
    backward_positions = np.asarray(backward_positions, dtype=float)
    backward_momenta = np.asarray(backward_momenta, dtype=float)

    end_position_forward = forward_positions[-1]
    end_momentum_forward = forward_momenta[-1]
    end_position_backward = backward_positions[-1]
    end_momentum_backward = backward_momenta[-1]

    span = end_position_forward - end_position_backward
    span_unit = span / np.linalg.norm(span)

    forward_positions_wiggle = forward_positions.copy()
    perpendicular = np.array([-span_unit[1], span_unit[0]], dtype=float)

    last_forward_index = len(forward_positions_wiggle) - 1

    for index in range(1, last_forward_index):
        fraction = index / last_forward_index
        amplitude = 0.18 * np.sin(2.0 * np.pi * fraction)
        forward_positions_wiggle[index] = (
            forward_positions_wiggle[index]
            + amplitude * perpendicular
        )

    rotation_angle = -8.0 * np.pi / 180.0
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ],
        dtype=float,
    )
    end_momentum_backward_rotated = rotation_matrix @ end_momentum_backward

    return {
        "start_position": start_position,
        "start_momentum": start_momentum,
        "forward_positions": forward_positions,
        "forward_momenta": forward_momenta,
        "backward_positions": backward_positions,
        "backward_momenta": backward_momenta,
        "forward_positions_wiggle": forward_positions_wiggle,
        "end_position_forward": end_position_forward,
        "end_momentum_forward": end_momentum_forward,
        "end_position_backward": end_position_backward,
        "end_momentum_backward_rotated": end_momentum_backward_rotated,
        "span": span,
        "span_unit": span_unit,
    }


def candidate_indices(
    forward_positions_wiggle: np.ndarray,
    forward_momenta: np.ndarray,
    backward_positions: np.ndarray,
    backward_momenta: np.ndarray,
) -> tuple[list[int], list[int], int | None]:
    """Return forward candidates, backward candidates, and selected index."""

    forward_indices = list(range(1, len(forward_positions_wiggle) - 1))
    backward_indices = list(range(1, len(backward_positions) - 1))

    forward_scores = [
        (
            index,
            log_joint_density( forward_positions_wiggle[index], forward_momenta[index] ),
        )
        for index in forward_indices
    ]

    backward_scores = [
        (
            index,
            log_joint_density( backward_positions[index], backward_momenta[index] ),
        )
        for index in backward_indices
    ]

    forward_scores.sort(key=lambda item: item[1], reverse=True)
    backward_scores.sort(key=lambda item: item[1], reverse=True)

    forward_candidates: list[int] = []

    for index, _score in forward_scores:
        if all(abs(index - kept_index) >= 2 for kept_index in forward_candidates):
            forward_candidates.append(index)

        if len(forward_candidates) == 3:
            break

    backward_candidates = [
        backward_scores[index][0]
        for index in range(min(2, len(backward_scores)))
    ]

    selected_index = None

    if backward_candidates:
        farthest_index = max(backward_candidates)
        selected_index = max(1, farthest_index - 1)

        if selected_index in backward_candidates and len(backward_candidates) > 1:
            nearest_index = min(backward_candidates)
            selected_index = max(1, nearest_index - 1)
            backward_candidates[backward_candidates.index(nearest_index)] = selected_index
        else:
            backward_candidates[backward_candidates.index(farthest_index)] = selected_index

        backward_candidates = sorted(set(backward_candidates))

    if selected_index is None and backward_candidates:
        selected_index = backward_candidates[0]

    return forward_candidates, backward_candidates, selected_index


def plot_nuts(
    out: str | Path | None = None,
    layout: str = "text",
    show: bool = True,
) -> plt.Figure:
    """Generate the NUTS sampler schematic."""

    trajectory = build_trajectory()

    start_position = trajectory["start_position"]
    forward_positions = trajectory["forward_positions"]
    forward_momenta = trajectory["forward_momenta"]
    backward_positions = trajectory["backward_positions"]
    backward_momenta = trajectory["backward_momenta"]
    forward_positions_wiggle = trajectory["forward_positions_wiggle"]
    end_position_forward = trajectory["end_position_forward"]
    end_momentum_forward = trajectory["end_momentum_forward"]
    end_position_backward = trajectory["end_position_backward"]
    end_momentum_backward_rotated = trajectory["end_momentum_backward_rotated"]
    span = trajectory["span"]
    span_unit = trajectory["span_unit"]

    height_to_width = 0.60 if str(layout).lower().strip() == "text" else 1.00
    size = figure_size(layout, height_to_width=height_to_width)

    fig, ax = plt.subplots(
        figsize=size.as_tuple(),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.04, top=0.99)

    all_x = np.concatenate(
        [ forward_positions[:, 0], backward_positions[:, 0], [end_position_forward[0], end_position_backward[0]] ]
    )
    all_y = np.concatenate(
        [ forward_positions[:, 1], backward_positions[:, 1], [end_position_forward[1], end_position_backward[1]] ]
    )

    xmin = np.min(all_x).item() - 1.5
    xmax = np.max(all_x).item() + 1.5
    ymin = np.min(all_y).item() - 1.5
    ymax = np.max(all_y).item() + 1.5

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 600),
        np.linspace(ymin, ymax, 600),
    )
    grid = np.stack([xx, yy], axis=-1)
    zz = np.exp(-np.vectorize(target_potential, signature="(n)->()")(grid))

    ax.contourf(xx, yy, zz, levels=40, cmap="Greys", alpha=0.35, zorder=0)
    ax.contour(xx, yy, zz, levels=12, colors="black", linewidths=0.45, zorder=1)

    forward_color = COLORS["vermillion"]
    backward_color = COLORS["blue"]
    candidate_color = "#E6B800"
    chosen_color = COLORS["purple"]
    span_color = COLORS["green"]
    momentum_color = COLORS["black"]

    path_linewidth = 1.55
    point_markersize = 5.2
    arrow_linewidth = 1.65
    arrow_mutation_scale = 17
    label_fontsize = 9.5

    ax.plot(
        backward_positions[:, 0],
        backward_positions[:, 1],
        color=backward_color,
        linestyle="--",
        linewidth=path_linewidth,
        zorder=3,
    )
    ax.plot(
        backward_positions[:, 0],
        backward_positions[:, 1],
        "o",
        color=backward_color,
        markersize=point_markersize,
        zorder=4,
    )

    ax.plot(
        forward_positions_wiggle[:, 0],
        forward_positions_wiggle[:, 1],
        color=forward_color,
        linewidth=path_linewidth,
        zorder=3,
    )
    ax.plot(
        forward_positions_wiggle[:, 0],
        forward_positions_wiggle[:, 1],
        "o",
        color=forward_color,
        markersize=point_markersize,
        zorder=4,
    )

    ax.plot(
        start_position[0],
        start_position[1],
        "o",
        color=momentum_color,
        markersize=5.0,
        zorder=5,
    )

    first_step_direction = forward_positions_wiggle[1] - forward_positions_wiggle[0]
    start_arrow_vector = first_step_direction / np.linalg.norm(first_step_direction)
    start_arrow_vector = 0.60 * start_arrow_vector

    ax.annotate(
        "",
        xy=start_position + start_arrow_vector + np.array([-0.05, 0.0]),
        xytext=start_position,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": arrow_linewidth,
            "color": momentum_color,
            "mutation_scale": arrow_mutation_scale,
        },
        zorder=6,
    )

    ax.text(
        start_position[0],
        start_position[1] - 0.30,
        r"$(\mathbf{x}_0,\mathbf{r}_0)$",
        fontsize=label_fontsize,
        color=momentum_color,
        ha="center",
        va="top",
        zorder=7,
    )

    draw_momentum_arrow(
        ax,
        end_position_forward,
        end_momentum_forward,
        length=0.90,
        linewidth=arrow_linewidth,
        mutation_scale=arrow_mutation_scale,
        color=momentum_color,
        zorder=8,
    )
    draw_momentum_arrow(
        ax,
        end_position_backward,
        end_momentum_backward_rotated,
        length=0.75,
        linewidth=arrow_linewidth,
        mutation_scale=arrow_mutation_scale,
        color=momentum_color,
        zorder=8,
    )

    ax.text(
        end_position_forward[0] + 0.10,
        end_position_forward[1] - 0.15,
        r"$(\mathbf{x}_+,\mathbf{r}_+)$",
        fontsize=label_fontsize,
        color=momentum_color,
        ha="left",
        va="bottom",
        zorder=9,
    )
    ax.text(
        end_position_backward[0] - 0.18,
        end_position_backward[1] + 0.22,
        r"$(\mathbf{x}_-,\mathbf{r}_-)$",
        fontsize=label_fontsize,
        color=momentum_color,
        ha="left",
        va="bottom",
        zorder=9,
    )

    ax.annotate(
        "",
        xy=end_position_forward,
        xytext=end_position_backward,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.95,
            "color": span_color,
            "mutation_scale": 22,
        },
        zorder=7,
    )

    extension = 0.90
    ax.plot(
        [ end_position_forward[0], end_position_forward[0] + span_unit[0] * extension ],
        [ end_position_forward[1], end_position_forward[1] + span_unit[1] * extension ],
        color=span_color,
        linestyle="--",
        linewidth=1.10,
        zorder=6,
    )
    ax.plot(
        [ end_position_backward[0], end_position_backward[0] - span_unit[0] * extension ],
        [ end_position_backward[1], end_position_backward[1] - span_unit[1] * extension ],
        color=span_color,
        linestyle="--",
        linewidth=1.10,
        zorder=6,
    )

    draw_angle_arc(
        ax,
        end_position_forward,
        end_momentum_forward,
        span,
        radius=0.50,
        linewidth=1.30,
        color=momentum_color,
        zorder=9,
    )
    draw_angle_arc(
        ax,
        end_position_backward,
        end_momentum_backward_rotated,
        span,
        radius=0.50,
        linewidth=1.30,
        color=momentum_color,
        zorder=9,
    )

    forward_candidates, backward_candidates, selected_index = candidate_indices(
        forward_positions_wiggle,
        forward_momenta,
        backward_positions,
        backward_momenta,
    )

    for index in forward_candidates:
        point = forward_positions_wiggle[index]
        ax.plot(
            point[0],
            point[1],
            "o",
            markersize=8.0,
            markerfacecolor="none",
            markeredgecolor=candidate_color,
            markeredgewidth=1.40,
            zorder=10,
        )

    for index in backward_candidates:
        point = backward_positions[index]
        ax.plot(
            point[0],
            point[1],
            "o",
            markersize=8.0,
            markerfacecolor="none",
            markeredgecolor=candidate_color,
            markeredgewidth=1.40,
            zorder=10,
        )

    if selected_index is not None:
        selected_point = backward_positions[selected_index]
        ax.plot(
            selected_point[0],
            selected_point[1],
            "D",
            markersize=7.5,
            color=chosen_color,
            markeredgecolor=momentum_color,
            markeredgewidth=0.85,
            zorder=11,
        )

    span_legend_arrow = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(1, 0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": span_color,
            "lw": 1.25,
        },
    )
    momentum_legend_arrow = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(1, 0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": momentum_color,
            "lw": 1.25,
        },
    )
    span_legend_arrow.set_visible(False)
    momentum_legend_arrow.set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            color=forward_color,
            marker="o",
            linewidth=path_linewidth,
            markersize=point_markersize,
        ),
        Line2D(
            [0],
            [0],
            color=backward_color,
            marker="o",
            linewidth=path_linewidth,
            markersize=point_markersize,
            linestyle="--",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=8.0,
            markerfacecolor="none",
            markeredgecolor=candidate_color,
            markeredgewidth=1.40,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=chosen_color,
            markeredgecolor=momentum_color,
            markeredgewidth=0.85,
            markersize=7.0,
            linestyle="None",
        ),
        span_legend_arrow,
        momentum_legend_arrow,
    ]

    labels = [
        r"Leapfrog points ($v=+1$)",
        r"Leapfrog points ($v=-1$)",
        r"Candidate nodes ($\geq u$)",
        r"Chosen sample",
        r"$\mathbf{x}_+-\mathbf{x}_-$ (span)",
        r"Momentum vectors $\mathbf{r}_0,\mathbf{r}_\pm$",
    ]

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.65",
        fontsize=7.3,
        handlelength=1.75,
        handletextpad=0.55,
        borderpad=0.35,
        labelspacing=0.75,
        handler_map={
            type(span_legend_arrow): AnnotationHandler(11),
            type(momentum_legend_arrow): AnnotationHandler(11),
        },
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])

    apply_axes_style(ax)

    save_or_show(fig, out, tag="nuts-sampler", show=show)
    return fig


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Create the NUTS sampler schematic.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--layout", choices=["column", "text"], default="text")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)

    args = parser.parse_args()

    plot_nuts(
        out=repo_path(args.out) if args.out else None,
        layout=args.layout,
        show=bool(args.show),
    )


if __name__ == "__main__":
    _cli()
