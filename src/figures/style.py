"""
Central Matplotlib style for paper figures.

This module defines the final publication style used by all figures.
All figure scripts use the same settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


PT_PER_INCH = 72.27

TEXTWIDTH_PT = 510.0
COLUMNWIDTH_PT = 246.0

TEXTWIDTH_IN = TEXTWIDTH_PT / PT_PER_INCH
COLUMNWIDTH_IN = COLUMNWIDTH_PT / PT_PER_INCH


@dataclass(frozen=True)
class FigureSize:
    """Figure size in inches."""

    width_in: float
    height_in: float

    def as_tuple(self) -> tuple[float, float]:
        """Return figure size as a Matplotlib-compatible tuple."""

        return (self.width_in, self.height_in)


COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "rose": "#EE6677",
    "wine": "#882255",
    "gray": "#666666",
    "light_gray": "#B8B8B8",
}

COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["vermillion"],
    COLORS["green"],
    COLORS["purple"],
    COLORS["orange"],
    COLORS["sky"],
    COLORS["rose"],
    COLORS["wine"],
]

FONT_SIZE = 10.0
AXIS_LABEL_SIZE = 10.0
TICK_LABEL_SIZE = 9.0
LEGEND_FONT_SIZE = 9.0
PANEL_LABEL_SIZE = 10.0

AXES_LINE_WIDTH = 0.8
TICK_WIDTH = 0.8
TICK_LENGTH = 3.0

LINE_WIDTH = 0.85
THIN_LINE_WIDTH = 0.55
BAND_EDGE_WIDTH = 0.35

SAVEFIG_DPI = 600
RASTERIZED_FILL = True


def figure_size(layout: str = "text", height_to_width: float = 0.62) -> FigureSize:
    """Return a figure size for a text-width or column-width figure."""

    layout = str(layout).strip().lower()

    if layout == "text":
        width = TEXTWIDTH_IN
    elif layout == "column":
        width = COLUMNWIDTH_IN
    else:
        raise ValueError("layout must be either 'text' or 'column'.")

    height = float(height_to_width) * width

    return FigureSize(width_in=width, height_in=height)


def apply_style() -> None:
    """Apply the shared publication Matplotlib style."""

    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}",
            "font.family": "serif",
            "font.size": FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": AXES_LINE_WIDTH,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": 3.0,
            "axes.prop_cycle": cycler(color=COLOR_CYCLE),
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "xtick.major.size": TICK_LENGTH,
            "ytick.major.size": TICK_LENGTH,
            "legend.frameon": False,
            "legend.fancybox": False,
            "savefig.format": "pdf",
            "savefig.dpi": SAVEFIG_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def apply_axes_style(ax: plt.Axes) -> None:
    """Apply consistent axis styling to one axes object."""

    ax.minorticks_off()
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=TICK_LENGTH,
        width=TICK_WIDTH,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        top=False,
        left=False,
        right=False,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(AXES_LINE_WIDTH)


apply_style()
