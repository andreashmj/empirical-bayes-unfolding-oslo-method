"""
Model-check figure.

The figure compares prior and posterior predictive checks for observed counts,
and folded-signal checks for the expected detected signal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ...analysis.envelopes import global_rank_envelope
from ...analysis.results import UnfoldingResults
from ..plot_utils import (
    add_panel_label,
    add_top_legend,
    add_zero_line,
    fill_step_band,
    rgba,
    save_or_show,
    set_symlog_y_with_room,
    steps,
    use_MeV_on_xaxis,
)
from ..style import COLORS, apply_axes_style, figure_size


def rank_envelope_from_array(
    values: np.ndarray,
    eg_axis: np.ndarray,
    mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return global rank envelope for an array with shape sample by Eg."""

    values = np.asarray(values, dtype=float)

    data_array = xr.DataArray(
        values,
        dims=("sample", "Eg"),
        coords={
            "sample": np.arange(values.shape[0]),
            "Eg": eg_axis,
        },
    )

    return global_rank_envelope(data_array, mass=mass)


def draw_matrix(data_array: xr.DataArray) -> np.ndarray:
    """Return a DataArray with dimensions sample and Eg as a NumPy matrix."""

    return np.asarray(data_array.transpose("sample", "Eg").values, dtype=float)


def poisson_predictive(rate: xr.DataArray, rng: np.random.Generator) -> np.ndarray:
    """Draw Poisson predictive counts from non-negative rate draws."""

    rate_values = draw_matrix(rate)
    rate_values = np.clip(rate_values, 0.0, None)

    return rng.poisson(rate_values).astype(float)


def scaled_deviation(
    values: np.ndarray,
    reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return scaled deviations using half-width of a posterior envelope."""

    half_width = np.maximum(0.5 * (upper - lower), 1.0e-12)
    return (values - reference[None, :]) / half_width[None, :]


def common_eg_axis(
    prior_results: UnfoldingResults,
    posterior_results: UnfoldingResults,
    ex_actual: float,
) -> np.ndarray:
    """Return common Eg axis after checking prior/posterior compatibility."""

    if prior_results.n_eg(ex_actual) != posterior_results.n_eg(ex_actual):
        raise ValueError("Eg length mismatch between prior and posterior result files.")

    eg_axis = posterior_results.eg(ex_actual)

    if not np.allclose(prior_results.eg(ex_actual), eg_axis, rtol=0.0, atol=1.0e-12):
        raise ValueError("Eg grid mismatch between prior and posterior result files.")

    return eg_axis


def set_count_axis_pair(
    axes: list[plt.Axes],
    arrays: list[np.ndarray],
    symlog_linthresh: float,
    top_factor: float = 7.0,
    y_lower: float = -1.0,
    zero_color: str = COLORS["black"],
) -> None:
    """Set a shared symlog count axis for one prior/posterior panel pair."""

    if not axes:
        return

    y_limits = set_symlog_y_with_room(
        axes[0],
        arrays=arrays,
        linthresh=symlog_linthresh,
        top_factor=top_factor,
        y_lower=y_lower,
    )

    add_zero_line(
        axes[0],
        color=zero_color,
        lw=0.45,
        ls=":",
        alpha=0.55,
        zorder=1.1,
    )

    apply_axes_style(axes[0])

    for ax in axes[1:]:
        ax.set_yscale("symlog", linthresh=float(symlog_linthresh))
        ax.set_ylim(*y_limits)

        add_zero_line(
            ax,
            color=zero_color,
            lw=0.45,
            ls=":",
            alpha=0.55,
            zorder=1.1,
        )

        apply_axes_style(ax)

def plot_band_with_reference(
    ax: plt.Axes,
    eg_band: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eg_reference: np.ndarray,
    reference: np.ndarray,
    band_color: str,
    mean_color: str,
    reference_color: str,
    band_alpha: float,
    mean_linewidth: float,
    reference_linewidth: float,
    edge_linewidth: float,
    label: str,
    ylabel: str | None = None,
    show_y_ticklabels: bool = True,
    box_label: bool = False,
) -> None:
    """Plot one predictive/rate band panel."""

    fill_step_band(
        ax,
        eg_band,
        lower,
        upper,
        facecolor=band_color,
        edgecolor=band_color,
        alpha=band_alpha,
        linewidth=edge_linewidth,
        zorder=2.0,
    )

    steps(
        ax,
        eg_band,
        mean,
        color=mean_color,
        lw=mean_linewidth,
        zorder=3.0,
    )

    steps(
        ax,
        eg_reference,
        reference,
        color=reference_color,
        lw=reference_linewidth,
        ls="--",
        zorder=3.1,
    )

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if not show_y_ticklabels:
        ax.tick_params(labelleft=False)

    add_panel_label(ax, label=label, box=box_label)
    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)


def plot_scaled_deviation(
    ax: plt.Axes,
    eg_axis: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    band_color: str,
    mean_color: str,
    zero_color: str,
    band_alpha: float,
    mean_linewidth: float,
    zero_linewidth: float,
    edge_linewidth: float,
    label: str,
    xlabel: bool = False,
) -> None:
    """Plot one scaled-deviation panel."""

    fill_step_band(
        ax,
        eg_axis,
        lower,
        upper,
        facecolor=band_color,
        edgecolor=band_color,
        alpha=band_alpha,
        linewidth=edge_linewidth,
        zorder=2.0,
    )

    steps(
        ax,
        eg_axis,
        mean,
        color=mean_color,
        lw=mean_linewidth,
        zorder=3.0,
    )

    ax.axhline(
        0.0,
        color=zero_color,
        ls=":",
        lw=zero_linewidth,
        alpha=0.85,
        zorder=1.5,
    )

    if xlabel:
        ax.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")

    ax.set_ylabel(r"$S(E_\gamma)$")

    add_panel_label(ax, label=label)
    use_MeV_on_xaxis(ax)
    apply_axes_style(ax)


def set_common_xticks(axes: list[plt.Axes], eg_axis: np.ndarray) -> None:
    """Apply a compact common tick set to all x axes."""

    x_right = np.max(eg_axis).item()
    ticks = [tick for tick in [0, 1000, 2000, 3000, 4000] if tick <= x_right]

    if not ticks:
        return

    for ax in axes:
        ax.set_xticks(ticks)


def plot_model_checks(
    prior_nc: str | Path,
    post_nc: str | Path,
    ex: float | None = None,
    mass: float = 0.95,
    seed: int = 123,
    out: str | Path | None = None,
    show: bool = True,
    layout: str = "text",
    symlog_linthresh: float = 5.0,
) -> plt.Figure:
    """Create the model-check figure."""

    prior_results = UnfoldingResults(prior_nc, expected_mode="prior")
    posterior_results = UnfoldingResults(post_nc, expected_mode="posterior")

    rng = np.random.default_rng(seed)

    ex_actual = posterior_results.ex_actual(ex)
    eg_axis = common_eg_axis(
        prior_results=prior_results,
        posterior_results=posterior_results,
        ex_actual=ex_actual,
    )

    n_obs = posterior_results.n_obs(ex_actual)
    n_off_obs = posterior_results.n_off_obs(ex_actual)

    prior_total_rate = prior_results.total_expected(ex_actual)
    posterior_total_rate = posterior_results.total_expected(ex_actual)

    prior_n_pred = poisson_predictive(prior_total_rate, rng=rng)
    posterior_n_pred = poisson_predictive(posterior_total_rate, rng=rng)

    eg_n_prior, n_prior_mean, n_prior_lower, n_prior_upper = rank_envelope_from_array(
        prior_n_pred,
        eg_axis,
        mass=mass,
    )
    eg_n_post, n_post_mean, n_post_lower, n_post_upper = rank_envelope_from_array(
        posterior_n_pred,
        eg_axis,
        mass=mass,
    )

    n_scaled = scaled_deviation(
        values=posterior_n_pred,
        reference=n_obs,
        lower=n_post_lower,
        upper=n_post_upper,
    )
    eg_sn, n_scaled_mean, n_scaled_lower, n_scaled_upper = rank_envelope_from_array(
        n_scaled,
        eg_axis,
        mass=mass,
    )

    prior_nu = prior_results.nu(ex_actual)
    posterior_nu = posterior_results.nu(ex_actual)
    nu_true = posterior_results.nu_true(ex_actual)

    prior_nu_values = draw_matrix(prior_nu)
    posterior_nu_values = draw_matrix(posterior_nu)

    eg_nu_prior, nu_prior_mean, nu_prior_lower, nu_prior_upper = rank_envelope_from_array(
        prior_nu_values,
        eg_axis,
        mass=mass,
    )
    eg_nu_post, nu_post_mean, nu_post_lower, nu_post_upper = rank_envelope_from_array(
        posterior_nu_values,
        eg_axis,
        mass=mass,
    )

    nu_scaled = scaled_deviation(
        values=posterior_nu_values,
        reference=nu_true,
        lower=nu_post_lower,
        upper=nu_post_upper,
    )
    eg_snu, nu_scaled_mean, nu_scaled_lower, nu_scaled_upper = rank_envelope_from_array(
        nu_scaled,
        eg_axis,
        mass=mass,
    )

    has_background = n_off_obs is not None

    if has_background:
        prior_bg = prior_results.bg(ex_actual)
        posterior_bg = posterior_results.bg(ex_actual)

        if prior_bg is None or posterior_bg is None:
            raise KeyError("n_off_obs exists, but bg_exp is missing in one of the files.")

        prior_bg_pred = poisson_predictive(prior_bg, rng=rng)
        posterior_bg_pred = poisson_predictive(posterior_bg, rng=rng)

        eg_bg_prior, bg_prior_mean, bg_prior_lower, bg_prior_upper = rank_envelope_from_array(
            prior_bg_pred,
            eg_axis,
            mass=mass,
        )
        eg_bg_post, bg_post_mean, bg_post_lower, bg_post_upper = rank_envelope_from_array(
            posterior_bg_pred,
            eg_axis,
            mass=mass,
        )

        bg_scaled = scaled_deviation(
            values=posterior_bg_pred,
            reference=n_off_obs,
            lower=bg_post_lower,
            upper=bg_post_upper,
        )
        eg_sbg, bg_scaled_mean, bg_scaled_lower, bg_scaled_upper = rank_envelope_from_array(
            bg_scaled,
            eg_axis,
            mass=mass,
        )

    prior_band = COLORS["vermillion"]
    prior_mean = COLORS["blue"]
    post_band = COLORS["sky"]
    post_mean = COLORS["orange"]
    dev_band = COLORS["green"]
    dev_mean = COLORS["purple"]
    reference_color = COLORS["black"]

    alpha_band = 0.28
    alpha_dev = 0.18

    mean_linewidth = 0.55
    dev_mean_linewidth = 0.30
    reference_linewidth = 0.40
    edge_linewidth = 0.35
    zero_linewidth = 0.50

    mass_percent = round(100.0 * mass)

    size = figure_size(layout, height_to_width=0.90)
    fig = plt.figure(figsize=size.as_tuple(), constrained_layout=False)

    grid = fig.add_gridspec(
        1,
        2,
        left=0.10,
        right=0.97,
        bottom=0.08,
        top=0.90,
        wspace=0.20,
        hspace=0.10,
        width_ratios=[2.0, 1.0],
    )

    grid_left = GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=grid[0],
        wspace=0.15,
    )
    grid_right = GridSpecFromSubplotSpec(
        3,
        1,
        subplot_spec=grid[1],
    )

    axes: list[plt.Axes] = []

    ax_a = fig.add_subplot(grid_left[0, 0])
    plot_band_with_reference(
        ax=ax_a,
        eg_band=eg_n_prior,
        mean=n_prior_mean,
        lower=n_prior_lower,
        upper=n_prior_upper,
        eg_reference=eg_axis,
        reference=n_obs,
        band_color=prior_band,
        mean_color=prior_mean,
        reference_color=reference_color,
        band_alpha=alpha_band,
        mean_linewidth=mean_linewidth,
        reference_linewidth=reference_linewidth,
        edge_linewidth=edge_linewidth,
        label="a",
        ylabel=r"$\mathbf{n}$ (counts)",
        box_label=False,
    )
    axes.append(ax_a)

    ax_b = fig.add_subplot(grid_left[0, 1], sharex=ax_a, sharey=ax_a)
    plot_band_with_reference(
        ax=ax_b,
        eg_band=eg_n_post,
        mean=n_post_mean,
        lower=n_post_lower,
        upper=n_post_upper,
        eg_reference=eg_axis,
        reference=n_obs,
        band_color=post_band,
        mean_color=post_mean,
        reference_color=reference_color,
        band_alpha=alpha_band,
        mean_linewidth=mean_linewidth,
        reference_linewidth=reference_linewidth,
        edge_linewidth=edge_linewidth,
        label="b",
        show_y_ticklabels=False,
    )
    axes.append(ax_b)

    set_count_axis_pair(
        [ax_a, ax_b],
        arrays=[n_prior_upper, n_post_upper, n_obs],
        symlog_linthresh=symlog_linthresh,
    )

    ax_c = fig.add_subplot(grid_right[0, 0], sharex=ax_a)
    plot_scaled_deviation(
        ax=ax_c,
        eg_axis=eg_sn,
        mean=n_scaled_mean,
        lower=n_scaled_lower,
        upper=n_scaled_upper,
        band_color=dev_band,
        mean_color=dev_mean,
        zero_color=reference_color,
        band_alpha=alpha_dev,
        mean_linewidth=dev_mean_linewidth,
        zero_linewidth=zero_linewidth,
        edge_linewidth=edge_linewidth,
        label="c",
    )
    axes.append(ax_c)

    if has_background:
        ax_d = fig.add_subplot(grid_left[1, 0], sharex=ax_a)
        plot_band_with_reference(
            ax=ax_d,
            eg_band=eg_bg_prior,
            mean=bg_prior_mean,
            lower=bg_prior_lower,
            upper=bg_prior_upper,
            eg_reference=eg_axis,
            reference=n_off_obs,
            band_color=prior_band,
            mean_color=prior_mean,
            reference_color=reference_color,
            band_alpha=alpha_band,
            mean_linewidth=mean_linewidth,
            reference_linewidth=reference_linewidth,
            edge_linewidth=edge_linewidth,
            label="d",
            ylabel=r"$\mathbf{n}_{\mathrm{off}}$ (counts)",
            box_label=False,
        )
        axes.append(ax_d)

        ax_e = fig.add_subplot(grid_left[1, 1], sharex=ax_a, sharey=ax_d)
        plot_band_with_reference(
            ax=ax_e,
            eg_band=eg_bg_post,
            mean=bg_post_mean,
            lower=bg_post_lower,
            upper=bg_post_upper,
            eg_reference=eg_axis,
            reference=n_off_obs,
            band_color=post_band,
            mean_color=post_mean,
            reference_color=reference_color,
            band_alpha=alpha_band,
            mean_linewidth=mean_linewidth,
            reference_linewidth=reference_linewidth,
            edge_linewidth=edge_linewidth,
            label="e",
            show_y_ticklabels=False,
        )
        axes.append(ax_e)

        set_count_axis_pair(
            [ax_d, ax_e],
            arrays=[bg_prior_upper, bg_post_upper, n_off_obs],
            symlog_linthresh=symlog_linthresh,
        )

        ax_f = fig.add_subplot(grid_right[1, 0], sharex=ax_a)
        plot_scaled_deviation(
            ax=ax_f,
            eg_axis=eg_sbg,
            mean=bg_scaled_mean,
            lower=bg_scaled_lower,
            upper=bg_scaled_upper,
            band_color=dev_band,
            mean_color=dev_mean,
            zero_color=reference_color,
            band_alpha=alpha_dev,
            mean_linewidth=dev_mean_linewidth,
            zero_linewidth=zero_linewidth,
            edge_linewidth=edge_linewidth,
            label="f",
        )
        axes.append(ax_f)

    else:
        ax_d = fig.add_subplot(grid_left[1, 0], sharex=ax_a)
        ax_e = fig.add_subplot(grid_left[1, 1], sharex=ax_a)
        ax_f = fig.add_subplot(grid_right[1, 0], sharex=ax_a)

        for ax in (ax_d, ax_e, ax_f):
            ax.axis("off")

    ax_g = fig.add_subplot(grid_left[2, 0], sharex=ax_a)
    plot_band_with_reference(
        ax=ax_g,
        eg_band=eg_nu_prior,
        mean=nu_prior_mean,
        lower=nu_prior_lower,
        upper=nu_prior_upper,
        eg_reference=eg_axis,
        reference=nu_true,
        band_color=prior_band,
        mean_color=prior_mean,
        reference_color=reference_color,
        band_alpha=alpha_band,
        mean_linewidth=mean_linewidth,
        reference_linewidth=reference_linewidth,
        edge_linewidth=edge_linewidth,
        label="g",
        ylabel=r"$\boldsymbol{\nu}$ (counts)",
        box_label=False,
    )
    ax_g.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    axes.append(ax_g)

    ax_h = fig.add_subplot(grid_left[2, 1], sharex=ax_a, sharey=ax_g)
    plot_band_with_reference(
        ax=ax_h,
        eg_band=eg_nu_post,
        mean=nu_post_mean,
        lower=nu_post_lower,
        upper=nu_post_upper,
        eg_reference=eg_axis,
        reference=nu_true,
        band_color=post_band,
        mean_color=post_mean,
        reference_color=reference_color,
        band_alpha=alpha_band,
        mean_linewidth=mean_linewidth,
        reference_linewidth=reference_linewidth,
        edge_linewidth=edge_linewidth,
        label="h",
        show_y_ticklabels=False,
    )
    ax_h.set_xlabel(r"Photon energy $E_\gamma$ (MeV)")
    axes.append(ax_h)

    set_count_axis_pair(
        [ax_g, ax_h],
        arrays=[nu_prior_upper, nu_post_upper, nu_true],
        symlog_linthresh=symlog_linthresh,
    )

    ax_i = fig.add_subplot(grid_right[2, 0], sharex=ax_a)
    plot_scaled_deviation(
        ax=ax_i,
        eg_axis=eg_snu,
        mean=nu_scaled_mean,
        lower=nu_scaled_lower,
        upper=nu_scaled_upper,
        band_color=dev_band,
        mean_color=dev_mean,
        zero_color=reference_color,
        band_alpha=alpha_dev,
        mean_linewidth=dev_mean_linewidth,
        zero_linewidth=zero_linewidth,
        edge_linewidth=edge_linewidth,
        label="i",
        xlabel=True,
    )
    axes.append(ax_i)

    for ax in (ax_a, ax_b, ax_c):
        ax.tick_params(labelbottom=False)

    if has_background:
        for ax in (ax_d, ax_e, ax_f):
            ax.tick_params(labelbottom=False)

    set_common_xticks(axes, eg_axis)

    prior_handle = (
        Patch(
            facecolor=rgba(prior_band, alpha_band),
            edgecolor=prior_band,
            linewidth=edge_linewidth,
        ),
        Line2D([0], [0], color=prior_mean, lw=mean_linewidth),
    )
    post_handle = (
        Patch(
            facecolor=rgba(post_band, alpha_band),
            edgecolor=post_band,
            linewidth=edge_linewidth,
        ),
        Line2D([0], [0], color=post_mean, lw=mean_linewidth),
    )
    dev_handle = (
        Patch(
            facecolor=rgba(dev_band, alpha_dev),
            edgecolor=dev_band,
            linewidth=edge_linewidth,
        ),
        Line2D([0], [0], color=dev_mean, lw=mean_linewidth),
    )
    reference_handle = Line2D(
        [0],
        [0],
        color=reference_color,
        lw=reference_linewidth,
        ls="--",
    )

    add_top_legend(
        fig,
        handles=[prior_handle, post_handle, reference_handle, dev_handle],
        labels=[
            rf"Prior: mean + {mass_percent}\% rank-env.",
            rf"Post.: mean + {mass_percent}\% rank-env.",
            "Ref.",
            rf"$S(E_\gamma)$: mean + {mass_percent}\% rank-env.",
        ],
        ncol=2,
        y=0.98,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        fontsize=9.0,
    )

    save_or_show(fig, out, tag="model-checks", show=show)
    return fig
