"""
CLI parsers for creating the figures in the methods section.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ...paths import repo_path

from .diagnostics import plot_diagnostics
from .ex_smearing import plot_ex_smearing
from .flowchart import plot_flowchart
from .data_generation import plot_data_generation
from .ill_posedness import plot_ill_posedness
from .nuts import plot_nuts
from .prior_draws import plot_prior_draws


def repo_path_or_none(path: str | None) -> Path | None:
    if path is None:
        return None
    return repo_path(path)


def add_common_figure_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate methods figures for the unfolding article."
    )

    parser.add_argument("--layout", choices=["text", "column"], default="text")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    ill_posedness = subparsers.add_parser("ill-posedness",
        help="Posterior draw comparison between x-space and eta-space.",
    )
    ill_posedness.add_argument("nc", help="Path to posterior draws.nc.")
    ill_posedness.add_argument("--ex", type=float, default=None)
    ill_posedness.add_argument("--max-draws", type=int, default=10)
    ill_posedness.add_argument("--seed", type=int, default=123)
    ill_posedness.add_argument("--symlog-linthresh", type=float, default=5.0)
    add_common_figure_args(ill_posedness)

    prior_draws = subparsers.add_parser("prior-draws",
        help="Prior draws in eta-space from a prior draws.nc file.",
    )
    prior_draws.add_argument("nc", help="Path to prior draws.nc.")
    prior_draws.add_argument("--ex", type=float, default=None)
    prior_draws.add_argument("--max-draws", type=int, default=250)
    prior_draws.add_argument("--seed", type=int, default=123)
    prior_draws.add_argument("--symlog-linthresh", type=float, default=5.0)
    add_common_figure_args(prior_draws)

    ex_smearing = subparsers.add_parser("ex-smearing",
        help="Excitation-energy smearing bias heatmaps.",
    )
    ex_smearing.add_argument("--mat-path", default="data/ExEg_1e8.npz")
    ex_smearing.add_argument("--response-db", default="OSCAR2020")
    ex_smearing.add_argument("--lower-eg-cut", default="60keV")
    ex_smearing.add_argument("--sigma-eg", default="30keV")
    ex_smearing.add_argument("--fwhm-ex-kev", type=float, default=150.0)
    ex_smearing.add_argument("--width-factor-left", type=float, default=1.0)
    ex_smearing.add_argument("--width-factor-right", type=float, default=1.5)
    add_common_figure_args(ex_smearing)

    generation = subparsers.add_parser("data-generation",
        help="Truth and observed matrices with two selected Ex slices.",
    )
    generation.add_argument("--mat-path", default="data/ExEg_1e8.npz")
    generation.add_argument("--ex-values", nargs=2, type=float, required=True)
    generation.add_argument("--response-db", default="OSCAR2020")
    generation.add_argument("--rebin-factors", nargs=2, type=int, default=[10, 2])
    generation.add_argument("--lower-eg-cut", default="60keV")
    generation.add_argument("--sigma-eg", default="30keV")
    generation.add_argument("--bg-fraction", type=float, default=0.15)
    generation.add_argument("--bg-flat-fraction", type=float, default=0.10)
    generation.add_argument("--mat-scale", type=float, default=1.0)
    generation.add_argument("--eg-tail-mass", type=float, default=1.0e-6)
    generation.add_argument("--rng-seed", type=int, default=100)
    generation.add_argument("--fwhm-ex-kev", type=float, default=150.0)

    generation_background = generation.add_mutually_exclusive_group()
    generation_background.add_argument("--include-background",
        dest="include_background",
        action="store_true",
    )
    generation_background.add_argument("--no-background",
        dest="include_background",
        action="store_false",
    )
    generation.set_defaults(include_background=True)

    generation_ex_smearing = generation.add_mutually_exclusive_group()
    generation_ex_smearing.add_argument("--include-ex-smearing",
        dest="include_ex_smearing",
        action="store_true",
    )
    generation_ex_smearing.add_argument("--no-ex-smearing",
        dest="include_ex_smearing",
        action="store_false",
    )
    generation.set_defaults(include_ex_smearing=False)

    add_common_figure_args(generation)

    diagnostics = subparsers.add_parser("diagnostics",
        help="Per-Eg R-hat and bulk ESS diagnostics from posterior draws.",
    )
    diagnostics.add_argument("nc", help="Path to posterior draws.nc.")
    diagnostics.add_argument("--ex", type=float, default=None)
    diagnostics.add_argument("--var", choices=["x", "eta"], default="eta")
    diagnostics.add_argument("--no-truth", action="store_true")
    add_common_figure_args(diagnostics)

    nuts = subparsers.add_parser("nuts",
        help="NUTS tree-expansion schematic.",
    )
    add_common_figure_args(nuts)

    flowchart = subparsers.add_parser("flowchart",
        help="Bayesian unfolding flowchart schematic.",
    )
    add_common_figure_args(flowchart)
    return parser


def run_command(args: argparse.Namespace) -> None:
    """Dispatch one methods-figure command."""
    if args.cmd == "ill-posedness":
        plot_ill_posedness(
            repo_path(args.nc),
            ex=args.ex,
            max_draws=args.max_draws,
            seed=args.seed,
            layout=args.layout,
            out=repo_path_or_none(args.out),
            symlog_linthresh=args.symlog_linthresh,
            show=bool(args.show),
        )
    elif args.cmd == "prior-draws":
        plot_prior_draws(
            repo_path(args.nc),
            ex=args.ex,
            max_draws=args.max_draws,
            seed=args.seed,
            symlog_linthresh=args.symlog_linthresh,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
        )

    elif args.cmd == "ex-smearing":
        plot_ex_smearing(
            mat_path=args.mat_path,
            response_db=args.response_db,
            lower_eg_cut=args.lower_eg_cut,
            sigma_eg=args.sigma_eg,
            fwhm_ex_kev=args.fwhm_ex_kev,
            width_factor_left=args.width_factor_left,
            width_factor_right=args.width_factor_right,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
        )

    elif args.cmd == "data-generation":
        ex_rebin_factor, eg_rebin_factor = args.rebin_factors
        plot_data_generation(
            mat_path=args.mat_path,
            ex_values=(args.ex_values[0], args.ex_values[1]),
            response_db=args.response_db,
            rebin_factors=(ex_rebin_factor, eg_rebin_factor),
            lower_eg_cut=args.lower_eg_cut,
            sigma_eg=args.sigma_eg,
            include_background=bool(args.include_background),
            bg_fraction=args.bg_fraction,
            bg_flat_fraction=args.bg_flat_fraction,
            mat_scale=args.mat_scale,
            eg_tail_mass=args.eg_tail_mass,
            rng_seed=args.rng_seed,
            include_ex_smearing=bool(args.include_ex_smearing),
            fwhm_ex_kev=args.fwhm_ex_kev,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
        )

    elif args.cmd == "diagnostics":
        plot_diagnostics(
            repo_path(args.nc),
            ex=args.ex,
            var=args.var,
            show_truth=(not args.no_truth),
            layout=args.layout,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
        )

    elif args.cmd == "nuts":
        plot_nuts(
            out=repo_path_or_none(args.out),
            layout=args.layout,
            show=bool(args.show),
        )

    elif args.cmd == "flowchart":
        plot_flowchart(
            out=repo_path_or_none(args.out),
            show=bool(args.show),
        )
    else:
        raise ValueError(f"Unknown command: {args.cmd}")

def main(argv: list[str] | None = None) -> None:
    """
    Run the methods-figure CLI.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    run_command(args)


if __name__ == "__main__":
    main()
