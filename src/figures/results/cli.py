"""
Command-line dispatcher for result figures.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from ...paths import repo_path

def repo_path_or_none(path: str | None) -> Path | None:
    """Return repo_path(path), or None when no output path is given."""

    if path is None:
        return None

    return repo_path(path)


def add_common_figure_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by result-figure commands."""

    parser.add_argument("--layout", choices=["text", "column"], default="text")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the result-figure command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate result figures for the unfolding article."
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    prior_spectrum = subparsers.add_parser("prior-spectrum",
        help="Prior mean and global rank-envelope bands for one excitation-energy row.",
    )
    prior_spectrum.add_argument("prior_nc", help="Path to prior draws.nc.")
    prior_spectrum.add_argument("--ex", type=float, default=None)
    prior_spectrum.add_argument("--var", choices=["eta", "x"], default="eta")
    prior_spectrum.add_argument("--mass", type=float, default=0.95)
    prior_spectrum.add_argument("--inner-mass", type=float, default=0.75)
    prior_spectrum.add_argument("--symlog-linthresh", type=float, default=5.0)
    add_common_figure_args(prior_spectrum)

    posterior_spectrum = subparsers.add_parser("posterior-spectrum",
        help="Posterior spectrum, zoom panels, and scaled deviation.",
    )
    posterior_spectrum.add_argument("post_nc", help="Path to posterior draws.nc.")
    posterior_spectrum.add_argument("--ex", type=float, default=None)
    posterior_spectrum.add_argument("--var", choices=["eta", "x"], default="eta")
    posterior_spectrum.add_argument("--mass", type=float, default=0.95)
    posterior_spectrum.add_argument("--zoom",
        default=None,
        help="Two zoom windows in keV, for example '650-900,1350-1650'.",
    )
    posterior_spectrum.add_argument("--symlog-linthresh", type=float, default=5.0)
    add_common_figure_args(posterior_spectrum)

    prior_posterior = subparsers.add_parser("prior-posterior",
        help="Prior and posterior means with global rank-envelope bands.",
    )
    prior_posterior.add_argument("prior_nc", help="Path to prior draws.nc.")
    prior_posterior.add_argument("post_nc", help="Path to posterior draws.nc.")
    prior_posterior.add_argument("--ex", type=float, default=None)
    prior_posterior.add_argument("--var", choices=["eta", "x"], default="eta")
    prior_posterior.add_argument("--mass", type=float, default=0.95)
    prior_posterior.add_argument("--symlog-linthresh", type=float, default=5.0)
    add_common_figure_args(prior_posterior)

    model_checks = subparsers.add_parser("model-checks",
        help="Prior/posterior model checks in observed and folded-signal space.",
    )
    model_checks.add_argument("prior_nc", help="Path to prior draws.nc.")
    model_checks.add_argument("post_nc", help="Path to posterior draws.nc.")
    model_checks.add_argument("--ex", type=float, default=None)
    model_checks.add_argument("--mass", type=float, default=0.95)
    model_checks.add_argument("--seed", type=int, default=123)
    add_common_figure_args(model_checks)

    multi_ex = subparsers.add_parser("multi-ex",
        help="Posterior spectra and scaled deviations for several excitation-energy rows.",
    )
    multi_ex.add_argument("post_nc", help="Path to posterior draws.nc.")
    multi_ex.add_argument("--ex-values", nargs="+", type=float, required=True)
    multi_ex.add_argument("--var", choices=["eta", "x"], default="eta")
    multi_ex.add_argument("--mass", type=float, default=0.95)
    add_common_figure_args(multi_ex)

    sensitivity = subparsers.add_parser("sensitivity",
        help="Posterior sensitivity comparison against a baseline run.",
    )
    sensitivity.add_argument("--baseline-post", required=True)
    sensitivity.add_argument("--alts-post", nargs="+", required=True)
    sensitivity.add_argument("--scheme",
        choices=[
            "alpha",
            "sigma",
            "rl",
            "prior",
            "prior-components",
            "lowstat-prior",
            "generic",
            "custom",
            ],
        default="generic",
    )
    sensitivity.add_argument("--labels", nargs="+", default=None)
    sensitivity.add_argument("--alphas", nargs="+", type=float, default=None)
    sensitivity.add_argument("--sigmas", nargs="+", default=None)
    sensitivity.add_argument("--rls", nargs="+", default=None)
    sensitivity.add_argument("--title-prefix", default=None)
    sensitivity.add_argument("--ex", type=float, default=None)
    sensitivity.add_argument("--var", choices=["eta", "x"], default="eta")
    sensitivity.add_argument("--mass", type=float, default=0.95)
    sensitivity.add_argument("--seed", type=int, default=123)
    add_common_figure_args(sensitivity)

    low_stat_prior_dependence = subparsers.add_parser("low-stat-prior-dependence",
        help=(
            "Absolute-scale posterior prior-dependence comparison for the "
            "low-statistics Ex ~= 9.5 MeV spectrum."),
    )
    low_stat_prior_dependence.add_argument("--baseline-post", required=True)
    low_stat_prior_dependence.add_argument("--alts-post", nargs=3, required=True)
    low_stat_prior_dependence.add_argument("--ex", type=float, default=None)
    low_stat_prior_dependence.add_argument("--var", choices=["eta", "x"], default="eta")
    low_stat_prior_dependence.add_argument("--mass", type=float, default=0.95)
    low_stat_prior_dependence.add_argument("--symlog-linthresh", type=float, default=5.0)
    low_stat_prior_dependence.add_argument("--out", default=None)
    low_stat_prior_dependence.add_argument("--no-show",
        dest="show",
        action="store_false",
    )
    low_stat_prior_dependence.set_defaults(show=True)

    robustness = subparsers.add_parser("robustness",
        help="Posterior robustness comparison against a baseline run.",
    )
    robustness.add_argument("--baseline-post", required=True)
    robustness.add_argument("--alts-post", nargs="+", required=True)
    robustness.add_argument("--labels", nargs="+", default=None)
    robustness.add_argument("--ex", type=float, default=None)
    robustness.add_argument("--var", choices=["eta", "x"], default="eta")
    robustness.add_argument("--mass", type=float, default=0.95)
    robustness.add_argument("--seed", type=int, default=123)
    add_common_figure_args(robustness)

    bayes_frequentist = subparsers.add_parser("bayes-frequentist",
        help="Bayesian posterior versus frequentist RMLE result.",
    )
    bayes_frequentist.add_argument("--bayes-nc",
        required=True,
        help="Path to Bayesian posterior draws.nc.",
    )
    bayes_frequentist.add_argument("--freq-dir",
        required=True,
        help="Frequentist output root containing Ex*/summary.npz.",
    )
    bayes_frequentist.add_argument("--ex", type=float, required=True)
    bayes_frequentist.add_argument("--var", choices=["eta", "x", "nu"], default="eta")
    bayes_frequentist.add_argument("--mass", type=float, default=0.95)
    bayes_frequentist.add_argument("--zoom1", nargs=2, type=float, default=None)
    bayes_frequentist.add_argument("--zoom2", nargs=2, type=float, default=None)
    bayes_frequentist.add_argument("--zoom3", nargs=2, type=float, default=None)
    add_common_figure_args(bayes_frequentist)

    return parser


def run_command(args: argparse.Namespace) -> None:
    """Dispatch one result-figure command."""

    if args.cmd == "prior-spectrum":
        from .prior_spectrum import plot_prior_spectrum

        inner_mass = args.inner_mass
        if inner_mass is not None and inner_mass <= 0.0:
            inner_mass = None

        plot_prior_spectrum(
            prior_nc=repo_path(args.prior_nc),
            ex=args.ex,
            var=args.var,
            mass=args.mass,
            inner_mass=inner_mass,
            symlog_linthresh=args.symlog_linthresh,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
            layout=args.layout,
        )

    elif args.cmd == "posterior-spectrum":
        from .posterior_spectrum import plot_posterior_spectrum

        plot_posterior_spectrum(
            post_nc=repo_path(args.post_nc),
            ex=args.ex,
            var=args.var,
            mass=args.mass,
            zoom=args.zoom,
            symlog_linthresh=args.symlog_linthresh,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
            layout=args.layout,
        )

    elif args.cmd == "prior-posterior":
        from .prior_posterior_comparison import plot_prior_posterior_comparison

        plot_prior_posterior_comparison(
            prior_nc=repo_path(args.prior_nc),
            post_nc=repo_path(args.post_nc),
            ex=args.ex,
            var=args.var,
            mass=args.mass,
            symlog_linthresh=args.symlog_linthresh,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
            layout=args.layout,
        )

    elif args.cmd == "model-checks":
        from .model_checks import plot_model_checks

        plot_model_checks(
            prior_nc=repo_path(args.prior_nc),
            post_nc=repo_path(args.post_nc),
            ex=args.ex,
            mass=args.mass,
            seed=args.seed,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
            layout=args.layout,
        )

    elif args.cmd == "multi-ex":
        from .multi_ex_results import plot_multi_ex_results

        plot_multi_ex_results(
            post_nc=repo_path(args.post_nc),
            ex_values=args.ex_values,
            var=args.var,
            mass=args.mass,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
            layout=args.layout,
        )

    elif args.cmd == "sensitivity":
        from .sensitivity import plot_sensitivity

        plot_sensitivity(
            baseline_post_nc=repo_path(args.baseline_post),
            alt_post_ncs=[repo_path(path) for path in args.alts_post],
            labels=args.labels,
            alphas=args.alphas,
            sigmas=args.sigmas,
            rls=args.rls,
            scheme=args.scheme,
            title_prefix=args.title_prefix,
            ex=args.ex,
            var=args.var,
            mass=args.mass,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
            layout=args.layout,
            seed=args.seed,
        )

    elif args.cmd == "low-stat-prior-dependence":
        from .low_stat_prior_dependence import plot_low_stat_prior_dependence

        plot_low_stat_prior_dependence(
            baseline_post_nc=repo_path(args.baseline_post),
            alt_post_ncs=[repo_path(path) for path in args.alts_post],
            ex=args.ex,
            var=args.var,
            mass=args.mass,
            symlog_linthresh=args.symlog_linthresh,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
        )

    elif args.cmd == "robustness":
        from .robustness import plot_robustness

        plot_robustness(
            baseline_post_nc=repo_path(args.baseline_post),
            alt_post_ncs=[repo_path(path) for path in args.alts_post],
            labels=args.labels,
            ex=args.ex,
            var=args.var,
            mass=args.mass,
            out=repo_path_or_none(args.out),
            show=bool(args.show),
            layout=args.layout,
            seed=args.seed,
        )

    elif args.cmd == "bayes-frequentist":
        from .bayes_frequentist_comparison import plot_bayes_frequentist_comparison

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
            out=repo_path_or_none(args.out),
            show=bool(args.show),
        )

    else:
        raise ValueError(f"Unknown command: {args.cmd}")


def main(argv: list[str] | None = None) -> None:
    """Run the result-figure CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    run_command(args)


if __name__ == "__main__":
    main()
