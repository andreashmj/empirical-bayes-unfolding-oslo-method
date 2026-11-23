"""
Commands for the paper's results figures.

"""

from __future__ import annotations

import argparse

from ...paths import repo_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)


    posterior = commands.add_parser("posterior-spectrum")
    posterior.add_argument("post_nc", type=repo_path)
    posterior.add_argument("--zoom", required=True, help="Two intervals in keV: '1100-1200,2800-3700'.")

    contraction = commands.add_parser("prior-posterior")
    contraction.add_argument("prior_nc", type=repo_path)
    contraction.add_argument("post_nc", type=repo_path)
    for command in (posterior, contraction):
        command.add_argument("--var", choices=("eta", "x"), default="eta")

    checks = commands.add_parser("model-checks")
    checks.add_argument("prior_nc", type=repo_path)
    checks.add_argument("post_nc", type=repo_path)

    multi = commands.add_parser("multi-ex")
    multi.add_argument("post_nc", type=repo_path)
    multi.add_argument("--ex-values", nargs=2, type=float, required=True)

    sensitivity = commands.add_parser("sensitivity")
    sensitivity.add_argument("--test", choices=("rl", "alpha", "sigma"), required=True)
    sensitivity.add_argument("--baseline-post", dest="baseline_post_nc", type=repo_path, required=True)
    sensitivity.add_argument("--alts-post", dest="alt_post_ncs", type=repo_path, nargs="+", required=True, help="Two files for RL/alpha, and four for sigma.")

    robustness = commands.add_parser("robustness")
    robustness.add_argument("--baseline-post", dest="baseline_post_nc", type=repo_path, required=True)
    robustness.add_argument("--alts-post", dest="alt_post_ncs", type=repo_path, nargs=4, required=True)

    low_prior = commands.add_parser("low-stat-prior-dependence")
    low_prior.add_argument("--baseline-post", dest="baseline_post_nc", type=repo_path, required=True)
    low_prior.add_argument("--alts-post", dest="alt_post_ncs", type=repo_path, nargs=3, required=True)

    comparison = commands.add_parser("bayes-frequentist")
    comparison.add_argument("--bayes-nc", type=repo_path, required=True)
    comparison.add_argument("--rmle-nc", type=repo_path, required=True)
    comparison.add_argument("--ex", type=float, required=True)

    experimental = commands.add_parser("experimental-comparison")
    experimental.add_argument("--config", dest="config_path", type=repo_path, required=True)
    experimental.add_argument("--bayes-nc", type=repo_path, required=True)
    experimental.add_argument("--rmle-nc", type=repo_path, required=True)
    experimental.add_argument("--standard-nc", type=repo_path, required=True)
    
    for command in (comparison, experimental):
        for number in (1, 2, 3):
            command.add_argument(f"--zoom{number}", nargs=2, type=float, required=True, metavar=("MIN_KEV", "MAX_KEV"))

    for command in (posterior, contraction, checks, sensitivity, robustness, low_prior, experimental):
        command.add_argument("--ex", type=float)
    for command in (posterior, contraction, checks, multi, sensitivity, robustness, low_prior, comparison, experimental):
        command.add_argument("--out", type=repo_path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "posterior-spectrum":
        from .posterior_spectrum import plot_posterior_spectrum as plot
    elif command == "prior-posterior":
        from .prior_posterior_comparison import plot_prior_posterior_comparison as plot
    elif command == "model-checks":
        from .model_checks import plot_model_checks as plot
    elif command == "multi-ex":
        from .multi_ex_results import plot_multi_ex_results as plot
    elif command == "sensitivity":
        from .sensitivity import plot_sensitivity as plot
        expected = 4 if args["test"] == "sigma" else 2
        if len(args["alt_post_ncs"]) != expected:
            parser.error(f"{args['test']} requires {expected} alternative files.")
    elif command == "robustness":
        from .robustness import plot_robustness as plot
    elif command == "low-stat-prior-dependence":
        from .low_stat_prior_dependence import plot_low_stat_prior_dependence as plot
    elif command == "bayes-frequentist":
        from .bayes_frequentist_comparison import plot_bayes_frequentist_comparison as plot
    else:
        from .experimental_method_comparison import plot_experimental_method_comparison as plot
    plot(**args)


if __name__ == "__main__":
    main()
