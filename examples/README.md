# Quickstart example

This example runs a small prior-only Bayesian unfolding demonstration for one
synthetic excitation-energy spectrum near \(E_x \approx 4.0\) MeV.

Run from the repository root with

    bash examples/quickstart.sh

The script

1. generates the synthetic ON/OFF data from `data/ExEg_1e8.npz`,
2. builds the Richardson-Lucy empirical-Bayes prior,
3. draws from the prior distribution,
4. writes `draws.nc`, and
5. creates a prior-spectrum figure.

The output files are

    results/examples/quickstart/prior_demo/prior/draws.nc
    figures/examples/quickstart_prior_4000.pdf

This is a smoke test and usage example. It is not one of the production runs
used for the paper figures.
