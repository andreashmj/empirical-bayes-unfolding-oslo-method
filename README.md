# Empirical-Bayes unfolding of γ-ray spectra

This repository contains the code and configuration files used for
empirical-Bayes unfolding of synthetic Oslo-method γ-ray spectra.

The framework uses a Poisson forward model, a Richardson-Lucy reference for
empirical-Bayes prior construction, and NUTS posterior sampling through PyMC.
It also includes a comparison with the frequentist regularized
maximum-likelihood implementation in OMpy.

The repository contains code and configuration files for

- synthetic-data generation,
- empirical-Bayes prior construction,
- Bayesian prior and posterior sampling,
- posterior diagnostics and predictive checks,
- methods and results figures,
- sampler-backend benchmark summaries,
- and the frequentist RMLE comparison.

The GitHub repository contains the source code, configuration files, tests,
examples, and other lightweight reproducibility material. The complete
generated result files are too large for normal Git version control and will
therefore be distributed through the associated Zenodo archive.

## Repository structure

```text
src/
    run_unfolding.py              Bayesian unfolding entry point
    pymc_unfolding.py             PyMC model and NUTS backends
    synthetic_data.py             Synthetic Oslo-method data generation
    config_utils.py               YAML configuration loading and profile mapping

    prior/                        Richardson-Lucy prior construction
    analysis/                     Result readers, diagnostics, benchmark summaries
    figures/                      Methods and results figure scripts
    frequentist/                  OMpy RMLE comparison runner

configs/
    runs/                         Run configurations
    priors/                       Prior configurations
    samplers/                     Sampler configurations

data/
    Input data files. Large archived inputs may be omitted from Git.

examples/
    Small example workflows and quick-test scripts.

tests/
    Unit tests for configuration handling, prior construction, response
    normalization, result-array conventions, and PyMC helper functions.

results/
    Generated Bayesian, RMLE, and benchmark outputs. Large result files are
    omitted from Git and included in the Zenodo archive.

figures/
    Generated paper figures.

environment.yml
    Conda environment definition.

requirements.txt
    Pinned Python dependencies, excluding OMpy.

pytest.ini
    Pytest configuration.
```

## Environment

Create and activate the project environment from the repository root:

```bash
conda env create -f environment.yml
conda activate empirical-bayes-unfolding
```

This installs Python 3.12 and the pinned Python dependencies listed in
`requirements.txt`.

### OMpy

OMpy is a required external dependency. This project uses the development
[`shapedev` branch](https://github.com/oslocyclotronlab/ompy/tree/shapedev).

Follow the installation instructions in the README for that branch. After
installing OMpy, fetch its external response-data files with

```bash
ompy data fetch
```

The exact OMpy commit used to produce the archived paper results will be
recorded in the Zenodo archive.

## CPU and benchmark settings

For CPU-only JAX runs, use

```bash
export JAX_PLATFORMS=cpu
```

For the controlled CPU benchmark runs reported in the paper, use

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_DISABLE_CUDA=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_force_host_platform_device_count=4
```

## Installation check

After activating the environment and installing OMpy, run the following
commands from the repository root:

```bash
python -m pip check
python -m compileall -q -f src tests
python -m pytest -q
```

The tests do not rerun the complete paper sampling workflow. They check
configuration handling, prior construction, response normalization,
result-array conventions, and smaller model helper functions.

The following commands only display the available command-line options. They
do not start an unfolding or figure-generation run:

```bash
python -m src.run_unfolding --help
python -m src.frequentist.run_rmle_on_synthetic --help
python -m src.figures.results.cli --help
python -m src.figures.methods.cli --help
```

## Quickstart posterior example

A small posterior example is provided in `examples/`. After creating and
activating the environment, run

```bash
bash examples/quickstart.sh
```

This performs a short PyMC/NUTS posterior run for one synthetic
$E_x\approx2.5\,\mathrm{MeV}$ spectrum and writes

```text
results/examples/quickstart/posterior_demo/posterior/draws.nc
figures/examples/quickstart_posterior_2500.pdf
```

The quickstart sampler settings are intentionally small. They are intended as
a quick test of the installation and execution pipeline, not as paper-quality
posterior sampling.

## Running Bayesian unfolding

Bayesian prior and posterior runs are controlled by YAML files under
`configs/runs/`.

A single run is executed with

```bash
python -m src.run_unfolding -c <config.yaml>
```

For example, the high-statistics baseline posterior run is

```bash
python -m src.run_unfolding \
  -c configs/runs/paper/high_stat/baseline/posterior.yaml
```

The corresponding prior run is

```bash
python -m src.run_unfolding \
  -c configs/runs/paper/high_stat/baseline/prior.yaml
```

The production configurations used for the article are located under

```text
configs/runs/paper/
```

## Frequentist RMLE comparison

The frequentist comparison uses the OMpy RMLE implementation on the same
synthetic one-dimensional excitation-energy slices.

Run the high-statistics RMLE comparison with

```bash
python -m src.frequentist.run_rmle_on_synthetic \
  --config configs/runs/paper/high_stat/rmle.yaml
```

Run the low-statistics RMLE comparison with

```bash
python -m src.frequentist.run_rmle_on_synthetic \
  --config configs/runs/paper/low_stat/rmle.yaml
```

The high-statistics RMLE configuration uses a sparsity-penalized fit. Its
penalty strength is selected by minimizing the Wasserstein-1 distance to the
known synthetic truth. This truth-based selection is used only for the
controlled synthetic comparison and is not a prescription for experimental
data.

The low-statistics RMLE configuration uses the unpenalized fit.

## Figure generation

The methods-figure command-line interface is

```bash
python -m src.figures.methods.cli --help
```

The results-figure command-line interface is

```bash
python -m src.figures.results.cli --help
```

The `--help` commands list the available figure subcommands and arguments
without generating a figure.

For example, the high-statistics Bayesian–frequentist comparison figure can be
regenerated with

```bash
python -m src.figures.results.cli bayes-frequentist \
  --bayes-nc results/paper/high_stat/baseline_pymc/posterior/draws.nc \
  --freq-dir results/frequentist/paper/high_stat/rmle \
  --ex 4000 \
  --var eta \
  --mass 0.95 \
  --zoom1 600 1200 \
  --zoom2 2200 3400 \
  --zoom3 6500 9000 \
  --out figures/results/bayes_frequentist_comparison_high_stat.pdf \
  --no-show
```

The low-statistics comparison figure can be regenerated with

```bash
python -m src.figures.results.cli bayes-frequentist \
  --bayes-nc results/paper/low_stat/baseline_pymc/posterior/draws.nc \
  --freq-dir results/frequentist/paper/low_stat/rmle \
  --ex 9500 \
  --var eta \
  --mass 0.95 \
  --zoom1 500 1500 \
  --zoom2 2500 4200 \
  --zoom3 7000 9500 \
  --out figures/results/bayes_frequentist_comparison_low_stat.pdf \
  --no-show
```

## Result files

The Bayesian `draws.nc` files are self-contained analysis products. Depending
on the run mode, they contain

- emitted-spectrum draws `x`,
- observed ON and OFF count vectors,
- synthetic validation truth vectors,
- Richardson-Lucy and prior-reference vectors,
- posterior sampling information,
- and the response operators `D` and `G_g` used in the run.

The stored arrays use the OMpy right-hand multiplication convention:

```python
eta = x @ G_g
nu = x @ D @ G_g
```

The response operators are stored in each result file so that these derived
quantities can be reconstructed exactly.

Large `draws.nc` files and the other complete paper outputs are not committed
to GitHub because of their size. They are included in the Zenodo archive.

## Benchmark summary tables

Sampler benchmark outputs are summarized with

```bash
python -m src.analysis.benchmark_summary \
  --result-root results \
  --out-dir results/benchmarks \
  --var eta
```

The script writes

```text
results/benchmarks/benchmark_raw_eta.csv
results/benchmarks/benchmark_formatted_eta.csv
```

The formatted table is used in the paper appendix.

## Tests

Run all tests with

```bash
python -m pytest -q
```

The tests are lightweight and do not rerun the complete paper sampling
workflow.

## Reproducing the paper workflow

The paper runs are configured under

```text
configs/runs/paper/
```

The complete workflow is:

1. Create and activate the Conda environment.
2. Install OMpy from the `shapedev` branch.
3. Fetch the OMpy data files with `ompy data fetch`.
4. Run the Bayesian prior and posterior configurations.
5. Run the high- and low-statistics RMLE configurations.
6. Generate the methods and results figures.
7. Generate the benchmark summary tables.

## Citation

The fixed code-and-results archive for version 1.0.0 is available on Zenodo:

> Andreas Halkjelsvik Mjøs, *Empirical-Bayes unfolding of γ-ray spectra:
> code and results*, version 1.0.0, Zenodo.  
> DOI: 10.5281/zenodo.20797045

Machine-readable citation metadata are provided in `CITATION.cff`.

The citation for the associated article will be added after publication.

## License

The source code and configuration scripts are distributed under the GNU
General Public License v3.0 or later. See `LICENSE` for details.

The generated synthetic data, result files, benchmark tables, and figures in
the Zenodo archive are distributed under the Creative Commons Attribution 4.0
International license.
