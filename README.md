# Empirical-Bayes unfolding in the Oslo method

This repository contains the code used for empirical-Bayes Bayesian unfolding
of synthetic Oslo-method gamma-ray spectra.

The method models the emitted spectrum with a Poisson forward model, uses a
Richardson-Lucy reference for empirical-Bayes prior construction, samples the
posterior with NUTS through PyMC, and compares the Bayesian result with an
OMpy RMLE frequentist unfolding.

The repository contains code for

- synthetic data generation,
- empirical-Bayes prior construction,
- Bayesian posterior sampling,
- posterior diagnostics,
- result and methods figures,
- sampler-backend benchmark summaries,
- and the frequentist RMLE comparison.

This repository is the code release candidate for the manuscript. A versioned
archival release and DOI will be created through Zenodo after the manuscript
and repository contents are finalized.

## Repository structure

```text
src/
    run_unfolding.py              Bayesian unfolding entry point
    pymc_unfolding.py             PyMC model and NUTS backends
    synthetic_data.py             Synthetic Oslo-method data generation
    config_utils.py               YAML configuration loading and profile mapping

    prior/                        Richardson-Lucy prior construction
    analysis/                     Result readers, diagnostics, benchmark summaries
    figures/                      Paper figure scripts
    frequentist/                  OMpy RMLE comparison runner

configs/
    runs/                         Run configurations
    priors/                       Prior configurations
    samplers/                     Sampler configurations

data/
    Input data files. Large files may be supplied separately.

examples/
    Small example workflow and quickstart scripts.

tests/
    Unit tests for configuration handling, prior construction, response
    normalization, result-array conventions, and PyMC helper functions.

results/
    Generated result files. Large posterior files are normally not committed.

figures/
    Generated paper figures. Final figures may be included intentionally, but
    large intermediate outputs should be regenerated or archived separately.
```

## Environment

Create the paper environment from the repository root with

```bash
conda env create -f environment.yml
conda activate unfolding-paper
```

This installs Python 3.12, pip, git, and the pinned Python dependencies listed
in `requirements.txt`.

OMpy is a required external dependency, but the `shapedev` version used for this
project is installed manually following the OMpy development-branch workflow.
It is not installed through a one-line pip requirement in `requirements.txt`.

Install OMpy with

```bash
mkdir -p ~/software
cd ~/software

git clone --branch shapedev https://github.com/oslocyclotronlab/ompy.git ompy-shapedev
cd ompy-shapedev

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Then fetch the external OMpy response/data cache:

```bash
ompy data fetch
```

For CPU-only JAX runs, use

```bash
export JAX_PLATFORMS=cpu
```

For controlled CPU benchmark runs, also use

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_DISABLE_CUDA=1
```

## Installation check

After activating the environment and installing OMpy, return to this repository
and run

```bash
cd ~/empirical-bayes-unfolding-oslo-method

python -m pip check
python -m compileall -q -f src tests
python -m src.run_unfolding --help
python -m src.frequentist.run_rmle_on_synthetic --help
python -m src.figures.results.cli --help
python -m src.figures.methods.cli --help
pytest -q
```

The tests do not rerun the full paper posterior sampling. They check the source
structure, configuration mapping, response normalization, prior construction,
result-array conventions, and small model helpers.

## Quickstart posterior example

A small posterior example is provided in `examples/`. After creating and
activating the environment, run

```bash
bash examples/quickstart.sh
```

This performs a short PyMC/NUTS posterior run for one synthetic
`E_x approximately 2.5 MeV` spectrum and writes

```text
results/examples/quickstart/posterior_demo/posterior/draws.nc
figures/examples/quickstart_posterior_2500.pdf
```

The quickstart sampler settings are intentionally small. They are meant as a
smoke test, not as paper-quality posterior sampling.

## Running Bayesian unfolding

Bayesian prior and posterior runs are controlled by YAML files in
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

A prior-only run is executed in the same way, using a prior-mode config:

```bash
python -m src.run_unfolding \
  -c configs/runs/paper/high_stat/baseline/prior.yaml
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

The high-statistics RMLE configuration uses the sparsity-penalized fit with
the penalty strength selected by a Wasserstein-1 comparison to the known
synthetic truth. The low-statistics RMLE configuration uses the unpenalized
fit.

## Figure generation

Methods figures are generated through

```bash
python -m src.figures.methods.cli --help
```

Result figures are generated through

```bash
python -m src.figures.results.cli --help
```

For example, the Bayesian-frequentist comparison figures can be regenerated
with

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

and

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

The Bayesian `draws.nc` files are self-contained analysis products. They store

- emitted-spectrum draws `x`,
- observed ON and OFF count vectors,
- synthetic validation truth vectors,
- prior-reference vectors,
- and the response operators `D` and `G_g` used in the run.

The response operators are deterministic for a given configuration, but they
are stored redundantly so that derived quantities can be reconstructed exactly:

```python
eta = x @ G_g
nu  = x @ D @ G_g
```

This is intentional. It makes archived result files independent of later
changes to response-cache handling or response-specialization code.

Large generated posterior files such as `draws.nc` and full NUTS trace files
are normally not stored directly in Git. They should be regenerated from the
configs or archived separately with the final Zenodo release.

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

The formatted table is intended for direct use in the manuscript appendix.

## Tests

Run the test suite with

```bash
pytest -q
```

The tests are lightweight. They do not rerun the full paper sampling workflow.

To check source syntax and imports before long reruns, use

```bash
find src tests -type d -name "__pycache__" -prune -exec rm -rf {} +
find src tests -name "*.pyc" -delete

python -m compileall -q -f src tests
pytest -q
```

## Reproducing the paper workflow

The paper runs are configured under

```text
configs/runs/paper/
```

The complete workflow is:

1. create and activate the Conda environment,
2. install OMpy from the `shapedev` branch,
3. fetch the OMpy data cache with `ompy data fetch`,
4. run the Bayesian prior and posterior configs,
5. run the RMLE comparison configs,
6. regenerate the paper figures,
7. regenerate the benchmark summary table.

Large generated outputs are not guaranteed to be tracked in Git. The final
archival release will specify which generated files are included and which
should be regenerated.

## Citation

A Zenodo DOI will be added after the final versioned release.
See `CITATION.cff`.

## License

See `LICENSE`.
