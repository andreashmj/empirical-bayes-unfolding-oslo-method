# Empirical-Bayes unfolding of γ-ray spectra

This repository contains the code used for the article "Empirical-Bayes unfolding of γ-ray spectra".

The code implements a Bayesian unfolding model for Oslo-method
γ-ray spectra. The method uses a Poisson forward model, an empirical-Bayes prior
based on a Richardson-Lucy reference estimate, and posterior sampling with
PyMC/NUTS.

The repository also contains the configuration files and plotting scripts used
for the article figures, as well as scripts for the comparison with the OMpy
RMLE unfolding method.

Large result files are not stored in this GitHub repository. They are
archived on Zenodo.

## Repository setup

The main parts of the repository are

```text
src/
    Source code for the Bayesian model, priors, uncertainty estimates, figures, and frequentist comparison

configs/
    Configuration files with settings for the runs, priors, and samplers

data/
    Input data files used to generate the results

examples/
    Small example runs

tests/
    Tests for the code base

reproducibility/
    OMpy version and local patch for compatibility
```

Result files are written to `results/`, and figures are stored in the folder `figures/`.

## Installation

Create and activate the Conda environment from the root of the repository

```bash
conda env create -f environment.yml
conda activate empirical-bayes-unfolding
```

OMpy is required for accessing the OSCAR response and for the comparison with the frequentist methods. The results in the article used the `shapedev` branch of OMpy at commit

```text
a2a55c124442c84a7015e70534a6637f7df69c41
```

with the local patch

```text
reproducibility/ompy-local-changes.patch
```

One way to install the same OMpy version is

```bash
PROJECT_ROOT="$(pwd)"
OMPY_DIR="../ompy-paper"

git clone --branch shapedev --single-branch \
  https://github.com/oslocyclotronlab/ompy.git \
  "$OMPY_DIR"

git -C "$OMPY_DIR" checkout --detach \
  a2a55c124442c84a7015e70534a6637f7df69c41

git -C "$OMPY_DIR" apply --check --whitespace=nowarn \
  "$PROJECT_ROOT/reproducibility/ompy-local-changes.patch"

git -C "$OMPY_DIR" apply --whitespace=nowarn \
  "$PROJECT_ROOT/reproducibility/ompy-local-changes.patch"

python -m pip install -e "$OMPY_DIR"
```

Then fetch the OMpy response data

```bash
ompy data fetch
```


## Running the Bayesian unfolding 

The YAML files in `configs/runs/` give the settings for the runs.

A run is started with

```bash
python -m src.run_unfolding -c <config.yaml>
```

For example, the high-statistics baseline posterior run can be executed by

```bash
python -m src.run_unfolding \
  -c configs/runs/paper/high_stat/baseline/posterior.yaml
```

The main configurations used for the article can be found in the folder

```text
configs/runs/paper/
```


## Figures

The scripts used to create the figures are in the folder `src/figures/`.

For more information on how to make the figures use the commands

```bash
python -m src.figures.methods.cli --help
python -m src.figures.results.cli --help
```

The figures are saved to `figures/`.

## Result files

The results are stored in `draws.nc` files. These main output is the posterior draws of the emitted spectrum. The files also include the observed spectra, truth vectors for synthetic runs, and the response matrices used in the run.

The code uses the right-handed convention for applying the response operators

```python
eta = x @ G_g
nu = x @ D @ G_g
```

Here `eta` is the resolution-limited spectrum and `nu` is the expected detected
signal. The code uses `x` for the emitted spectrum variable, which is `mu` in the article. 

## Citation

The archived code and results for version 1.0.0 are available on Zenodo:

> Andreas Halkjelsvik Mjøs, *Empirical-Bayes unfolding of γ-ray spectra:
> code and results*, version 1.0.0, Zenodo.  
> DOI: [10.5281/zenodo.20797045](https://doi.org/10.5281/zenodo.20797045)

Citation metadata are provided in `CITATION.cff`.

## License

Copyright © 2026 Andreas Halkjelsvik Mjøs.

This project is licensed under the GNU General Public License, version 3 or
later. See `LICENSE.md` for the full license text.

Third-party software and data retain their original licenses.
