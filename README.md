# Empirical-Bayes unfolding of γ-ray spectra

In this repository, you find the code used to produce the results and figures for the article "Empirical-Bayes unfolding of γ-ray spectra".

The empirical-Bayes unfolding method targets unfolding of Oslo-method γ-ray spectra. It utilizes a joint ON/OFF Poisson likelihood, an empirical-Bayes
prior based on a Richardson-Lucy reference estimate, and PyMC/NUTS for posterior sampling.

The repository includes all the configuration files and plotting scripts used for the runs and figure creation. Moreover, it contains the scripts used to run 
RMLE and the standard iterative unfolding for the same cases as the Bayesian method.


For access to the generated result files, see the archive on Zenodo.

## Repository setup

The main parts of the repository are

```text
src/
    Source code for the Bayesian model, priors, adaptive maximum-deviation envelopes, figures, and frequentist comparison

configs/
    Configuration files with settings for the runs, priors, and samplers

data/
    Input data files used to generate the results

ompy_reproducibility/
    OMpy version and patch with modifications to the OMpy source code.
```

Result files are written to `results/`, and figures are stored in the folder `figures/`.

## Installation

Create and activate the Conda environment from the root of the repository

```bash
conda env create -f environment.yml
conda activate empirical-bayes-unfolding
```

OMpy is required for accessing the OSCAR response and for the comparison with the frequentist methods.

```text
a2a55c124442c84a7015e70534a6637f7df69c41
```

with the local patch

```text
ompy_reproducibility/ompy-local-changes.patch
```

One way to install the same OMpy version is

```bash
ROOT="$(pwd)"
OMPY_DIR="../ompy-paper"

git clone --branch shapedev --single-branch \
  https://github.com/oslocyclotronlab/ompy.git \
  "$OMPY_DIR"

git -C "$OMPY_DIR" checkout --detach \
  a2a55c124442c84a7015e70534a6637f7df69c41

git -C "$OMPY_DIR" apply --check --whitespace=nowarn \
  "$ROOT/ompy_reproducibility/ompy-local-changes.patch"

git -C "$OMPY_DIR" apply --whitespace=nowarn \
  "$ROOT/ompy_reproducibility/ompy-local-changes.patch"

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
  -c configs/runs/synthetic/high_stat/baseline/posterior.yaml
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

The results are stored in `draws.nc` files. The main output is the posterior draws of the excitation-energy-smeared emitted spectrum. 

The code uses the right-hand convention for applying the response operators

```python
eta = x @ G_g
nu = x @ D @ G_g
```

Here `eta` is the resolution-limited spectrum and `nu` is the expected detected
signal.

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
