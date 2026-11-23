"""
Run Poisson resamples and save one eta posterior per resample.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from ..analysis.diagnostics import rhat_ess_by_eg
from ..config_utils import dict_to_namespace, load_run_config
from ..paths import repo_path
from ..run_unfolding import UnfoldManager
from ..synthetic_data import SyntheticDataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ex", type=float, required=True)
    p.add_argument("--resamples", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    out = repo_path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = dict_to_namespace(load_run_config(repo_path(a.config)))
    cfg.ex_energies = [a.ex]
    cfg.mode = "posterior"
    cfg.save_trace = False

    loader = SyntheticDataLoader.from_config(cfg)
    i = loader.n.index_X(a.ex)
    active = loader.eg_slice_for_index(i)
    eg = np.asarray(loader.Eg, dtype=float)[:active.stop]
    truth = np.asarray(loader.eta.iloc[i, active].values, dtype=float)
    off_expected = np.asarray(loader.background_expected.values)
    on_expected = np.asarray(loader.nu.values) + off_expected

    for r in range(1, a.resamples + 1):
        path = out / f"resample{r:03d}.nc"
        if path.exists():
            continue

        seed = 100 + r
        rng = np.random.default_rng(seed)
        loader._n_off.values[:] = rng.poisson(off_expected)
        loader._n.values[:] = rng.poisson(on_expected)
        loader.rng_seed = seed

        print(f"Resample {r}/{a.resamples}, seed={seed}", flush=True)

        manager = UnfoldManager(loader, cfg)
        manager._run_one_ex_unfolding(0, manager.selected_indices[0], "posterior")

        eta = manager.x_draws_by_ex[0] @ manager.G_g_by_ex[0]
        rhat, ess = rhat_ess_by_eg(eta, eg, name="eta")

        ds = xr.Dataset(
            data_vars={
                "eta": (("chain", "draw", "Eg"), eta.astype("float32")),
                "eta_true": ("Eg", truth),
                "n_obs": ("Eg", np.asarray(loader._n.values[i, active])),
                "n_off_obs": ("Eg", np.asarray(loader._n_off.values[i, active])),
                "rhat": ("Eg", rhat),
                "ess_bulk": ("Eg", ess),
            },
            coords={"Eg": eg},
            attrs={
                "ex_requested_kev": a.ex,
                "ex_actual_kev": manager.ex_actual_values[0],
                "resample": r,
                "observation_seed": seed,
            },
        )

        temp = out / f"resample{r:03d}.tmp.nc"
        ds.to_netcdf(temp, engine="h5netcdf", encoding={"eta": {"zlib": True, "complevel": 2}})
        temp.replace(path)
        ds.close()

        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
