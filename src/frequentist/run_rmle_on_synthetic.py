"""
Run 1D RMLE for same case as the Bayesian method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import ompy as om
import xarray as xr
from ompy.unfolding.rmle.rmle import RMLE
from ompy.unfolding.rmle.rmle1d import BackgroundModel
from ompy.unfolding.rmle.lossmodel import ModelLoss

from ..config_utils import load_run_config
from ..paths import repo_path



def read_row(path: Path, ex: float) -> dict:
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        row = ds.swap_dims({"Ex": "Ex_req"}).sel(Ex_req=ex)
        length = int(row["Eg_len"].item())
        include_background = "n_off_obs" in ds
        return {
            "Ex": float(row["Ex"].item()),
            "Eg": np.asarray(ds["Eg"].values[:length], dtype=float),
            "n": np.asarray(row["n_obs"].values[:length], dtype=float),
            "n_off": (
                np.asarray(row["n_off_obs"].values[:length], dtype=float)
                if include_background else np.zeros(length)
            ),
            "D": np.asarray(row["D"].values[:length, :length], dtype=float),
            "G_g": np.asarray(row["G_g"].values[:length, :length], dtype=float),
            "eta_true": np.asarray(row["eta_true"].values[:length], dtype=float),
            "include_background": include_background,
        }


def run_row(path: Path, ex: float, config: dict) -> xr.Dataset:
    data = read_row(path, ex)
    Eg, G_g = data["Eg"], data["G_g"]
    iterations = int(config["freq_iterations"])
    samples = int(config["freq_bootstrap"])
    seed = int(config["freq_bootstrap_seed"])

    n = om.Vector(X=Eg, values=data["n"], dtype=np.float32)
    D_op = om.Matrix(X=Eg, Y=Eg, values=data["D"], dtype=np.float32)
    G_g_op = om.Matrix(X=Eg, Y=Eg, values=G_g, dtype=np.float32)
    rmle = RMLE(D_eg=D_op, G_eg=G_g_op)
    background = BackgroundModel(
        backgrounds=(data["n_off"].astype(np.float32),)
        if data["include_background"] else (),
        do_fold=False,
    )

    print(f"RMLE 1D, no penalty, Ex={data['Ex']:.6g} keV, Eg bins={Eg.size}, iterations={iterations}, bootstrap={samples}", flush=True)
    result = rmle.unfold(
        n,
        background=background,
        initial=n.clone(values=np.maximum(n.values, 0.1)),
        mask=np.ones(Eg.size, dtype=bool),
        loss=ModelLoss(),
        iterations=iterations,
        leave_tqdm=False,
    )
    x_hat = np.asarray(result.best().values, dtype=float)
    eta_hat = x_hat @ G_g

    print("Fit complete. Starting bootstrap.", flush=True)
    np.random.seed(seed)
    boot = result.resample(samples, base="nu", bootstrap_background=True, background_base="beta")
    eta_boot = np.asarray(boot.ubox, dtype=float) @ G_g
    

    vectors = dict(
        n_obs=data["n"], n_off_obs=data["n_off"], x_hat=x_hat,
        eta_hat=eta_hat, eta_true=data["eta_true"]
    )
    output = xr.Dataset(
        {name: ("Eg", values) for name, values in vectors.items()},
        coords={"Eg": Eg, "bootstrap": np.arange(samples),
                "iteration": np.arange(len(result.cost))},
        attrs=dict(bootstrap_seed=seed, penalty="none", iterations=iterations),
    )
    output["eta_boot"] = (("bootstrap", "Eg"), eta_boot)
    output["loss_history"] = ("iteration", np.asarray(result.cost))
    output["bootstrap_end_losses"] = ("bootstrap", np.asarray(boot.costs)[:, -1])
    output["Ex_req"] = ex
    output["Eg_len"] = len(Eg)
    output["Eg"].attrs["units"] = "keV"
    return output.expand_dims(Ex=[data["Ex"]])


def run(args: argparse.Namespace) -> None:
    config = load_run_config(repo_path(args.config))
    bayes_nc = repo_path(args.bayes_nc)
    rows = [run_row(bayes_nc, float(ex), config) for ex in config["ex_energies"]]
    
    output = xr.concat(rows, dim="Ex", join="outer", combine_attrs="override")
    output["Ex"].attrs["units"] = "keV"
    output["Ex_req"].attrs["units"] = "keV"
    
    path = repo_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(path, engine="h5netcdf", encoding={ name: {"zlib": True, "complevel": 4} for name in output.data_vars})
    print(f"Saved {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayes-nc", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="One results.nc file for all requested Ex rows.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
