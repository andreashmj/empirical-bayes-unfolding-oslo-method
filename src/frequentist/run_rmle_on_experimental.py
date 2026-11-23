"""
Run 1D RMLE on experimental data.

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



def read_row(ds: xr.Dataset, ex: float) -> dict:
    row = ds.swap_dims({"Ex": "Ex_req"}).sel(Ex_req=ex)
    length = int(row["Eg_len"].item())
    Eg = np.asarray(row["Eg"].values[:length], dtype=float)
    raw = np.asarray(row["n_obs"].values[:length], dtype=float)
    off = np.asarray(row["n_off_obs"].values[:length], dtype=float)
    D = np.asarray(row["D"].values[:length, :length], dtype=float)
    G_g = np.asarray(row["G_g"].values[:length, :length], dtype=float)
    return dict(
        ex_requested=ex, ex_actual=float(row["Ex"].item()), Eg=Eg,
        raw=raw, off=off,
        n=om.Vector(X=Eg, values=raw, dtype=np.float32),
        D=om.Matrix(X=Eg, Y=Eg, values=D, dtype=np.float64),
        G_g=om.Matrix(X=Eg, Y=Eg, values=G_g, dtype=np.float64),
    )


def run_rmle(data: dict, config: dict) -> xr.Dataset:
    n, G_g = data["n"], np.asarray(data["G_g"].values, dtype=float)
    background = BackgroundModel(
        backgrounds=(data["off"].astype(np.float32),) if config["include_background"] else (),
        do_fold=False,
    )
    samples = int(config["freq_bootstrap"])
    print(f"RMLE 1D: Ex={data['ex_actual']:g} keV, bootstrap={samples}", flush=True)
    result = RMLE(D_eg=data["D"], G_eg=data["G_g"]).unfold(
        n, 
        background=background, 
        loss=ModelLoss(), 
        iterations=int(config["freq_iterations"]),
        initial=n.clone(values=np.maximum(n.values, 0.1)),
        mask=np.ones(len(data["Eg"]), 
        dtype=bool)
    )
    
    x_hat = np.asarray(result.best().values, dtype=float)
    eta_hat = x_hat @ G_g
    
    seed = int(config["freq_bootstrap_seed"])
    np.random.seed(seed)
    
    boot = result.resample( samples, base="nu", bootstrap_background=True, background_base="beta")
    eta_boot = np.asarray(boot.ubox, dtype=float) @ G_g
    
    R = data["D"].values @ G_g
    nu_hat = x_hat @ R
    nu_boot = np.asarray(boot.ubox, dtype=float) @ R
    
    bg_hat = (np.asarray(result.best_beta().values, dtype=float) if config["include_background"] else np.zeros_like(x_hat))
    bg_boot = (np.stack([np.asarray(beta.values, dtype=float) for beta in boot.betas]) if config["include_background"] else np.zeros_like(nu_boot))
    
    
    vectors = dict(n_obs=data["raw"], n_off_obs=data["off"], x_hat=x_hat, eta_hat=eta_hat, nu_hat=nu_hat, bg_hat=bg_hat)
    
    ds = xr.Dataset(
        {name: ("Eg", values) for name, values in vectors.items()},
        coords={"Eg": data["Eg"], "bootstrap": np.arange(samples),
                "iteration": np.arange(len(result.cost))},
        attrs=dict(bootstrap_seed=seed, penalty="none", iterations=int(config["freq_iterations"])))
    ds["eta_boot"] = (("bootstrap", "Eg"), eta_boot)
    ds["nu_boot"] = (("bootstrap", "Eg"), nu_boot)
    ds["bg_boot"] = (("bootstrap", "Eg"), bg_boot)
    ds["loss_history"] = ("iteration", np.asarray(result.cost))
    ds["Ex_req"] = data["ex_requested"]
    ds["Eg_len"] = len(data["Eg"])
    ds["Eg"].attrs["units"] = "keV"
    
    return ds.expand_dims(Ex=[data["ex_actual"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayes-nc", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="One results.nc file for all selected Ex rows.")
    args = parser.parse_args()
    config = load_run_config(repo_path(args.config))

    rows = []
    with xr.open_dataset(repo_path(args.bayes_nc), engine="h5netcdf") as ds:
        for energy in config["ex_energies"]:
            ex = float(energy)
            data = read_row(ds, ex)
            rows.append(run_rmle(data, config))
        response_attrs = {key: ds.attrs[key] for key in
                          ("response_db",  "fwhm_eg_unfold")}

    output = xr.concat(rows, dim="Ex", join="outer", combine_attrs="override")
    output.attrs.update(response_attrs, bayes_nc=str(args.bayes_nc))
    output["Ex"].attrs["units"] = "keV"
    output["Ex_req"].attrs["units"] = "keV"
    path = repo_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(path, engine="h5netcdf", encoding={
        name: {"zlib": True, "complevel": 4} for name in output.data_vars
    })
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
