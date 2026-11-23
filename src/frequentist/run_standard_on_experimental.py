"""
Run standard iterative Oslo unfolding with Compton subtraction on the same experimental inputs as the Bayesian method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd


iterations = 40
resolution_scale = 0.1
fwhm_tweaks = {"fe": 1.0, "se": 1.1, "de": 1.3, "511": 0.9}
table_columns = ["E", "fwhm_abs", "eff_tot", "pFE", "pSE", "pDE", "p511"]


def prepare(args):
    from ompy import Vector
    from ompy.response import Response
    from ompy.response.responsedata import Components
    from ompy.units import Quantity
    
    with xr.open_dataset(args.bayes_nc, engine="h5netcdf") as ds:
        output = ds[["Ex_req", "Eg_len", "n_obs", "n_off_obs"]].load().copy(deep=True)
        output.attrs = {key: ds.attrs[key] for key in ("response_db", "fwhm_eg_unfold")}
        
        Ex, Eg = output["Ex"].values, output["Eg"].values
        output["Eg_mid"] = ("Eg", Eg + 0.5 * (Eg[1] - Eg[0]))
        
        for name in table_columns[1:]:
            output[name] = (("Ex", "Eg"), np.full((len(Ex), len(Eg)), np.nan))
            
        output = output.assign_coords(Eg_in=Eg, Eg_out=Eg)
        output["R_unfold"] = (("Ex", "Eg_in", "Eg_out"), np.full((len(Ex), len(Eg), len(Eg)), np.nan))
        fwhm = ds.attrs["fwhm_eg_unfold"]
        output.attrs["resolution_scale"] = resolution_scale
        response = Response.from_db(ds.attrs["response_db"]).normalize_FWHM("1330keV", fwhm)
        reduced = response.normalize_FWHM("1330keV", Quantity(fwhm) * resolution_scale)

        for i, length in enumerate(output["Eg_len"].values):
            length = int(length)
            row = ds.isel(Ex=i)
            E = Eg[:length]
            on = np.asarray(row["n_obs"].values[:length], dtype=float)
            off = np.asarray(row["n_off_obs"].values[:length], dtype=float)
            D = np.asarray(row["D"].values[:length, :length], dtype=float)
            G_g = np.asarray(row["G_g"].values[:length, :length], dtype=float)
            spectrum = Vector(X=E, values=on, unit="keV", edge="left")

            unnormalized = response.discrete_like(spectrum, normalize=False).values
            norm = unnormalized.sum(axis=1)
            
            
            if np.any(norm <= 0):
                raise ValueError("Detector response contains rows with zero normalization")

            np.testing.assert_allclose(unnormalized / norm[:, None], D, rtol=1e-5, atol=1e-7)

            np.testing.assert_allclose(response.gaussian_like(spectrum).values, G_g, rtol=1e-5, atol=1e-7)

            probabilities = []
            for name in ("FE", "SE", "DE", "AP"):
                weights = dict(FE=0.0, SE=0.0, DE=0.0, AP=0.0, compton=0.0)
                weights[name] = getattr(response.components, name)
                component = response.discrete_like(
                    spectrum, components=Components(**weights), normalize=False,
                ).values
                probabilities.append(component.sum(axis=1) / norm)

            E_mid = np.asarray(spectrum.X_index.to_mid().bins, dtype=float)
            fwhm_bins = reduced.interpolation.FWHM(
                E_mid + 2.0 * (E_mid[1] - E_mid[0])
            )
            output["R_unfold"].values[i, :length, :length] = D @ reduced.gaussian_like(spectrum).values
            for name, values in zip(table_columns[1:], [fwhm_bins, np.ones(length), *probabilities]):
                output[name].values[i, :length] = values

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(args.out, engine="h5netcdf", encoding={
        name: {"zlib": True, "complevel": 4} for name in output.data_vars
    })
    print(f"Saved {args.out}")


def run(args):
    
    import ompy.unfolder as om_unfolder
    from ompy.matrix import Matrix
    smooth = om_unfolder.gauss_smoothing_matrix_1D

    def writable_smoothing(matrix, energy, fwhm, **kwargs):
        return smooth(
            np.array(matrix, dtype=float, order="C", copy=True),
            np.array(energy, dtype=float, order="C", copy=True),
            np.array(fwhm, dtype=float, order="C", copy=True),
            **kwargs,
        )

    om_unfolder.gauss_smoothing_matrix_1D = writable_smoothing
    try:
        with xr.open_dataset(args.input, engine="h5netcdf") as data:
            output = data[["Ex_req", "Eg_len"]].load().copy(deep=True).assign_coords(Eg=data["Eg"].values)
            output.attrs.update(input_nc=str(args.input), iterations=iterations)
            
            eta = np.full((data.sizes["Ex"], data.sizes["Eg"]), np.nan)
        
            selected = np.empty(data.sizes["Ex"], dtype=int)
            
            for i, ex in enumerate(data["Ex"].values):
                length = int(data["Eg_len"].values[i])
                row = data.isel(Ex=i, Eg=slice(0, length))
                E_mid = np.array(row["Eg_mid"].values, dtype=float, copy=True)
                
                table = pd.DataFrame({"E": E_mid, **{name: np.array(row[name].values, copy=True) for name in table_columns[1:]}})
                
                signal = np.maximum(row["n_obs"].values - row["n_off_obs"].values, 0.0)
                
                raw = Matrix(values=signal[None, :], Eg=E_mid, Ex=np.array([ex]), state="raw")
                R = Matrix(values=data["R_unfold"].values[i, :length, :length].copy(),
                           Eg=E_mid, Ex=E_mid)

                unfolder = om_unfolder.Unfolder(num_iter=iterations, response=R)
                unfolder.response_tab = table
                unfolder.use_compton_subtraction = True
                unfolder.FWHM_tweak_multiplier = fwhm_tweaks.copy()
                unfolded = unfolder(raw)

                eta[i, :len(signal)] = unfolded.values[0]

    
                selected[i] = unfolder.iscores[0]
                print(f"Ex={ex:.2f} keV, input={signal.sum():.6g}, unfolded={unfolded.values.sum():.6g}, iteration, index={selected[i]}")
    finally:
        om_unfolder.gauss_smoothing_matrix_1D = smooth

    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    output["eta_hat"] = (("Ex", "Eg"), eta)

    output["selected_iteration_index"] = ("Ex", selected)
    output.to_netcdf(args.out, engine="h5netcdf", encoding={name: {"zlib": True, "complevel": 4} for name in output.data_vars})
    print(f"Saved {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    
    p = commands.add_parser("prepare", help="Use the OMpy shapedev branch.")
    p.add_argument("--bayes-nc", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(function=prepare)
    p = commands.add_parser("run", help="Use the OMpy master branch.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(function=run)
    
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
