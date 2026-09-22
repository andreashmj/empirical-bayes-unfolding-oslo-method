from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..config_utils import load_run_config
from ..experimental_data import ExperimentalDataLoader
from ..paths import repo_path


DEFAULT_SPARSITY_THRESHOLD = 0.1
DEFAULT_SPARSITY_SMOOTHING = 100.0


def cfg(config: dict, key: str, default: Any = None) -> Any:
    return config[key] if key in config else default


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def vector_values(vector: Any) -> np.ndarray:
    if hasattr(vector, "values"):
        return np.asarray(vector.values, dtype=float)
    return np.asarray(vector, dtype=float)


def maybe_float32(values: Any) -> Any:
    if hasattr(values, "astype"):
        try:
            return values.astype("float32")
        except Exception:
            return values
    return values


@contextmanager
def numpy_seed(seed: int | None):
    if seed is None:
        yield
        return

    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        yield
    finally:
        np.random.set_state(state)


def selected_ex_values(config: dict, cli_values: list[float] | None) -> list[float]:
    if cli_values:
        return [float(value) for value in cli_values]

    values = config.get("ex_energies")
    if values is None:
        raise ValueError("The experimental config must contain ex_energies.")
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def build_loader(config: dict) -> ExperimentalDataLoader:
    return ExperimentalDataLoader(
        prompt_path=config["prompt_path"],
        background_path=config.get("background_path"),
        response_db=str(cfg(config, "response_db", "OSCAR2020")),
        rebin_factors=tuple(cfg(config, "rebin_factors", (4, 1))),
        lower_eg_cut=str(cfg(config, "lower_eg_cut", "200keV")),
        include_background=bool(cfg(config, "include_background", True)),
        sigma_eg=str(cfg(config, "sigma_eg_unfold", cfg(config, "sigma_eg_gen", "30keV"))),
        eg_margin_kev=float(cfg(config, "eg_margin_kev", 200.0)),
    )


def unfolding_response(config: dict, loader: ExperimentalDataLoader):
    sigma = str(cfg(config, "sigma_eg_unfold", loader.resp_norm_sigma))
    if sigma == str(loader.resp_norm_sigma):
        return loader.resp

    import ompy as om

    return (
        om.response.Response.from_db(str(cfg(config, "response_db", "OSCAR2020")))
        .normalize_sigma(loader.resp_norm_anchor, sigma)
    )


def import_rmle():
    try:
        from ompy.unfolding.rmle import RMLE
    except Exception:
        from ompy.unfolding.rmle.rmle import RMLE

    from ompy.unfolding.rmle.lossmodel import ModelLoss
    from ompy.unfolding.rmle.penalty import Sparsity
    from ompy.unfolding.rmle.rmle1d import BackgroundModel
    from ompy.unfolding.resampling.confidence import make_ci

    return RMLE, BackgroundModel, ModelLoss, Sparsity, make_ci


def ex_index(loader: ExperimentalDataLoader, ex_kev: float) -> int:
    try:
        return int(loader.n.index_X(ex_kev))
    except Exception:
        ex_axis = np.asarray(loader.Ex, dtype=float)
        return int(np.argmin(np.abs(ex_axis - float(ex_kev))))


def prepare_slice(loader: ExperimentalDataLoader, response_unfold: Any, ex_kev: float) -> dict:
    index = ex_index(loader, ex_kev)
    active = loader.eg_slice_for_index(index)
    n_bins = int(active.stop)

    n_cut = loader.n.iloc[index, active]
    n_off_cut = loader.n_off.iloc[index, active]
    D_cut, G_cut = response_unfold.specialize_like(n_cut)

    D = np.asarray(D_cut.values, dtype=float)
    G = np.asarray(G_cut.values, dtype=float)

    return {
        "ex_requested": float(ex_kev),
        "ex_actual": float(np.asarray(loader.Ex, dtype=float)[index]),
        "n_vec": maybe_float32(n_cut),
        "raw": vector_values(n_cut),
        "background": vector_values(n_off_cut),
        "D_use": D_cut,
        "G_use": G_cut,
        "D": D,
        "G": G,
        "R": D @ G,
        "Eg_keV": np.asarray(loader.Eg[:n_bins], dtype=float),
    }


def loss_model(
    ModelLoss: Any,
    Sparsity: Any,
    penalty: str,
    alpha: float | None,
    threshold: float,
    smoothing: float,
):
    penalty = str(penalty).strip().lower()
    if penalty == "none":
        return ModelLoss()
    if penalty != "sparsity":
        raise ValueError("freq_penalty must be 'none' or 'sparsity'.")
    if alpha is None:
        raise ValueError("Set freq_sparsity_alpha for experimental sparsity RMLE.")

    return ModelLoss(
        penalty=Sparsity(
            alpha=float(alpha),
            threshold=float(threshold),
            smoothing=float(smoothing),
            target="mu_normalized",
        )
    )


def run_rmle(
    prepared: dict,
    include_background: bool,
    iterations: int,
    mask: Any,
    initial: Any,
    penalty: str,
    sparsity_alpha: float | None,
    sparsity_threshold: float,
    sparsity_smoothing: float,
) -> dict:
    RMLE, BackgroundModel, ModelLoss, Sparsity, make_ci = import_rmle()

    if include_background:
        background_model = BackgroundModel(backgrounds=(prepared["background"],), do_fold=False)
    else:
        background_model = BackgroundModel(backgrounds=(), do_fold=False)

    rmle = RMLE(D_eg=prepared["D_use"], G_eg=prepared["G_use"])
    result = rmle.unfold(
        prepared["n_vec"],
        background=background_model,
        iterations=int(iterations),
        mask=mask,
        initial=initial,
        loss=loss_model(
            ModelLoss,
            Sparsity,
            penalty,
            sparsity_alpha,
            sparsity_threshold,
            sparsity_smoothing,
        ),
    )

    return {
        "result": result,
        "make_ci": make_ci,
        "mu_hat": vector_values(result.best()),
        "eta_hat": vector_values(result.best_eta()),
        "nu_hat": vector_values(result.best_folded()),
        "total_hat": vector_values(result.folded_total()),
    }


def ci_center(box: np.ndarray, summary: str) -> np.ndarray:
    summary = str(summary).strip().lower()
    if summary == "mean":
        return np.mean(box, axis=0)
    if summary == "median":
        return np.median(box, axis=0)
    raise ValueError("freq_ci_summary must be 'mean' or 'median'.")


def ci_arrays(
    result: Any,
    run: dict,
    n_bootstrap: int,
    seed: int | None,
    alpha: float,
    method: str,
    summary: str,
    make_ci: Any,
) -> dict[str, np.ndarray]:
    if int(n_bootstrap) <= 0:
        return {}

    with numpy_seed(seed):
        boot = result.resample(int(n_bootstrap))

    arrays: dict[str, np.ndarray] = {}
    for name, box, original in [
        ("mu", np.asarray(boot.ubox, dtype=float), run["mu_hat"]),
        ("eta", np.asarray(boot.etabox, dtype=float), run["eta_hat"]),
        ("nu", np.asarray(boot.nubox, dtype=float), run["nu_hat"]),
    ]:
        center = ci_center(box, summary)
        try:
            lo, hi = make_ci(box, original=original, alpha=float(alpha), method=method)
        except Exception:
            lo, hi = make_ci(box, original=original, alpha=float(alpha), method="standard")
        arrays[f"ci_{name}_center"] = np.asarray(center, dtype=float)
        arrays[f"ci_{name}_lo"] = np.asarray(lo, dtype=float)
        arrays[f"ci_{name}_hi"] = np.asarray(hi, dtype=float)

    return arrays


def save_summary(path: Path, prepared: dict, arrays: dict[str, np.ndarray]) -> None:
    ensure_dir(path)
    payload = {
        "ex_requested_keV": np.asarray(prepared["ex_requested"]),
        "ex_actual_keV": np.asarray(prepared["ex_actual"]),
        "Eg_keV": np.asarray(prepared["Eg_keV"], dtype=float),
        "raw": np.asarray(prepared["raw"], dtype=float),
        "bg_off": np.asarray(prepared["background"], dtype=float),
    }
    payload.update({key: np.asarray(value) for key, value in arrays.items()})
    np.savez(path / "summary.npz", **payload)


def run_rmle_one(
    config: dict,
    loader: ExperimentalDataLoader,
    response_unfold: Any,
    ex_kev: float,
    out_root: Path,
) -> None:
    prepared = prepare_slice(loader, response_unfold, ex_kev)
    include_background = bool(cfg(config, "include_background", True))

    run = run_rmle(
        prepared=prepared,
        include_background=include_background,
        iterations=int(cfg(config, "freq_iterations", 10000)),
        mask=cfg(config, "freq_mask", "last nonzero"),
        initial=cfg(config, "freq_initial", "raw"),
        penalty=str(cfg(config, "freq_penalty", "none")),
        sparsity_alpha=(
            None
            if cfg(config, "freq_sparsity_alpha", None) is None
            else float(cfg(config, "freq_sparsity_alpha"))
        ),
        sparsity_threshold=float(cfg(config, "freq_sparsity_threshold", DEFAULT_SPARSITY_THRESHOLD)),
        sparsity_smoothing=float(cfg(config, "freq_sparsity_smoothing", DEFAULT_SPARSITY_SMOOTHING)),
    )

    ci = ci_arrays(
        result=run["result"],
        run=run,
        n_bootstrap=int(cfg(config, "freq_bootstrap", 1000)),
        seed=int(cfg(config, "freq_bootstrap_seed", 123)),
        alpha=float(cfg(config, "freq_ci_alpha", 0.05)),
        method=str(cfg(config, "freq_ci_method", "bonferroni percentile")),
        summary=str(cfg(config, "freq_ci_summary", "mean")),
        make_ci=run["make_ci"],
    )

    out_dir = ensure_dir(out_root / f"Ex{round(ex_kev)}")
    save_summary(
        out_dir,
        prepared,
        {
            "mu_hat": run["mu_hat"],
            "eta_hat": run["eta_hat"],
            "nu_hat": run["nu_hat"],
            "total_hat": run["total_hat"],
            **ci,
        },
    )
    print(f"[RMLE] Ex={prepared['ex_actual']:.0f} keV -> {out_dir}")


def iterative_unfold(raw: np.ndarray, background: np.ndarray, R: np.ndarray, n_iter: int) -> np.ndarray:
    target = np.clip(np.asarray(raw, dtype=float) - np.asarray(background, dtype=float), 0.0, None)
    R = np.asarray(R, dtype=float)
    sensitivity = np.maximum(np.sum(R, axis=1), 1.0e-12)

    mu = np.maximum(target, 1.0e-12)
    if float(np.sum(mu)) <= 0.0:
        mu = np.ones_like(target, dtype=float)

    for _ in range(int(n_iter)):
        pred = mu @ R + 1.0e-12
        ratio = target / pred
        mu *= (R @ ratio) / sensitivity
        mu = np.maximum(mu, 1.0e-12)

    return mu


def run_iterative_one(
    config: dict,
    loader: ExperimentalDataLoader,
    response_unfold: Any,
    ex_kev: float,
    out_root: Path,
) -> None:
    prepared = prepare_slice(loader, response_unfold, ex_kev)
    mu = iterative_unfold(
        prepared["raw"],
        prepared["background"],
        prepared["R"],
        int(cfg(config, "iterative_iterations", 200)),
    )
    eta = mu @ prepared["G"]
    nu = mu @ prepared["R"]

    out_dir = ensure_dir(out_root / f"Ex{round(ex_kev)}")
    save_summary(
        out_dir,
        prepared,
        {
            "mu_hat": mu,
            "eta_hat": eta,
            "nu_hat": nu,
            "total_hat": nu + np.clip(prepared["background"], 0.0, None),
        },
    )
    print(f"[iterative] Ex={prepared['ex_actual']:.0f} keV -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experimental RMLE and iterative unfolding.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["rmle", "iterative", "both"], default="both")
    parser.add_argument("--ex", nargs="*", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_run_config(repo_path(args.config))

    output_dir = repo_path(cfg(config, "output_dir", "results"))
    dataset_id = str(cfg(config, "dataset_id", "experimental/28Si"))
    ex_values = selected_ex_values(config, args.ex)

    loader = build_loader(config)
    response_unfold = unfolding_response(config, loader)

    if args.method in {"rmle", "both"}:
        rmle_root = ensure_dir(output_dir / dataset_id / "rmle")
        for ex_kev in ex_values:
            run_rmle_one(config, loader, response_unfold, float(ex_kev), rmle_root)

    if args.method in {"iterative", "both"}:
        iterative_root = ensure_dir(output_dir / dataset_id / "ompy_iterative")
        for ex_kev in ex_values:
            run_iterative_one(config, loader, response_unfold, float(ex_kev), iterative_root)


if __name__ == "__main__":
    main()
