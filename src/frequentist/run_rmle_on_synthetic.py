"""
Run the paper RMLE comparison on synthetic one-dimensional Ex slices.

The frequentist comparison is intentionally narrow. It runs the OMpy RMLE
implementation with the configured response, ON/OFF background input, and
bootstrap confidence bands. The only optional regularization used for the paper
comparison is the sparsity penalty with W1 selection of the penalty strength;
otherwise the standard unpenalized RMLE loss is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config_utils import load_run_config
from ..paths import repo_path
from ..synthetic_data import SyntheticDataLoader


DEFAULT_SPARSITY_ALPHA_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
DEFAULT_SPARSITY_THRESHOLD = 0.1
DEFAULT_SPARSITY_SMOOTHING = 100.0


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_value(config: dict, key: str, default):
    """Return a configuration value or a default."""

    return config[key] if key in config else default


def vector_values(vector: Any) -> np.ndarray:
    """Return numeric values from an OMpy object or array-like value."""

    if hasattr(vector, "values"):
        return np.asarray(vector.values, dtype=float)

    return np.asarray(vector, dtype=float)


def maybe_float32(values: Any) -> Any:
    """Cast OMpy arrays to float32 when the object supports it."""

    if hasattr(values, "astype"):
        try:
            return values.astype("float32")
        except Exception:
            return values

    return values


@contextmanager
def numpy_random_seed(seed: int | None):
    """Temporarily seed NumPy's legacy RNG around OMpy bootstrap resampling.

    OMpy's RMLE bootstrap path uses 'np.random' internally and does not expose
    a seed argument at the public resampling call. The wrapper therefore seeds
    NumPy around the resampling call and restores the previous state afterwards.
    """

    if seed is None:
        yield
        return

    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        yield
    finally:
        np.random.set_state(state)


def parse_ex_values(config: dict, cli_ex: list[float] | None) -> list[float]:
    """Return requested excitation-energy slices from CLI or run config."""

    if cli_ex:
        return [float(value) for value in cli_ex]

    values = config.get("ex_energies")
    if values is not None:
        if isinstance(values, (list, tuple)):
            return [float(value) for value in values]
        return [float(values)]

    if "ex_start_energy" in config and "ex_end_energy" in config:
        start = float(config["ex_start_energy"])
        stop = float(config["ex_end_energy"])
        if start > stop:
            raise ValueError("ex_start_energy must be <= ex_end_energy.")
        return [start, stop]

    return [4000.0]


def build_synthetic_loader(config: dict) -> SyntheticDataLoader:
    """Build the synthetic-data loader from the run configuration."""

    return SyntheticDataLoader(
        mat_path=config["mat_path"],
        response_db=config_value(config, "response_db", "OSCAR2020"),
        rebin_factors=tuple(config_value(config, "rebin_factors", (10, 2))),
        mat_scale=float(config_value(config, "mat_scale", 1.0)),
        lower_eg_cut=str(config_value(config, "lower_eg_cut", "60keV")),
        include_background=bool(config_value(config, "include_background", True)),
        rng_seed=int(config_value(config, "rng_seed", 100)),
        sigma_eg=str(config_value(config, "sigma_eg_gen", "30keV")),
        bg_fraction=float(config_value(config, "bg_fraction", 0.15)),
        bg_flat_fraction=float(config_value(config, "bg_flat_fraction", 0.10)),
        eg_tail_mass=float(config_value(config, "eg_tail_mass", 1.0e-6)),
        include_ex_smearing=bool(config_value(config, "include_ex_smearing", False)),
        fwhm_ex_kev=float(config_value(config, "fwhm_ex_kev", 150.0)),
    )


def unfolding_response(config: dict, loader: SyntheticDataLoader):
    """Return the detector response used for the RMLE unfolding."""

    sigma_eg_unfold = str(
        config_value(config, "sigma_eg_unfold", loader.resp_norm_sigma)
    )

    if sigma_eg_unfold == str(loader.resp_norm_sigma):
        return loader.resp

    import ompy as om

    return (
        om.response.Response.from_db(str(config_value(config, "response_db", "OSCAR2020")))
        .normalize_sigma(loader.resp_norm_anchor, sigma_eg_unfold)
    )


def import_ompy_rmle():
    """Import the OMpy RMLE classes used in the paper comparison."""

    try:
        from ompy.unfolding.rmle import RMLE
    except Exception:
        from ompy.unfolding.rmle.rmle import RMLE

    from ompy.unfolding.rmle.rmle1d import BackgroundModel
    from ompy.unfolding.rmle.lossmodel import ModelLoss
    from ompy.unfolding.rmle.penalty import Sparsity
    from ompy.unfolding.resampling.confidence import make_ci

    return RMLE, BackgroundModel, ModelLoss, Sparsity, make_ci


def resolve_ex_index(loader: SyntheticDataLoader, ex_keV: float) -> int:
    """Return the nearest loader row index for one requested Ex value."""

    try:
        return int(loader.n.index_X(ex_keV))
    except Exception:
        ex_axis = np.asarray(loader.Ex, dtype=float)
        return int(np.argmin(np.abs(ex_axis - ex_keV)))


def prepare_ex_slice(
    loader: SyntheticDataLoader,
    response_unfold: Any,
    ex_keV: float,
) -> dict:
    """Prepare one active excitation-energy slice for OMpy RMLE."""

    ex_index = resolve_ex_index(loader, ex_keV)
    ex_axis = np.asarray(loader.Ex, dtype=float)
    ex_actual = float(ex_axis[ex_index])

    active_eg_slice = loader.eg_slice_for_index(ex_index)
    n_bins = int(active_eg_slice.stop)

    n_cut = loader.n.iloc[ex_index, active_eg_slice]
    D_cut, G_cut = response_unfold.specialize_like(n_cut)

    if loader.n_off is None:
        background_observed = np.zeros(n_bins, dtype=float)
    else:
        background_observed = vector_values(
            loader.n_off.iloc[ex_index, active_eg_slice]
        )

    if loader.background_expected is None:
        background_expected = np.zeros(n_bins, dtype=float)
    else:
        background_expected = vector_values(
            loader.background_expected.iloc[ex_index, active_eg_slice]
        )

    x_true = vector_values(loader.x_true.iloc[ex_index, active_eg_slice])

    D_values = np.asarray(D_cut.values, dtype=float)
    G_values = np.asarray(G_cut.values, dtype=float)
    response_matrix = D_values @ G_values

    return {
        "ex_index": ex_index,
        "ex_actual": ex_actual,
        "active_eg_length": n_bins,
        "n_vec": maybe_float32(n_cut),
        "background_observed": maybe_float32(background_observed),
        "background_expected": background_expected,
        "D_use": D_cut,
        "G_use": G_cut,
        "x_true": x_true,
        "eta_true": x_true @ G_values,
        "nu_true": x_true @ response_matrix,
        "eg_axis": np.asarray(loader.Eg[:n_bins], dtype=float),
    }


def normalised_w1(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> float:
    """Return discrete W1 distance after normalizing two non-negative spectra."""

    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    y = np.clip(np.asarray(y, dtype=float), 0.0, None)
    grid = np.asarray(grid, dtype=float)

    n = min(x.size, y.size, grid.size)
    x = x[:n]
    y = y[:n]
    grid = grid[:n]

    x_sum = float(np.sum(x))
    y_sum = float(np.sum(y))

    if x_sum <= 0.0 or y_sum <= 0.0:
        return float("inf")

    x_cdf = np.cumsum(x / x_sum)
    y_cdf = np.cumsum(y / y_sum)

    if n == 1:
        return float(abs(x_cdf[0] - y_cdf[0]))

    spacing = np.diff(grid)
    spacing = np.where(np.isfinite(spacing), spacing, 0.0)
    return float(np.sum(np.abs(x_cdf[:-1] - y_cdf[:-1]) * spacing))


def build_loss_model(
    ModelLoss: Any,
    Sparsity: Any,
    penalty: str,
    sparsity_alpha: float | None,
    sparsity_threshold: float,
    sparsity_smoothing: float,
):
    """Build the OMpy loss model for the paper RMLE comparison."""

    penalty = str(penalty).strip().lower()

    if penalty == "none":
        return ModelLoss(), {"penalty": "none"}

    if penalty != "sparsity":
        raise ValueError("freq_penalty must be either 'none' or 'sparsity'.")

    if sparsity_alpha is None:
        raise ValueError("sparsity_alpha is required when freq_penalty='sparsity'.")

    penalty_term = Sparsity(
        alpha=float(sparsity_alpha),
        threshold=float(sparsity_threshold),
        smoothing=float(sparsity_smoothing),
        target="mu_normalized",
    )
    return ModelLoss(penalty=penalty_term), {
        "penalty": "sparsity",
        "alpha_selection": "w1",
        "sparsity_alpha": float(sparsity_alpha),
        "sparsity_threshold": float(sparsity_threshold),
        "sparsity_smoothing": float(sparsity_smoothing),
        "sparsity_target": "mu_normalized",
    }


def run_rmle_fit(
    RMLE: Any,
    BackgroundModel: Any,
    ModelLoss: Any,
    Sparsity: Any,
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
    """Run one OMpy RMLE fit."""

    if include_background:
        background_model = BackgroundModel(
            backgrounds=(prepared["background_observed"],),
            do_fold=False,
        )
    else:
        background_model = BackgroundModel(backgrounds=(), do_fold=False)

    loss_model, loss_meta = build_loss_model(
        ModelLoss=ModelLoss,
        Sparsity=Sparsity,
        penalty=penalty,
        sparsity_alpha=sparsity_alpha,
        sparsity_threshold=sparsity_threshold,
        sparsity_smoothing=sparsity_smoothing,
    )

    rmle = RMLE(D_eg=prepared["D_use"], G_eg=prepared["G_use"])
    result = rmle.unfold(
        prepared["n_vec"],
        background=background_model,
        iterations=int(iterations),
        mask=mask,
        initial=initial,
        loss=loss_model,
    )

    beta_hat = None
    if hasattr(result, "best_beta"):
        beta = result.best_beta()
        if beta is not None:
            beta_hat = vector_values(beta)

    return {
        "result": result,
        "mu_hat": vector_values(result.best()),
        "eta_hat": vector_values(result.best_eta()),
        "nu_hat": vector_values(result.best_folded()),
        "total_hat": vector_values(result.folded_total()),
        "beta_hat": beta_hat,
        "loss_meta": loss_meta,
    }


def select_sparsity_alpha_by_w1(
    RMLE: Any,
    BackgroundModel: Any,
    ModelLoss: Any,
    Sparsity: Any,
    prepared: dict,
    include_background: bool,
    iterations: int,
    mask: Any,
    initial: Any,
    alpha_grid: Sequence[float],
    sparsity_threshold: float,
    sparsity_smoothing: float,
) -> tuple[float, list[dict[str, float]]]:
    """Select sparsity alpha by W1 distance to the synthetic eta truth."""

    scores: list[dict[str, float]] = []

    for alpha in alpha_grid:
        run = run_rmle_fit(
            RMLE=RMLE,
            BackgroundModel=BackgroundModel,
            ModelLoss=ModelLoss,
            Sparsity=Sparsity,
            prepared=prepared,
            include_background=include_background,
            iterations=iterations,
            mask=mask,
            initial=initial,
            penalty="sparsity",
            sparsity_alpha=float(alpha),
            sparsity_threshold=sparsity_threshold,
            sparsity_smoothing=sparsity_smoothing,
        )
        score = normalised_w1(
            run["eta_hat"],
            prepared["eta_true"],
            prepared["eg_axis"],
        )
        scores.append({"alpha": float(alpha), "w1_eta": float(score)})

    best = min(scores, key=lambda row: row["w1_eta"])
    return float(best["alpha"]), scores


def ci_center(box: np.ndarray, summary: str) -> np.ndarray:
    """Return the bootstrap center curve."""

    summary = str(summary).strip().lower()
    if summary == "mean":
        return np.mean(box, axis=0)
    if summary == "median":
        return np.median(box, axis=0)
    raise ValueError("freq_ci_summary must be 'mean' or 'median'.")


def make_confidence_interval(
    box: np.ndarray,
    original: np.ndarray,
    alpha: float,
    method: str,
    summary: str,
    make_ci: Any,
):
    """Return center, lower, upper, and effective confidence-band method."""

    center = ci_center(box, summary)

    try:
        lower, upper = make_ci(
            box,
            original=original,
            alpha=float(alpha),
            method=method,
        )
        return center, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), method
    except Exception as error:
        fallback = "standard"
        print(
            f"[WARN] CI method {method!r} failed "
            f"({type(error).__name__}: {error}). Falling back to {fallback!r}.",
            file=sys.stderr,
        )
        lower, upper = make_ci(
            box,
            original=original,
            alpha=float(alpha),
            method=fallback,
        )
        return center, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), fallback


def maybe_expand_beta_box(beta_box: np.ndarray, n_bins: int) -> np.ndarray:
    """Broadcast bootstrap background beta to sample-by-Eg shape."""

    beta_box = np.asarray(beta_box, dtype=float)

    if beta_box.ndim == 1:
        return beta_box[:, None] + np.zeros((beta_box.shape[0], n_bins), dtype=float)

    if beta_box.ndim == 2:
        if beta_box.shape[1] == n_bins:
            return beta_box
        if beta_box.shape[0] == n_bins:
            return beta_box.T

    raise ValueError(f"Could not broadcast beta_box with shape {beta_box.shape}.")


def add_ci_payload(
    payload: dict,
    key: str,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Store confidence-band arrays under stable output names."""

    payload[f"ci_{key}_center"] = np.asarray(center, dtype=float)
    payload[f"ci_{key}_lo"] = np.asarray(lower, dtype=float)
    payload[f"ci_{key}_hi"] = np.asarray(upper, dtype=float)


def save_slice_summary(
    out_dir: str | Path,
    ex_keV: float,
    eg_keV: np.ndarray,
    raw: np.ndarray,
    background_observed: np.ndarray,
    background_expected: np.ndarray,
    mu_hat: np.ndarray,
    eta_hat: np.ndarray,
    nu_hat: np.ndarray,
    total_hat: np.ndarray,
    beta_hat: np.ndarray | None,
    truth: dict[str, np.ndarray],
    bootstrap_boxes: dict[str, np.ndarray] | None,
    ci_payload: dict[str, np.ndarray] | None,
    meta_extra: dict[str, Any],
) -> None:
    """Save one excitation-energy summary as npz, csv, and metadata json."""

    import pandas as pd

    out_dir = ensure_dir(out_dir)

    meta = {
        "ex_keV": float(ex_keV),
        "n_bins": int(len(eg_keV)),
        "has_beta": beta_hat is not None,
        "has_truth": True,
        "has_bootstrap": bootstrap_boxes is not None,
        "has_ci": ci_payload is not None,
        "files": {"npz": "summary.npz", "csv": "data.csv"},
    }
    meta.update(meta_extra)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    payload = {
        "ex_keV": float(ex_keV),
        "Eg_keV": np.asarray(eg_keV, dtype=float),
        "raw": np.asarray(raw, dtype=float),
        "bg_off": np.asarray(background_observed, dtype=float),
        "bg_exp": np.asarray(background_expected, dtype=float),
        "mu_hat": np.asarray(mu_hat, dtype=float),
        "eta_hat": np.asarray(eta_hat, dtype=float),
        "nu_hat": np.asarray(nu_hat, dtype=float),
        "total_hat": np.asarray(total_hat, dtype=float),
    }

    if beta_hat is not None:
        payload["beta_hat"] = np.asarray(beta_hat, dtype=float)

    for key, values in truth.items():
        payload[f"truth_{key}"] = np.asarray(values, dtype=float)

    if bootstrap_boxes is not None:
        for key, values in bootstrap_boxes.items():
            payload[f"boot_{key}_box"] = np.asarray(values, dtype=float)

    if ci_payload is not None:
        for key, values in ci_payload.items():
            payload[key] = np.asarray(values, dtype=float)

    np.savez(out_dir / "summary.npz", **payload)

    table = pd.DataFrame(
        {
            "Eg_keV": np.asarray(eg_keV, dtype=float),
            "raw": np.asarray(raw, dtype=float),
            "bg_off": np.asarray(background_observed, dtype=float),
            "bg_exp": np.asarray(background_expected, dtype=float),
            "mu_hat": np.asarray(mu_hat, dtype=float),
            "eta_hat": np.asarray(eta_hat, dtype=float),
            "nu_hat": np.asarray(nu_hat, dtype=float),
            "total_hat": np.asarray(total_hat, dtype=float),
        }
    )

    if beta_hat is not None and np.asarray(beta_hat).shape == table["Eg_keV"].shape:
        table["beta_hat"] = np.asarray(beta_hat, dtype=float)

    for key, values in truth.items():
        values = np.asarray(values, dtype=float)
        if values.shape == table["Eg_keV"].shape:
            table[f"truth_{key}"] = values

    if ci_payload is not None:
        for key, values in ci_payload.items():
            values = np.asarray(values, dtype=float)
            if values.shape == table["Eg_keV"].shape:
                table[key] = values

    table.to_csv(out_dir / "data.csv", index=False)


def bootstrap_summary(
    result: Any,
    run: dict,
    bootstrap: int,
    bootstrap_seed: int | None,
    alpha: float,
    ci_method: str,
    ci_summary: str,
    make_ci: Any,
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None, str]:
    """Build bootstrap boxes and confidence bands when requested."""

    if bootstrap <= 0:
        return None, None, ci_method

    with numpy_random_seed(bootstrap_seed):
        bootstraps = result.resample(int(bootstrap))

    mu_box = np.asarray(bootstraps.ubox, dtype=float)
    eta_box = np.asarray(bootstraps.etabox, dtype=float)
    nu_box = np.asarray(bootstraps.nubox, dtype=float)

    boxes: dict[str, np.ndarray] = {
        "mu": mu_box,
        "eta": eta_box,
        "nu": nu_box,
    }

    beta_box = getattr(bootstraps, "betabox", None)
    if beta_box is not None:
        try:
            boxes["total"] = nu_box + maybe_expand_beta_box(
                np.asarray(beta_box, dtype=float),
                n_bins=nu_box.shape[1],
            )
        except Exception:
            pass

    ci_payload: dict[str, np.ndarray] = {}
    effective_method = ci_method

    for key, box, original in [
        ("mu", mu_box, run["mu_hat"]),
        ("eta", eta_box, run["eta_hat"]),
        ("nu", nu_box, run["nu_hat"]),
    ]:
        center, lower, upper, effective_method = make_confidence_interval(
            box=box,
            original=original,
            alpha=alpha,
            method=effective_method,
            summary=ci_summary,
            make_ci=make_ci,
        )
        add_ci_payload(ci_payload, key, center, lower, upper)

    if "total" in boxes:
        center, lower, upper, effective_method = make_confidence_interval(
            box=boxes["total"],
            original=run["total_hat"],
            alpha=alpha,
            method=effective_method,
            summary=ci_summary,
            make_ci=make_ci,
        )
        add_ci_payload(ci_payload, "total", center, lower, upper)

    return boxes, ci_payload, effective_method


def run_one_ex(
    config: dict,
    loader: SyntheticDataLoader,
    response_unfold: Any,
    ex_keV: float,
    out_root: Path,
    iterations: int,
    bootstrap: int,
    bootstrap_seed: int | None,
    ci_method: str,
    ci_summary: str,
    ci_alpha: float,
    mask: Any,
    initial: Any,
    penalty: str,
    alpha_selection: str,
    alpha_grid: Sequence[float],
    sparsity_threshold: float,
    sparsity_smoothing: float,
) -> None:
    """Run OMpy RMLE and save output for one excitation-energy row."""

    RMLE, BackgroundModel, ModelLoss, Sparsity, make_ci = import_ompy_rmle()

    prepared = prepare_ex_slice(
        loader=loader,
        response_unfold=response_unfold,
        ex_keV=ex_keV,
    )

    include_background = bool(config_value(config, "include_background", True))

    print(
        f"[RMLE] Ex_req={round(ex_keV)} keV | "
        f"Ex_grid={round(prepared['ex_actual'])} keV | "
        f"N_bins={prepared['active_eg_length']}"
    )
    print(f"[RMLE] iterations={iterations} | bootstrap={bootstrap}")
    print(f"[RMLE] background={'on' if include_background else 'off'} | background.do_fold=False")
    print(f"[RMLE] ci_method={ci_method!r} | ci_alpha={ci_alpha} | ci_summary={ci_summary}")
    print(f"[RMLE] mask={mask!r} | initial={initial!r}")

    selected_alpha = None
    alpha_scores: list[dict[str, float]] = []

    if penalty == "none":
        print("[RMLE] loss=ModelLoss() without additional penalty")
    elif penalty == "sparsity":
        if alpha_selection != "w1":
            raise ValueError("freq_alpha_selection must be 'w1' when freq_penalty='sparsity'.")

        selected_alpha, alpha_scores = select_sparsity_alpha_by_w1(
            RMLE=RMLE,
            BackgroundModel=BackgroundModel,
            ModelLoss=ModelLoss,
            Sparsity=Sparsity,
            prepared=prepared,
            include_background=include_background,
            iterations=iterations,
            mask=mask,
            initial=initial,
            alpha_grid=alpha_grid,
            sparsity_threshold=sparsity_threshold,
            sparsity_smoothing=sparsity_smoothing,
        )
        print(
            f"[RMLE] loss=ModelLoss(Sparsity) | alpha_selection=w1 | "
            f"selected_alpha={selected_alpha:g}"
        )
        print("[RMLE] alpha sweep eta-space W1:")
        for row in alpha_scores:
            print(f"         alpha={row['alpha']:<8g} W1={row['w1_eta']:.6g}")
    else:
        raise ValueError("freq_penalty must be either 'none' or 'sparsity'.")

    run = run_rmle_fit(
        RMLE=RMLE,
        BackgroundModel=BackgroundModel,
        ModelLoss=ModelLoss,
        Sparsity=Sparsity,
        prepared=prepared,
        include_background=include_background,
        iterations=iterations,
        mask=mask,
        initial=initial,
        penalty=penalty,
        sparsity_alpha=selected_alpha,
        sparsity_threshold=sparsity_threshold,
        sparsity_smoothing=sparsity_smoothing,
    )

    boxes, ci_payload, effective_ci_method = bootstrap_summary(
        result=run["result"],
        run=run,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
        alpha=ci_alpha,
        ci_method=ci_method,
        ci_summary=ci_summary,
        make_ci=make_ci,
    )

    meta_extra = {
        **run["loss_meta"],
        "alpha_scores": alpha_scores,
        "iterations": int(iterations),
        "bootstrap": int(bootstrap),
        "bootstrap_seed": None if bootstrap_seed is None else int(bootstrap_seed),
        "bootstrap_seed_strategy": "numpy_random_seed_around_result_resample",
        "ci_method_requested": ci_method,
        "ci_method_effective": effective_ci_method,
        "ci_summary": ci_summary,
        "ci_alpha": float(ci_alpha),
        "mask": str(mask),
        "initial": str(initial),
        "include_background": include_background,
        "background_do_fold": False,
    }

    ex_dir = ensure_dir(out_root / f"Ex{round(ex_keV)}")

    save_slice_summary(
        out_dir=ex_dir,
        ex_keV=ex_keV,
        eg_keV=prepared["eg_axis"],
        raw=vector_values(prepared["n_vec"]),
        background_observed=vector_values(prepared["background_observed"]),
        background_expected=vector_values(prepared["background_expected"]),
        mu_hat=run["mu_hat"],
        eta_hat=run["eta_hat"],
        nu_hat=run["nu_hat"],
        total_hat=run["total_hat"],
        beta_hat=run["beta_hat"],
        truth={
            "mu": prepared["x_true"],
            "eta": prepared["eta_true"],
            "nu": prepared["nu_true"],
        },
        bootstrap_boxes=boxes,
        ci_payload=ci_payload,
        meta_extra=meta_extra,
    )

    print(f"[OK] Ex={round(ex_keV)} keV -> {ex_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the RMLE paper comparison."""

    parser = argparse.ArgumentParser(
        description="Run the OMpy RMLE paper comparison on synthetic Ex slices."
    )
    parser.add_argument("--config", required=True, help="Path to YAML run config.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    parser.add_argument("--ex", nargs="*", type=float, default=None, help="Ex slices in keV.")
    parser.add_argument("--iterations", type=int, default=None, help="Override freq_iterations.")
    parser.add_argument("--bootstrap", type=int, default=None, help="Override freq_bootstrap.")
    parser.add_argument("--bootstrap-seed", type=int, default=None, help="Override freq_bootstrap_seed.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the frequentist comparison pipeline."""

    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_run_config(repo_path(args.config))

    output_dir = repo_path(args.output_dir or config_value(config, "output_dir", "results"))
    dataset_id = str(config_value(config, "dataset_id", "dataset"))
    out_root = ensure_dir(output_dir / "frequentist" / dataset_id / "rmle")

    iterations = (
        int(args.iterations)
        if args.iterations is not None
        else int(config_value(config, "freq_iterations", 10000))
    )
    bootstrap = (
        int(args.bootstrap)
        if args.bootstrap is not None
        else int(config_value(config, "freq_bootstrap", 1000))
    )
    bootstrap_seed = (
        int(args.bootstrap_seed)
        if args.bootstrap_seed is not None
        else int(config_value(config, "freq_bootstrap_seed", 123))
    )

    ci_method = str(config_value(config, "freq_ci_method", "bonferroni percentile"))
    ci_summary = str(config_value(config, "freq_ci_summary", "mean"))
    ci_alpha = float(config_value(config, "freq_ci_alpha", 0.05))
    mask = config_value(config, "freq_mask", "last nonzero")
    initial = config_value(config, "freq_initial", "raw")

    penalty = str(config_value(config, "freq_penalty", "none")).strip().lower()
    if penalty not in {"none", "sparsity"}:
        raise ValueError("freq_penalty must be either 'none' or 'sparsity'.")

    alpha_selection = str(config_value(config, "freq_alpha_selection", "w1")).strip().lower()
    if penalty == "none":
        alpha_selection = "none"
    elif alpha_selection != "w1":
        raise ValueError("Only W1 alpha selection is supported for freq_penalty='sparsity'.")

    alpha_grid = [
        float(value)
        for value in config_value(config, "freq_sparsity_alpha_grid", DEFAULT_SPARSITY_ALPHA_GRID)
    ]
    sparsity_threshold = float(
        config_value(config, "freq_sparsity_threshold", DEFAULT_SPARSITY_THRESHOLD)
    )
    sparsity_smoothing = float(
        config_value(config, "freq_sparsity_smoothing", DEFAULT_SPARSITY_SMOOTHING)
    )

    provenance = {
        "config_path": str(repo_path(args.config)),
        "output_root": str(out_root),
        "dataset_id": dataset_id,
        "rng_seed": int(config_value(config, "rng_seed", 100)),
        "rebin_factors": config_value(config, "rebin_factors", None),
        "mat_scale": float(config_value(config, "mat_scale", 1.0)),
        "include_background": bool(config_value(config, "include_background", True)),
        "sigma_eg_gen": config_value(config, "sigma_eg_gen", "30keV"),
        "sigma_eg_unfold": config_value(
            config,
            "sigma_eg_unfold",
            config_value(config, "sigma_eg_gen", "30keV"),
        ),
        "include_ex_smearing": bool(config_value(config, "include_ex_smearing", False)),
        "background_do_fold": False,
        "penalty": penalty,
        "alpha_selection": alpha_selection,
        "sparsity_alpha_grid": alpha_grid if penalty == "sparsity" else None,
        "sparsity_threshold": sparsity_threshold if penalty == "sparsity" else None,
        "sparsity_smoothing": sparsity_smoothing if penalty == "sparsity" else None,
        "sparsity_target": "mu_normalized" if penalty == "sparsity" else None,
        "iterations": iterations,
        "bootstrap": bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "ci_method": ci_method,
        "ci_summary": ci_summary,
        "ci_alpha": ci_alpha,
        "mask": str(mask),
        "initial": str(initial),
    }
    (out_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    loader = build_synthetic_loader(config)
    response_unfold = unfolding_response(config, loader)

    for ex_keV in parse_ex_values(config, args.ex):
        run_one_ex(
            config=config,
            loader=loader,
            response_unfold=response_unfold,
            ex_keV=float(ex_keV),
            out_root=out_root,
            iterations=iterations,
            bootstrap=bootstrap,
            bootstrap_seed=bootstrap_seed,
            ci_method=ci_method,
            ci_summary=ci_summary,
            ci_alpha=ci_alpha,
            mask=mask,
            initial=initial,
            penalty=penalty,
            alpha_selection=alpha_selection,
            alpha_grid=alpha_grid,
            sparsity_threshold=sparsity_threshold,
            sparsity_smoothing=sparsity_smoothing,
        )

    print(f"\nDone. Frequentist results in: {out_root}")


if __name__ == "__main__":
    main()
