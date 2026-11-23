"""
Helpers for project paths.
"""

from __future__ import annotations

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"



def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    resolved = repo_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def result_dir(config: dict, family: str) -> Path:
    kind = config.get("dataset_kind", "synthetic")
    root = repo_path(config.get("output_dir", "results")) / kind
    dataset = str(config["dataset_id"])
    return root / family / dataset if kind == "synthetic" else root / dataset / family


def bayes_run_dir(config: dict) -> Path:
    return result_dir(config, "bayes") / config["run_id"] / config["mode"]
