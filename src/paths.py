"""Project-root path helpers."""

from __future__ import annotations

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"



def repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""

    path = Path(path).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    """Resolve a directory path and create it if needed."""

    resolved = repo_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
