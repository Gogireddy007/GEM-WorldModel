"""YAML config loading, resolved relative to the repo root."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict:
    """Load a config file by stem name, e.g. load_config('data') -> configs/data.yaml."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No such config: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a path from a config file relative to the repo root."""
    return REPO_ROOT / relative
