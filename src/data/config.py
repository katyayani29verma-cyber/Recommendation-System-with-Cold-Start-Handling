"""Load the shared data/eval configs so every track reads the same settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config by filename stem, e.g. load_config('data')."""
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
