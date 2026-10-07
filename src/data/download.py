"""Download and extract the MovieLens dataset specified in configs/data.yaml.

Usage:
    python -m src.data.download
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

from src.data.config import REPO_ROOT, load_config


def download_movielens(force: bool = False) -> Path:
    """Download and extract the configured MovieLens release.

    Returns the directory containing the extracted CSVs
    (e.g. data/raw/ml-latest-small/).
    """
    cfg = load_config("data")["dataset"]
    raw_dir = REPO_ROOT / cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = raw_dir / cfg["name"]
    if extracted_dir.exists() and not force:
        print(f"Already present: {extracted_dir}")
        return extracted_dir

    print(f"Downloading {cfg['url']} ...")
    resp = requests.get(cfg["url"], timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(raw_dir)

    print(f"Extracted to {extracted_dir}")
    return extracted_dir


if __name__ == "__main__":
    download_movielens()
