"""Configuration loading. Every hyperparameter lives in configs/*.yaml."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
SECTIONS = ("data", "surrogate", "calibration", "heston", "market", "ablations")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(quick: bool = False, config_dir: Path | None = None) -> dict[str, Any]:
    """Load all config sections; with ``quick`` merge configs/quick.yaml on top."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg: dict[str, Any] = {}
    for name in SECTIONS:
        with open(cdir / f"{name}.yaml", encoding="utf-8") as fh:
            cfg[name] = yaml.safe_load(fh) or {}
    if quick:
        with open(cdir / "quick.yaml", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    cfg["quick"] = quick
    cfg["results_dir"] = "results/quick" if quick else "results"
    cfg["assets_dir"] = "results/quick/assets" if quick else "assets"
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
