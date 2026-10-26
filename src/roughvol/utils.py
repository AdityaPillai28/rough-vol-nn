"""Small shared helpers: seeding, devices, timing, JSON results."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from roughvol.config import resolve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (seeds the legacy global RNG used by libraries)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    """``with Timer() as t: ...`` then read ``t.elapsed`` (seconds)."""

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        self.elapsed = float("nan")
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.floating | float):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_json(obj: dict[str, Any], path: str | Path) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(obj), fh, indent=2)
    os.replace(tmp, p)
    return p


def load_json(path: str | Path) -> dict[str, Any]:
    with open(resolve(path), encoding="utf-8") as fh:
        return json.load(fh)


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    info["cpu_count"] = os.cpu_count()
    return info
