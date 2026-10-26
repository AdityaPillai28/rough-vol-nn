"""Training-set generation: parameter sets -> Monte Carlo implied-vol grids.

Generation is chunked and resumable. The full parameter table is a deterministic
function of the seed, and sample i always uses MC seed ``seed + i``, so a chunk is
bit-for-bit reproducible no matter when (or after how many interruptions) it is
computed. Each chunk is written atomically (temp file + rename) and carries a
fingerprint of the generating config; resuming with a different config is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from roughvol.config import resolve
from roughvol.grid import PARAM_NAMES, Grid, param_bounds
from roughvol.rbergomi import RoughBergomiMC
from roughvol.utils import get_device, hardware_info, save_json


def sample_params(n: int, lo: np.ndarray, hi: np.ndarray, seed: int) -> np.ndarray:
    """Uniform samples in the parameter box, shape (n, 4) in PARAM_NAMES order."""
    rng = np.random.default_rng(seed)
    return lo + (hi - lo) * rng.random((n, len(lo)))


def _fingerprint(dcfg: dict, n_paths: int, seed: int) -> str:
    keys = {
        "grid": dcfg["grid"],
        "params": dcfg["params"],
        "seed": seed,
        "n_paths": n_paths,
        "mc": {k: dcfg["mc"][k] for k in ("steps_per_year", "estimator", "dtype")},
    }
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:16]


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _pricer(dcfg: dict, grid: Grid, device: torch.device) -> RoughBergomiMC:
    dtype = getattr(torch, dcfg["mc"]["dtype"])
    return RoughBergomiMC(grid.maturities, dcfg["mc"]["steps_per_year"], device, dtype)


def price_params(
    mc: RoughBergomiMC,
    grid: Grid,
    params: np.ndarray,
    seeds: np.ndarray,
    n_paths: int,
    estimator: str,
    path_batch: int,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    nT, nK = grid.shape
    out = {
        name: np.empty((len(params), nT, nK), np.float32)
        for name in ("iv", "iv_se", "price", "price_se")
    }
    out["forward"] = np.empty((len(params), nT), np.float32)
    k = grid.log_moneyness
    it = tqdm(range(len(params)), leave=False, disable=not progress)
    for i in it:
        H, eta, rho, xi0 = params[i]
        r = mc.price(
            H,
            eta,
            rho,
            xi0,
            k,
            n_paths,
            seed=int(seeds[i]),
            estimator=estimator,
            path_batch=path_batch,
        )
        out["iv"][i], out["iv_se"][i] = r.iv, r.iv_se
        out["price"][i], out["price_se"][i] = r.price, r.price_se
        out["forward"][i] = r.forward
    return out


def generate_dataset(cfg: dict, device: str = "auto", max_chunks: int | None = None) -> dict:
    """Generate (or resume) the chunked training set, then merge it."""
    dcfg = cfg["data"]
    gcfg = dcfg["generation"]
    grid = Grid.from_config(dcfg)
    lo, hi = param_bounds(dcfg)
    seed = int(dcfg["seed"])
    n, cs = int(gcfg["n_samples"]), int(gcfg["chunk_size"])
    n_paths = int(dcfg["mc"]["n_paths"])
    fp = _fingerprint(dcfg, n_paths, seed)
    params = sample_params(n, lo, hi, seed)
    raw = resolve(gcfg["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    dev = get_device(device)
    mc = _pricer(dcfg, grid, dev)

    n_chunks = (n + cs - 1) // cs
    computed, seconds = 0, 0.0
    bar = tqdm(range(n_chunks), desc="chunks")
    for c in bar:
        path = raw / f"chunk_{c:05d}.npz"
        if path.exists():
            with np.load(path) as z:
                if str(z["fingerprint"]) != fp:
                    raise RuntimeError(
                        f"{path} was generated with a different config; "
                        "delete the raw directory to regenerate"
                    )
            continue
        if max_chunks is not None and computed >= max_chunks:
            break
        sl = slice(c * cs, min((c + 1) * cs, n))
        t0 = time.perf_counter()
        res = price_params(
            mc,
            grid,
            params[sl],
            seed + np.arange(sl.start, sl.stop),
            n_paths,
            dcfg["mc"]["estimator"],
            int(dcfg["mc"]["path_batch"]),
        )
        dt = time.perf_counter() - t0
        _atomic_savez(
            path,
            params=params[sl].astype(np.float32),
            index=np.arange(sl.start, sl.stop),
            fingerprint=np.array(fp),
            seconds=np.array(dt),
            **res,
        )
        computed += 1
        seconds += dt
        bar.set_postfix(ms_per_sample=f"{1e3 * dt / (sl.stop - sl.start):.1f}")
    return merge_chunks(cfg, fp, n_chunks)


def merge_chunks(cfg: dict, fp: str, n_chunks: int) -> dict:
    dcfg = cfg["data"]
    gcfg = dcfg["generation"]
    raw = resolve(gcfg["raw_dir"])
    files = [raw / f"chunk_{c:05d}.npz" for c in range(n_chunks)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f"[data] {len(missing)} of {n_chunks} chunks still missing; rerun to resume.")
        return {"complete": False, "missing_chunks": len(missing)}
    parts: dict[str, list[np.ndarray]] = {}
    total_seconds = 0.0
    for f in files:
        with np.load(f) as z:
            for key in ("params", "index", "iv", "iv_se", "price", "price_se", "forward"):
                parts.setdefault(key, []).append(z[key])
            total_seconds += float(z["seconds"])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    assert np.array_equal(data["index"], np.arange(len(data["index"])))
    n = len(data["index"])
    f_tr, f_va, _ = gcfg["split"]
    n_tr, n_va = int(f_tr * n), int(f_va * n)
    split = np.zeros(n, np.int8)  # 0 train, 1 val, 2 test
    split[n_tr : n_tr + n_va] = 1
    split[n_tr + n_va :] = 2
    grid = Grid.from_config(dcfg)
    out = resolve(gcfg["processed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        out,
        **data,
        split=split,
        maturities=grid.maturities,
        z=grid.z,
        param_names=np.array(PARAM_NAMES),
        fingerprint=np.array(fp),
    )

    iv, iv_se = data["iv"], data["iv_se"]
    rel_se = iv_se / iv
    summary = {
        "n_samples": n,
        "n_paths": int(dcfg["mc"]["n_paths"]),
        "steps_per_year": int(dcfg["mc"]["steps_per_year"]),
        "estimator": dcfg["mc"]["estimator"],
        "grid_shape": list(grid.shape),
        "nan_fraction": float(np.mean(~np.isfinite(iv))),
        "rel_iv_se_mean": float(np.nanmean(rel_se)),
        "rel_iv_se_median": float(np.nanmedian(rel_se)),
        "rel_iv_se_mean_grid": np.nanmean(rel_se, axis=0),
        "forward_abs_dev_mean": float(np.mean(np.abs(data["forward"] - 1.0))),
        "gpu_seconds_total": total_seconds,
        "ms_per_sample": 1e3 * total_seconds / n,
        "hardware": hardware_info(),
    }
    save_json(summary, Path(cfg["results_dir"]) / "data_generation.json")
    print(f"[data] merged {n} samples -> {out} ({total_seconds / 60:.1f} min of pricing)")
    return {"complete": True, **summary}


def generate_reference(cfg: dict, device: str = "auto") -> dict:
    """High-precision surfaces (many more paths) used as ground truth."""
    dcfg = cfg["data"]
    rcfg = dcfg["reference"]
    out = resolve(rcfg["out"])
    if out.exists():
        print(f"[data] reference set exists: {out}")
        return {"complete": True}
    grid = Grid.from_config(dcfg)
    lo, hi = param_bounds(dcfg)
    seed = int(dcfg["seed"]) + int(rcfg["seed_offset"])
    params = sample_params(int(rcfg["n_samples"]), lo, hi, seed)
    mc = _pricer(dcfg, grid, get_device(device))
    t0 = time.perf_counter()
    res = price_params(
        mc,
        grid,
        params,
        10_000_000 + seed + np.arange(len(params)),
        int(rcfg["n_paths"]),
        dcfg["mc"]["estimator"],
        int(dcfg["mc"]["path_batch"]),
        progress=True,
    )
    dt = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        out,
        params=params.astype(np.float32),
        maturities=grid.maturities,
        z=grid.z,
        seconds=np.array(dt),
        n_paths=np.array(int(rcfg["n_paths"])),
        **res,
    )
    print(f"[data] reference set: {len(params)} surfaces in {dt:.0f}s -> {out}")
    return {"complete": True, "seconds": dt}


def load_dataset(cfg: dict) -> dict[str, np.ndarray]:
    with np.load(resolve(cfg["data"]["generation"]["processed"])) as z:
        return {k: z[k] for k in z.files}


def load_reference(cfg: dict) -> dict[str, np.ndarray]:
    with np.load(resolve(cfg["data"]["reference"]["out"])) as z:
        return {k: z[k] for k in z.files}
