"""Training loop for the surrogates (Adam, plateau LR schedule, early stopping)."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from roughvol.config import resolve
from roughvol.data import load_dataset
from roughvol.grid import param_bounds
from roughvol.surrogate.models import Surrogate, build_network
from roughvol.utils import get_device, hardware_info, save_json, set_seed


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = torch.where(mask, pred - target, torch.zeros_like(pred))
    return (diff * diff).sum() / mask.sum().clamp_min(1)


def train_one(name: str, spec: dict, ds: dict, cfg: dict) -> dict:
    scfg = cfg["surrogate"]
    tcfg = scfg["train"]
    set_seed(int(scfg["seed"]))
    device = get_device(scfg["device"])
    lo, hi = param_bounds(cfg["data"])
    nT, nK = len(ds["maturities"]), len(ds["z"])

    iv = ds["iv"].reshape(len(ds["iv"]), -1).astype(np.float32)
    tr, va = ds["split"] == 0, ds["split"] == 1
    y_mean = np.nanmean(iv[tr], axis=0)
    y_std = np.nanstd(iv[tr], axis=0)
    net = build_network(spec, 4, nT * nK)
    model = Surrogate(net, spec, lo, hi, y_mean, y_std, ds["maturities"], ds["z"]).to(device)

    def tensors(m):
        x = model.scale_x(torch.as_tensor(ds["params"][m], device=device))
        y = torch.as_tensor(iv[m], device=device)
        mask = torch.isfinite(y)
        y = torch.where(mask, (y - model.y_mean) / model.y_std, torch.zeros_like(y))
        return x, y, mask

    xtr, ytr, mtr = tensors(tr)
    xva, yva, mva = tensors(va)
    opt = torch.optim.Adam(
        model.net.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        factor=float(tcfg["lr_factor"]),
        patience=int(tcfg["lr_patience"]),
        min_lr=float(tcfg["min_lr"]),
    )
    bs = int(tcfg["batch_size"])
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(int(tcfg["epochs"])):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        tot = 0.0
        for i in range(0, len(xtr), bs):
            idx = perm[i : i + bs]
            loss = masked_mse(model.forward_scaled(xtr[idx]), ytr[idx], mtr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = masked_mse(model.forward_scaled(xva), yva, mva).item()
        sched.step(val)
        history.append(
            {"epoch": epoch, "train": tot / len(xtr), "val": val, "lr": opt.param_groups[0]["lr"]}
        )
        if val < best * (1 - 1e-4):
            best, best_state, best_epoch, bad = val, copy.deepcopy(model.net.state_dict()), epoch, 0
        else:
            bad += 1
        if bad >= int(tcfg["early_stop_patience"]):
            break
    seconds = time.perf_counter() - t0
    model.net.load_state_dict(best_state)
    out_dir = resolve(scfg["out_dir"])
    path = out_dir / f"{name}.pt"
    summary = {
        "name": name,
        "spec": spec,
        "n_parameters": model.n_parameters(),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_mse_std_units": best,
        "train_seconds": seconds,
        "device": str(device),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "hardware": hardware_info(),
    }
    model.save(path, extra=summary)
    save_json(
        {**summary, "history": history}, Path(cfg["results_dir"]) / f"surrogate_train_{name}.json"
    )
    print(
        f"[train] {name}: {model.n_parameters()} params, best val {best:.3e} at epoch "
        f"{best_epoch}, {seconds:.0f}s -> {path}"
    )
    return summary


def train_all(cfg: dict) -> dict:
    ds = load_dataset(cfg)
    return {
        name: train_one(name, spec, ds, cfg) for name, spec in cfg["surrogate"]["models"].items()
    }


def load_surrogate(cfg: dict, name: str, device: str = "cpu") -> Surrogate:
    return Surrogate.load(resolve(cfg["surrogate"]["out_dir"]) / f"{name}.pt", device)
