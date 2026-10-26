"""Calibration through the surrogate (step 2 of the two-step approach) and, for the speed
comparison, through the Monte Carlo pricer itself.

Objective (all methods): sum over quoted cells of (sigma_model - sigma_market)^2, with
optional per-cell weights.

* ``lm_calibrate``   batched Levenberg-Marquardt in torch. Parameters live in the box via
  theta = lo + (hi - lo) * sigmoid(u); the exact Jacobian d sigma / d u comes from
  forward-mode autodiff through the network. Marquardt's diagonal scaling makes the step
  invariant to the per-parameter rescaling introduced by the sigmoid.
* ``lbfgs_calibrate`` scipy L-BFGS-B with box constraints and autodiff gradients.
* ``mc_calibrate``   scipy trust-region least squares with the MC pricer in the loop
  (common random numbers, finite-difference Jacobian).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from scipy.optimize import least_squares, minimize
from torch.func import jacfwd, vmap

from roughvol.surrogate.models import Surrogate


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def random_starts(
    n: int, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator, include_center: bool = True
) -> np.ndarray:
    pts = lo + (hi - lo) * rng.random((n, len(lo)))
    if include_center and n > 0:
        pts[0] = 0.5 * (lo + hi)
    return pts


def lm_calibrate(
    model: Surrogate,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    n_starts: int = 4,
    max_iter: int = 100,
    lambda0: float = 1e-2,
    lambda_up: float = 10.0,
    lambda_down: float = 0.3,
    tol: float = 1e-10,
    seed: int = 0,
    predict_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    """Calibrate B surfaces at once.

    target: (B, nT, nK) grid implied vols, or (B, M) quotes when ``predict_fn`` is given
    (NaN = not quoted). ``predict_fn`` maps params (n, 4) to model vols (n, M); by default
    it is the surrogate grid itself. Used for market quotes off the training grid.
    """
    dev = model.lo.device
    lo, hi = model.lo.double(), model.hi.double()
    B = target.shape[0]
    target = target.reshape(B, -1)
    m = np.isfinite(target) if mask is None else (mask.reshape(B, -1) & np.isfinite(target))
    w = np.ones_like(target, dtype=float) if weights is None else weights.reshape(B, -1)
    tgt = torch.as_tensor(np.nan_to_num(target), device=dev, dtype=torch.float32)
    sw = torch.as_tensor(np.sqrt(w) * m, device=dev, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    starts = random_starts(n_starts, lo.cpu().numpy(), hi.cpu().numpy(), rng)
    S = len(starts)
    theta0 = torch.as_tensor(np.tile(starts, (B, 1)), device=dev, dtype=torch.float64)
    u = _logit((theta0 - lo) / (hi - lo)).float()
    rep = torch.arange(B, device=dev).repeat_interleave(S)
    tgt_r, sw_r = tgt[rep], sw[rep]
    lo_f, span = model.lo, model.hi - model.lo
    predict = predict_fn or (lambda th: model(th).reshape(th.shape[0], -1))

    def resid_single(uu, t, s):
        return (predict(lo_f + span * torch.sigmoid(uu[None]))[0] - t) * s

    jac_fn = vmap(jacfwd(resid_single, argnums=0), in_dims=(0, 0, 0))
    res_fn = vmap(resid_single, in_dims=(0, 0, 0))

    with torch.no_grad():
        r = res_fn(u, tgt_r, sw_r)
        cost = (r * r).sum(1).double()
    lam = torch.full((len(u),), lambda0, device=dev, dtype=torch.float64)
    active = torch.ones(len(u), dtype=torch.bool, device=dev)
    n_iter = torch.zeros(len(u), dtype=torch.int32, device=dev)
    eye = torch.eye(4, device=dev, dtype=torch.float64)
    for _ in range(max_iter):
        J = jac_fn(u, tgt_r, sw_r).double()  # (BS, M, 4)
        with torch.no_grad():
            JtJ = J.transpose(1, 2) @ J
            g = (J.transpose(1, 2) @ r.double()[..., None]).squeeze(-1)
            diag = torch.diagonal(JtJ, dim1=1, dim2=2).clamp_min(1e-12)
            A = JtJ + lam[:, None, None] * torch.diag_embed(diag) + 1e-12 * eye
            step = torch.linalg.solve(A, -g)
            u_new = (u.double() + step).clamp(-12, 12).float()
            r_new = res_fn(u_new, tgt_r, sw_r)
            cost_new = (r_new * r_new).sum(1).double()
            accept = (cost_new < cost) & active
            rel = (cost - cost_new) / cost.clamp_min(1e-30)
            u = torch.where(accept[:, None], u_new, u)
            r = torch.where(accept[:, None], r_new, r)
            converged = accept & (rel < tol)
            cost = torch.where(accept, cost_new, cost)
            lam = torch.where(accept, lam * lambda_down, lam * lambda_up).clamp(1e-12, 1e12)
            n_iter += active.int()
            active &= ~converged & (lam < 1e11)
            if not active.any():
                break
    with torch.no_grad():
        theta = (lo + (hi - lo) * torch.sigmoid(u.double())).reshape(B, S, 4)
        cost = cost.reshape(B, S)
        best = cost.argmin(1)
        ar = torch.arange(B, device=dev)
        n_q = torch.as_tensor(m.sum(1), device=dev).clamp_min(1)
        return {
            "theta": theta[ar, best].cpu().numpy(),
            "cost": cost[ar, best].cpu().numpy(),
            "rmse": torch.sqrt(cost[ar, best] / n_q).cpu().numpy(),
            "n_iter": n_iter.reshape(B, S)[ar, best].cpu().numpy(),
        }


def lbfgs_calibrate(
    model: Surrogate,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    n_starts: int = 4,
    max_iter: int = 300,
    seed: int = 0,
) -> dict:
    """Single surface (nT, nK); box-constrained L-BFGS-B with autodiff gradients."""
    dev = model.lo.device
    m = np.isfinite(target) if mask is None else (mask & np.isfinite(target))
    tgt = torch.as_tensor(np.nan_to_num(target), device=dev, dtype=torch.float32)
    mk = torch.as_tensor(m, device=dev)
    lo, hi = model.lo.cpu().numpy().astype(float), model.hi.cpu().numpy().astype(float)

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        th = torch.as_tensor(x, device=dev, dtype=torch.float32).requires_grad_(True)
        d = torch.where(mk, model(th[None])[0] - tgt, torch.zeros_like(tgt))
        c = (d * d).sum()
        c.backward()
        return float(c), th.grad.detach().cpu().numpy().astype(float)

    best = None
    for x0 in random_starts(n_starts, lo, hi, np.random.default_rng(seed)):
        res = minimize(
            fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi, strict=True)),
            options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-10},
        )
        if best is None or res.fun < best.fun:
            best = res
    return {
        "theta": best.x,
        "cost": best.fun,
        "rmse": np.sqrt(best.fun / max(m.sum(), 1)),
        "nfev": best.nfev,
    }


def mc_calibrate(
    mc,
    log_moneyness: np.ndarray,
    target: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    n_paths: int,
    x0: np.ndarray,
    max_nfev: int = 60,
    seed: int = 0,
    path_batch: int | None = None,
) -> dict:
    """Calibrate with Monte Carlo in the loop (common random numbers across evaluations)."""
    m = np.isfinite(target)
    nfev = [0]

    def resid(x: np.ndarray) -> np.ndarray:
        nfev[0] += 1
        r = mc.price(*x, log_moneyness, n_paths, seed=seed, path_batch=path_batch)
        out = np.where(m, np.nan_to_num(r.iv, nan=1.0) - np.nan_to_num(target), 0.0)
        return out.ravel()

    t0 = time.perf_counter()
    res = least_squares(
        resid, x0, bounds=(lo, hi), method="trf", diff_step=1e-3, max_nfev=max_nfev, x_scale=hi - lo
    )
    return {
        "theta": res.x,
        "cost": 2 * res.cost,
        "rmse": np.sqrt(2 * res.cost / m.sum()),
        "nfev": nfev[0],
        "seconds": time.perf_counter() - t0,
    }
