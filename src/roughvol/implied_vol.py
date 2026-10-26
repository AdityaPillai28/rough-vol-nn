"""Robust, vectorised Black implied-volatility inversion.

Safeguarded Newton iteration on the *log* of the out-of-the-money price, with a
bisection fallback whenever a Newton step leaves the current bracket. Working on
ln(price) makes the objective close to linear in s for deep out-of-the-money options
(where the price is exponentially small), which is the regime where plain Newton on the
price diverges. Prices outside the no-arbitrage bounds return NaN rather than a guess.
"""

from __future__ import annotations

import numpy as np

from roughvol.bs import black_np, black_vega_np

S_MIN, S_MAX = 1e-8, 20.0


def to_otm(price: np.ndarray, k: np.ndarray, is_call: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalised prices to the out-of-the-money side via put-call parity
    (C - P = 1 - e^k in forward units). Returns (otm_price, otm_is_call)."""
    price, k = np.asarray(price, float), np.asarray(k, float)
    is_call = np.broadcast_to(np.asarray(is_call, bool), price.shape)
    otm_call = k >= 0.0
    parity = 1.0 - np.exp(k)  # C - P
    p = np.where(is_call == otm_call, price, np.where(is_call, price - parity, price + parity))
    return p, np.broadcast_to(otm_call, p.shape)


def implied_total_sd(
    price: np.ndarray,
    k: np.ndarray,
    is_call: np.ndarray | bool,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    """Total implied standard deviation s = sigma*sqrt(T) from a normalised price."""
    price, k = np.broadcast_arrays(np.asarray(price, float), np.asarray(k, float))
    p, call = to_otm(price, k, is_call)
    upper = np.where(call, 1.0, np.exp(k))
    valid = np.isfinite(p) & (p > 0.0) & (p < upper)
    p = np.where(valid, p, 0.5 * upper)
    log_p = np.log(p)

    lo = np.full(p.shape, S_MIN)
    hi = np.full(p.shape, S_MAX)
    # Start at the inflection point of s -> price (max vega), Jaeckel's classic choice.
    s = np.clip(np.sqrt(2.0 * np.abs(k)), 0.05, 2.0)
    active = valid.copy()
    for _ in range(max_iter):
        b = black_np(k, s, call)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.log(b) - log_p
            step = f * b / black_vega_np(k, s)
        hi = np.where(f > 0, s, hi)
        lo = np.where(f <= 0, s, lo)
        s_new = s - step
        small = np.abs(step) <= tol * (1.0 + s)
        bad = (~np.isfinite(s_new) | (s_new < lo) | (s_new > hi)) & ~small
        mid = np.where(hi / lo > 4.0, np.sqrt(lo * hi), 0.5 * (lo + hi))
        s_new = np.where(bad, mid, s_new)
        done = small | (hi - lo <= tol * (1.0 + s))
        s = np.where(active, s_new, s)
        active &= ~done
        if not active.any():
            break
    return np.where(valid, s, np.nan)


def implied_vol(price, k, T, is_call) -> np.ndarray:
    """Black implied vol from a normalised (price / F, undiscounted) option price."""
    T = np.asarray(T, float)
    return implied_total_sd(price, k, is_call) / np.sqrt(T)


def implied_vol_se(iv: np.ndarray, price_se: np.ndarray, k: np.ndarray, T) -> np.ndarray:
    """Delta-method standard error of an implied vol from a price standard error."""
    s = iv * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        return price_se / (black_vega_np(k, s) * np.sqrt(T))
