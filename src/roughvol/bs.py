"""Black (forward) formulas in normalised units.

All prices are undiscounted and divided by the forward F, so an option is described by
its log-moneyness k = ln(K/F) and total standard deviation s = sigma * sqrt(T).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import ndtr


def black_np(k: np.ndarray, s: np.ndarray, is_call: np.ndarray | bool) -> np.ndarray:
    """Normalised Black price, numpy float64."""
    k, s = np.asarray(k, float), np.asarray(s, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k / s + 0.5 * s
    d2 = d1 - s
    call = ndtr(d1) - np.exp(k) * ndtr(d2)
    put = np.exp(k) * ndtr(-d2) - ndtr(-d1)
    return np.where(is_call, call, put)


def black_vega_np(k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """d price / d s (same for calls and puts)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -np.asarray(k, float) / s + 0.5 * s
    return np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)


def bs_price_np(F, K, T, sigma, is_call) -> np.ndarray:
    """Undiscounted Black price in currency units."""
    F, K = np.asarray(F, float), np.asarray(K, float)
    return F * black_np(np.log(K / F), np.asarray(sigma) * np.sqrt(T), is_call)


def black_torch(F: torch.Tensor, K: torch.Tensor, v: torch.Tensor, is_call: torch.Tensor):
    """Black price for forward F, strike K and total variance v (broadcasting, torch)."""
    sd = torch.sqrt(v.clamp_min(1e-300))
    d1 = (torch.log(F / K) + 0.5 * v) / sd
    d2 = d1 - sd
    call = F * torch.special.ndtr(d1) - K * torch.special.ndtr(d2)
    put = K * torch.special.ndtr(-d2) - F * torch.special.ndtr(-d1)
    return torch.where(is_call, call, put)
