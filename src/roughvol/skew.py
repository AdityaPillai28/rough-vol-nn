"""At-the-money skew term structure psi(T) = |d sigma_BS / d k| at k = 0.

For rough Bergomi with flat forward variance, the small-time limit is (Alos, Leon &
Vives 2007; Fukasawa 2011; Bayer, Friz, Gulisashvili, Horvath & Stemper 2019)

    psi(T) ~ |rho| * eta * sqrt(2H) / (2 (H + 1/2)(H + 3/2)) * T^(H - 1/2),

i.e. a power law that explodes as T -> 0 when H < 1/2. In any classical (Markovian,
H = 1/2) stochastic-volatility model such as Heston, psi(T) tends to a constant
(rho * sigma / (4 sqrt(v0)) for Heston).
"""

from __future__ import annotations

import numpy as np


def atm_skew(iv: np.ndarray, z: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """Central-difference ATM skew from grids (..., nT, nK) with k = z * sqrt(T)."""
    j = int(np.argmin(np.abs(z)))
    if j == 0 or j == len(z) - 1:
        raise ValueError("need strikes on both sides of the money")
    dk = (z[j + 1] - z[j - 1]) * np.sqrt(np.asarray(maturities, float))
    return np.abs(iv[..., :, j + 1] - iv[..., :, j - 1]) / dk


def theoretical_skew(T, H: float, eta: float, rho: float) -> np.ndarray:
    c = abs(rho) * eta * np.sqrt(2.0 * H) / (2.0 * (H + 0.5) * (H + 1.5))
    return c * np.asarray(T, float) ** (H - 0.5)


def heston_short_skew(v0: float, sigma: float, rho: float) -> float:
    return abs(rho) * sigma / (4.0 * np.sqrt(v0))


def fit_power_law(T: np.ndarray, psi: np.ndarray, t_max: float | None = None) -> dict:
    """OLS fit of log psi = log c + slope * log T (optionally only for T <= t_max)."""
    T, psi = np.asarray(T, float), np.asarray(psi, float)
    sel = np.isfinite(psi) & (psi > 0)
    if t_max is not None:
        sel &= t_max >= T
    X = np.log(T[sel])
    Y = np.log(psi[sel])
    A = np.vstack([np.ones_like(X), X]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    dof = max(len(X) - 2, 1)
    cov = np.linalg.inv(A.T @ A) * (resid @ resid) / dof
    return {
        "slope": float(coef[1]),
        "slope_se": float(np.sqrt(cov[1, 1])),
        "log_c": float(coef[0]),
        "n_points": int(sel.sum()),
    }
