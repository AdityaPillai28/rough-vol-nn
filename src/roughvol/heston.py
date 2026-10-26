"""Heston baseline priced with the COS method (Fang & Oosterlee 2008).

Forward-normalised, zero rates: X_T = ln(S_T / F) with
    dX = -v/2 dt + sqrt(v) dB,   dv = kappa (theta - v) dt + sigma sqrt(v) dW,  d<B,W> = rho dt.

The characteristic function uses the "little Heston trap" form (Albrecher et al. 2007),
which avoids the branch-cut discontinuity of the complex log for long maturities. Puts
are priced by COS (their payoff is bounded, so the expansion is insensitive to the
truncation range) and calls follow from put-call parity.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from roughvol.implied_vol import implied_vol

HESTON_NAMES = ("v0", "kappa", "theta", "sigma", "rho")


def heston_cf(u: np.ndarray, T: float, v0, kappa, theta, sigma, rho) -> np.ndarray:
    """E[exp(i u X_T)] for X_T = ln(S_T / F)."""
    iu = 1j * u
    beta = kappa - rho * sigma * iu
    d = np.sqrt(beta * beta + sigma * sigma * (iu + u * u))
    g = (beta - d) / (beta + d)
    edt = np.exp(-d * T)
    C = kappa * theta / sigma**2 * ((beta - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    D = (beta - d) / sigma**2 * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(C + D * v0)


def heston_cumulants(T, v0, kappa, theta, sigma, rho) -> tuple[float, float]:
    """Mean (closed form) and variance of X_T. The variance is taken from a central
    difference of ln(phi) at u = 0, which is accurate to ~1e-8 and sidesteps the
    typo-prone closed-form c2 found in the literature."""
    e = np.exp(-kappa * T)
    c1 = (1.0 - e) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * T
    h = 1e-3
    lp = np.log(heston_cf(np.array([h, -h]), T, v0, kappa, theta, sigma, rho))
    c2 = -(lp[0] + lp[1]).real / (h * h)
    return c1, abs(float(c2))


def cos_put(k: np.ndarray, T: float, params, n_terms: int = 512, L: float = 16.0) -> np.ndarray:
    """Normalised (price / F) Heston puts at log-moneyness k for one maturity."""
    k = np.atleast_1d(np.asarray(k, float))
    c1, c2 = heston_cumulants(T, *params)
    x = -k  # x = ln(F / K)
    width = L * np.sqrt(c2 + 1e-12)
    a = c1 + x.min() - width
    b = c1 + x.max() + width
    n = np.arange(n_terms)
    w = n * np.pi / (b - a)
    # Put payoff K (1 - e^y)^+ on y = ln(S_T / K) in [a, 0]:  V_n = 2/(b-a) K (psi - chi).
    # chi_n(a, 0) = int_a^0 e^y cos(w (y - a)) dy,  psi_n(a, 0) = int_a^0 cos(w (y - a)) dy
    chi = (np.cos(w * a) - np.exp(a) - w * np.sin(w * a)) / (1.0 + w * w)
    psi = np.empty(n_terms)
    psi[0] = -a
    psi[1:] = -np.sin(w[1:] * a) / w[1:]
    V = 2.0 / (b - a) * (psi - chi)
    phi = heston_cf(w, T, *params)
    terms = (phi[None, :] * np.exp(1j * w[None, :] * (x[:, None] - a))).real * V[None, :]
    terms[:, 0] *= 0.5
    K = np.exp(k)
    return K * terms.sum(axis=1)


def heston_iv_grid(params, maturities, log_moneyness, n_terms: int = 512, L: float = 16.0):
    """Implied vols on a (nT, nK) log-moneyness grid (OTM options)."""
    out = np.empty_like(log_moneyness, dtype=float)
    for i, T in enumerate(maturities):
        k = log_moneyness[i]
        put = cos_put(k, T, params, n_terms, L)
        price = np.where(k >= 0, put + 1.0 - np.exp(k), put)  # parity: C = P + F - K
        out[i] = implied_vol(price, k, T, k >= 0)
    return out


def heston_iv_quotes(
    params, T: np.ndarray, k: np.ndarray, n_terms: int = 512, L: float = 16.0
) -> np.ndarray:
    """Implied vols at arbitrary quotes (T[m], k[m]); groups by maturity internally."""
    T, k = np.asarray(T, float), np.asarray(k, float)
    out = np.empty_like(k)
    for t in np.unique(T):
        sel = T == t
        out[sel] = heston_iv_grid(params, [t], k[sel][None], n_terms, L)[0]
    return out


def calibrate_heston(
    T: np.ndarray,
    k: np.ndarray,
    target: np.ndarray,
    lower,
    upper,
    n_starts: int = 3,
    max_nfev: int = 400,
    n_terms: int = 512,
    L: float = 16.0,
    seed: int = 0,
) -> dict:
    """Least-squares fit of Heston implied vols to quotes (T, k, iv); NaN = not quoted.
    Grids can be passed flattened (``np.repeat`` the maturities)."""
    T, k, target = (np.asarray(a, float).ravel() for a in (T, k, target))
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    m = np.isfinite(target)
    near = np.abs(k) <= np.quantile(np.abs(k[m]), 0.2)
    atm_var = float(np.mean(target[m & near] ** 2))

    def resid(p):
        iv = heston_iv_quotes(p, T, k, n_terms, L)
        return np.where(m, np.nan_to_num(iv, nan=2.0) - np.nan_to_num(target), 0.0)

    rng = np.random.default_rng(seed)
    starts = [np.clip([atm_var, 2.0, atm_var, 1.0, -0.7], lower, upper)]
    starts += list(lower + (upper - lower) * rng.random((max(n_starts - 1, 0), 5)))
    best = None
    for x0 in starts:
        try:
            r = least_squares(
                resid,
                x0,
                bounds=(lower, upper),
                method="trf",
                max_nfev=max_nfev,
                x_scale=upper - lower,
            )
        except (ValueError, FloatingPointError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    return {
        "params": best.x,
        "rmse": float(np.sqrt(2 * best.cost / m.sum())),
        "nfev": int(best.nfev),
    }
