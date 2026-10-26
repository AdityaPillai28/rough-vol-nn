"""Rough Bergomi Monte Carlo with the hybrid scheme (Bennedsen, Lunde & Pakkanen 2017).

Model (Bayer, Friz & Gatheral 2016), forward-normalised, zero rates, S_0 = 1:

    dS_t / S_t = sqrt(V_t) dB_t,           B = rho W + sqrt(1 - rho^2) W_perp
    V_t        = xi0 * exp(eta * Y_t - eta^2/2 * Var[Y_t])
    Y_t        = sqrt(2H) * int_0^t (t - s)^(H - 1/2) dW_s       (Riemann-Liouville fBm)

The Volterra process Y is simulated with the kappa = 1 hybrid scheme: the kernel is
integrated exactly on the cell adjacent to the diagonal (where it is singular when
H < 1/2) and approximated by a step function evaluated at the optimal points b_k
elsewhere. The far-field sum is a discrete convolution, done by FFT in O(N s log s).

Two price estimators are provided:

* ``plain``  - simulate S on the grid (log-Euler) and average payoffs.
* ``mixing`` - condition on the volatility driver W (Romano & Touzi 1997). Given W,
  ln S_T is Gaussian, so the payoff expectation is a Black price with forward
  exp(rho*I - rho^2/2*Q) and total variance (1 - rho^2)*Q, where I = int sqrt(V) dW and
  Q = int V dt. This removes the W_perp noise exactly and gives smooth, strictly
  positive prices even far from the money.

Both use antithetic variates (Z, -Z) and report standard errors from the antithetic
pair means, which are i.i.d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from roughvol.bs import black_torch
from roughvol.implied_vol import implied_vol, implied_vol_se


def hybrid_kernel(H: float, n_steps: int, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights of the kappa = 1 hybrid scheme.

    Returns
    -------
    G : (n_steps + 1,) far-field weights, G[0] = G[1] = 0, G[k] = (b_k * dt)^alpha.
    chol : (2, 2) Cholesky factor of Cov(dW_i, dW~_i), with
        dW~_i = int_{t_i}^{t_{i+1}} (t_{i+1} - u)^alpha dW_u.
    var_Y : (n_steps + 1,) exact variance of the *discretised* Y at each grid point,
        used in the compensator so that E[V_t] = xi0 holds exactly on the grid.
    """
    a = H - 0.5
    k = np.arange(n_steps + 1, dtype=float)
    G = np.zeros(n_steps + 1)
    if abs(a) < 1e-12:
        G[2:] = 1.0
    else:
        kk = k[2:]
        b = ((kk ** (a + 1) - (kk - 1) ** (a + 1)) / (a + 1)) ** (1.0 / a)
        G[2:] = (b * dt) ** a
    c11 = dt
    c12 = dt ** (a + 1) / (a + 1)
    c22 = dt ** (2 * a + 1) / (2 * a + 1)
    l11 = np.sqrt(c11)
    l21 = c12 / l11
    l22 = np.sqrt(max(c22 - l21 * l21, 0.0))
    chol = np.array([[l11, 0.0], [l21, l22]])
    var_Y = 2 * H * (c22 + np.concatenate([[0.0], np.cumsum(G[1:] ** 2) * dt]))
    var_Y[0] = 0.0
    return G, chol, var_Y


@dataclass
class MCResult:
    price: np.ndarray  # (nT, nK) normalised OTM prices
    price_se: np.ndarray
    iv: np.ndarray  # (nT, nK)
    iv_se: np.ndarray
    forward: np.ndarray  # (nT,) Monte Carlo estimate of E[S_T] (should be 1)
    forward_se: np.ndarray
    n_paths: int


class RoughBergomiMC:
    """Hybrid-scheme simulator on a fixed time grid covering a set of maturities."""

    def __init__(
        self,
        maturities,
        steps_per_year: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.maturities = np.asarray(maturities, float)
        self.n = int(steps_per_year)
        self.dt = 1.0 / self.n
        idx = np.rint(self.maturities * self.n).astype(int)
        if np.any(np.abs(idx / self.n - self.maturities) > 1e-9) or np.any(idx < 1):
            raise ValueError("every maturity must be a positive multiple of 1/steps_per_year")
        self.mat_steps = idx
        self.n_steps = int(idx.max())
        self.device = torch.device(device)
        self.dtype = dtype
        self.fft_len = 1 << int(np.ceil(np.log2(2 * self.n_steps + 1)))

    # ------------------------------------------------------------------ paths
    def _normals(self, n_half: int, n_dims: int, gen: torch.Generator) -> torch.Tensor:
        z = torch.randn(
            n_half, self.n_steps, n_dims, generator=gen, device=self.device, dtype=self.dtype
        )
        return torch.cat([z, -z], dim=0)  # antithetic partners: rows i and i + n_half

    def simulate(
        self,
        H: float,
        eta: float,
        xi0: float,
        n_paths: int,
        gen: torch.Generator,
        with_perp: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Simulate the variance on the grid.

        Returns dW (N, s), V (N, s + 1) at grid points t_0..t_s, Y (N, s + 1) and, when
        ``with_perp``, the independent increments dW_perp (N, s).
        """
        if n_paths % 2:
            raise ValueError("n_paths must be even (antithetic pairs)")
        s, dt = self.n_steps, self.dt
        G, chol, var_Y = hybrid_kernel(H, s, dt)
        z = self._normals(n_paths // 2, 3 if with_perp else 2, gen)
        dW = chol[0, 0] * z[..., 0]
        dWt = chol[1, 0] * z[..., 0] + chol[1, 1] * z[..., 1]

        # Far field: Y2[:, j] = sum_{k=2}^{j} G[k] dW[:, j - k], j = 1..s  (FFT convolution)
        L = self.fft_len
        g = torch.as_tensor(G, device=self.device, dtype=self.dtype)
        conv = torch.fft.irfft(torch.fft.rfft(dW, n=L) * torch.fft.rfft(g, n=L), n=L)
        Y = torch.zeros(n_paths, s + 1, device=self.device, dtype=self.dtype)
        Y[:, 1:] = np.sqrt(2 * H) * (dWt + conv[:, 1 : s + 1])

        vy = torch.as_tensor(var_Y, device=self.device, dtype=self.dtype)
        V = xi0 * torch.exp(eta * Y - 0.5 * eta * eta * vy)
        out = {"dW": dW, "V": V, "Y": Y}
        if with_perp:
            out["dW_perp"] = np.sqrt(dt) * z[..., 2]
        return out

    def _integrals(self, dW: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """I(T) = int_0^T sqrt(V) dW and Q(T) = int_0^T V dt at each maturity (left points)."""
        Vl = V[:, :-1]
        cols = torch.as_tensor(self.mat_steps - 1, device=self.device)
        I = torch.cumsum(torch.sqrt(Vl) * dW, dim=1)[:, cols]
        Q = torch.cumsum(Vl, dim=1)[:, cols] * self.dt
        return I.double(), Q.double()

    # ---------------------------------------------------------------- pricing
    def price(
        self,
        H: float,
        eta: float,
        rho: float,
        xi0: float,
        log_moneyness: np.ndarray,
        n_paths: int,
        seed: int = 0,
        estimator: str = "mixing",
        path_batch: int | None = None,
    ) -> MCResult:
        """Price OTM options on the grid k[i, j] (maturity i, strike j)."""
        k = np.asarray(log_moneyness, float)
        if k.shape[0] != len(self.maturities):
            raise ValueError("log_moneyness must have one row per maturity")
        K = torch.as_tensor(np.exp(k), device=self.device, dtype=torch.float64)
        is_call = torch.as_tensor(k >= 0, device=self.device)
        batch = min(path_batch or n_paths, n_paths)
        batch -= batch % 2
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))

        sums = torch.zeros(2, *k.shape, device=self.device, dtype=torch.float64)
        fsums = torch.zeros(2, len(self.maturities), device=self.device, dtype=torch.float64)
        n_pairs = 0
        done = 0
        while done < n_paths:
            m = min(batch, n_paths - done)
            m -= m % 2
            sim = self.simulate(H, eta, xi0, m, gen, with_perp=estimator == "plain")
            I, Q = self._integrals(sim["dW"], sim["V"])
            if estimator == "mixing":
                fwd = torch.exp(rho * I - 0.5 * rho * rho * Q)  # (m, nT)
                var = (1.0 - rho * rho) * Q
                pay = black_torch(fwd[..., None], K, var[..., None], is_call)
            elif estimator == "plain":
                Vl = sim["V"][:, :-1]
                cols = torch.as_tensor(self.mat_steps - 1, device=self.device)
                Ip = torch.cumsum(torch.sqrt(Vl) * sim["dW_perp"], dim=1)[:, cols].double()
                fwd = torch.exp(rho * I + np.sqrt(1 - rho * rho) * Ip - 0.5 * Q)
                intrinsic = torch.where(is_call, fwd[..., None] - K, K - fwd[..., None])
                pay = intrinsic.clamp_min(0.0)
            else:
                raise ValueError(f"unknown estimator {estimator!r}")
            h = m // 2
            pair = 0.5 * (pay[:h] + pay[h:])
            fpair = 0.5 * (fwd[:h] + fwd[h:])
            sums[0] += pair.sum(0)
            sums[1] += (pair * pair).sum(0)
            fsums[0] += fpair.sum(0)
            fsums[1] += (fpair * fpair).sum(0)
            n_pairs += h
            done += m

        mean = (sums[0] / n_pairs).cpu().numpy()
        var = ((sums[1] / n_pairs).cpu().numpy() - mean**2).clip(min=0.0)
        se = np.sqrt(var / max(n_pairs - 1, 1))
        fmean = (fsums[0] / n_pairs).cpu().numpy()
        fse = np.sqrt(
            ((fsums[1] / n_pairs).cpu().numpy() - fmean**2).clip(min=0.0) / max(n_pairs - 1, 1)
        )
        T = self.maturities[:, None]
        iv = implied_vol(mean, k, T, k >= 0)
        iv_se = implied_vol_se(iv, se, k, T)
        return MCResult(mean, se, iv, iv_se, fmean, fse, 2 * n_pairs)
