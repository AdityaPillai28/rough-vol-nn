"""The (maturity x strike) grid the surrogate is trained on.

Strikes are expressed as log-moneyness k = ln(K / F) = z * sqrt(T), with z fixed across
maturities, so each column of the grid sits at a comparable distance from the money in
units of sqrt(T).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    maturities: np.ndarray  # (nT,)
    z: np.ndarray  # (nK,)

    @classmethod
    def from_config(cls, cfg: dict) -> Grid:
        g = cfg["grid"] if "grid" in cfg else cfg
        return cls(np.asarray(g["maturities"], float), np.asarray(g["z"], float))

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.maturities), len(self.z)

    @property
    def size(self) -> int:
        return len(self.maturities) * len(self.z)

    @property
    def log_moneyness(self) -> np.ndarray:
        """k[i, j] = z[j] * sqrt(T[i]), shape (nT, nK)."""
        return self.z[None, :] * np.sqrt(self.maturities)[:, None]

    @property
    def strikes(self) -> np.ndarray:
        """K / F, shape (nT, nK)."""
        return np.exp(self.log_moneyness)

    @property
    def atm_index(self) -> int:
        return int(np.argmin(np.abs(self.z)))


PARAM_NAMES = ("H", "eta", "rho", "xi0")


def param_bounds(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    p = cfg["params"]
    lo = np.array([p[n][0] for n in PARAM_NAMES], float)
    hi = np.array([p[n][1] for n in PARAM_NAMES], float)
    return lo, hi
