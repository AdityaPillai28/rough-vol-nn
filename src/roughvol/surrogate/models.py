"""Grid-based surrogates of the pricing map theta = (H, eta, rho, xi0) -> IV grid.

"Grid-based" (Horvath, Muguruza & Tomas 2021): the network outputs the whole
(maturity x strike) implied-vol grid at once, rather than taking (T, k) as inputs. One
forward pass therefore prices the full surface, and the Jacobian with respect to the four
model parameters (needed by Levenberg-Marquardt) is cheap and exact via autodiff.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

ACTIVATIONS = {"elu": nn.ELU, "silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: list[int], activation: str = "elu"):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), ACTIVATIONS[activation]()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str):
        super().__init__()
        self.act = ACTIVATIONS[activation]()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(self.act(self.fc1(self.act(h))))


class ResMLP(nn.Module):
    """Small pre-activation residual MLP (no normalisation layers: they add kinks in
    parameter space that hurt the smoothness gradient-based calibration relies on)."""

    def __init__(
        self, n_in: int, n_out: int, width: int = 128, n_blocks: int = 3, activation: str = "silu"
    ):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, activation) for _ in range(n_blocks)])
        self.act = ACTIVATIONS[activation]()
        self.out = nn.Linear(width, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.blocks(self.inp(x))))


def build_network(spec: dict, n_in: int, n_out: int) -> nn.Module:
    if spec["kind"] == "mlp":
        return MLP(n_in, n_out, list(spec["hidden"]), spec.get("activation", "elu"))
    if spec["kind"] == "resmlp":
        return ResMLP(
            n_in, n_out, int(spec["width"]), int(spec["n_blocks"]), spec.get("activation", "silu")
        )
    raise ValueError(f"unknown network kind {spec['kind']!r}")


class Surrogate(nn.Module):
    """Network plus input/output scaling. ``forward`` maps raw params to IVs (B, nT, nK)."""

    def __init__(self, net: nn.Module, spec: dict, lo, hi, y_mean, y_std, maturities, z):
        super().__init__()
        self.net = net
        self.spec = dict(spec)
        f32 = torch.float32
        self.register_buffer("lo", torch.as_tensor(np.asarray(lo), dtype=f32))
        self.register_buffer("hi", torch.as_tensor(np.asarray(hi), dtype=f32))
        self.register_buffer("y_mean", torch.as_tensor(np.asarray(y_mean), dtype=f32))
        self.register_buffer("y_std", torch.as_tensor(np.asarray(y_std), dtype=f32))
        self.maturities = np.asarray(maturities, float)
        self.z = np.asarray(z, float)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return len(self.maturities), len(self.z)

    def scale_x(self, theta: torch.Tensor) -> torch.Tensor:
        return 2.0 * (theta - self.lo) / (self.hi - self.lo) - 1.0

    def forward_scaled(self, x: torch.Tensor) -> torch.Tensor:
        """Standardised outputs from scaled inputs (used in training)."""
        return self.net(x)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        y = self.net(self.scale_x(theta)) * self.y_std + self.y_mean
        return y.reshape(*theta.shape[:-1], *self.grid_shape)

    @torch.no_grad()
    def predict(self, theta) -> np.ndarray:
        t = torch.as_tensor(np.asarray(theta, np.float32), device=self.lo.device)
        return self(t).cpu().numpy()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "spec": self.spec,
                "state": self.net.state_dict(),
                "lo": self.lo.cpu().numpy(),
                "hi": self.hi.cpu().numpy(),
                "y_mean": self.y_mean.cpu().numpy(),
                "y_std": self.y_std.cpu().numpy(),
                "maturities": self.maturities,
                "z": self.z,
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> Surrogate:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        n_out = len(ck["maturities"]) * len(ck["z"])
        net = build_network(ck["spec"], 4, n_out)
        net.load_state_dict(ck["state"])
        s = cls(
            net,
            ck["spec"],
            ck["lo"],
            ck["hi"],
            ck["y_mean"],
            ck["y_std"],
            ck["maturities"],
            ck["z"],
        )
        return s.to(device).eval()
