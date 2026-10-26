"""Statistical tests of the rough Bergomi Monte Carlo pricer.

Tolerances are expressed in Monte Carlo standard errors; seeds are fixed so the tests
are deterministic.
"""

import numpy as np
import pytest
import torch

from roughvol.bs import black_np
from roughvol.rbergomi import RoughBergomiMC, hybrid_kernel

MATS = np.array([0.02, 0.1, 0.5, 1.0])
Z = np.array([-0.3, -0.1, 0.0, 0.1, 0.25])
K_GRID = Z[None, :] * np.sqrt(MATS)[:, None]


@pytest.fixture(scope="module")
def mc():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return RoughBergomiMC(MATS, steps_per_year=200, device=dev)


def test_kernel_brownian_limit():
    # H = 1/2: the Volterra process is Brownian motion, so Var[Y_t] = t exactly.
    G, _, var_Y = hybrid_kernel(0.5, 100, 0.01)
    np.testing.assert_allclose(G[2:], 1.0)
    np.testing.assert_allclose(var_Y, np.arange(101) * 0.01, atol=1e-14)


@pytest.mark.parametrize("H", [0.05, 0.1, 0.3])
def test_kernel_variance_close_to_fbm(H):
    # The discretised variance should approach t^{2H} away from t = 0.
    n = 500
    _, _, var_Y = hybrid_kernel(H, n, 1.0 / n)
    t = np.arange(n + 1) / n
    rel = np.abs(var_Y[50:] / t[50:] ** (2 * H) - 1)
    assert rel.max() < 0.02


def test_volterra_empirical_variance(mc):
    gen = torch.Generator(device=mc.device).manual_seed(3)
    sim = mc.simulate(0.1, 1.0, 0.04, 40000, gen)
    _, _, var_Y = hybrid_kernel(0.1, mc.n_steps, mc.dt)
    emp = sim["Y"].double().var(0).cpu().numpy()
    np.testing.assert_allclose(emp[1:], var_Y[1:], rtol=0.03)


def test_antithetic_pairs(mc):
    gen = torch.Generator(device=mc.device).manual_seed(4)
    z = mc._normals(10, 2, gen)
    assert torch.equal(z[:10], -z[10:])


def test_forward_variance_is_flat(mc):
    # E[V_t] = xi0 on the grid by construction of the compensator.
    gen = torch.Generator(device=mc.device).manual_seed(5)
    V = mc.simulate(0.07, 2.5, 0.04, 40000, gen)["V"].double()
    m, se = V.mean(0), V.std(0) / np.sqrt(V.shape[0])
    z = ((m - 0.04) / se)[1:].abs()
    assert z.max() < 5.0  # 200 correlated grid points, so allow a wide band


@pytest.mark.parametrize("estimator", ["plain", "mixing"])
def test_eta_zero_is_black_scholes(mc, estimator):
    xi0 = 0.04
    r = mc.price(0.1, 0.0, -0.7, xi0, K_GRID, 40000, seed=11, estimator=estimator)
    bs = black_np(K_GRID, np.sqrt(xi0 * MATS)[:, None], K_GRID >= 0)
    z = (r.price - bs) / r.price_se
    assert np.abs(z).max() < 4.0
    np.testing.assert_allclose(r.iv, 0.2, atol=3e-3)


@pytest.mark.parametrize(
    "params", [(0.1, 1.9, -0.9, 0.04), (0.04, 3.5, -0.5, 0.1), (0.45, 1.0, -0.2, 0.02)]
)
def test_martingale(mc, params):
    H, eta, rho, xi0 = params
    r = mc.price(H, eta, rho, xi0, K_GRID, 60000, seed=21, estimator="plain")
    z = (r.forward - 1.0) / r.forward_se
    assert np.abs(z).max() < 4.0, (r.forward, r.forward_se)


def test_mixing_agrees_with_plain(mc):
    a = mc.price(0.1, 1.9, -0.9, 0.04, K_GRID, 60000, seed=1, estimator="plain")
    b = mc.price(0.1, 1.9, -0.9, 0.04, K_GRID, 60000, seed=2, estimator="mixing")
    z = (a.price - b.price) / np.hypot(a.price_se, b.price_se)
    assert np.abs(z).max() < 4.5
    # Conditioning on W can only reduce the variance.
    assert np.all(b.price_se <= a.price_se * 1.05)


def test_standard_errors_are_honest(mc):
    """The reported SE must match the spread of independent estimates."""
    est, ses = [], []
    for seed in range(40):
        r = mc.price(0.1, 1.5, -0.7, 0.04, K_GRID, 4000, seed=100 + seed)
        est.append(r.price)
        ses.append(r.price_se)
    ratio = np.std(est, axis=0, ddof=1) / np.mean(ses, axis=0)
    # With 40 replications the sample std has ~11% relative error.
    assert 0.65 < np.median(ratio) < 1.35
    assert np.all((ratio > 0.5) & (ratio < 1.6))


def test_se_scales_like_inverse_sqrt_n(mc):
    a = mc.price(0.1, 1.5, -0.7, 0.04, K_GRID, 8000, seed=7)
    b = mc.price(0.1, 1.5, -0.7, 0.04, K_GRID, 32000, seed=8, path_batch=8000)
    assert np.median(a.price_se / b.price_se) == pytest.approx(2.0, rel=0.1)


def test_negative_rho_gives_negative_skew(mc):
    r = mc.price(0.1, 1.9, -0.9, 0.04, K_GRID, 20000, seed=9)
    assert np.all(np.diff(r.iv, axis=1)[:, :2] < 0)


def test_rejects_off_grid_maturity():
    with pytest.raises(ValueError):
        RoughBergomiMC([0.013], steps_per_year=100)
