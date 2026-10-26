import numpy as np
import pytest
from scipy.integrate import quad

from roughvol.bs import black_np
from roughvol.heston import calibrate_heston, cos_put, heston_cf, heston_cumulants, heston_iv_grid
from roughvol.skew import atm_skew, fit_power_law, heston_short_skew, theoretical_skew

P = (0.04, 1.5, 0.05, 0.6, -0.7)


def gil_pelaez_put(k: float, T: float, params) -> float:
    """Independent reference: Gil-Pelaez inversion with adaptive quadrature."""

    def p_j(j):
        def f(u):
            if j == 1:
                v = heston_cf(u - 1j, T, *params) / heston_cf(-1j, T, *params)
            else:
                v = heston_cf(u, T, *params)
            return (np.exp(-1j * u * k) * v / (1j * u)).real

        return 0.5 + quad(f, 1e-10, 400, limit=2000)[0] / np.pi

    call = p_j(1) - np.exp(k) * p_j(2)
    return call - 1.0 + np.exp(k)


@pytest.mark.parametrize("T", [0.02, 0.25, 1.5])
def test_cos_matches_quadrature(T):
    k = np.array([-0.3, -0.1, 0.0, 0.1]) * np.sqrt(T)
    cos = cos_put(k, T, P, n_terms=512)
    ref = np.array([gil_pelaez_put(kk, T, P) for kk in k])
    np.testing.assert_allclose(cos, ref, atol=1e-8)


def test_cf_martingale_and_cumulants():
    T = 0.7
    assert abs(heston_cf(np.array([-1j]), T, *P)[0] - 1.0) < 1e-12  # E[S_T / F] = 1
    c1, c2 = heston_cumulants(T, *P)
    h = 1e-2
    lp = np.log(heston_cf(np.array([h, -h]), T, *P))
    assert c1 == pytest.approx(((lp[0] - lp[1]) / (2 * h)).imag, abs=1e-6)
    assert c2 == pytest.approx(-((lp[0] + lp[1]) / h**2).real, rel=1e-4)


def test_heston_tiny_volvol_is_black_scholes():
    p = (0.04, 1.0, 0.04, 1e-3, 0.0)
    k = np.array([[-0.1, 0.0, 0.1]])
    iv = heston_iv_grid(p, [0.5], k)
    np.testing.assert_allclose(iv, 0.2, atol=2e-5)
    put = cos_put(k[0], 0.5, p)
    np.testing.assert_allclose(put, black_np(k[0], 0.2 * np.sqrt(0.5), False), atol=2e-6)


def test_heston_calibration_recovers_params():
    mats = np.array([0.1, 0.5, 1.0])
    k = np.array([-0.3, -0.15, 0.0, 0.15, 0.3])[None, :] * np.sqrt(mats)[:, None]
    target = heston_iv_grid(P, mats, k)
    TT = np.repeat(mats[:, None], k.shape[1], axis=1)
    res = calibrate_heston(
        TT, k, target, [1e-3, 0.1, 1e-3, 0.05, -0.99], [0.5, 10, 0.5, 3, 0.5], n_starts=2
    )
    assert res["rmse"] < 1e-6


def test_skew_tools():
    T = np.array([0.01, 0.02, 0.05, 0.1])
    psi = theoretical_skew(T, 0.1, 1.9, -0.9)
    fit = fit_power_law(T, psi)
    assert fit["slope"] == pytest.approx(-0.4, abs=1e-10)
    z = np.array([-0.1, 0.0, 0.1])
    iv = 0.2 - 0.5 * z[None, :] * np.sqrt(T)[:, None]  # sigma = 0.2 - 0.5 k
    np.testing.assert_allclose(atm_skew(iv, z, T), 0.5)
    assert heston_short_skew(0.04, 0.6, -0.7) == pytest.approx(0.7 * 0.6 / 0.8)


def test_heston_short_dated_skew_is_flat():
    # Heston's ATM skew converges to rho*sigma/(4 sqrt(v0)) as T -> 0 instead of exploding.
    T = np.array([0.002, 0.005])
    z = np.array([-0.05, 0.0, 0.05])
    iv = heston_iv_grid(P, T, z[None, :] * np.sqrt(T)[:, None], n_terms=1024)
    psi = atm_skew(iv, z, T)
    np.testing.assert_allclose(psi, heston_short_skew(P[0], P[3], P[4]), rtol=0.05)
