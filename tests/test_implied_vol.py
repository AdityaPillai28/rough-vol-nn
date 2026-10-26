import numpy as np
import pytest

from roughvol.bs import black_np, bs_price_np
from roughvol.implied_vol import implied_vol, implied_vol_se, to_otm


def test_roundtrip_random():
    rng = np.random.default_rng(1)
    n = 20000
    k = rng.uniform(-1.5, 1.0, n)
    T = rng.uniform(0.002, 3.0, n)
    sigma = rng.uniform(0.02, 2.5, n)
    call = k >= 0
    price = black_np(k, sigma * np.sqrt(T), call)
    ok = price > 1e-280
    iv = implied_vol(price, k, T, call)
    assert np.isfinite(iv[ok]).all()
    assert np.max(np.abs(iv[ok] - sigma[ok]) / sigma[ok]) < 1e-9


@pytest.mark.parametrize(
    "k,T,sigma",
    [(-0.5, 0.01, 0.15), (0.3, 0.02, 0.1), (-2.0, 0.1, 0.3), (0.0, 1e-4, 0.2), (0.0, 5.0, 1.5)],
)
def test_roundtrip_extreme(k, T, sigma):
    call = k >= 0
    price = black_np(k, sigma * np.sqrt(T), call)
    assert price > 0
    assert implied_vol(price, k, T, call) == pytest.approx(sigma, rel=1e-8)


def test_itm_inputs_use_parity():
    k = np.array([-0.2, 0.2])
    s = np.array([0.1, 0.1])
    itm = black_np(k, s, np.array([True, False]))  # ITM call at k<0, ITM put at k>0
    iv = implied_vol(itm, k, 1.0, np.array([True, False]))
    np.testing.assert_allclose(iv, 0.1, rtol=1e-8)
    _, c = to_otm(itm, k, np.array([True, False]))
    np.testing.assert_array_equal(c, [False, True])


def test_arbitrage_violations_return_nan():
    # Negative price, zero price and price above the upper bound (1 for calls, e^k for puts).
    iv = implied_vol(
        np.array([-1e-3, 0.0, 1.2, np.exp(-0.1) + 1e-3]),
        np.array([0.1, 0.1, 0.1, -0.1]),
        0.5,
        np.array([True, True, True, False]),
    )
    assert np.isnan(iv).all()


def test_currency_units_and_vega_se():
    F, K, T, sig = 22000.0, 21500.0, 0.25, 0.14
    put = bs_price_np(F, K, T, sig, False)
    assert implied_vol(put / F, np.log(K / F), T, False) == pytest.approx(sig, rel=1e-10)
    # Delta method: a price shift of vega * dsigma maps to dsigma.
    k = np.log(K / F)
    dsig = 1e-4
    dp = black_np(k, (sig + dsig) * np.sqrt(T), False) - black_np(k, sig * np.sqrt(T), False)
    assert implied_vol_se(np.array(sig), np.array(dp), k, T) == pytest.approx(dsig, rel=1e-3)
