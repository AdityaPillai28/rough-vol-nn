"""Market loader tests on SYNTHETIC chains generated inside the test (Heston prices with a
known forward and rate). These fixtures exist only to test the code path; they are never
written to the repo or used as market data."""

import numpy as np
import pandas as pd
import pytest
import torch

from roughvol.bs import bs_price_np
from roughvol.config import load_config
from roughvol.heston import heston_iv_grid
from roughvol.market import (
    MarketDataError,
    QuoteInterpolator,
    build_surface,
    implied_forward,
    read_chain,
)

F0, R = 22000.0, 0.07
HP = (0.02, 2.0, 0.025, 0.5, -0.7)
VAL = pd.Timestamp("2025-01-06")


def _synthetic_chain(days=(10, 38, 94)):
    rows = []
    for d in days:
        T = d / 365
        K = np.arange(18000, 26001, 100.0)
        k = np.log(K / F0)
        iv = heston_iv_grid(HP, [T], k[None])[0]
        D = np.exp(-R * T)
        for cp in ("C", "P"):
            px = D * bs_price_np(F0, K, T, iv, cp == "C")
            for Ki, p in zip(K, px, strict=True):
                rows.append(
                    {
                        "date": VAL.date(),
                        "expiry": (VAL + pd.Timedelta(days=d)).date(),
                        "strike": Ki,
                        "option_type": cp + "E",
                        "bid": p * 0.99,
                        "ask": p * 1.01,
                        "volume": 100,
                        "open_interest": 1000,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def mcfg():
    return load_config()["market"]


def test_forward_from_parity():
    df = _synthetic_chain()
    df["price"] = 0.5 * (df["bid"] + df["ask"])
    g = df[df["expiry"] == df["expiry"].iloc[0]]
    calls = g[g["option_type"] == "CE"][["strike", "price"]]
    puts = g[g["option_type"] == "PE"][["strike", "price"]]
    F, D, _ = implied_forward(calls, puts)
    assert F == pytest.approx(F0, rel=1e-6)
    assert D == pytest.approx(np.exp(-R * 10 / 365), rel=1e-6)


def test_build_surface_recovers_ivs_and_filters(tmp_path, mcfg):
    df = _synthetic_chain()
    # One arbitrage violation (a call priced above a lower-strike call) and one
    # illiquid quote with a 60% spread.
    first = df["expiry"] == df["expiry"].iloc[0]
    i = df.index[first & (df["option_type"] == "CE") & (df["strike"] == 22600)][0]
    df.loc[i, ["bid", "ask"]] = df.loc[i, ["bid", "ask"]] * 3
    j = df.index[first & (df["option_type"] == "PE") & (df["strike"] == 21500)][0]
    df.loc[j, "bid"] = df.loc[j, "ask"] * 0.4
    path = tmp_path / "chain.csv"
    df.to_csv(path, index=False)

    chain = read_chain(path)
    assert chain["format"].iloc[0] == "canonical"
    surf = build_surface(chain, mcfg, t_min=0.02, t_max=1.5)
    rep = surf.report
    assert rep["quotes_removed_arbitrage"] >= 1
    assert rep["rows_after_liquidity"] < rep["rows_in"]
    np.testing.assert_allclose(surf.forwards["forward"], F0, rtol=1e-4)
    q = surf.quotes
    t0 = 10 / 365
    assert not ((q["strike"] == 22600) & (q["T"] == t0)).any()  # arbitrage
    assert not ((q["strike"] == 21500) & (q["T"] == t0)).any()  # wide spread
    ref = np.concatenate(
        [heston_iv_grid(HP, [T], g["k"].to_numpy()[None])[0] for T, g in q.groupby("T", sort=False)]
    )
    np.testing.assert_allclose(q["iv"].to_numpy(), ref, atol=2e-3)  # bid/ask mid of +-1%
    assert set(q["type"]) == {"C", "P"}
    assert (q.loc[q["type"] == "P", "k"] < 0).all()


def test_nse_udiff_detection(tmp_path):
    df = _synthetic_chain(days=(38,))
    ud = pd.DataFrame(
        {
            "TradDt": df["date"],
            "FinInstrmTp": "IDO",
            "TckrSymb": "NIFTY",
            "XpryDt": df["expiry"],
            "StrkPric": df["strike"],
            "OptnTp": df["option_type"],
            "ClsPric": 0.5 * (df["bid"] + df["ask"]),
            "SttlmPric": 0.5 * (df["bid"] + df["ask"]),
            "TtlTradgVol": 10,
            "OpnIntrst": 100,
            "UndrlygPric": F0,
        }
    )
    path = tmp_path / "udiff.csv"
    ud.to_csv(path, index=False)
    chain = read_chain(path, underlying="NIFTY")
    assert chain["format"].iloc[0] == "nse_udiff"
    assert len(chain) == len(df)


def test_unknown_layout_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(p, index=False)
    with pytest.raises(MarketDataError):
        read_chain(p)


def test_quote_interpolator_exact_on_grid_nodes():
    gT = np.array([0.1, 0.5, 1.0])
    gz = np.array([-0.2, 0.0, 0.2])
    grid = torch.tensor(np.random.default_rng(0).uniform(0.1, 0.3, (2, 3, 3)), dtype=torch.float32)
    T = np.array([0.1, 0.5, 1.0, 0.5])
    z = np.array([-0.2, 0.0, 0.2, 0.2])
    out = QuoteInterpolator(gT, gz, T, z)(grid)
    expect = grid[:, [0, 1, 2, 1], [0, 1, 2, 2]]
    torch.testing.assert_close(out, expect, rtol=1e-6, atol=1e-6)
    # Between maturities total variance is linear in T.
    mid = QuoteInterpolator(gT, gz, [0.3], [0.0])(grid)
    w = 0.5 * (grid[:, 0, 1] ** 2 * 0.1 + grid[:, 1, 1] ** 2 * 0.5)
    torch.testing.assert_close(mid[:, 0], torch.sqrt(w / 0.3), rtol=1e-6, atol=1e-6)
