"""Index option-chain loader: forwards from put-call parity, liquidity and no-arbitrage
filters, implied vols. Nothing here generates market data; it only reads a user file.

Supported layouts (``format: auto`` detects them from the header):

* ``canonical``    columns: expiry, strike, option_type (C/P/CE/PE), and price columns
                   bid/ask and/or one of mid/settle/close/last; optional volume,
                   open_interest, date (valuation date), underlying (spot).
* ``nse_bhavcopy`` legacy NSE F&O bhavcopy (INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR,
                   OPTION_TYP, CLOSE, SETTLE_PR, CONTRACTS, OPEN_INT, TIMESTAMP).
* ``nse_udiff``    NSE UDiFF F&O bhavcopy (TradDt, FinInstrmTp, TckrSymb, XpryDt, StrkPric,
                   OptnTp, ClsPric, SttlmPric, TtlTradgVol, OpnIntrst, UndrlygPric).

Bhavcopies carry settlement/close prices only (no bid/ask), so the spread filter is
skipped for them and the volume / open-interest filters do the liquidity work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from roughvol.implied_vol import implied_vol

CANONICAL = [
    "valuation_date",
    "expiry",
    "strike",
    "type",
    "price",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "spot",
]


class MarketDataError(ValueError):
    pass


def _detect(cols: set[str]) -> str:
    if {"EXPIRY_DT", "STRIKE_PR", "OPTION_TYP"} <= cols:
        return "nse_bhavcopy"
    if {"XpryDt", "StrkPric", "OptnTp"} <= cols:
        return "nse_udiff"
    if {"expiry", "strike"} <= {c.lower() for c in cols}:
        return "canonical"
    raise MarketDataError(f"unrecognised option-chain layout; columns: {sorted(cols)}")


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.strip(), dayfirst=False, format="mixed").dt.normalize()


def read_chain(
    path: str | Path,
    fmt: str = "auto",
    underlying: str | None = None,
    valuation_date: str | None = None,
) -> pd.DataFrame:
    """Read a CSV into the canonical long format (one row per option)."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    fmt = _detect(set(raw.columns)) if fmt == "auto" else fmt
    df = pd.DataFrame(index=raw.index)
    if fmt == "nse_bhavcopy":
        sel = raw["INSTRUMENT"].astype(str).str.strip().eq("OPTIDX")
        if underlying:
            sel &= raw["SYMBOL"].astype(str).str.strip().eq(underlying)
        raw = raw[sel]
        df = pd.DataFrame(
            {
                "valuation_date": _dates(raw["TIMESTAMP"]),
                "expiry": _dates(raw["EXPIRY_DT"]),
                "strike": raw["STRIKE_PR"].astype(float),
                "type": raw["OPTION_TYP"].astype(str).str.strip(),
                "price": raw["SETTLE_PR"]
                .astype(float)
                .where(raw["SETTLE_PR"] > 0, raw["CLOSE"].astype(float)),
                "volume": raw["CONTRACTS"].astype(float),
                "open_interest": raw["OPEN_INT"].astype(float),
            }
        )
    elif fmt == "nse_udiff":
        sel = raw["FinInstrmTp"].astype(str).str.strip().eq("IDO")
        if underlying:
            sel &= raw["TckrSymb"].astype(str).str.strip().eq(underlying)
        raw = raw[sel]
        df = pd.DataFrame(
            {
                "valuation_date": _dates(raw["TradDt"]),
                "expiry": _dates(raw["XpryDt"]),
                "strike": raw["StrkPric"].astype(float),
                "type": raw["OptnTp"].astype(str).str.strip(),
                "price": raw["SttlmPric"]
                .astype(float)
                .where(raw["SttlmPric"] > 0, raw["ClsPric"].astype(float)),
                "volume": raw["TtlTradgVol"].astype(float),
                "open_interest": raw["OpnIntrst"].astype(float),
                "spot": raw["UndrlygPric"].astype(float),
            }
        )
    elif fmt == "canonical":
        low = {c.lower(): c for c in raw.columns}

        def col(*names):
            for n in names:
                if n in low:
                    return raw[low[n]]
            return None

        df["expiry"] = _dates(col("expiry"))
        df["strike"] = col("strike").astype(float)
        df["type"] = col("option_type", "type", "right", "cp").astype(str).str.strip()
        for name, alts in {
            "bid": ("bid",),
            "ask": ("ask", "offer"),
            "volume": ("volume", "contracts"),
            "open_interest": ("open_interest", "oi", "openinterest"),
            "spot": ("underlying", "spot", "underlying_price"),
        }.items():
            c = col(*alts)
            if c is not None:
                df[name] = c.astype(float)
        p = col("mid", "settle", "settlement", "close", "last", "ltp", "price")
        if p is not None:
            df["price"] = p.astype(float)
        if "bid" in df and "ask" in df:
            mid = 0.5 * (df["bid"] + df["ask"])
            ok = (df["bid"] > 0) & (df["ask"] > df["bid"])
            df["price"] = mid.where(ok, df.get("price", np.nan))
        if "price" not in df:
            raise MarketDataError("no price column (bid/ask, mid, settle, close or last)")
        d = col("date", "valuation_date", "trade_date", "timestamp")
        if d is not None:
            df["valuation_date"] = _dates(d)
    else:
        raise MarketDataError(f"unknown format {fmt!r}")

    t = df["type"].str.upper().str[0]
    df["type"] = t.map({"C": "C", "P": "P"})
    df = df.dropna(subset=["type", "strike", "price", "expiry"])
    if valuation_date is not None:
        df["valuation_date"] = pd.Timestamp(valuation_date)
    if "valuation_date" not in df or df["valuation_date"].isna().all():
        raise MarketDataError("valuation date unknown: set market.valuation_date in the config")
    if df["valuation_date"].nunique() != 1:
        raise MarketDataError("file contains several trade dates; pass one day at a time")
    for c in CANONICAL:
        if c not in df:
            df[c] = np.nan
    df["format"] = fmt
    return df[[*CANONICAL, "format"]].reset_index(drop=True)


def implied_forward(calls: pd.DataFrame, puts: pd.DataFrame, n_strikes: int = 8):
    """Put-call parity C - P = D (F - K): regress C - P on K over the strikes nearest the
    money. Returns (F, D, n_used) or None when there is not enough information."""
    both = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    if len(both) < 3:
        return None
    both["diff"] = both["price_c"] - both["price_p"]
    both = both.iloc[np.argsort(np.abs(both["diff"].to_numpy()))[:n_strikes]]
    if len(both) < 3:
        return None
    slope, intercept = np.polyfit(both["strike"], both["diff"], 1)
    D = -slope
    if not (0.8 < D < 1.05):  # discount factor implausible -> bad quotes
        return None
    return intercept / D, D, len(both)


def _arbitrage_mask(K: np.ndarray, c: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Keep-mask for normalised call prices c(K) (sorted by K): bounds, monotonicity and
    convexity. Offending quotes are removed greedily until the remaining set is clean."""
    keep = (c >= np.maximum(1.0 - K, 0.0) - tol) & (c <= 1.0 + tol)
    for _ in range(len(K)):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        k, v = K[idx], c[idx]
        slope = np.diff(v) / np.diff(k)
        bad_mono = np.flatnonzero(slope > tol)  # call prices must decrease in strike
        bad_conv = np.flatnonzero(np.diff(slope) < -tol)  # and be convex
        if len(bad_mono) == 0 and len(bad_conv) == 0:
            break
        # Drop the quote at the centre of the first violation.
        j = bad_mono[0] + 1 if len(bad_mono) else bad_conv[0] + 1
        keep[idx[j]] = False
    return keep


@dataclass
class MarketSurface:
    quotes: pd.DataFrame  # expiry, T, strike, k, z, type, price, iv, forward, discount
    forwards: pd.DataFrame
    report: dict = field(default_factory=dict)


def build_surface(df: pd.DataFrame, cfg: dict, t_min: float, t_max: float) -> MarketSurface:
    """Filter a canonical chain and turn it into OTM implied vols on (T, k)."""
    f = cfg["filters"]
    rep: dict = {
        "rows_in": len(df),
        "format": str(df["format"].iloc[0]),
        "valuation_date": str(df["valuation_date"].iloc[0].date()),
    }
    val = df["valuation_date"].iloc[0]
    df = df.copy()
    df["days"] = (df["expiry"] - val).dt.days
    df["T"] = df["days"] / 365.0
    liq = (df["price"] >= f["min_price"]) & (df["days"] >= f["min_days"])
    if df["volume"].notna().any():
        liq &= df["volume"].fillna(0) >= f["min_volume"]
    if df["open_interest"].notna().any():
        liq &= df["open_interest"].fillna(0) >= f["min_open_interest"]
    if df["bid"].notna().any() and df["ask"].notna().any():
        spread = (df["ask"] - df["bid"]) / df["price"]
        liq &= spread.isna() | (spread <= f["max_rel_spread"])
    rep["rows_after_liquidity"] = int(liq.sum())
    df = df[liq & (df["T"] <= f["max_years"])]

    rows, fwd_rows = [], []
    n_parity_fail = n_arb = 0
    for expiry, g in df.groupby("expiry"):
        calls = g[g["type"] == "C"][["strike", "price"]]
        puts = g[g["type"] == "P"][["strike", "price"]]
        res = implied_forward(calls, puts, int(cfg["parity"]["n_strikes"]))
        if res is None:
            n_parity_fail += 1
            continue
        F, D, n_used = res
        T = float(g["T"].iloc[0])
        fwd_rows.append(
            {
                "expiry": expiry,
                "T": T,
                "forward": F,
                "discount": D,
                "implied_rate": -np.log(D) / T,
                "n_parity_strikes": n_used,
                "spot": float(g["spot"].iloc[0]) if g["spot"].notna().any() else None,
            }
        )
        # OTM side only: puts below the forward, calls at/above it.
        otm = pd.concat(
            [
                puts[puts["strike"] < F].assign(type="P"),
                calls[calls["strike"] >= F].assign(type="C"),
            ]
        ).sort_values("strike")
        Kn = otm["strike"].to_numpy() / F
        p = otm["price"].to_numpy() / (D * F)  # undiscounted, forward-normalised
        c_equiv = np.where(otm["type"] == "C", p, p + 1.0 - Kn)
        keep = _arbitrage_mask(Kn, c_equiv)
        n_arb += int((~keep).sum())
        otm, Kn, p = otm[keep], Kn[keep], p[keep]
        k = np.log(Kn)
        iv = implied_vol(p, k, T, otm["type"].to_numpy() == "C")
        rows.append(
            pd.DataFrame(
                {
                    "expiry": expiry,
                    "T": T,
                    "strike": otm["strike"],
                    "k": k,
                    "z": k / np.sqrt(T),
                    "type": otm["type"],
                    "price": p,
                    "iv": iv,
                    "forward": F,
                    "discount": D,
                }
            )
        )
    rep["expiries_parity_failed"] = n_parity_fail
    rep["quotes_removed_arbitrage"] = n_arb
    q = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "expiry",
                "T",
                "strike",
                "k",
                "z",
                "type",
                "price",
                "iv",
                "forward",
                "discount",
            ]
        )
    )
    n0 = len(q)
    q = q[np.isfinite(q["iv"])]
    rep["quotes_iv_failed"] = n0 - len(q)
    zlo, zhi = f["z_range"]
    inside = (q["z"] >= zlo) & (q["z"] <= zhi) & (q["T"] >= t_min - 1e-9) & (q["T"] <= t_max)
    rep["quotes_outside_surrogate_domain"] = int((~inside).sum())
    q = q[inside]
    counts = q.groupby("expiry")["iv"].transform("size")
    q = q[counts >= f["min_quotes_per_expiry"]].reset_index(drop=True)
    rep["quotes_final"] = len(q)
    rep["expiries_final"] = int(q["expiry"].nunique())
    q_t = q.groupby("expiry").size()
    rep["calendar_violations"] = _calendar_violations(q) if len(q_t) > 1 else 0
    return MarketSurface(q, pd.DataFrame(fwd_rows), rep)


def _calendar_violations(q: pd.DataFrame) -> int:
    """Count ATM-interpolated total-variance decreases across expiries (reported only)."""
    w = []
    for _, g in q.groupby("T"):
        g = g.sort_values("k")
        if g["k"].min() <= 0 <= g["k"].max():
            w.append(float(np.interp(0.0, g["k"], g["iv"] ** 2 * g["T"])))
    return int(np.sum(np.diff(w) < 0))


class QuoteInterpolator:
    """Differentiable map from the surrogate grid (B, nT, nK) to quotes (B, M).

    Linear in z = k / sqrt(T) on each grid maturity, then linear in total implied
    variance across maturities (which preserves the absence of calendar arbitrage
    between grid maturities).
    """

    def __init__(self, grid_T, grid_z, T, z):
        gT, gz = np.asarray(grid_T, float), np.asarray(grid_z, float)
        T, z = np.asarray(T, float), np.asarray(z, float)
        if T.min() < gT[0] - 1e-9 or T.max() > gT[-1] + 1e-9:
            raise ValueError("quote maturities outside the grid")
        jz = np.clip(np.searchsorted(gz, z) - 1, 0, len(gz) - 2)
        wz = (z - gz[jz]) / (gz[jz + 1] - gz[jz])
        iT = np.clip(np.searchsorted(gT, T) - 1, 0, len(gT) - 2)
        wT = (T - gT[iT]) / (gT[iT + 1] - gT[iT])
        self.jz, self.iT = torch.as_tensor(jz), torch.as_tensor(iT)
        self.wz = torch.as_tensor(wz, dtype=torch.float32)
        self.wT = torch.as_tensor(wT, dtype=torch.float32)
        self.gT = torch.as_tensor(gT, dtype=torch.float32)
        self.T = torch.as_tensor(T, dtype=torch.float32)

    def to(self, device):
        for name in ("jz", "iT", "wz", "wT", "gT", "T"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def __call__(self, grid_iv):
        def at(i):  # vol at maturity row i (M,), interpolated in z
            lo = grid_iv[:, i, self.jz]
            hi = grid_iv[:, i, self.jz + 1]
            return lo + self.wz * (hi - lo)

        s0, s1 = at(self.iT), at(self.iT + 1)
        w0 = s0 * s0 * self.gT[self.iT]
        w1 = s1 * s1 * self.gT[self.iT + 1]
        w = w0 + self.wT * (w1 - w0)
        return (w / self.T).clamp_min(1e-10).sqrt()
