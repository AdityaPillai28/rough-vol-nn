"""Interactive rough Bergomi surface explorer: every slider move is one surrogate forward
pass (well under a millisecond), so the surface redraws instantly.

Run: python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from roughvol.config import load_config, resolve
from roughvol.grid import PARAM_NAMES, param_bounds
from roughvol.plotting import BLUE, ORANGE, PLOTLY_LAYOUT, SERIES, TEXT_2
from roughvol.skew import atm_skew, theoretical_skew
from roughvol.surrogate.models import Surrogate

st.set_page_config(page_title="rough-vol-nn", layout="wide")


@st.cache_resource
def load_model(name: str):
    for quick in (False, True):
        cfg = load_config(quick=quick)
        path = resolve(cfg["surrogate"]["out_dir"]) / f"{name}.pt"
        if path.exists():
            return Surrogate.load(path, "cpu"), cfg, quick
    return None, load_config(), False


st.markdown("#### Rough Bergomi implied-vol surface, priced by a neural surrogate")
model_name = st.sidebar.radio(
    "Surrogate",
    ["resmlp", "mlp"],
    format_func={"resmlp": "Residual MLP", "mlp": "Grid MLP (paper)"}.get,
)
model, cfg, quick = load_model(model_name)
if model is None:
    st.error(
        "No trained surrogate found. Run `make data train` (or `python -m roughvol all "
        "--quick`) first."
    )
    st.stop()
if quick:
    st.warning("Showing the --quick (CI) model: tiny training set, not representative.")

lo, hi = param_bounds(cfg["data"])
defaults = {"H": 0.1, "eta": 1.9, "rho": -0.9, "xi0": 0.04}
labels = {
    "H": "Hurst exponent H",
    "eta": "vol-of-vol η",
    "rho": "spot-vol correlation ρ",
    "xi0": "forward variance ξ₀",
}
theta = []
for i, n in enumerate(PARAM_NAMES):
    theta.append(
        st.sidebar.slider(
            labels[n],
            float(lo[i]),
            float(hi[i]),
            defaults[n],
            step=float((hi[i] - lo[i]) / 200),
            format="%.3f",
            key=n,
        )
    )
theta = np.array(theta)

iv = model.predict(theta[None])[0]
times = []
for _ in range(20):
    t0 = time.perf_counter()
    model.predict(theta[None])
    times.append(time.perf_counter() - t0)
dt = float(np.median(times))
T = model.maturities
k = model.z[None, :] * np.sqrt(T)[:, None]
st.sidebar.metric("Surrogate evaluation (median of 20)", f"{1e3 * dt:.2f} ms")
st.sidebar.caption(
    f"ATM vol {np.sqrt(theta[3]):.1%} = √ξ₀. Training box: "
    + ", ".join(f"{n}∈[{a:g}, {b:g}]" for n, a, b in zip(PARAM_NAMES, lo, hi, strict=True))
)

left, right = st.columns([3, 2])
with left:
    fig = go.Figure(
        go.Surface(
            x=k,
            y=np.repeat(T[:, None], k.shape[1], 1),
            z=iv,
            colorscale="Blues",
            showscale=False,
            contours={"z": {"show": True, "color": "white", "width": 1}},
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=560,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        scene={
            "xaxis_title": "log-moneyness k",
            "yaxis_title": "T (years)",
            "zaxis_title": "implied vol",
            "camera": {"eye": {"x": -1.6, "y": -1.5, "z": 0.8}},
        },
        title="Implied-volatility surface",
    )
    st.plotly_chart(fig, use_container_width=True)

LEGEND = {"orientation": "h", "y": -0.28, "x": 0, "font": {"size": 10, "color": TEXT_2}}

with right:
    sm = go.Figure()
    for c, r in zip(SERIES, [0, 2, 4, len(T) - 1], strict=True):
        sm.add_trace(
            go.Scatter(
                x=k[r], y=iv[r], mode="lines+markers", line={"color": c}, name=f"T = {T[r]:g}"
            )
        )
    sm.update_layout(
        **PLOTLY_LAYOUT,
        height=270,
        title="Smiles",
        legend=LEGEND,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis_title="k",
        yaxis_title="implied vol",
    )
    st.plotly_chart(sm, use_container_width=True)

    psi = atm_skew(iv, model.z, T)
    tt = np.geomspace(T[0], T[-1], 50)
    sk = go.Figure(
        [
            go.Scatter(
                x=T,
                y=psi,
                mode="markers",
                marker={"color": BLUE, "size": 9},
                name="surrogate ATM skew",
            ),
            go.Scatter(
                x=tt,
                y=theoretical_skew(tt, *theta[:3]),
                mode="lines",
                line={"color": ORANGE, "dash": "dash"},
                name="small-T theory ∝ T^(H−1/2)",
            ),
        ]
    )
    sk.update_layout(
        **PLOTLY_LAYOUT,
        height=270,
        title="ATM skew term structure (log-log)",
        legend=LEGEND,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis={"type": "log", "title": "T"},
        yaxis={"type": "log"},
    )
    st.plotly_chart(sk, use_container_width=True)

st.caption(
    "The theory line is the small-time asymptotic (Alòs-León-Vives / Fukasawa); it is "
    "expected to match the short end only. Grid: strikes at z = k/√T ∈ "
    f"[{model.z[0]:g}, {model.z[-1]:g}], maturities {T[0]:g}–{T[-1]:g}y."
)
