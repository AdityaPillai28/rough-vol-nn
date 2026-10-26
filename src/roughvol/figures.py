"""All README figures, built only from results/*.json and results/arrays/*.npz."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from roughvol.config import resolve
from roughvol.data import load_reference
from roughvol.grid import PARAM_NAMES, Grid
from roughvol.plotting import (
    AQUA,
    BLUE,
    DIV_CMAP,
    ORANGE,
    SEQ_CMAP,
    SERIES,
    TEXT,
    TEXT_2,
    save,
    set_style,
)
from roughvol.utils import load_json

LABELS = {"H": "H", "eta": r"$\eta$", "rho": r"$\rho$", "xi0": r"$\xi_0$"}
MODEL_NAMES = {"mlp": "Grid MLP (3x30, paper)", "resmlp": "Residual MLP"}


def _npz(cfg, name):
    p = resolve(cfg["results_dir"]) / "arrays" / f"{name}.npz"
    return dict(np.load(p)) if p.exists() else None


def _json(cfg, name):
    p = resolve(cfg["results_dir"]) / f"{name}.json"
    return load_json(p) if p.exists() else None


def _heat(ax, M, z, T, title, cmap, norm=None, fmt="{:.2f}", vmax=None):
    im = ax.imshow(
        M,
        cmap=cmap,
        norm=norm,
        vmin=None if norm else 0,
        vmax=None if norm else vmax,
        aspect="auto",
        origin="lower",
    )
    ax.set_xticks(range(len(z)), [f"{v:+.2f}" for v in z], rotation=60)
    ax.set_yticks(range(len(T)), [f"{v:g}" for v in T])
    ax.set_xlabel(r"$z = k/\sqrt{T}$")
    ax.set_title(title)
    ax.grid(False)
    lim = norm.vmax if norm is not None else (vmax or np.nanmax(M))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            dark = (
                (v > 0.6 * lim)
                if norm is None
                else abs(v - norm.vcenter) > 0.6 * (lim - norm.vcenter)
            )
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if dark else TEXT,
            )
    return im


def fig_error_heatmaps(cfg, out):
    a = _npz(cfg, "surrogate_errors")
    if a is None:
        return
    z, T = a["z"], a["maturities"]
    names = [n for n in ("mlp", "resmlp") if f"{n}_err_ref_grid" in a]
    grids = [100 * a["noise_floor_grid"]] + [100 * a[f"{n}_err_ref_grid"] for n in names]
    vmax = max(np.nanmax(g) for g in grids)
    fig, axes = plt.subplots(
        1, len(grids) + 1, figsize=(4.3 * (len(grids) + 1), 4.2), constrained_layout=True
    )
    ev = _json(cfg, "surrogate_eval") or {}
    n_ref, n_lab = ev.get("n_reference", "?"), cfg["data"]["mc"]["n_paths"]
    titles = [f"MC noise floor: |MC {n_lab} paths - ref| / ref"] + [
        MODEL_NAMES[n] + ": |NN - ref| / ref" for n in names
    ]
    for ax, g, t in zip(axes, grids, titles, strict=False):
        im = _heat(ax, g, z, T, t, SEQ_CMAP, vmax=vmax)
    axes[0].set_ylabel("maturity T (years)")
    fig.colorbar(im, ax=axes[: len(grids)], label="mean relative IV error (%)", shrink=0.85)
    best = names[-1]
    ratio = a[f"{best}_err_ref_grid"] / a["noise_floor_grid"]
    norm = TwoSlopeNorm(vcenter=1.0, vmin=min(0.2, ratio.min()), vmax=max(2.0, ratio.max()))
    im2 = _heat(
        axes[-1], ratio, z, T, f"{MODEL_NAMES[best]} error / noise floor", DIV_CMAP, norm=norm
    )
    fig.colorbar(im2, ax=axes[-1], label="ratio (<1: below MC noise)", shrink=0.85)
    fig.suptitle(
        "Surrogate error over strike x maturity vs. the Monte Carlo noise floor "
        f"({n_ref} high-precision reference surfaces)",
        color=TEXT,
        fontsize=11,
    )
    save(fig, out / "error_heatmaps.png")


def fig_recovery(cfg, out):
    a = _npz(cfg, "recovery")
    if a is None:
        return
    name = cfg["calibration"]["model"]
    true, est = a["true"], a[f"{name}_lm"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), constrained_layout=True)
    for i, (ax, n) in enumerate(zip(axes, PARAM_NAMES, strict=True)):
        ax.scatter(true[:, i], est[:, i], s=4, alpha=0.35, color=BLUE, linewidths=0)
        lo, hi = true[:, i].min(), true[:, i].max()
        ax.plot([lo, hi], [lo, hi], color=TEXT_2, lw=1, ls="--")
        mae = np.mean(np.abs(est[:, i] - true[:, i]))
        ax.set_title(f"{LABELS[n]}   MAE {mae:.3g}")
        ax.set_xlabel(f"true {LABELS[n]}")
        ax.set_ylabel(f"calibrated {LABELS[n]}")
    fig.suptitle(
        f"Parameter recovery on {len(true)} held-out MC surfaces (LM through {MODEL_NAMES[name]})",
        color=TEXT,
    )
    save(fig, out / "recovery_scatter.png")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    clean = a[f"{name}_clean"]
    for i, (ax, n) in enumerate(zip(axes, PARAM_NAMES, strict=True)):
        e1 = est[:, i] - true[:, i]
        e0 = clean[:, i] - true[:, i]
        lim = np.quantile(np.abs(e1), 0.99)
        bins = np.linspace(-lim, lim, 60)
        ax.hist(e1, bins=bins, color=BLUE, alpha=0.85, label="MC targets (noisy)")
        ax.hist(e0, bins=bins, color=ORANGE, alpha=0.7, label="surrogate targets (noise-free)")
        ax.set_xlabel(f"calibrated - true {LABELS[n]}")
        ax.set_title(LABELS[n])
    axes[0].set_ylabel("count")
    axes[0].legend(loc="upper left")
    fig.suptitle("Calibration error distributions", color=TEXT)
    save(fig, out / "recovery_errors.png")


def fig_atm_skew(cfg, out):
    s = _json(cfg, "atm_skew")
    if s is None:
        return
    T = np.array(s["maturities"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    ax = axes[0]
    for c, (key, cur) in zip(SERIES, s["curves"].items(), strict=False):
        psi = np.array(cur["psi"])
        ax.errorbar(
            T,
            psi,
            yerr=2 * np.array(cur["psi_se"]),
            fmt="o",
            ms=5,
            color=c,
            capsize=0,
            label=f"H = {key}: MC, fitted slope "
            f"{cur['fit_short']['slope']:+.2f} (theory {cur['theory_slope']:+.2f})",
        )
        ax.plot(T, cur["theory"], color=c, lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel(r"ATM skew $|\partial_k \sigma_{BS}|$")
    ax.set_title(r"rBergomi ATM skew vs. $T^{H-1/2}$ (dashed: small-time asymptotics)")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    base = f"{s['params']['H']:g}"
    cur = s["curves"][base]
    ax.plot(T, cur["psi"], "o", color=BLUE, label=f"rBergomi MC (H = {base})")
    ax.plot(T, cur["theory"], color=BLUE, lw=1.2, ls="--", label=r"theory $\propto T^{H-1/2}$")
    sg = s["surrogate"]
    ax.plot(
        sg["maturities"],
        sg["psi"],
        "s",
        mfc="none",
        color=AQUA,
        mew=1.5,
        ms=7,
        label="surrogate (grid maturities)",
    )
    h = s["heston"]
    ax.plot(T, h["psi"], color=ORANGE, marker="^", label="Heston fitted to the same surface")
    ax.axhline(h["short_limit"], color=ORANGE, lw=1, ls=":")
    ax.text(
        T[0],
        h["short_limit"] * 0.86,
        r"Heston $T\to 0$ limit $\rho\sigma/(4\sqrt{v_0})$",
        color=TEXT_2,
        fontsize=7.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("maturity T (years)")
    ax.set_title("Only a rough model reproduces the short-end skew explosion")
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, out / "atm_skew_loglog.png")


def fig_heston(cfg, out):
    h = _json(cfg, "heston_baseline")
    if h is None:
        return
    r = h["rmse_by_maturity"]
    T = np.array(r["maturities"])
    x = np.arange(len(T))
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    ax.bar(
        x - 0.2,
        100 * np.array(r["rbergomi"]),
        0.38,
        color=BLUE,
        label="rBergomi (via surrogate, 4 params)",
    )
    ax.bar(x + 0.2, 100 * np.array(r["heston"]), 0.38, color=ORANGE, label="Heston (COS, 5 params)")
    ax.set_xticks(x, [f"{t:g}" for t in T])
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("IV RMSE (vol points)")
    ax.set_title(f"Fit to {h['n_surfaces']} rBergomi reference surfaces, by maturity")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save(fig, out / "heston_vs_rbergomi.png")


def fig_training(cfg, out):
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    found = False
    for c, n in zip(SERIES, ("mlp", "resmlp"), strict=False):
        t = _json(cfg, f"surrogate_train_{n}")
        if t is None:
            continue
        found = True
        h = t["history"]
        ax.plot([e["epoch"] for e in h], [e["val"] for e in h], color=c, label=f"{MODEL_NAMES[n]}")
        ax.plot([e["epoch"] for e in h], [e["train"] for e in h], color=c, lw=1, alpha=0.5)
    if not found:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardised IV)")
    ax.set_title("Training (thin) and validation (thick) loss")
    ax.legend()
    save(fig, out / "training_curves.png")


def _surface_axes(fig, pos, title):
    ax = fig.add_subplot(pos, projection="3d")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("log-moneyness k", fontsize=8)
    ax.set_ylabel("T (years)", fontsize=8)
    ax.set_zlabel("implied vol", fontsize=8)
    ax.view_init(elev=24, azim=-128)
    ax.tick_params(labelsize=7)
    return ax


def fig_surfaces(cfg, out):
    """3D surfaces. With market data: market quotes vs calibrated rBergomi. Without:
    a synthetic MC reference surface vs the surrogate calibrated to it (labelled so)."""
    from roughvol.surrogate.train import load_surrogate

    grid = Grid.from_config(cfg["data"])
    model = load_surrogate(cfg, cfg["calibration"]["model"], "cpu")
    mk = _json(cfg, "market")
    qpath = resolve(cfg["results_dir"]) / "arrays" / "market_quotes.csv"
    fig = plt.figure(figsize=(13, 5.4), constrained_layout=True)
    if mk and mk.get("status") == "ok" and qpath.exists():
        import pandas as pd

        q = pd.read_csv(qpath)
        theta = np.array([mk["rbergomi"]["params"][n] for n in PARAM_NAMES])
        title_l = f"Market quotes ({mk['report']['valuation_date']})"
        title_r = "rBergomi calibrated through the surrogate"
        kq, Tq, ivq = q["k"], q["T"], q["iv"]
    else:
        ref = load_reference(cfg)
        h = _npz(cfg, "heston")
        i = 0
        theta = h["rb_theta"][i] if h is not None else ref["params"][i]
        title_l = "SYNTHETIC 'market': MC reference surface (no real data supplied)"
        title_r = "rBergomi calibrated through the surrogate"
        K = grid.log_moneyness
        kq, Tq = K.ravel(), np.repeat(grid.maturities, grid.shape[1])
        ivq = ref["iv"][i].ravel()
    fit = model.predict(theta[None])[0]
    KK = grid.log_moneyness
    TT = np.repeat(grid.maturities[:, None], grid.shape[1], axis=1)
    ax = _surface_axes(fig, 121, title_l)
    ax.scatter(kq, Tq, ivq, s=10, color=ORANGE, depthshade=False)
    ax2 = _surface_axes(fig, 122, title_r)
    ax2.plot_surface(KK, TT, fit, cmap=SEQ_CMAP, edgecolor=BLUE, linewidth=0.3, alpha=0.85)
    ax2.scatter(kq, Tq, ivq, s=8, color=ORANGE, depthshade=False, label="target")
    for a_ in (ax, ax2):
        a_.set_zlim(min(np.min(ivq), fit.min()) * 0.95, max(np.max(ivq), fit.max()) * 1.03)
    pt = ", ".join(f"{LABELS[n]}={v:.3g}" for n, v in zip(PARAM_NAMES, theta, strict=True))
    fig.suptitle(f"Implied-vol surface: target vs. model ({pt})", color=TEXT)
    save(fig, out / "iv_surface_3d.png")


def fig_smiles(cfg, out):
    h = _npz(cfg, "heston")
    if h is None:
        return
    from roughvol.heston import heston_iv_grid
    from roughvol.surrogate.train import load_surrogate

    grid = Grid.from_config(cfg["data"])
    ref = load_reference(cfg)
    model = load_surrogate(cfg, cfg["calibration"]["model"], "cpu")
    i = 0
    rb = model.predict(h["rb_theta"][i : i + 1])[0]
    hv = heston_iv_grid(
        h["heston_params"][i],
        grid.maturities,
        grid.log_moneyness,
        int(cfg["heston"]["cos"]["n_terms"]),
        float(cfg["heston"]["cos"]["L"]),
    )
    rows = [0, 2, 4, len(grid.maturities) - 1]
    fig, axes = plt.subplots(1, len(rows), figsize=(15, 3.6), constrained_layout=True)
    for ax, r in zip(axes, rows, strict=True):
        k = grid.log_moneyness[r]
        ax.errorbar(
            k,
            ref["iv"][i, r],
            yerr=2 * ref["iv_se"][i, r],
            fmt="o",
            color=TEXT_2,
            ms=4,
            label="MC reference (+-2 SE)",
        )
        ax.plot(k, rb[r], color=BLUE, label="rBergomi surrogate fit")
        ax.plot(k, hv[r], color=ORANGE, label="Heston fit")
        ax.set_title(f"T = {grid.maturities[r]:g}")
        ax.set_xlabel("log-moneyness k")
    axes[0].set_ylabel("implied vol")
    axes[0].legend(loc="upper right", fontsize=7.5)
    th = ref["params"][i]
    fig.suptitle(
        "Smiles of one synthetic reference surface (true "
        + ", ".join(f"{LABELS[n]}={v:.3g}" for n, v in zip(PARAM_NAMES, th, strict=True))
        + ")",
        color=TEXT,
    )
    save(fig, out / "smiles.png")


def fig_market_smiles(cfg, out):
    mk = _json(cfg, "market")
    qpath = resolve(cfg["results_dir"]) / "arrays" / "market_quotes.csv"
    if not mk or mk.get("status") != "ok" or not qpath.exists():
        return
    import pandas as pd

    q = pd.read_csv(qpath)
    Ts = sorted(q["T"].unique())
    n = min(len(Ts), 6)
    pick = [Ts[round(j)] for j in np.linspace(0, len(Ts) - 1, n)]
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), constrained_layout=True, squeeze=False)
    for ax, T in zip(axes[0], pick, strict=True):
        g = q[q["T"] == T].sort_values("k")
        ax.plot(g["k"], g["iv"], "o", color=TEXT_2, ms=4, label="market")
        ax.plot(g["k"], g["rbergomi_iv"], color=BLUE, label="rBergomi")
        ax.plot(g["k"], g["heston_iv"], color=ORANGE, label="Heston")
        ax.set_title(f"T = {T * 365:.0f} days")
        ax.set_xlabel("log-moneyness k")
    axes[0][0].set_ylabel("implied vol")
    axes[0][0].legend(fontsize=7.5)
    fig.suptitle(f"Market smiles {mk['report']['valuation_date']}", color=TEXT)
    save(fig, out / "market_smiles.png")


def make_all(cfg: dict) -> None:
    set_style()
    out = resolve(cfg["assets_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for f in (
        fig_error_heatmaps,
        fig_recovery,
        fig_atm_skew,
        fig_heston,
        fig_training,
        fig_surfaces,
        fig_smiles,
        fig_market_smiles,
    ):
        f(cfg, out)
        print(f"[figures] {f.__name__}")
