"""Evaluation: every number reported in the README is produced here and written to
results/*.json (arrays for figures go to results/arrays/*.npz)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from roughvol.calibration import lbfgs_calibrate, lm_calibrate, mc_calibrate
from roughvol.config import resolve
from roughvol.data import load_dataset, load_reference, price_params
from roughvol.grid import PARAM_NAMES, Grid, param_bounds
from roughvol.heston import calibrate_heston, heston_iv_grid, heston_iv_quotes
from roughvol.rbergomi import RoughBergomiMC
from roughvol.skew import atm_skew, fit_power_law, heston_short_skew, theoretical_skew
from roughvol.surrogate.train import load_surrogate
from roughvol.utils import get_device, hardware_info, save_json, set_seed


def _arrays_path(cfg: dict, name: str) -> Path:
    p = resolve(cfg["results_dir"]) / "arrays" / f"{name}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / np.abs(b)


def _stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def _lm_kwargs(cfg: dict) -> dict:
    c = cfg["calibration"]["lm"]
    return {
        "n_starts": c["n_starts"],
        "max_iter": c["max_iter"],
        "lambda0": c["lambda0"],
        "lambda_up": c["lambda_up"],
        "lambda_down": c["lambda_down"],
        "tol": c["tol"],
        "lambda_max": c["lambda_max"],
    }


def _mc(cfg: dict, device, maturities=None, steps=None) -> RoughBergomiMC:
    d = cfg["data"]
    grid = Grid.from_config(d)
    return RoughBergomiMC(
        grid.maturities if maturities is None else maturities,
        d["mc"]["steps_per_year"] if steps is None else steps,
        device,
        getattr(torch, d["mc"]["dtype"]),
    )


# --------------------------------------------------------------------------- step 2
def reference_lowres(cfg: dict, device) -> np.ndarray:
    """Reference parameters re-priced at *training* path count with independent seeds:
    the empirical Monte Carlo noise floor of a single training label."""
    path = _arrays_path(cfg, "reference_lowres")
    if path.exists():
        with np.load(path) as z:
            return z["iv"]
    d = cfg["data"]
    ref = load_reference(cfg)
    grid = Grid.from_config(d)
    res = price_params(
        _mc(cfg, device),
        grid,
        ref["params"].astype(float),
        20_000_000 + np.arange(len(ref["params"])),
        int(d["mc"]["n_paths"]),
        d["mc"]["estimator"],
        int(d["mc"]["path_batch"]),
    )
    np.savez(path, iv=res["iv"], iv_se=res["iv_se"])
    return res["iv"]


def eval_surrogates(cfg: dict) -> dict:
    device = get_device(cfg["surrogate"]["device"])
    ds, ref = load_dataset(cfg), load_reference(cfg)
    test = ds["split"] == 2
    iv_lo = reference_lowres(cfg, device)
    ref_iv = ref["iv"].astype(float)
    noise_rel = _rel(iv_lo, ref_iv)  # |MC_40k - MC_ref| / MC_ref
    ref_self = ref["iv_se"] / ref["iv"]  # reference's own relative SE
    label_se = ds["iv_se"][test] / ds["iv"][test]

    out: dict = {
        "n_test": int(test.sum()),
        "n_reference": len(ref_iv),
        "reference_n_paths": int(ref["n_paths"]),
        "mc_noise_floor": {
            "vs_reference": _stats(noise_rel),
            "label_rel_se": _stats(label_se),
            "reference_rel_se": _stats(ref_self),
        },
        "models": {},
    }
    arrays = {
        "noise_floor_grid": noise_rel.mean(0),
        "label_se_grid": np.nanmean(label_se, 0),
        "maturities": ds["maturities"],
        "z": ds["z"],
    }
    for name in cfg["surrogate"]["models"]:
        model = load_surrogate(cfg, name, device)
        pred_test = model.predict(ds["params"][test])
        pred_ref = model.predict(ref["params"])
        e_test = _rel(pred_test, ds["iv"][test])
        e_ref = _rel(pred_ref, ref_iv)
        # Inference timing: one surface on CPU (what a calibrator sees) and a big batch.
        cpu = load_surrogate(cfg, name, "cpu")
        x1 = torch.as_tensor(ref["params"][:1])
        with torch.no_grad():
            for _ in range(20):
                cpu(x1)
            t0 = time.perf_counter()
            for _ in range(500):
                cpu(x1)
            t_single = (time.perf_counter() - t0) / 500
            xb = torch.as_tensor(ds["params"][test], device=device)
            model(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_batch = (time.perf_counter() - t0) / len(xb)
        out["models"][name] = {
            "n_parameters": model.n_parameters(),
            "vs_mc_labels_test": _stats(e_test),
            "vs_reference": _stats(e_ref),
            "ratio_to_noise_floor": float(np.mean(e_ref) / np.mean(noise_rel)),
            "inference_seconds_single_cpu": t_single,
            "inference_seconds_per_surface_batched": t_batch,
            "inference_device_batched": str(device),
        }
        arrays[f"{name}_err_ref_grid"] = e_ref.mean(0)
        arrays[f"{name}_err_ref_p95_grid"] = np.quantile(e_ref, 0.95, axis=0)
        arrays[f"{name}_err_test_grid"] = np.nanmean(e_test, 0)
        print(
            f"[eval] {name}: mean rel err vs reference {e_ref.mean():.4%} "
            f"(MC noise floor {noise_rel.mean():.4%})"
        )
    np.savez(_arrays_path(cfg, "surrogate_errors"), **arrays)
    save_json(out, Path(cfg["results_dir"]) / "surrogate_eval.json")
    return out


# --------------------------------------------------------------------------- step 3
def _param_err(theta: np.ndarray, true: np.ndarray) -> dict:
    d = {}
    for i, n in enumerate(PARAM_NAMES):
        e = theta[:, i] - true[:, i]
        d[n] = {
            "mae": float(np.mean(np.abs(e))),
            "median_ae": float(np.median(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e * e))),
            "bias": float(np.mean(e)),
            "median_rel": float(np.median(np.abs(e) / np.abs(true[:, i]))),
        }
    return d


def eval_recovery(cfg: dict) -> dict:
    ccfg = cfg["calibration"]
    device = get_device(cfg["surrogate"]["device"])
    ds, ref = load_dataset(cfg), load_reference(cfg)
    idx = np.flatnonzero(ds["split"] == 2)[: int(ccfg["recovery"]["n_surfaces"])]
    true = ds["params"][idx].astype(float)
    target = ds["iv"][idx].astype(float)
    out: dict = {"n_surfaces": len(idx), "target": "held-out MC test surfaces", "models": {}}
    arrays: dict = {"true": true}
    for name in cfg["surrogate"]["models"]:
        model = load_surrogate(cfg, name, device)
        t0 = time.perf_counter()
        res = lm_calibrate(model, target, seed=int(ccfg["seed"]), **_lm_kwargs(cfg))
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # Same surrogate, noise-free targets: isolates identifiability from MC/NN error.
        clean = lm_calibrate(model, model.predict(true), seed=int(ccfg["seed"]), **_lm_kwargs(cfg))
        refres = lm_calibrate(
            model, ref["iv"].astype(float), seed=int(ccfg["seed"]), **_lm_kwargs(cfg)
        )
        out["models"][name] = {
            "lm_noisy": {
                "param_errors": _param_err(res["theta"], true),
                "fit_rmse": _stats(res["rmse"]),
                "median_iterations": float(np.median(res["n_iter"])),
                "batched_seconds_total": dt,
                "batched_seconds_per_surface": dt / len(idx),
            },
            "lm_surrogate_targets": {
                "param_errors": _param_err(clean["theta"], true),
                "fit_rmse": _stats(clean["rmse"]),
            },
            "lm_reference_targets": {
                "param_errors": _param_err(refres["theta"], ref["params"].astype(float)),
                "fit_rmse": _stats(refres["rmse"]),
                "n_surfaces": len(ref["iv"]),
            },
        }
        arrays[f"{name}_lm"] = res["theta"]
        arrays[f"{name}_lm_rmse"] = res["rmse"]
        arrays[f"{name}_clean"] = clean["theta"]
    # L-BFGS-B on a subset (sequential, one surface at a time).
    model = load_surrogate(cfg, ccfg["model"], device)
    n_lb = min(int(ccfg["recovery"]["n_lbfgs"]), len(idx))
    lb = [
        lbfgs_calibrate(
            model,
            target[i],
            n_starts=int(ccfg["lbfgs"]["n_starts"]),
            max_iter=int(ccfg["lbfgs"]["max_iter"]),
            seed=int(ccfg["seed"]),
        )
        for i in range(n_lb)
    ]
    th = np.array([r["theta"] for r in lb])
    out["lbfgs_subset"] = {
        "model": ccfg["model"],
        "n_surfaces": n_lb,
        "param_errors": _param_err(th, true[:n_lb]),
        "fit_rmse": _stats(np.array([r["rmse"] for r in lb])),
        "lm_fit_rmse_same_subset": _stats(arrays[f"{ccfg['model']}_lm_rmse"][:n_lb]),
    }
    arrays["lbfgs"] = th
    np.savez(_arrays_path(cfg, "recovery"), **arrays)
    save_json(out, Path(cfg["results_dir"]) / "calibration_recovery.json")
    m = out["models"][ccfg["model"]]["lm_noisy"]["param_errors"]
    print("[eval] recovery MAE: " + ", ".join(f"{n}={m[n]['mae']:.4f}" for n in PARAM_NAMES))
    return out


def eval_speed(cfg: dict) -> dict:
    """Wall-clock per calibration: surrogate (LM, L-BFGS) vs MC-in-the-loop."""
    ccfg = cfg["calibration"]
    device = get_device(cfg["surrogate"]["device"])
    ref = load_reference(cfg)
    lo, hi = param_bounds(cfg["data"])
    grid = Grid.from_config(cfg["data"])
    cpu_model = load_surrogate(cfg, ccfg["model"], "cpu")
    n_single = min(int(ccfg["speed"]["n_single"]), len(ref["iv"]))
    single = {**_lm_kwargs(cfg), "jacobian": ccfg["lm"]["jacobian_single"]}
    threads = torch.get_num_threads()
    torch.set_num_threads(int(ccfg["lm"]["cpu_threads"]))
    t_lm, t_lb = [], []
    for i in range(n_single):
        tgt = ref["iv"][i : i + 1].astype(float)
        t0 = time.perf_counter()
        lm_calibrate(cpu_model, tgt, seed=i, **single)
        t_lm.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        lbfgs_calibrate(
            cpu_model,
            tgt[0],
            n_starts=int(ccfg["lbfgs"]["n_starts"]),
            max_iter=int(ccfg["lbfgs"]["max_iter"]),
            seed=i,
        )
        t_lb.append(time.perf_counter() - t0)
    torch.set_num_threads(threads)

    mcc = ccfg["mc_calibration"]
    mc = _mc(cfg, device)
    n_mc = int(mcc["n_surfaces"])
    mc_runs = []
    for i in range(n_mc):
        tgt = ref["iv"][i].astype(float)
        r = mc_calibrate(
            mc,
            grid.log_moneyness,
            tgt,
            lo,
            hi,
            int(mcc["n_paths"]),
            0.5 * (lo + hi),
            int(mcc["max_nfev"]),
            seed=1234 + i,
            path_batch=int(cfg["data"]["mc"]["path_batch"]),
        )
        s = lm_calibrate(cpu_model, tgt[None], seed=i, **single)
        true = ref["params"][i].astype(float)
        mc_runs.append(
            {
                "seconds": r["seconds"],
                "nfev": r["nfev"],
                "fit_rmse": r["rmse"],
                "abs_param_error": np.abs(r["theta"] - true),
                "surrogate_abs_param_error": np.abs(s["theta"][0] - true),
                "surrogate_fit_rmse": float(s["rmse"][0]),
            }
        )
        print(f"[eval] MC calibration {i + 1}/{n_mc}: {r['seconds']:.1f}s, {r['nfev']} evals")
    t_mc = np.array([r["seconds"] for r in mc_runs])
    with torch.no_grad():
        t0 = time.perf_counter()
        mc.price(
            *ref["params"][0].astype(float),
            grid.log_moneyness,
            int(mcc["n_paths"]),
            seed=0,
            path_batch=int(cfg["data"]["mc"]["path_batch"]),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_price = time.perf_counter() - t0
    rec = resolve(cfg["results_dir"]) / "calibration_recovery.json"
    batched = None
    if rec.exists():
        from roughvol.utils import load_json

        batched = load_json(rec)["models"][ccfg["model"]]["lm_noisy"]["batched_seconds_per_surface"]
    out = {
        "surrogate_model": ccfg["model"],
        "lm_single_cpu_seconds": {
            "median": float(np.median(t_lm)),
            "p90": float(np.quantile(t_lm, 0.9)),
            "n": n_single,
            "n_starts": int(ccfg["lm"]["n_starts"]),
            "jacobian": single["jacobian"],
            "cpu_threads": int(ccfg["lm"]["cpu_threads"]),
        },
        "lbfgs_single_cpu_seconds": {
            "median": float(np.median(t_lb)),
            "p90": float(np.quantile(t_lb, 0.9)),
            "n": n_single,
        },
        "mc_calibration": {
            "device": str(device),
            "n_paths": int(mcc["n_paths"]),
            "median_seconds": float(np.median(t_mc)),
            "median_nfev": float(np.median([r["nfev"] for r in mc_runs])),
            "runs": mc_runs,
        },
        "mc_single_surface_price_seconds": t_price,
        "speedup_mc_over_lm": float(np.median(t_mc) / np.median(t_lm)),
        "batched_gpu_seconds_per_surface": batched,
        "speedup_mc_over_batched": float(np.median(t_mc) / batched) if batched else None,
        "hardware": hardware_info(),
    }
    save_json(out, Path(cfg["results_dir"]) / "calibration_speed.json")
    print(f"[eval] speedup MC / surrogate-LM: {out['speedup_mc_over_lm']:.0f}x")
    return out


# --------------------------------------------------------------------------- Heston & skew
def eval_heston(cfg: dict) -> dict:
    hcfg = cfg["heston"]
    device = get_device(cfg["surrogate"]["device"])
    ref = load_reference(cfg)
    grid = Grid.from_config(cfg["data"])
    n = min(int(hcfg["calibration"]["n_surfaces"]), len(ref["iv"]))
    model = load_surrogate(cfg, cfg["calibration"]["model"], device)
    TT = np.repeat(grid.maturities[:, None], grid.shape[1], axis=1)
    k = grid.log_moneyness
    targets = ref["iv"][:n].astype(float)
    rb = lm_calibrate(model, targets, seed=0, **_lm_kwargs(cfg))
    rb_fit = model.predict(rb["theta"])
    h_err, h_params, h_rmse = [], [], []
    for i in range(n):
        r = calibrate_heston(
            TT,
            k,
            targets[i],
            hcfg["bounds"]["lower"],
            hcfg["bounds"]["upper"],
            int(hcfg["calibration"]["n_starts"]),
            int(hcfg["calibration"]["max_nfev"]),
            int(hcfg["cos"]["n_terms"]),
            float(hcfg["cos"]["L"]),
            seed=i,
        )
        fit = heston_iv_quotes(
            r["params"], TT.ravel(), k.ravel(), int(hcfg["cos"]["n_terms"]), float(hcfg["cos"]["L"])
        ).reshape(grid.shape)
        h_err.append(np.abs(fit - targets[i]))
        h_params.append(r["params"])
        h_rmse.append(r["rmse"])
    h_err = np.array(h_err)
    rb_err = np.abs(rb_fit - targets)
    out = {
        "n_surfaces": n,
        "targets": "high-precision MC rBergomi reference surfaces",
        "rbergomi_surrogate_rmse": _stats(rb["rmse"]),
        "heston_rmse": _stats(np.array(h_rmse)),
        "rmse_by_maturity": {
            "maturities": grid.maturities,
            "rbergomi": np.sqrt((rb_err**2).mean(axis=(0, 2))),
            "heston": np.sqrt((h_err**2).mean(axis=(0, 2))),
        },
        "heston_params_median": dict(
            zip(("v0", "kappa", "theta", "sigma", "rho"), np.median(h_params, 0), strict=True)
        ),
    }
    np.savez(
        _arrays_path(cfg, "heston"),
        heston_abs_err=h_err.mean(0),
        rb_abs_err=rb_err.mean(0),
        heston_params=np.array(h_params),
        rb_theta=rb["theta"],
        targets=targets,
    )
    save_json(out, Path(cfg["results_dir"]) / "heston_baseline.json")
    print(
        f"[eval] Heston RMSE {np.mean(h_rmse):.4f} vs rBergomi-surrogate {np.mean(rb['rmse']):.4f}"
    )
    return out


def eval_skew(cfg: dict) -> dict:
    """ATM-skew term structure from dense-maturity MC runs vs the T^(H-1/2) power law,
    plus the skew of a Heston model fitted to the same surface."""
    s = cfg["heston"]["skew"]
    hcfg = cfg["heston"]
    device = get_device(cfg["surrogate"]["device"])
    grid = Grid.from_config(cfg["data"])
    mats = np.array(s["maturities"], float)
    dz = float(s["dz"])
    zz = np.array([-dz, 0.0, dz])
    kk = zz[None, :] * np.sqrt(mats)[:, None]
    base = dict(s["params"])
    H_values = s.get("H_values", [base["H"]])
    mc = _mc(cfg, device, mats, int(s["steps_per_year"]))
    curves = {}
    for i, H in enumerate(H_values):
        r = mc.price(
            H,
            base["eta"],
            base["rho"],
            base["xi0"],
            kk,
            int(s["n_paths"]),
            seed=555 + i,
            path_batch=int(s.get("path_batch", s["n_paths"])),
        )
        psi = atm_skew(r.iv, zz, mats)
        psi_se = np.hypot(r.iv_se[:, 0], r.iv_se[:, 2]) / (2 * dz * np.sqrt(mats))
        short = fit_power_law(mats, psi, t_max=0.1)
        full = fit_power_law(mats, psi)
        curves[f"{H:g}"] = {
            "H": H,
            "psi": psi,
            "psi_se": psi_se,
            "theory": theoretical_skew(mats, H, base["eta"], base["rho"]),
            "fit_short": short,
            "fit_all": full,
            "theory_slope": H - 0.5,
        }
        print(
            f"[eval] skew H={H:g}: slope {short['slope']:.3f} +- {short['slope_se']:.3f} "
            f"(theory {H - 0.5:.3f})"
        )
    # Heston fitted to the base-H surface on the standard grid.
    mcg = _mc(cfg, device)
    r = mcg.price(
        base["H"],
        base["eta"],
        base["rho"],
        base["xi0"],
        grid.log_moneyness,
        int(s["n_paths"]),
        seed=999,
        path_batch=int(s.get("path_batch", s["n_paths"])),
    )
    TT = np.repeat(grid.maturities[:, None], grid.shape[1], axis=1)
    hfit = calibrate_heston(
        TT,
        grid.log_moneyness,
        r.iv,
        hcfg["bounds"]["lower"],
        hcfg["bounds"]["upper"],
        int(hcfg["calibration"]["n_starts"]),
        int(hcfg["calibration"]["max_nfev"]),
        int(hcfg["cos"]["n_terms"]),
        float(hcfg["cos"]["L"]),
    )
    h_iv = heston_iv_grid(hfit["params"], mats, kk, 2048, float(hcfg["cos"]["L"]))
    h_psi = atm_skew(h_iv, zz, mats)
    v0, _, _, sig, rho = hfit["params"]
    # Surrogate skew on the grid maturities at the base parameters.
    model = load_surrogate(cfg, cfg["calibration"]["model"], "cpu")
    sur = model.predict(np.array([[base[n] for n in PARAM_NAMES]]))[0]
    out = {
        "params": base,
        "maturities": mats,
        "dz": dz,
        "n_paths": int(s["n_paths"]),
        "steps_per_year": int(s["steps_per_year"]),
        "curves": curves,
        "heston": {
            "params": hfit["params"],
            "fit_rmse": hfit["rmse"],
            "psi": h_psi,
            "short_limit": heston_short_skew(v0, sig, rho),
            "fit_short": fit_power_law(mats, h_psi, t_max=0.1),
        },
        "surrogate": {
            "maturities": grid.maturities,
            "psi": atm_skew(sur, grid.z, grid.maturities),
            "mc_grid_psi": atm_skew(r.iv, grid.z, grid.maturities),
        },
    }
    save_json(out, Path(cfg["results_dir"]) / "atm_skew.json")
    return out


def eval_market(cfg: dict, csv_path: str | None) -> dict:
    from roughvol.market import MarketDataError, QuoteInterpolator, build_surface, read_chain

    mcfg = cfg["market"]
    path = csv_path or mcfg.get("csv_path")
    out_path = Path(cfg["results_dir"]) / "market.json"
    if not path or not resolve(path).exists():
        msg = (
            "no option-chain CSV supplied (pass --market path/to/chain.csv or set "
            "market.csv_path); real-data calibration SKIPPED. No market data is invented."
        )
        print(f"[eval] {msg}")
        save_json({"status": "skipped", "reason": msg}, out_path)
        return {"status": "skipped"}
    try:
        chain = read_chain(
            resolve(path), mcfg["format"], mcfg.get("underlying"), mcfg.get("valuation_date")
        )
        grid = Grid.from_config(cfg["data"])
        surf = build_surface(chain, mcfg, float(grid.maturities[0]), float(grid.maturities[-1]))
    except MarketDataError as e:
        print(f"[eval] market data rejected: {e}")
        save_json({"status": "error", "reason": str(e)}, out_path)
        return {"status": "error"}
    q = surf.quotes
    if len(q) < 10:
        save_json({"status": "insufficient", "report": surf.report}, out_path)
        return {"status": "insufficient"}
    device = get_device(cfg["surrogate"]["device"])
    model = load_surrogate(cfg, cfg["calibration"]["model"], device)
    interp = QuoteInterpolator(grid.maturities, grid.z, q["T"].to_numpy(), q["z"].to_numpy())
    interp.to(device)
    target = q["iv"].to_numpy()[None]
    t0 = time.perf_counter()
    res = lm_calibrate(
        model,
        target,
        seed=0,
        predict_fn=lambda th: interp(model(th)),
        **{**_lm_kwargs(cfg), "n_starts": 16},
    )
    t_rb = time.perf_counter() - t0
    theta = res["theta"][0]
    with torch.no_grad():
        rb_fit = (
            interp(model(torch.as_tensor(theta[None], dtype=torch.float32, device=device)))
            .cpu()
            .numpy()[0]
        )
    hcfg = cfg["heston"]
    t0 = time.perf_counter()
    h = calibrate_heston(
        q["T"].to_numpy(),
        q["k"].to_numpy(),
        q["iv"].to_numpy(),
        hcfg["bounds"]["lower"],
        hcfg["bounds"]["upper"],
        5,
        int(hcfg["calibration"]["max_nfev"]),
        int(hcfg["cos"]["n_terms"]),
        float(hcfg["cos"]["L"]),
    )
    t_h = time.perf_counter() - t0
    h_fit = heston_iv_quotes(h["params"], q["T"].to_numpy(), q["k"].to_numpy())
    q = q.assign(rbergomi_iv=rb_fit, heston_iv=h_fit)
    q.to_csv(_arrays_path(cfg, "market_quotes").with_suffix(".csv"), index=False)
    per_exp = (
        q.groupby("T")
        .apply(
            lambda g: {
                "T": float(g["T"].iloc[0]),
                "n": len(g),
                "rb_rmse": float(np.sqrt(np.mean((g["rbergomi_iv"] - g["iv"]) ** 2))),
                "heston_rmse": float(np.sqrt(np.mean((g["heston_iv"] - g["iv"]) ** 2))),
            },
            include_groups=False,
        )
        .tolist()
    )
    mats_q = np.array(sorted(q["T"].unique()))
    out = {
        "status": "ok",
        "file": str(path),
        "report": surf.report,
        "forwards": surf.forwards.assign(expiry=surf.forwards["expiry"].astype(str)).to_dict(
            "records"
        ),
        "rbergomi": {
            "params": dict(zip(PARAM_NAMES, theta, strict=True)),
            "rmse": float(res["rmse"][0]),
            "seconds": t_rb,
        },
        "heston": {
            "params": dict(zip(("v0", "kappa", "theta", "sigma", "rho"), h["params"], strict=True)),
            "rmse": h["rmse"],
            "seconds": t_h,
        },
        "per_expiry": per_exp,
        "maturities": mats_q,
    }
    save_json(out, out_path)
    print(
        f"[eval] market: {len(q)} quotes, rBergomi RMSE {out['rbergomi']['rmse']:.4f}, "
        f"Heston RMSE {h['rmse']:.4f}"
    )
    return out


def eval_ablations(cfg: dict) -> dict:
    """(a) time-discretisation bias vs MC noise; (b) plain vs mixing estimator variance."""
    a = cfg["ablations"]
    device = get_device(cfg["surrogate"]["device"])
    grid = Grid.from_config(cfg["data"])
    lo, hi = param_bounds(cfg["data"])
    rng = np.random.default_rng(int(a["seed"]))
    params = lo + (hi - lo) * rng.random((int(a["n_param_sets"]), 4))
    params[0] = [0.05, 2.5, -0.9, 0.04]  # rough corner: where discretisation bites hardest
    k = grid.log_moneyness
    d = a["discretisation"]
    steps = [int(s) for s in d["steps_per_year"]]
    ivs, ses = {}, {}
    for n in steps:
        mc = _mc(cfg, device, steps=n)
        out = [
            mc.price(*p, k, int(d["n_paths"]), seed=77 + i, path_batch=int(d["path_batch"]))
            for i, p in enumerate(params)
        ]
        ivs[n] = np.array([r.iv for r in out])
        ses[n] = np.array([r.iv_se for r in out])
    finest = steps[-1]
    disc = {}
    for n in steps[:-1]:
        diff = np.abs(ivs[n] - ivs[finest])
        disc[str(n)] = {
            "mean_abs_iv_diff_vs_finest": float(diff.mean()),
            "max_abs_iv_diff_vs_finest": float(diff.max()),
            "mean_abs_diff_short_end_T<=0.1": float(diff[:, grid.maturities <= 0.1].mean()),
            "mean_combined_se": float(np.hypot(ses[n], ses[finest]).mean()),
            "rough_corner_mean_abs_diff": float(diff[0].mean()),
        }
    e = a["estimator"]
    mc = _mc(cfg, device)
    ratios = []
    for i, p in enumerate(params):
        pl = mc.price(*p, k, int(e["n_paths"]), seed=5 + i, estimator="plain")
        mx = mc.price(*p, k, int(e["n_paths"]), seed=5 + i, estimator="mixing")
        ratios.append(pl.price_se / mx.price_se)
    ratios = np.array(ratios)
    out = {
        "params": params,
        "discretisation": {
            "finest_steps_per_year": finest,
            "n_paths": int(d["n_paths"]),
            "vs_finest": disc,
        },
        "estimator_se_ratio_plain_over_mixing": {
            "median": float(np.median(ratios)),
            "min": float(ratios.min()),
            "max": float(ratios.max()),
            "per_param_set_median": np.median(ratios, axis=(1, 2)),
            "equivalent_path_multiplier_median": float(np.median(ratios**2)),
        },
    }
    save_json(out, Path(cfg["results_dir"]) / "ablations.json")
    print(f"[eval] ablations: plain/mixing SE ratio median {np.median(ratios):.2f}")
    return out


def run_all(cfg: dict, market_csv: str | None = None) -> None:
    set_seed(int(cfg["calibration"]["seed"]))
    eval_surrogates(cfg)
    eval_recovery(cfg)
    eval_speed(cfg)
    eval_heston(cfg)
    eval_skew(cfg)
    eval_ablations(cfg)
    eval_market(cfg, market_csv)
