"""Regenerate every number in README.md from results/*.json.

README.md contains marker pairs  <!-- BEGIN:name --> ... <!-- END:name -->; the content
between them is replaced by the table/text produced by the function ``block_<name>``.
A missing result file renders as "pending" - numbers are never typed by hand.

Usage: python scripts/build_readme_tables.py [--results results] [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PENDING = "pending"
NAMES = {"mlp": "Grid MLP (3x30 ELU, paper)", "resmlp": "Residual MLP (3 blocks x 128)"}
PARAMS = ("H", "eta", "rho", "xi0")
PLABEL = {"H": "H", "eta": "η", "rho": "ρ", "xi0": "ξ₀"}


def _load(results: Path, name: str) -> dict | None:
    p = results / f"{name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def pct(x, d=2):
    return PENDING if x is None else f"{100 * x:.{d}f}%"


def num(x, d=3):
    return PENDING if x is None else f"{x:.{d}g}"


def secs(x):
    if x is None:
        return PENDING
    if x < 1e-3:
        return f"{x * 1e6:.0f} µs"
    if x < 1:
        return f"{x * 1e3:.1f} ms"
    return f"{x:.1f} s"


def block_data(r: Path) -> str:
    d = _load(r, "data_generation")
    if d is None:
        return f"Training-set generation: {PENDING}."
    h = d["hardware"]
    return (
        "| quantity | value |\n|---|---|\n"
        f"| parameter sets | {d['n_samples']:,} |\n"
        f"| grid | {d['grid_shape'][0]} maturities x {d['grid_shape'][1]} strikes |\n"
        f"| paths per set (incl. antithetic) | {d['n_paths']:,} |\n"
        f"| time steps per year | {d['steps_per_year']} |\n"
        f"| estimator | {d['estimator']} (conditional on W, antithetic) |\n"
        f"| mean relative IV standard error of a label | {pct(d['rel_iv_se_mean'], 3)} |\n"
        f"| failed IV inversions | {pct(d['nan_fraction'], 3)} |\n"
        f"| GPU pricing time | {d['ms_per_sample_median_chunk']:.1f} ms per surface (median "
        f"over chunks) → {d['hours_at_median_rate']:.2f} h for the set on {h.get('gpu', 'CPU')} |"
    )


def block_surrogate(r: Path) -> str:
    e = _load(r, "surrogate_eval")
    if e is None:
        return f"Surrogate evaluation: {PENDING}."
    nf = e["mc_noise_floor"]
    rows = [
        "| model | params | mean rel. err vs ref | p95 | p99 | max | error / MC noise floor "
        "| 1 surface (CPU) |",
        "|---|---|---|---|---|---|---|---|",
        f"| MC, {(_load(r, 'data_generation') or {}).get('n_paths', '?'):,} paths "
        "(noise floor) | – | "
        f"{pct(nf['vs_reference']['mean'], 3)} | {pct(nf['vs_reference']['p95'], 3)} | "
        f"{pct(nf['vs_reference']['p99'], 3)} | {pct(nf['vs_reference']['max'], 3)} | 1.00 | – |",
    ]
    for name, m in e["models"].items():
        v = m["vs_reference"]
        rows.append(
            f"| {NAMES.get(name, name)} | {m['n_parameters']:,} | {pct(v['mean'], 3)} | "
            f"{pct(v['p95'], 3)} | {pct(v['p99'], 3)} | {pct(v['max'], 3)} | "
            f"{m['ratio_to_noise_floor']:.2f} | {secs(m['inference_seconds_single_cpu'])} |"
        )
    rows.append("")
    rows.append(
        f"Errors are relative IV errors over all {e['n_reference']} x 88 cells of the "
        f"high-precision reference set ({e['reference_n_paths']:,} paths, own relative SE "
        f"{pct(nf['reference_rel_se']['mean'], 3)} on average)."
    )
    return "\n".join(rows)


def block_recovery(r: Path) -> str:
    c = _load(r, "calibration_recovery")
    if c is None:
        return f"Parameter recovery: {PENDING}."
    rows = [
        "| model / targets | "
        + " | ".join(f"MAE {PLABEL[p]}" for p in PARAMS)
        + " | median fit RMSE (vol pts) |",
        "|---" * 6 + "|",
    ]
    for name, m in c["models"].items():
        for key, lab in (
            ("lm_noisy", "MC test surfaces"),
            ("lm_reference_targets", "high-precision reference"),
            ("lm_surrogate_targets", "noise-free (own output)"),
        ):
            pe = m[key]["param_errors"]
            rows.append(
                f"| {NAMES.get(name, name)} — {lab} | "
                + " | ".join(num(pe[p]["mae"]) for p in PARAMS)
                + f" | {100 * m[key]['fit_rmse']['median']:.3f} |"
            )
    lb = c.get("lbfgs_subset")
    if lb:
        pe = lb["param_errors"]
        rows.append(
            f"| L-BFGS-B ({lb['model']}, {lb['n_surfaces']} surfaces) | "
            + " | ".join(num(pe[p]["mae"]) for p in PARAMS)
            + f" | {100 * lb['fit_rmse']['median']:.3f} |"
        )
    rows.append("")
    rows.append(f"LM, {c['n_surfaces']:,} held-out surfaces, multi-start, batched on GPU.")
    return "\n".join(rows)


def block_speed(r: Path) -> str:
    s = _load(r, "calibration_speed")
    if s is None:
        return f"Calibration speed: {PENDING}."
    mc = s["mc_calibration"]
    return (
        "| calibration | wall-clock per surface | notes |\n|---|---|---|\n"
        f"| LM through surrogate (CPU, 1 surface) | {secs(s['lm_single_cpu_seconds']['median'])} "
        f"| median of {s['lm_single_cpu_seconds']['n']}, {s['lm_single_cpu_seconds']['n_starts']} "
        "starts |\n"
        f"| L-BFGS-B through surrogate (CPU) | {secs(s['lbfgs_single_cpu_seconds']['median'])} "
        f"| median of {s['lbfgs_single_cpu_seconds']['n']} |\n"
        f"| Monte Carlo in the loop ({mc['device']}, {mc['n_paths']:,} paths) | "
        f"{secs(mc['median_seconds'])} | median of {len(mc['runs'])}, "
        f"{mc['median_nfev']:.0f} pricer calls |\n"
        f"| **speed-up (MC / surrogate LM)** | **{s['speedup_mc_over_lm']:,.0f}x** | |"
    )


def block_heston(r: Path) -> str:
    h = _load(r, "heston_baseline")
    if h is None:
        return f"Heston baseline: {PENDING}."
    m = h["rmse_by_maturity"]
    head = "| | " + " | ".join(f"T={t:g}" for t in m["maturities"]) + " | all |"
    sep = "|---" * (len(m["maturities"]) + 2) + "|"
    rb = (
        "| rBergomi (surrogate) | "
        + " | ".join(f"{100 * v:.2f}" for v in m["rbergomi"])
        + f" | {100 * h['rbergomi_surrogate_rmse']['mean']:.2f} |"
    )
    he = (
        "| Heston (COS) | "
        + " | ".join(f"{100 * v:.2f}" for v in m["heston"])
        + f" | {100 * h['heston_rmse']['mean']:.2f} |"
    )
    return (
        f"IV RMSE in vol points on {h['n_surfaces']} rBergomi reference surfaces:\n\n"
        + "\n".join([head, sep, rb, he])
    )


def block_skew(r: Path) -> str:
    s = _load(r, "atm_skew")
    if s is None:
        return f"ATM skew: {PENDING}."
    rows = ["| H | fitted slope (T ≤ 0.1) | theory H − 1/2 |", "|---|---|---|"]
    for key, c in s["curves"].items():
        f = c["fit_short"]
        rows.append(
            f"| {key} | {f['slope']:+.3f} ± {f['slope_se']:.3f} | {c['theory_slope']:+.3f} |"
        )
    hs = s["heston"]["fit_short"]
    rows.append(
        f"| Heston fit (to H={s['params']['H']:g}) | {hs['slope']:+.3f} ± "
        f"{hs['slope_se']:.3f} | 0 (flat) |"
    )
    return "\n".join(rows)


def block_market(r: Path) -> str:
    m = _load(r, "market")
    if m is None:
        return f"Real-data calibration: {PENDING}."
    if m["status"] != "ok":
        return (
            "Real-data calibration: **not run** — no option-chain file was supplied "
            "(see *Real data* below). No market data is invented."
        )
    rep = m["report"]
    return (
        f"{rep['valuation_date']}, {rep['quotes_final']} quotes on {rep['expiries_final']} "
        f"expiries ({rep['format']}).\n\n| model | IV RMSE (vol pts) | calibration time |\n"
        f"|---|---|---|\n| rBergomi via surrogate | {100 * m['rbergomi']['rmse']:.2f} | "
        f"{secs(m['rbergomi']['seconds'])} |\n| Heston (COS) | {100 * m['heston']['rmse']:.2f} | "
        f"{secs(m['heston']['seconds'])} |\n\nCalibrated rBergomi: "
        + ", ".join(f"{PLABEL[p]} = {m['rbergomi']['params'][p]:.3f}" for p in PARAMS)
    )


def block_ablations(r: Path) -> str:
    a = _load(r, "ablations")
    e = _load(r, "surrogate_eval")
    rows = ["| ablation | result |", "|---|---|"]
    if a is None:
        return f"Ablations: {PENDING}."
    d = a["discretisation"]
    for n, v in d["vs_finest"].items():
        rows.append(
            f"| time steps: {n}/yr vs {d['finest_steps_per_year']}/yr | mean abs IV diff "
            f"{100 * v['mean_abs_iv_diff_vs_finest']:.3f} vol pts (T ≤ 0.1: "
            f"{100 * v['mean_abs_diff_short_end_T<=0.1']:.3f}; rough corner H=0.05: "
            f"{100 * v['rough_corner_mean_abs_diff']:.3f}); MC SE of the difference "
            f"{100 * v['mean_combined_se']:.3f} |"
        )
    s = a["estimator_se_ratio_plain_over_mixing"]
    rows.append(
        f"| estimator: plain / mixing price SE | median {s['median']:.2f} (range "
        f"{s['min']:.2f}–{s['max']:.2f}) ≈ {s['equivalent_path_multiplier_median']:.1f}x "
        "fewer paths for equal error |"
    )
    if e:
        m = e["models"]
        if "mlp" in m and "resmlp" in m:
            rows.append(
                f"| network: paper MLP vs residual MLP | mean rel. error "
                f"{pct(m['mlp']['vs_reference']['mean'], 3)} vs "
                f"{pct(m['resmlp']['vs_reference']['mean'], 3)} |"
            )
    c = _load(r, "calibration_recovery")
    if c:
        for name, m in c["models"].items():
            a1 = m["lm_noisy"]["param_errors"]["H"]["mae"]
            a0 = m["lm_surrogate_targets"]["param_errors"]["H"]["mae"]
            rows.append(
                f"| targets: noise-free vs MC-noisy ({NAMES.get(name, name)}) | "
                f"MAE(H) {a0:.4f} vs {a1:.4f} |"
            )
    return "\n".join(rows)


def render(text: str, results: Path) -> str:
    def sub(mt: re.Match) -> str:
        name = mt.group(1)
        fn = globals().get(f"block_{name}")
        body = fn(results) if fn else f"(unknown block {name})"
        return f"<!-- BEGIN:{name} -->\n{body}\n<!-- END:{name} -->"

    return re.sub(r"<!-- BEGIN:(\w+) -->.*?<!-- END:\1 -->", sub, text, flags=re.S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--check", action="store_true", help="fail if README is out of date")
    a = ap.parse_args()
    readme = Path(a.readme)
    old = readme.read_text(encoding="utf-8")
    new = render(old, Path(a.results))
    if a.check:
        sys.exit(0 if new == old else 1)
    readme.write_text(new, encoding="utf-8")
    print(f"updated {readme}")


if __name__ == "__main__":
    main()
