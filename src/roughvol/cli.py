"""Command-line entry point: ``python -m roughvol {data,train,eval,figures,all} [--quick]``."""

from __future__ import annotations

import argparse
import time

from roughvol.config import load_config
from roughvol.utils import set_seed


def cmd_data(cfg: dict, args: argparse.Namespace) -> None:
    from roughvol.data import generate_dataset, generate_reference

    res = generate_dataset(cfg, device=args.device, max_chunks=args.max_chunks)
    if res.get("complete"):
        generate_reference(cfg, device=args.device)


def cmd_train(cfg: dict, args: argparse.Namespace) -> None:
    from roughvol.surrogate.train import train_all

    train_all(cfg)


def cmd_eval(cfg: dict, args: argparse.Namespace) -> None:
    from roughvol.evaluate import run_all

    run_all(cfg, market_csv=args.market)


def cmd_figures(cfg: dict, args: argparse.Namespace) -> None:
    from roughvol.figures import make_all

    make_all(cfg)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="roughvol", description=__doc__)
    p.add_argument("command", choices=["data", "train", "eval", "figures", "all"])
    p.add_argument("--quick", action="store_true", help="tiny CPU run (CI)")
    p.add_argument("--device", default="auto")
    p.add_argument("--market", default=None, help="path to an option-chain CSV")
    p.add_argument("--max-chunks", type=int, default=None, help="stop data generation early")
    args = p.parse_args(argv)
    cfg = load_config(quick=args.quick)
    if args.quick:
        args.device = "cpu"
    set_seed(int(cfg["data"]["seed"]))
    steps = ["data", "train", "eval", "figures"] if args.command == "all" else [args.command]
    t0 = time.perf_counter()
    for step in steps:
        print(f"=== {step} {'(quick)' if args.quick else ''}")
        globals()[f"cmd_{step}"](cfg, args)
    print(f"done in {time.perf_counter() - t0:.1f}s")
