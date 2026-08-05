# ruff: noqa: N802, N803, N806, E501

"""The ``make-demo-data`` subcommand: SCAR demo CSV generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..preprocessing.pu_labeling import make_scar_dataset

__all__ = ["build_demo_parser", "run_demo"]


def build_demo_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``make-demo-data`` subcommand to *sub* (side-effect only)."""
    parser = sub.add_parser("make-demo-data", help="generate SCAR demo CSVs (X, y_pu, y_true)")
    parser.add_argument("--out-dir", type=str, required=True, help="output directory")
    parser.add_argument("--n", type=int, default=200, help="samples per class (total = 2n)")
    parser.add_argument("--c", type=float, default=0.5, help="SCAR labeling propensity (0, 1]")
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--separation", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=run_demo)


def run_demo(args: argparse.Namespace) -> None:
    """Generate demo CSVs into ``--out-dir``."""
    X, y_pu, y_true = make_scar_dataset(
        n=args.n,
        c=args.c,
        n_features=args.n_features,
        separation=args.separation,
        random_state=args.seed,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_csv(out / "X.csv", index=False)
    pd.DataFrame({"label": y_pu}).to_csv(out / "y_pu.csv", index=False)
    pd.DataFrame({"label": y_true}).to_csv(out / "y_true.csv", index=False)
    print(f"Wrote X.csv, y_pu.csv, y_true.csv to {out} ({len(y_pu)} samples)")
