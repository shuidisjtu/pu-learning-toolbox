# ruff: noqa: N802, N803, N806, E501

"""The ``make-demo-data`` subcommand: SCAR demo CSV generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..core.random import check_random_state
from ..preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset

__all__ = ["build_demo_parser", "run_demo"]


def build_demo_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``make-demo-data`` subcommand to *sub* (side-effect only)."""
    parser = sub.add_parser("make-demo-data", help="generate SCAR demo CSVs (X, y_pu, y_true)")
    parser.add_argument("--out-dir", type=str, required=True, help="output directory")
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="samples per class (total = 2n); must be >= 3 so enough positives get labeled",
    )
    parser.add_argument("--c", type=float, default=0.5, help="SCAR labeling propensity (0, 1]")
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument(
        "--separation",
        type=float,
        default=1.0,
        help="distance between the two class centres; 1.0 keeps the demo in "
        "the overlap regime where prior estimation stays reasonably unbiased "
        "(pen_l1 ~0.44 at sep=1.0 vs ~0.33 at sep=4.0 on n=200 data)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=run_demo)


def run_demo(args: argparse.Namespace) -> None:
    """Generate demo CSVs into ``--out-dir``."""
    rng = check_random_state(args.seed)
    result = simulate_pu_dataset(
        GaussianSourceConfig(
            2 * args.n, args.n_features, 0.5, args.separation, rng, row_order="class_blocked"
        ),
        PULabelingConfig("scar", "fixed_count", rng, c=args.c),
    )
    observed = result.observations
    n_labeled = result.manifest["labeling"]["n_labeled_actual"]
    if n_labeled < 5:
        raise ValueError(
            f"Demo data requires at least 5 labeled positives; --n/--c produce {n_labeled}. "
            f"Increase --n or --c (got n={args.n}, c={args.c})."
        )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # String feature names: numeric headers are indistinguishable from data
    # rows when a CLI reader checks for a missing header.
    features = observed.features
    pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])]).to_csv(
        out / "X.csv", index=False
    )
    pd.DataFrame({"label": observed.labels}).to_csv(out / "y_pu.csv", index=False)
    pd.DataFrame({"sample_id": observed.sample_ids}).to_csv(out / "sample_ids.csv", index=False)
    pd.DataFrame({"label": result.audit.true_labels}).to_csv(out / "y_true.csv", index=False)
    manifest = {**result.manifest, "cli_seed": args.seed}
    (out / "construction.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote PU observations, audit truth and construction record to {out} ({len(features)} rows)"
    )
