# ruff: noqa: N802, N803, N806, E501

"""The ``profile`` subcommand: profile data and diagnose SCAR/SAR assumptions.

pu-workflow step 1: reads X/y_pu CSV (or .npy images) and writes a
strict-JSON profile (``profile.json``) that the recommend step consumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..preprocessing.data_profiler import profile_pu_data
from .run import _load_features, _load_label_column

__all__ = ["build_profile_parser", "run_profile"]


def build_profile_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``profile`` subcommand to *sub* (side-effect only)."""
    parser = sub.add_parser(
        "profile",
        help="profile data and diagnose SCAR/SAR assumptions (pu-workflow step 1)",
    )
    parser.add_argument("--data", required=True, help="feature matrix CSV (header row required)")
    parser.add_argument("--labels", required=True, help="PU labels CSV, single column {1, 0}")
    parser.add_argument("--true-labels", default=None, help="true labels CSV {0, 1} (optional)")
    parser.add_argument("--out-dir", default=".", help="output directory (default: current)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.set_defaults(func=run_profile)


def run_profile(args: argparse.Namespace) -> None:
    """Write ``profile.json`` into ``--out-dir``; raise on user errors."""
    X = _load_features(Path(args.data))
    y_pu = _load_label_column(Path(args.labels), "labels")
    y_true = (
        _load_label_column(Path(args.true_labels), "true-labels")
        if args.true_labels is not None
        else None
    )
    profile = profile_pu_data(X, y_pu, y_true=y_true, random_state=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (out_dir / "profile.json").write_text(payload, encoding="utf-8")
