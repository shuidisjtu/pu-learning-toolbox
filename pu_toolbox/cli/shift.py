# ruff: noqa: N802, N803, N806

"""The ``shift-audit`` subcommand for source-to-target PU drift."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..diagnostics import analyze_pu_shift
from .run import _load_features, _load_label_column

__all__ = ["build_shift_parser", "run_shift_audit"]


def build_shift_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``shift-audit`` subcommand to *sub*."""
    parser = sub.add_parser(
        "shift-audit",
        help="audit source-to-target drift and export bounded covariate weights",
    )
    parser.add_argument(
        "--source-data",
        required=True,
        help="source features: headered CSV table or .npy 4-D NCHW array",
    )
    parser.add_argument(
        "--source-labels",
        required=True,
        help="source PU labels: headered one-column CSV {1, 0}",
    )
    parser.add_argument(
        "--target-data",
        required=True,
        help="target features with the same flattened feature count as source",
    )
    parser.add_argument(
        "--target-labels",
        default=None,
        help="optional target PU labels: headered one-column CSV {1, 0}",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="output directory for shift report and source weights",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="relative density-ratio parameter in (0, 1] (default: 0.1)",
    )
    parser.add_argument(
        "--probability-clip",
        type=float,
        default=1e-6,
        help="domain-probability clipping value in (0, 0.5) (default: 1e-6)",
    )
    parser.add_argument("--cv", type=int, default=5, help="domain OOF folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true", help="suppress terminal summary")
    parser.set_defaults(func=run_shift_audit)


def run_shift_audit(args: argparse.Namespace) -> None:
    """Run the drift audit and write its three stable artifacts."""
    source = _load_features(Path(args.source_data))
    source_labels = _load_label_column(Path(args.source_labels), "source-labels")
    target = _load_features(Path(args.target_data))
    target_labels = (
        _load_label_column(Path(args.target_labels), "target-labels")
        if args.target_labels is not None
        else None
    )
    report = analyze_pu_shift(
        source,
        source_labels,
        target,
        y_target_pu=target_labels,
        alpha=args.alpha,
        probability_clip=args.probability_clip,
        cv=args.cv,
        random_state=args.seed,
    )
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"note: {out_dir} already exists; files will be overwritten", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save(out_dir / "shift_report.json")
    report.save(out_dir / "shift_report.md")
    report.save(out_dir / "source_importance_weights.csv")
    if not args.quiet:
        print(
            "PU shift audit: "
            f"domain_auc={report.domain_auc:.3f}, "
            f"severity={report.severity}, "
            f"adaptation_ready={report.adaptation_ready}"
        )
