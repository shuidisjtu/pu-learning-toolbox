# ruff: noqa: N802, N803, N806

"""The ``shift-audit`` subcommand for source-to-target PU drift."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..diagnostics import analyze_pu_shift
from ..workflows import PUPipeline, ShiftAwarePUPipeline
from .run import _load_features, _load_label_column, _parse_classifier_params

__all__ = ["build_shift_parser", "run_shift_audit", "run_shift_run"]


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

    run = sub.add_parser(
        "shift-run",
        help="compare unweighted and covariate-weighted PU models on one target domain",
    )
    for action in parser._actions[1:]:
        if action.dest in {"help", "probability_clip", "quiet"}:
            continue
        required = action.dest in {"source_data", "source_labels", "target_data", "out_dir"}
        kwargs = {
            "dest": action.dest,
            "default": action.default,
            "required": required,
            "help": action.help,
        }
        if action.type is not None:
            kwargs["type"] = action.type
        run.add_argument(*action.option_strings, **kwargs)
    run.get_default("target_labels")
    for action in run._actions:
        if action.dest == "target_labels":
            action.required = True
            action.help = "target PU labels: headered one-column CSV {1, 0}"
    run.add_argument("--source-true-labels", default=None, help="optional source truth CSV {0, 1}")
    run.add_argument("--target-true-labels", default=None, help="optional target truth CSV {0, 1}")
    run.add_argument("--class-prior", type=float, default=None, help="source class prior")
    run.add_argument("--target-class-prior", type=float, default=None, help="target class prior")
    run.add_argument("--prior-estimator", default="pen_l1", help="source prior estimator")
    run.add_argument("--classifier", default="elkan_noto", help="weighted-capable classifier")
    run.add_argument("--classifier-param", action="append", default=None, metavar="KEY=VALUE")
    run.add_argument("--metrics", default=None, help="comma-separated metrics")
    run.add_argument("--primary-metric", default=None, help="target metric used for recommendation")
    run.add_argument(
        "--min-improvement", type=float, default=0.0, help="minimum target improvement"
    )
    run.add_argument(
        "--allow-unstable", action="store_true", help="run weighted arm despite overlap warning"
    )
    run.add_argument("--quiet", action="store_true", help="suppress terminal summary")
    run.set_defaults(func=run_shift_run)


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


def run_shift_run(args: argparse.Namespace) -> None:
    """Run a paired baseline/weighted comparison and save stable artifacts."""
    source = _load_features(Path(args.source_data))
    source_labels = _load_label_column(Path(args.source_labels), "source-labels")
    target = _load_features(Path(args.target_data))
    target_labels = _load_label_column(Path(args.target_labels), "target-labels")
    source_truth = (
        _load_label_column(Path(args.source_true_labels), "source-true-labels")
        if args.source_true_labels is not None
        else None
    )
    target_truth = (
        _load_label_column(Path(args.target_true_labels), "target-true-labels")
        if args.target_true_labels is not None
        else None
    )
    metrics = (
        [item.strip() for item in args.metrics.split(",") if item.strip()] if args.metrics else None
    )
    pipeline = PUPipeline(
        classifier=args.classifier,
        classifier_params=_parse_classifier_params(args.classifier_param),
        prior_estimator=None if args.prior_estimator == "none" else args.prior_estimator,
        cv=args.cv,
        metrics=metrics,
        random_state=args.seed,
    )
    comparison = ShiftAwarePUPipeline(
        pipeline=pipeline,
        alpha=args.alpha,
        shift_cv=args.cv,
        allow_unstable=args.allow_unstable,
    ).compare(
        source,
        source_labels,
        target,
        y_target_pu=target_labels,
        y_true_source=source_truth,
        y_true_target=target_truth,
        class_prior=args.class_prior,
        target_class_prior=args.target_class_prior,
        primary_metric=args.primary_metric,
        min_improvement=args.min_improvement,
    )
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"note: {out_dir} already exists; files will be overwritten", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.save(out_dir / "shift_comparison.json")
    comparison.save(out_dir / "shift_comparison.md")
    comparison.shift.save(out_dir / "shift_report.json")
    comparison.shift.save(out_dir / "source_importance_weights.csv")
    rows = {"baseline_prediction": comparison.baseline.target_predictions}
    if comparison.weighted is not None:
        rows["weighted_prediction"] = comparison.weighted.target_predictions
    pd.DataFrame(rows).to_csv(out_dir / "target_predictions.csv", index=False)
    if not args.quiet:
        print(
            "PU shift comparison: "
            f"recommendation={comparison.recommendation}, "
            f"primary_metric={comparison.primary_metric or 'unavailable'}"
        )
