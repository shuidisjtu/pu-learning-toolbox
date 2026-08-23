# ruff: noqa: N802, N803, N806

"""Deployment monitoring and active-review CLI commands."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from ..diagnostics import PUShiftMonitor, analyze_pu_uncertainty
from .run import _load_features, _load_label_column

__all__ = ["build_deployment_parser", "run_review", "run_shift_monitor"]


def build_deployment_parser(sub: argparse._SubParsersAction) -> None:
    monitor = sub.add_parser(
        "shift-monitor", help="append one target window to a persistent PU shift history"
    )
    monitor.add_argument("--reference-data", required=True)
    monitor.add_argument("--reference-labels", required=True)
    monitor.add_argument("--window-data", required=True)
    monitor.add_argument("--window-labels", default=None)
    monitor.add_argument("--window-id", required=True)
    monitor.add_argument("--timestamp", default=None)
    monitor.add_argument("--history", default=None, help="optional previous monitor history JSON")
    monitor.add_argument("--out-dir", required=True)
    monitor.add_argument("--alpha", type=float, default=0.1)
    monitor.add_argument("--cv", type=int, default=5)
    monitor.add_argument("--seed", type=int, default=42)
    monitor.add_argument("--auc-jump-threshold", type=float, default=0.1)
    monitor.add_argument("--label-rate-jump-threshold", type=float, default=0.05)
    monitor.add_argument("--quiet", action="store_true")
    monitor.set_defaults(func=run_shift_monitor)

    review = sub.add_parser(
        "review", help="create selective predictions and an active human-review queue"
    )
    review.add_argument(
        "--model", required=True, help="trusted model.pkl produced by pu-toolbox run"
    )
    review.add_argument("--data", required=True)
    review.add_argument(
        "--labels", default=None, help="optional PU labels; excludes labeled positives"
    )
    review.add_argument("--true-labels", default=None)
    review.add_argument("--importance-weights", default=None, help="headered one-column CSV")
    review.add_argument("--out-dir", required=True)
    review.add_argument("--min-confidence", type=float, default=0.5)
    review.add_argument("--query-budget", type=int, default=20)
    review.add_argument(
        "--query-strategy",
        choices=["uncertainty", "shift_weighted", "diverse_uncertainty"],
        default="uncertainty",
    )
    review.add_argument("--seed", type=int, default=42)
    review.add_argument("--quiet", action="store_true")
    review.set_defaults(func=run_review)


def run_shift_monitor(args: argparse.Namespace) -> None:
    reference = _load_features(Path(args.reference_data))
    reference_labels = _load_label_column(Path(args.reference_labels), "reference-labels")
    window_data = _load_features(Path(args.window_data))
    window_labels = (
        _load_label_column(Path(args.window_labels), "window-labels")
        if args.window_labels is not None
        else None
    )
    monitor = PUShiftMonitor(
        reference,
        reference_labels,
        alpha=args.alpha,
        cv=args.cv,
        random_state=args.seed,
        auc_jump_threshold=args.auc_jump_threshold,
        label_rate_jump_threshold=args.label_rate_jump_threshold,
    )
    if args.history is not None:
        monitor.load_history(args.history)
    window, shift = monitor.update(
        window_data,
        y_window_pu=window_labels,
        window_id=args.window_id,
        timestamp=args.timestamp,
    )
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"note: {out_dir} already exists; files will be overwritten", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor.save_history(out_dir / "shift_history.json")
    shift.save(out_dir / "window_shift.json")
    shift.save(out_dir / "window_shift.md")
    shift.save(out_dir / "source_importance_weights.csv")
    (out_dir / "window_summary.json").write_text(
        json.dumps(window.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        print(
            f"PU shift window: id={window.window_id}, alert={window.alert_level}, "
            f"codes={','.join(window.alert_codes) or 'none'}"
        )


def _load_weights(path: str | None) -> object | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if frame.shape[1] != 1:
        raise ValueError("importance-weights CSV must contain exactly one column.")
    return frame.iloc[:, 0].to_numpy(dtype=float)


def run_review(args: argparse.Namespace) -> None:
    # Pickle can execute code while loading. The CLI deliberately names this a
    # trusted artifact and never downloads or auto-discovers model files.
    with Path(args.model).open("rb") as handle:
        model = pickle.load(handle)  # noqa: S301
    data = _load_features(Path(args.data))
    labels = _load_label_column(Path(args.labels), "labels") if args.labels is not None else None
    truth = (
        _load_label_column(Path(args.true_labels), "true-labels")
        if args.true_labels is not None
        else None
    )
    report = analyze_pu_uncertainty(
        model,
        data,
        y_pu=labels,
        y_true=truth,
        min_confidence=args.min_confidence,
        query_budget=args.query_budget,
        query_strategy=args.query_strategy,
        importance_weight=_load_weights(args.importance_weights),
        random_state=args.seed,
    )
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"note: {out_dir} already exists; files will be overwritten", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save(out_dir / "uncertainty.json")
    report.save(out_dir / "uncertainty.md")
    report.save(out_dir / "uncertainty_rows.csv")
    if not args.quiet:
        print(
            f"PU review: coverage={report.summary['coverage']:.3f}, "
            f"queries={report.summary['n_queries']}"
        )
