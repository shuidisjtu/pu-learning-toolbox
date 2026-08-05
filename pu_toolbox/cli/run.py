# ruff: noqa: N802, N803, N806, E501

"""The ``run`` subcommand: one-command end-to-end PU workflow."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.exceptions import PULearningError
from ..workflows import PUPipeline

__all__ = ["build_run_parser", "run_run"]


def build_run_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``run`` subcommand to *sub* (side-effect only)."""
    parser = sub.add_parser("run", help="run the full PU workflow on CSV data")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="feature matrix CSV (rows = samples; first row must be a header)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="PU labels CSV, single column {1, 0} (first row must be a header)",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="output directory (report.json + report.md)"
    )
    parser.add_argument(
        "--true-labels", type=str, default=None, help="true labels CSV {0, 1} for oracle metrics"
    )
    parser.add_argument(
        "--class-prior", type=float, default=None, help="explicit class prior (0, 1)"
    )
    parser.add_argument(
        "--prior-estimator", type=str, default="recpe", help="recpe | pen_l1 | km1 | km2 | none"
    )
    parser.add_argument(
        "--classifier", type=str, default="auto", help="registered method name or 'auto'"
    )
    parser.add_argument("--cv", type=int, default=5, help="number of CV folds (default: 5)")
    parser.add_argument("--metrics", type=str, default=None, help="comma-separated metric names")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--save-model", action="store_true", help="also save the fitted model to model.pkl"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the summary printout")
    parser.set_defaults(func=run_run)


def run_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` subcommand; maps user errors to exit code 1."""
    try:
        _run(args)
    except (PULearningError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)

    X = _load_features(data_path)
    y_pu = _load_label_column(labels_path, "labels")
    y_true = _load_label_column(Path(args.true_labels), "true-labels") if args.true_labels else None

    prior_estimator: str | None = None if args.prior_estimator == "none" else args.prior_estimator
    metrics = args.metrics.split(",") if args.metrics else None

    pipe = PUPipeline(
        classifier=args.classifier,
        prior_estimator=prior_estimator,
        cv=args.cv,
        metrics=metrics,
        random_state=args.seed,
    )
    report = pipe.fit_evaluate(X, y_pu, y_true=y_true, class_prior=args.class_prior)

    out_dir.mkdir(parents=True, exist_ok=True)
    report.save(out_dir / "report.json")
    report.save(out_dir / "report.md")
    if args.save_model:
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(report.final_model, f)
    if not args.quiet:
        print(report.summary())


def _load_features(path: Path) -> np.ndarray:
    """Read a feature-matrix CSV into a float ndarray."""
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"cannot parse CSV {path}: {exc}") from exc
    return df.to_numpy(dtype=float)


def _load_label_column(path: Path, what: str) -> np.ndarray:
    """Read a single-column labels CSV; header (if any) is ignored."""
    if not path.exists():
        raise FileNotFoundError(f"{what} file not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"cannot parse CSV {path}: {exc}") from exc
    return df.iloc[:, 0].to_numpy(dtype=float)
