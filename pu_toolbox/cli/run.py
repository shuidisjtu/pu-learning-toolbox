# ruff: noqa: N802, N803, N806, E501

"""The ``run`` subcommand: one-command end-to-end PU workflow."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflows import PUPipeline

__all__ = ["build_run_parser", "run_run"]


def _coerce_value(value: str) -> int | float | str:
    """Coerce a CLI value to int, then float, falling back to str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_prior_params(items: list[str] | None) -> dict[str, int | float | str]:
    """Parse repeated ``--prior-param KEY=VALUE`` into a dict."""
    params: dict[str, int | float | str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--prior-param expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--prior-param expects a non-empty key, got {item!r}")
        params[key] = _coerce_value(value)
    return params


def build_run_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``run`` subcommand to *sub* (side-effect only)."""
    parser = sub.add_parser("run", help="run the full PU workflow on CSV data")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="feature matrix: CSV table (rows = samples; first row must be a header) "
        "or .npy 4-D NCHW image array",
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
        "--prior-estimator",
        type=str,
        default="pen_l1",
        help="prior-estimator name or alias (see 'list-priors'); 'none' disables estimation",
    )
    parser.add_argument(
        "--prior-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="prior-estimator parameter, repeatable (e.g. --prior-param sigma=3.0); "
        "values are coerced to int/float when possible",
    )
    parser.add_argument(
        "--classifier", type=str, default="auto", help="registered method name or 'auto'"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mlp",
        choices=["mlp", "cnn"],
        help="network architecture for deep classifiers: 'mlp' (table data) "
        "or 'cnn' (4-D NCHW images; requires --classifier wconpu/infomax_pu)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["cnn13", "resnet18", "resnet50"],
        help="CNN backbone when --architecture cnn (default: cnn13)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="torch device for deep classifiers: auto/cpu/cuda (default: auto)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="max epochs for deep estimators whose signature accepts it "
        "(e.g. wconpu, self_pu, nnpu)",
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
    """Execute the ``run`` subcommand.

    Error mapping (``error: <msg>`` + exit 1) happens once in the top-level
    :func:`pu_toolbox.cli.main`; this subcommand no longer duplicates it.
    """
    _run(args)


def _run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)

    X = _load_features(data_path)
    y_pu = _load_label_column(labels_path, "labels")
    y_true = (
        _load_label_column(Path(args.true_labels), "true-labels")
        if args.true_labels is not None
        else None
    )

    prior_estimator: str | None = None if args.prior_estimator == "none" else args.prior_estimator
    prior_params = _parse_prior_params(args.prior_param)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None

    if args.architecture == "mlp" and args.backbone is not None:
        raise ValueError("--backbone is only valid with --architecture cnn")

    pipe = PUPipeline(
        classifier=args.classifier,
        prior_estimator=prior_estimator,
        prior_params=prior_params,
        cv=args.cv,
        metrics=metrics,
        random_state=args.seed,
        architecture=args.architecture,
        backbone=args.backbone or "cnn13",
        device=args.device,
        max_epochs=args.max_epochs,
    )
    report = pipe.fit_evaluate(X, y_pu, y_true=y_true, class_prior=args.class_prior)

    if out_dir.exists():
        print(f"note: {out_dir} already exists; files will be overwritten", file=sys.stderr)
        if (out_dir / "model.pkl").exists() and not args.save_model:
            print(
                f"note: stale {out_dir / 'model.pkl'} from a previous run remains "
                "and does not match this run",
                file=sys.stderr,
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save(out_dir / "report.json")
    report.save(out_dir / "report.md")
    if args.save_model:
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(report.final_model, f)
    if not args.quiet:
        print(report.summary())


def _read_csv(path: Path, what: str) -> pd.DataFrame:
    """Read a CSV, rejecting headerless files instead of dropping a row.

    ``pd.read_csv(header=0)`` treats the first row as column names, so a
    headerless numeric CSV silently loses its first sample.  A genuine
    header row (column names) is non-numeric; when the first row is all
    numeric we cannot tell it apart from data and fail with a clear error.
    """
    if not path.exists():
        raise FileNotFoundError(f"{what} file not found: {path}")
    try:
        raw = pd.read_csv(path, header=None)
    except pd.errors.ParserError as exc:
        raise ValueError(f"cannot parse CSV {path}: {exc}") from exc
    try:
        raw.iloc[0].astype(float)
    except (ValueError, TypeError):
        # Non-numeric first row -> a real header; read with header=0.
        return pd.read_csv(path, header=0)
    raise ValueError(
        f"{what} CSV {path}: the first row is numeric, so it cannot be "
        "distinguished from data. Add a header row (non-numeric column "
        "names) -- otherwise the first sample is consumed as column names."
    )


def _load_features(path: Path) -> np.ndarray:
    """Read a feature matrix: CSV table (2-D) or .npy image array (4-D NCHW)."""
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if not isinstance(array, np.ndarray):
            # np.load on a .npz archive returns an NpzFile (no .ndim),
            # which would surface as a bare AttributeError outside the
            # CLI's error mapping.
            raise ValueError(
                f"image data file {path} must be a .npy array; got "
                f"{type(array).__name__}. Only .npy (not .npz) archives are supported."
            )
        if array.ndim != 4:
            raise ValueError(
                f"image data file {path} must be a 4-D NCHW float array; "
                f"got ndim={array.ndim}. Table data must use CSV."
            )
        array = array.astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"image data file {path} contains NaN or Inf values.")
        return array
    array = _read_csv(path, "data").to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(
            f"data CSV {path} contains NaN or Inf values. "
            "Impute or remove them before running."
        )
    return array


def _load_label_column(path: Path, what: str) -> np.ndarray:
    """Read a single-column labels CSV; header (if any) is tolerated."""
    df = _read_csv(path, what)
    if df.shape[1] != 1:
        raise ValueError(f"{what} CSV {path} must be a single column; got {df.shape[1]} columns.")
    return df.iloc[:, 0].to_numpy(dtype=float)
