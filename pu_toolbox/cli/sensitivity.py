# ruff: noqa: N802, N803, N806, E501

"""The ``sensitivity`` subcommand: assumption sensitivity analysis.

pu-workflow step 4: fits a fast classifier (default elkan_noto), then
audits the fixed model outputs over class-prior / label-propensity
assumptions.  Writes ``sensitivity.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..diagnostics.sensitivity import analyze_pu_sensitivity
from ..registry import get_algorithm, get_metadata
from ..registry.builtin_methods import register_all_builtin_methods
from .run import _load_features, _load_label_column

__all__ = ["build_sensitivity_parser", "run_sensitivity"]

_DEFAULT_CLASS_PRIORS = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"


def _parse_floats(text: str, what: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"cannot parse {what} grid {text!r}: values must be floats") from exc


def build_sensitivity_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``sensitivity`` subcommand to *sub* (side-effect only)."""
    parser = sub.add_parser(
        "sensitivity",
        help="assumption sensitivity analysis on fitted predictions (pu-workflow step 4)",
    )
    parser.add_argument("--data", required=True, help="feature matrix CSV (header row required)")
    parser.add_argument("--labels", required=True, help="PU labels CSV, single column {1, 0}")
    parser.add_argument(
        "--classifier",
        default="elkan_noto",
        help="registered classifier to fit (must not require a class prior; default: elkan_noto)",
    )
    parser.add_argument(
        "--class-priors",
        default=_DEFAULT_CLASS_PRIORS,
        help=f"comma-separated grid (default: {_DEFAULT_CLASS_PRIORS})",
    )
    parser.add_argument(
        "--label-propensities", default=None, help="comma-separated grid (optional)"
    )
    parser.add_argument("--out-dir", default=".", help="output directory (default: current)")
    parser.set_defaults(func=run_sensitivity)


def run_sensitivity(args: argparse.Namespace) -> None:
    """Write ``sensitivity.json`` into ``--out-dir``; raise on user errors."""
    X = _load_features(Path(args.data))
    y_pu = _load_label_column(Path(args.labels), "labels")
    register_all_builtin_methods()
    meta = get_metadata(args.classifier)
    if meta.requires_class_prior:
        raise ValueError(
            f"classifier {args.classifier!r} requires a class prior at fit "
            "time, but sensitivity analysis fits without one; choose a "
            "prior-free classifier (e.g. elkan_noto, nnpu, pusb)"
        )
    clf = get_algorithm(args.classifier)()
    clf.fit(X, y_pu)
    y_pred = clf.predict(X)
    priors = _parse_floats(args.class_priors, "class-priors")
    propensities = (
        _parse_floats(args.label_propensities, "label-propensities")
        if args.label_propensities
        else None
    )
    analysis = analyze_pu_sensitivity(
        y_pu, y_pred, class_priors=priors, label_propensities=propensities
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sensitivity.json").write_text(analysis.to_json() + "\n", encoding="utf-8")
