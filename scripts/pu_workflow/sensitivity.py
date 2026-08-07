#!/usr/bin/env python3
# ruff: noqa: N802, N803, N806, E402, E501
"""pu-workflow step 4: assumption sensitivity analysis on fitted predictions.

Fits a fast classifier (default elkan_noto), then audits the fixed model
outputs over class-prior / label-propensity assumptions.  Writes
``sensitivity.json``.  Exit codes: 0 success; 1 user/input error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_toolbox.cli.run import _load_features, _load_label_column
from pu_toolbox.core.exceptions import PULearningError
from pu_toolbox.diagnostics.sensitivity import analyze_pu_sensitivity
from pu_toolbox.registry import get_algorithm
from pu_toolbox.registry.builtin_methods import register_all_builtin_methods

_DEFAULT_CLASS_PRIORS = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"


def _parse_floats(text: str, what: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"cannot parse {what} grid {text!r}: values must be floats") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pu-workflow-sensitivity")
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
    args = parser.parse_args(argv)

    try:
        X = _load_features(Path(args.data))
        y_pu = _load_label_column(Path(args.labels), "labels")
        register_all_builtin_methods()
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
    except (OSError, ValueError, PULearningError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
