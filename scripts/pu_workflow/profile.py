#!/usr/bin/env python3
# ruff: noqa: N802, N803, N806, E501
"""pu-workflow step 1: profile data and diagnose SCAR/SAR assumptions.

Reads X/y_pu CSV (or .npy images) and writes a strict-JSON profile
(``profile.json``) that the recommend step consumes.  Exit codes: 0
success; 1 user/input error (clear stderr message, no traceback).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pu_toolbox.cli.run import _load_features, _load_label_column
from pu_toolbox.preprocessing.data_profiler import profile_pu_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pu-workflow-profile")
    parser.add_argument("--data", required=True, help="feature matrix CSV (header row required)")
    parser.add_argument("--labels", required=True, help="PU labels CSV, single column {1, 0}")
    parser.add_argument("--true-labels", default=None, help="true labels CSV {0, 1} (optional)")
    parser.add_argument("--out-dir", default=".", help="output directory (default: current)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    args = parser.parse_args(argv)

    try:
        X = _load_features(Path(args.data))
        y_pu = _load_label_column(Path(args.labels), "labels")
        y_true = (
            _load_label_column(Path(args.true_labels), "true-labels")
            if args.true_labels is not None
            else None
        )
        profile = profile_pu_data(X, y_pu, y_true=y_true, random_state=args.seed)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (out_dir / "profile.json").write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
