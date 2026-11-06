#!/usr/bin/env python
"""Export a result summary or recorded manifests to a workbook, cards and conclusions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pu_toolbox.experiment.reporting import export, write_summary
from pu_toolbox.experiment.results import build_summary
from pu_toolbox.experiment.survey_summary import SummaryError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--summary", help="a finished summary.json")
    source.add_argument("--manifests", nargs="+", help="manifest files or result directories")
    parser.add_argument("--expected-seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--pending", nargs="*", default=[], help="unreviewed inputs, excluded from means"
    )
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--literature", help="optional JSON {'facts': [...]} of cited literature facts"
    )
    parser.add_argument(
        "--native-views", help="optional JSON method-to-os/ts/both mapping for diagnostics"
    )
    args = parser.parse_args(argv)
    try:
        if args.summary and (args.pending or args.exclude):
            raise ValueError(
                "--pending and --exclude require --manifests; summaries are already aggregated"
            )
        literature = []
        if args.literature:
            literature = json.loads(Path(args.literature).read_text(encoding="utf-8"))["facts"]
        native_view = None
        if args.native_views:
            native_view = json.loads(Path(args.native_views).read_text(encoding="utf-8"))
            if not isinstance(native_view, dict) or any(
                value not in {"os", "ts", "both"} for value in native_view.values()
            ):
                raise ValueError("native views must map methods to os, ts or both")
        if args.manifests:
            report = build_summary(
                args.manifests,
                expected_seeds=[int(seed) for seed in args.expected_seeds.split(",")],
                pending_paths=args.pending,
                excluded_paths=args.exclude,
            )
            for path in write_summary(report, args.out_dir):
                print(f"wrote {path}")
        else:
            report = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        for path in export(report, args.out_dir, literature=literature, native_view=native_view):
            print(f"wrote {path}")
    except (SummaryError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"overall: {report.get('overall')}; rows: {len(report.get('rows', []))}")
    return 0 if report.get("overall") != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
