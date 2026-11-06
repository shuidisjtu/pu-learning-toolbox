#!/usr/bin/env python
"""Summarize recorded v3 manifests without batch, protocol or approval files."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from pu_toolbox.experiment.reporting import write_summary
from pu_toolbox.experiment.results import build_summary
from pu_toolbox.experiment.survey_provenance import resolve_code_commit, with_code_commit
from pu_toolbox.experiment.survey_summary import SummaryError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests", nargs="+", required=True, help="manifest files or result directories"
    )
    parser.add_argument(
        "--expected-seeds", default="0,1,2,3,4", help="comma-separated expected seeds"
    )
    parser.add_argument(
        "--pending", nargs="*", default=[], help="unreviewed inputs, excluded from means"
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="excluded inputs, retained in coverage"
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    try:
        report = build_summary(
            args.manifests,
            expected_seeds=[int(seed) for seed in args.expected_seeds.split(",")],
            pending_paths=args.pending,
            excluded_paths=args.exclude,
        )
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        commit, dirty = resolve_code_commit()
        report["provenance"] = with_code_commit(report.get("provenance"), commit, dirty=dirty)
        for path in write_summary(report, args.out_dir):
            print(f"wrote {path}")
    except (SummaryError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"overall: {report['overall']}; rows: {len(report['rows'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
