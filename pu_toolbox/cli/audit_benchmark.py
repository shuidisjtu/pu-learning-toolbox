"""The ``audit-benchmark`` subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path

from pu_toolbox.diagnostics.benchmark import audit_benchmark_results

__all__ = ["build_audit_benchmark_parser", "run_audit_benchmark"]


def build_audit_benchmark_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the benchmark artifact auditor to the top-level CLI."""
    parser = sub.add_parser(
        "audit-benchmark",
        help="validate benchmark completeness, provenance, metrics, and PU splits",
    )
    parser.add_argument("--result-dir", required=True, help="benchmark output directory")
    parser.add_argument("--output", default=None, help="optional path for the JSON audit report")
    parser.set_defaults(func=run_audit_benchmark)


def run_audit_benchmark(args: argparse.Namespace) -> None:
    """Run the audit, print a concise summary, and optionally persist JSON."""
    report = audit_benchmark_results(args.result_dir)
    if args.output is not None:
        report.save(Path(args.output))

    status = "PASS" if report.passed else "FAIL"
    print(
        f"Benchmark audit: {status} "
        f"({report.n_trials} trials, {report.n_methods} methods, {report.n_seeds} seeds)"
    )
    for message in report.errors:
        print(f"ERROR: {message}")
    for message in report.warnings:
        print(f"WARNING: {message}")
    if not report.passed:
        raise ValueError("benchmark audit failed")
