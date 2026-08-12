"""Command-line interface for the PU Learning Toolbox.

The CLI is a thin wrapper over :class:`~pu_toolbox.workflows.PUPipeline`:
it parses arguments, reads CSV inputs, delegates to the pipeline, and
maps errors to exit codes.  All learning logic lives in the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..core.exceptions import PULearningError
from .audit_benchmark import build_audit_benchmark_parser
from .demo import build_demo_parser
from .info import build_info_parser
from .profile import build_profile_parser
from .recommend import build_recommend_parser
from .run import build_run_parser
from .sensitivity import build_sensitivity_parser
from .skill import build_skill_parser

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with all subcommands attached."""
    parser = argparse.ArgumentParser(
        prog="pu-toolbox",
        description="Positive-Unlabeled learning toolbox: one-command end-to-end PU workflows.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    build_run_parser(sub)
    build_info_parser(sub)
    build_demo_parser(sub)
    build_profile_parser(sub)
    build_recommend_parser(sub)
    build_sensitivity_parser(sub)
    build_skill_parser(sub)
    build_audit_benchmark_parser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (console script).

    User errors map to ``error: <msg>`` on stderr + exit 1 (no traceback)
    for every subcommand.  Unknown exceptions keep their traceback for
    debugging.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (PULearningError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
