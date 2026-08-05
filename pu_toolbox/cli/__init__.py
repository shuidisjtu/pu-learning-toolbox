"""Command-line interface for the PU Learning Toolbox.

The CLI is a thin wrapper over :class:`~pu_toolbox.workflows.PUPipeline`:
it parses arguments, reads CSV inputs, delegates to the pipeline, and
maps errors to exit codes.  All learning logic lives in the pipeline.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .demo import build_demo_parser
from .info import build_info_parser
from .run import build_run_parser

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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (console script)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
