#!/usr/bin/env python3
# ruff: noqa: E402
"""pu-workflow step 4: assumption sensitivity analysis on fitted predictions.

Compatibility wrapper: delegates to the ``pu-toolbox sensitivity``
subcommand.  Exit codes: 0 success; 1 user/input error.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_toolbox.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    cli_main(["sensitivity", *(argv if argv is not None else sys.argv[1:])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
