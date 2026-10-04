#!/usr/bin/env python3
"""Ruff lint + format gate (6th quality gate).

Runs ``ruff check`` and ``ruff format --check`` over the full source
surface (pu_toolbox tests benchmarks examples scripts) — the same
scope the CI workflow uses — so the local gate can never diverge from
CI again.  (A local commit that skipped ``ruff format --check`` passed
the 5 legacy gates and then failed the CI format step on 2026-08-09.)

``SCOPE`` is derived from the git index, not hand-maintained: it is the
set of top-level components of the tracked ``*.py`` files
(``git ls-files '*.py' | cut -d/ -f1 | sort -u`` → today exactly
``benchmarks examples pu_toolbox scripts tests``).  A new top-level
source root is therefore linted as soon as it is staged, with no list
to remember to update — the gate answers "does it cover the new files?"
by construction.

The index decides which *roots* are checked, not which *files*: the
roots are passed to ruff as directories, so ruff checks the ``.py``
files present in the working tree under them.  A file that exists only
locally is therefore linted here but is absent from a CI checkout — a
local green is not by itself proof that CI is green.

When git is unavailable (source tarball, stripped image) the derivation
falls back to ``FALLBACK_SCOPE`` and says so on stderr.

Usage::

    uv run python scripts/check_format.py

Exit 0 when both checks pass, 1 otherwise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Used when the git index cannot be read (source tarball, stripped image);
# the five roots this repository tracked when the derivation was introduced.
FALLBACK_SCOPE = ("benchmarks", "examples", "pu_toolbox", "scripts", "tests")


def _tracked_source_roots() -> tuple[str, ...]:
    """Top-level components of the tracked ``*.py`` files, in stable order."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        print(
            f"check_format: git unavailable ({exc}); using FALLBACK_SCOPE={FALLBACK_SCOPE}.",
            file=sys.stderr,
        )
        return FALLBACK_SCOPE
    roots = {line.split("/", 1)[0] for line in proc.stdout.splitlines() if line.strip()}
    if proc.returncode != 0 or not roots:
        print(
            "check_format: `git ls-files '*.py'` yielded no tracked roots; "
            f"using FALLBACK_SCOPE={FALLBACK_SCOPE}.",
            file=sys.stderr,
        )
        return FALLBACK_SCOPE
    return tuple(sorted(roots))


SCOPE = _tracked_source_roots()


def run_ruff(
    args: list[str], *, cwd: Path = PROJECT_ROOT, scope: tuple[str, ...] = SCOPE
) -> tuple[bool, str]:
    """Run one ruff subcommand over *scope*; return (ok, combined output)."""
    cmd = [sys.executable, "-m", "ruff", *args, *scope]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def main(argv: list[str] | None = None) -> int:
    """Run lint + format checks; return 0 when clean."""
    sys.stdout.reconfigure(encoding="utf-8")
    lint_ok, lint_out = run_ruff(["check"])
    fmt_ok, fmt_out = run_ruff(["format", "--check"])
    if lint_ok and fmt_ok:
        print("Format gate passed: ruff check + ruff format --check clean.")
        return 0
    print("Format gate failed:")
    if not lint_ok:
        print("  ruff check:")
        for line in lint_out.splitlines():
            print(f"    {line}")
    if not fmt_ok:
        print("  ruff format --check:")
        for line in fmt_out.splitlines():
            print(f"    {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
