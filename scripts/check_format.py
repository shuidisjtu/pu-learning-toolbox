#!/usr/bin/env python3
"""Ruff lint + format gate (6th quality gate).

Runs ``ruff check`` and ``ruff format --check`` over the full source
surface (pu_toolbox tests benchmarks examples scripts) — the same
scope the CI workflow uses — so the local gate can never diverge from
CI again.  (A local commit that skipped ``ruff format --check`` passed
the 5 legacy gates and then failed the CI format step on 2026-08-09.)

``SCOPE`` is a hand-maintained inventory, and it must stay one: "the
project's source surface" is a policy decision, not a filesystem fact,
so no derivation from the tree can reproduce it.  The obvious dynamic
replacement — enumerate the top-level directories — is *wider*, not
stricter.  ``data/`` and ``dist/`` are gitignored local-only trees
(``.gitignore:5`` / ``:68``) that are absent from a CI checkout, yet
naming them explicitly makes ruff lint them::

    $ printf 'import os\\n' > data/scratch_probe.py     # gitignored scratch file
    $ uv run ruff check benchmarks data dist docs examples pu_toolbox scripts tests
     --> data\\scratch_probe.py:1:8       # F401 in a gitignored, local-only tree

That turns the verdict into a function of the working tree, so the same
commit can fail locally and pass in CI — exactly the divergence this
gate was written to prevent.  A new top-level source root is therefore
linted only after it is added to ``SCOPE`` by hand.

Usage::

    uv run python scripts/check_format.py

Exit 0 when both checks pass, 1 otherwise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep in sync with the CI scope in .github/workflows/tests.yml.
SCOPE = ("pu_toolbox", "tests", "benchmarks", "examples", "scripts")


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
