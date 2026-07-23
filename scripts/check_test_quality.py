#!/usr/bin/env python3
"""Test quality gate - enforce per-module limits, markers, and coverage.

Rules (aligned with ``docs/project_structure.md`` §3):
1. **Count**: <= 15 test methods per paper / module file.
2. **Markers**: every test class or method must carry a registered pytest
   marker (``unit``, ``math``, ``property``, ``contract``, ``slow``,
   ``paper``).
3. **Coverage**: each file should touch all four categories:
   - *basic* — smoke / functional correctness
   - *param*  — parameter validation / error paths
   - *edge*   — boundary conditions / empty inputs / extremes
   - *determ* — determinism / seed reproducibility

Usage::

    uv run python scripts/check_test_quality.py [--max 15] [--strict]

Exit 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple

# ═════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"

# Registered markers from pyproject.toml (must stay in sync).
REGISTERED_MARKERS: set[str] = {
    "unit",
    "math",
    "property",
    "contract",
    "slow",
    "paper",
}

# Keywords used to classify test intent.  A test name match counts
# toward that category.  One test can satisfy multiple categories.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "basic": [
        "basic",
        "fit",
        "predict",
        "output",
        "shape",
        "score",
        "smoke",
        "runs",
        "works",
        "estimate",
        "delegation",
        "counts",
        "positive",
        "class_prior",
        "generate",
    ],
    "param": [
        "invalid",
        "raises",
        "error",
        "validation",
    ],
    "edge": [
        "edge",
        "boundary",
        "empty",
        "zero",
        "none",
        "single",
        "extreme",
        "all_",
    ],
    "determ": [
        "determin",
        "seed",
        "reproduc",
        "consistent",
    ],
}

# Files exempt from the ≤15 limit (cross-cutting contract / registry).
UNLIMITED_FILES: set[str] = {
    "test_classifier_baseline.py",  # contract: unified API + baseline for all NATIVE
    "test_builtin_methods.py",  # registry metadata completeness
    "test_registry.py",  # registry mechanics
    "test_import.py",  # smoke imports
}


# ═════════════════════════════════════════════════════════════════════
# Data types
# ═════════════════════════════════════════════════════════════════════


class TestMethod(NamedTuple):
    name: str
    lineno: int
    has_marker: bool  # decorated directly or inherits from class


class ModuleReport(NamedTuple):
    path: Path
    n_tests: int
    has_marker_violations: list[TestMethod]
    categories_found: set[str]
    categories_missing: set[str]


# ═════════════════════════════════════════════════════════════════════
# AST visitor
# ═════════════════════════════════════════════════════════════════════


def _has_marker(decorator_list: list[ast.expr]) -> bool:
    """Return True if any decorator is ``@pytest.mark.<name>``."""
    for dec in decorator_list:
        if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Attribute):
            # @pytest.mark.xxx
            outer = dec.value
            if (
                isinstance(outer.value, ast.Name)
                and outer.value.id == "pytest"
                and outer.attr == "mark"
            ):
                return dec.attr in REGISTERED_MARKERS
    return False


def _classify_name(name: str) -> set[str]:
    """Return the coverage categories that *name* belongs to."""
    found: set[str] = set()
    lower = name.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            found.add(cat)
    return found


def analyse_file(filepath: Path) -> ModuleReport:
    """Parse a test file and return its quality report."""
    tree = ast.parse(filepath.read_text(encoding="utf-8"))

    methods: list[TestMethod] = []
    current_class_marker = False

    for node in ast.walk(tree):
        # Track class-level markers
        if isinstance(node, ast.ClassDef):
            current_class_marker = _has_marker(node.decorator_list)

        # Collect test methods
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # A method inside a class inherits the class marker
            owns_marker = _has_marker(node.decorator_list) or current_class_marker
            methods.append(TestMethod(node.name, node.lineno, owns_marker))

    # Marker violations
    violations = [m for m in methods if not m.has_marker]

    # Category coverage
    all_categories: set[str] = set()
    for m in methods:
        all_categories |= _classify_name(m.name)
    missing = {"basic", "param", "edge", "determ"} - all_categories

    return ModuleReport(
        path=filepath,
        n_tests=len(methods),
        has_marker_violations=violations,
        categories_found=all_categories,
        categories_missing=missing,
    )


# ═════════════════════════════════════════════════════════════════════
# Report & main
# ═════════════════════════════════════════════════════════════════════


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main(max_tests: int = 15, strict: bool = False) -> int:
    """Run all checks and return exit code (0 = clean, 1 = issues found)."""
    # Ensure UTF-8 output on Windows terminals.
    sys.stdout.reconfigure(encoding="utf-8")
    test_files = sorted(TESTS_DIR.rglob("test_*.py"))
    if not test_files:
        print("No test files found.")
        return 1

    reports: list[ModuleReport] = []
    for fp in test_files:
        reports.append(analyse_file(fp))

    # ── Summary header ──────────────────────────────────────────────
    total = sum(r.n_tests for r in reports)
    print(f"Checking {len(reports)} test files ({total} test methods) …\n")

    n_issues = 0

    # ── 1. Count check ──────────────────────────────────────────────
    print("─" * 62)
    print(f"{'File':<48} {'Tests':>6} {'Limit':>6}")
    print("─" * 62)
    for r in reports:
        rel = _relative(r.path)
        status = ""
        if r.path.name in UNLIMITED_FILES:
            status = "  (unlimited)"
        elif r.n_tests > max_tests:
            status = f"  !! OVER LIMIT (>{max_tests})"
            n_issues += 1
        print(f"{rel:<48} {r.n_tests:>6} {max_tests:>6}{status}")

    # ── 2. Marker check ─────────────────────────────────────────────
    print("\n─ Marker compliance ─")
    marker_ok = True
    for r in reports:
        if r.has_marker_violations:
            marker_ok = False
            n_issues += 1
            rel = _relative(r.path)
            names = [m.name for m in r.has_marker_violations]
            print(
                f"  {rel}: {len(names)} unmarked — {', '.join(names[:5])}"
                f"{' …' if len(names) > 5 else ''}"
            )
    if marker_ok:
        print("  ✓ all test methods have a registered marker")

    # ── 3. Coverage check ───────────────────────────────────────────
    print("\n─ Coverage categories (basic / param / edge / determ) ─")
    coverage_ok = True
    for r in reports:
        if r.categories_missing and (strict or len(r.categories_missing) > 1):
            coverage_ok = False
            n_issues += 1
            rel = _relative(r.path)
            print(f"  {rel}: missing {sorted(r.categories_missing)}")
    if coverage_ok:
        print("  ✓ all files cover required categories (or missing ≤1 in relaxed mode)")

    # ── Final verdict ───────────────────────────────────────────────
    print()
    if n_issues == 0:
        print("✓ All checks passed.")
        return 0
    else:
        print(f"✗ {n_issues} issue(s) found.")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test quality gate for PU Learning Toolbox",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=15,
        help="Maximum test methods per module file (default: 15)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require ALL four coverage categories in every file",
    )
    args = parser.parse_args()
    sys.exit(main(max_tests=args.max, strict=args.strict))
