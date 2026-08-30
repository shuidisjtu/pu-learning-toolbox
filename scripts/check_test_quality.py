#!/usr/bin/env python3
"""Test quality gate - enforce per-module limits, markers, and coverage.

Rules (aligned with ``docs/project_structure.md`` §3):
1. **Count**: <= 15 test methods per paper / module file.
2. **Markers**: every test class or method must carry a registered pytest
   marker (``unit``, ``math``, ``property``, ``contract``, ``slow``,
   ``paper``).
3. **Coverage**: each file must touch all four categories (strict by
   default; ``--lenient`` allows at most one missing):
   - *basic* — smoke / functional correctness
   - *param*  — parameter validation / error paths
   - *edge*   — boundary conditions / empty inputs / extremes
   - *determ* — determinism / seed reproducibility
   Categories a file legitimately cannot cover may be declared in
   ``PARTIAL_COVERAGE`` (per category, with a reason) instead of forcing
   filler tests.  Declarations are reprinted every run and flagged as
   removable once the category is actually covered, so the list stays
   honest and shrinkable.
4. **Exemptions**: the three hand-maintained lists are reprinted (with reasons).
   Stale entries fail the gate once their exemption is no longer necessary,
   providing an enforced exit path instead of an ever-growing allowlist.

Usage::

    uv run python scripts/check_test_quality.py [--max 15] [--lenient]

Exit 0 when all checks pass, 1 otherwise.  Strict (all four coverage
categories required) is the default for both local runs and CI, so the
two can never drift apart; ``--lenient`` is the explicit opt-out.
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
    "integration",
    "e2e",
}

# Keywords used to classify test intent.  A test name match counts
# toward that category.  One test can satisfy multiple categories.
# The repo naming convention prefixes tests with ``test_basic_`` /
# ``test_param_`` / ``test_edge_`` / ``test_determ_``, so the ``param``
# and ``determ`` families also carry their literal prefixes as keywords
# (``basic`` / ``edge`` already did).
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
        "param",
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
        "determ",
        "determin",
        "seed",
        "reproduc",
        "consistent",
    ],
}

# Files exempt from the ≤15 limit (cross-cutting contract / registry),
# keyed by file name with the reason each exemption was granted.  The
# exemption review (``review_exemptions``) fails entries that no longer
# need the exemption, so the lists must shrink.
UNLIMITED_FILES: dict[str, str] = {
    "test_generate_structure.py": (
        "structure generator unit tests (find_blocks/parse_tree/build/merge/generate/CLI)"
    ),
    "test_kldce_math.py": "MATH formula verification (includes merged QP oracle tests)",
    "test_kldce_smo.py": (
        "paired-SMO math verification: analytic-step hand cases, "
        "KKT interval feasibility, oracle cross-check"
    ),
    "test_classification.py": (
        "PU + planned supervised/calibration metrics with hand-computed MATH golden values"
    ),
    "test_pipeline.py": (
        "cross-cutting end-to-end integration suite for the PUPipeline workflow "
        "(report contents, prior resolution, auto mode, error paths, determinism, "
        "provenance call-site mapping)"
    ),
}

# Files whose algorithms are fully covered by contract tests
# (test_classifier_baseline.py), so they do not need to independently
# cover all 4 categories (basic / param / edge / determ).
CONTRACT_COVERED_FILES: dict[str, str] = {
    "test_bias_aware.py": "algorithm covered by contract tests",
    "test_dist_pu.py": "algorithm covered by contract tests",
    "test_import.py": "smoke imports",
    "test_kldce_property.py": "algorithm covered by contract tests",
}

# Files whose tests legitimately cannot cover all four categories.
# Declared per missing category with a reason.  Entries are reprinted
# every run and flagged as removable once the category is covered, so
# the list stays honest and shrinkable.
PARTIAL_COVERAGE: dict[str, dict[str, str]] = {
    "test_architecture_capability.py": {
        "basic": (
            "single-purpose gate unit tests on synthetic classifier stubs; "
            "the pipeline fit/predict flow is covered by test_pipeline(_deep) "
            "integration tests"
        ),
        "edge": (
            "the gate's own boundary scenarios (capability/signature mismatch) "
            "are these error-path tests; input boundary validation lives in "
            "encoder and pipeline tests"
        ),
        "determ": (
            "check_architecture_capability is a pure function over metadata "
            "(no randomness, no seed state); deep estimator seed determinism "
            "is unit-tested with fixed seeds"
        ),
    },
    "test_build_encoder_export.py": {
        "basic": (
            "cross-classifier export-contract suite: no fit/predict behavioral "
            "smoke; classifier parameter and boundary behavior are covered by "
            "test_classifier_baseline and the deep unit tests"
        ),
        "determ": (
            "export-contract test compares structure and forward shape only, "
            "never trained weights (random init); no RNG/seed dependence"
        ),
    },
    "test_capability_declarations.py": {
        "basic": (
            "declaration-consistency contract tests (legality / registry-sync / "
            "derivation / signature); fit-predict behavioral smoke is covered by "
            "test_classifier_baseline.py and the algorithm unit suites"
        ),
        "param": (
            "declaration validity is asserted via the four invariants, not via "
            "constructor error paths; parameter validation lives in per-algorithm "
            "unit tests"
        ),
    },
    "test_cli_deep_save_model.py": {
        "param": (
            "end-to-end success-path regression; CLI parameter error "
            "paths live in existing CLI tests"
        ),
        "edge": (
            "success-path regression; boundary inputs are covered by existing CLI/model tests"
        ),
        "determ": (
            "GPU/CUDA deep training is not bit-reproducible; determinism is asserted at unit level"
        ),
    },
    "test_cnn_candidates.py": {
        "basic": (
            "候选集推导为 registry 元数据查询，无 fit/predict 行为；训练流程由 "
            "test_pipeline(_deep) 集成测试覆盖"
        ),
        "param": "无参数校验表面；能力声明合法性由 test_capability_declarations 契约测试覆盖",
        "edge": "断言固定的当前声明集与双向一致性；空集/声明边界由 registry 单测覆盖",
        "determ": (
            "cnn_candidates 为纯函数（无随机性、无种子状态）；深度估计器种子确定性"
            "由既有固定种子单测覆盖"
        ),
    },
    "test_cv_fold_isolation.py": {
        "basic": (
            "单一意图隔离锁测试：2 折各真实训练 1 epoch 并断言权重/对象比较，"
            "无 fit/predict 输出冒烟；管道路径行为由 test_pipeline(_deep) 集成测试覆盖"
        ),
        "param": "无参数校验表面（构造参数固定）；参数校验与错误路径由 pipeline/WConPU 单测覆盖",
        "edge": "无输入边界场景；边界/校验行为由 validate_pu_X_y 与 encoder 单测覆盖",
        "determ": (
            "固定 random_state 播种但不断言确定性（被测对象是隔离而非复现）；"
            "深度训练种子确定性由 test_pipeline_deep TestPipelineDeepSeedReproducibility 覆盖"
        ),
    },
    "test_ui_history_flow.py": {
        "param": (
            "AppTest wiring test is a success-path end-to-end; parameter errors are unit-level"
        ),
        "edge": ("AppTest wiring test is a success-path end-to-end; boundary cases are unit-level"),
        "determ": (
            "UI flow includes background-thread timing; history module determinism is unit-tested"
        ),
    },
    "test_traditional_pu_resume.py": {
        "param": (
            "no parameter validation surface here; runner config validation is "
            "covered by TestLoadConfig"
        ),
        "edge": (
            "interrupted-run semantics only; boundary inputs are covered by "
            "TestRunnerMini / TestPnuRowRecords"
        ),
        "determ": (
            "same-seed determinism is asserted in "
            "test_traditional_pu_benchmark_runner.py TestDeterminism"
        ),
    },
    "test_deep_vision_pickle.py": {
        "param": "factory parameter validation is covered by existing vision tests",
        "edge": "factory boundary validation is covered by existing vision tests",
        "determ": (
            "pickle roundtrip recovery is asserted by value comparison; module has no randomness"
        ),
    },
    "test_encoder_validation.py": {
        "determ": (
            "validate_encoder_features is a pure function (no randomness, no seed "
            "state); determinism of the consuming deep estimators is unit-tested "
            "with fixed seeds"
        ),
    },
    "test_history.py": {
        "param": "history.append accepts any mapping; there is no input validation path",
        "determ": "history module is deterministic by construction (no randomness)",
    },
    "test_report_provenance.py": {
        "basic": (
            "provenance field assembly assertions (plain dict key/value asserts, "
            "not a fit/predict smoke); end-to-end pipeline behavior is covered "
            "by test_pipeline(_deep) integration tests"
        ),
        "param": (
            "build_pipeline_report is called directly with fixed arguments; "
            "field-value validation and error paths are covered by "
            "pipeline/device tests"
        ),
        "edge": (
            "only two representative combinations (mlp bare, cnn full) are "
            "asserted; None/empty/invalid boundaries are covered by device and "
            "pipeline tests"
        ),
        "determ": (
            "build_pipeline_report is a pure assembly function (no randomness, "
            "no seed state); device resolution is only asserted to land in "
            "{cpu, cuda}, never pinned to a fixed value"
        ),
    },
}


# ═════════════════════════════════════════════════════════════════════
# Data types
# ═════════════════════════════════════════════════════════════════════


class TestMethod(NamedTuple):
    name: str
    lineno: int
    has_marker: bool  # decorated directly, or inherits from class/module


class ModuleReport(NamedTuple):
    path: Path
    n_tests: int
    has_marker_violations: list[TestMethod]
    categories_found: set[str]
    categories_missing: set[str]


# ═════════════════════════════════════════════════════════════════════
# AST visitor
# ═════════════════════════════════════════════════════════════════════


def _is_registered_marker(node: ast.expr) -> bool:
    """Return True if *node* is ``pytest.mark.<registered>``."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        # pytest.mark.xxx
        outer = node.value
        return (
            isinstance(outer.value, ast.Name)
            and outer.value.id == "pytest"
            and outer.attr == "mark"
            and node.attr in REGISTERED_MARKERS
        )
    return False


def _has_marker(decorator_list: list[ast.expr]) -> bool:
    """Return True if any decorator is ``@pytest.mark.<registered>``."""
    return any(_is_registered_marker(dec) for dec in decorator_list)


def _module_pytestmark(tree: ast.Module) -> bool:
    """Return True if the module assigns a registered marker to ``pytestmark``.

    Module-level ``pytestmark = pytest.mark.unit`` (or a list/tuple of
    them, e.g. ``pytestmark = [pytest.mark.unit, pytest.mark.paper]``)
    applies the marker to every test in the module, so those tests count
    as marked even though they carry no per-function decorator.
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        if _is_registered_marker(node.value):
            return True
        if isinstance(node.value, ast.List | ast.Tuple) and any(
            _is_registered_marker(el) for el in node.value.elts
        ):
            return True
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
    module_marked = _module_pytestmark(tree)

    # Module-level test functions.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            methods.append(
                TestMethod(
                    node.name,
                    node.lineno,
                    _has_marker(node.decorator_list) or module_marked,
                )
            )

    # Methods inside classes: inherit the *owning* class marker.
    # (ast.walk order cannot be relied on, so the marker must be
    # resolved per class rather than through a shared variable.)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_marker = _has_marker(node.decorator_list)
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name.startswith("test_"):
                    methods.append(
                        TestMethod(
                            child.name,
                            child.lineno,
                            _has_marker(child.decorator_list) or class_marker or module_marked,
                        )
                    )

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


def _effective_missing(report: ModuleReport) -> set[str]:
    """Missing categories after deducting declared PARTIAL_COVERAGE entries.

    A file may declare categories it legitimately cannot cover (see
    ``PARTIAL_COVERAGE``); those are subtracted here so the gate only
    judges what the tests can realistically provide.
    """
    declared = set(PARTIAL_COVERAGE.get(report.path.name, {}))
    return report.categories_missing - declared


def review_exemptions(reports: list[ModuleReport], max_tests: int = 15) -> list[str]:
    """Print exemptions and return stale entries that must be removed.

    Governance aid for the three hand-maintained lists: they are
    reprinted (with reasons) every run. An ``UNLIMITED_FILES`` entry is stale
    once its count is within the normal limit. A ``CONTRACT_COVERED_FILES``
    entry is stale once the file independently satisfies the coverage rule.
    A ``PARTIAL_COVERAGE`` category is stale once the category appears in the
    file. Every stale entry contributes to the gate's non-zero exit code.
    """
    stale: list[str] = []
    print("\n─ Exemption review ─")
    print("  UNLIMITED_FILES (exempt from the ≤15 test limit):")
    for name, reason in sorted(UNLIMITED_FILES.items()):
        print(f"    {name} — {reason}")
    print("  CONTRACT_COVERED_FILES (covered by contract tests):")
    for name, reason in sorted(CONTRACT_COVERED_FILES.items()):
        print(f"    {name} — {reason}")
    print("  PARTIAL_COVERAGE (declared missing categories with reasons):")
    for name, categories in sorted(PARTIAL_COVERAGE.items()):
        for cat, reason in sorted(categories.items()):
            print(f"    {name} [{cat}] — {reason}")
    for r in reports:
        declared = set(PARTIAL_COVERAGE.get(r.path.name, {}))
        now_covered = sorted(declared & r.categories_found)
        if now_covered:
            rel = _relative(r.path)
            message = f"{rel} must drop declared {now_covered} from PARTIAL_COVERAGE (now covered)"
            stale.append(message)
            print(f"  ERROR: {message}")
    for r in reports:
        rel = _relative(r.path)
        if r.path.name in UNLIMITED_FILES and r.n_tests <= max_tests:
            message = (
                f"{rel} must be removed from UNLIMITED_FILES "
                f"({r.n_tests} tests <= limit {max_tests})"
            )
            stale.append(message)
            print(f"  ERROR: {message}")
        if r.path.name in CONTRACT_COVERED_FILES and not _effective_missing(r):
            message = (
                f"{rel} must be removed from CONTRACT_COVERED_FILES "
                "(independent coverage is complete)"
            )
            stale.append(message)
            print(f"  ERROR: {message}")
    return stale


def main(max_tests: int = 15, strict: bool = True) -> int:
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
        if r.path.name in CONTRACT_COVERED_FILES:
            continue  # covered by contract tests
        missing = _effective_missing(r)
        if missing and (strict or len(missing) > 1):
            coverage_ok = False
            n_issues += 1
            rel = _relative(r.path)
            print(f"  {rel}: missing {sorted(missing)}")
    if coverage_ok:
        if strict:
            print("  ✓ all files cover all four required categories")
        else:
            print("  ✓ all files cover required categories (missing ≤1 allowed in lenient mode)")

    # ── 4. Exemption review (stale entries fail the gate) ───────────
    n_issues += len(review_exemptions(reports, max_tests))

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
        "--lenient",
        action="store_true",
        help=(
            "Allow a file to miss at most one coverage category (default: ALL four are required)"
        ),
    )
    args = parser.parse_args()
    sys.exit(main(max_tests=args.max, strict=not args.lenient))
