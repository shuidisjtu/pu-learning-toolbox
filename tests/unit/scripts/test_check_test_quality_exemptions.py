"""Tests for the test-quality gate's exemption-review (exit mechanism).

The gate maintains three hand-grown exemption lists. Every run, the
exemption-review section reprints them with reasons and fails any entry
whose exemption is no longer necessary, so the lists cannot grow forever.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_test_quality as gate  # noqa: E402
from check_test_quality import (  # noqa: E402
    _effective_missing,
    analyse_file,
    review_exemptions,
)


def _exempt_report(tmp_path, name: str, test_names: list[str]):
    """Build a ModuleReport for an exemption-listed file in tmp_path."""
    source = "".join(f"def {t}():\n    pass\n\n" for t in test_names)
    fp = tmp_path / name
    fp.write_text(source, encoding="utf-8")
    return analyse_file(fp)


@pytest.mark.unit
def test_exemption_reporting(tmp_path, capsys):
    """A count exemption fails once the file is within the normal limit."""
    report = _exempt_report(
        tmp_path,
        "test_generate_structure.py",  # member of UNLIMITED_FILES
        ["test_basic_fit", "test_param_invalid", "test_edge_empty", "test_determ_seed"],
    )
    stale = review_exemptions([report])
    out = capsys.readouterr().out
    assert "must be removed from UNLIMITED_FILES (4 tests <= limit 15)" in out
    assert len(stale) == 1


@pytest.mark.unit
def test_basic_contract_list_member_flagged(tmp_path, capsys):
    """A CONTRACT_COVERED_FILES member satisfying coverage is flagged too."""
    report = _exempt_report(
        tmp_path,
        "test_bias_aware.py",  # member of CONTRACT_COVERED_FILES
        ["test_basic_fit", "test_param_invalid", "test_edge_empty", "test_determ_seed"],
    )
    stale = review_exemptions([report])
    out = capsys.readouterr().out
    assert "must be removed from CONTRACT_COVERED_FILES" in out
    assert len(stale) == 1


@pytest.mark.unit
def test_edge_missing_one_category_keeps_contract_exemption(tmp_path, capsys):
    """A contract exemption remains valid while one category is missing."""
    report = _exempt_report(
        tmp_path,
        "test_bias_aware.py",
        ["test_basic_fit", "test_param_invalid", "test_edge_empty"],
    )
    stale = review_exemptions([report])
    out = capsys.readouterr().out
    assert "must be removed from CONTRACT_COVERED_FILES" not in out
    assert stale == []


@pytest.mark.unit
def test_edge_high_test_count_keeps_unlimited_hint_quiet(tmp_path, capsys):
    """A UNLIMITED_FILES member is only flagged when within the count limit.

    Full coverage alone must not suggest de-listing a file that still
    needs the exemption for the ≤15-test rule (e.g. a 20-test file would
    immediately break the gate if removed from UNLIMITED_FILES).
    """
    names = ["test_basic_fit", "test_param_invalid", "test_edge_empty", "test_determ_seed"]
    report = _exempt_report(tmp_path, "test_builtin_methods.py", names * 5)  # 20 tests
    stale = review_exemptions([report])
    out = capsys.readouterr().out
    assert "must be removed from UNLIMITED_FILES" not in out
    assert stale == []


@pytest.mark.unit
def test_param_count_exemption_exit_is_independent_of_coverage(tmp_path, capsys):
    """UNLIMITED_FILES governs count only, so low coverage does not retain it."""
    report = _exempt_report(
        tmp_path,
        "test_generate_structure.py",
        ["test_basic_fit", "test_param_invalid"],
    )
    stale = review_exemptions([report])
    out = capsys.readouterr().out
    assert "must be removed from UNLIMITED_FILES" in out
    assert len(stale) == 1


@pytest.mark.unit
def test_param_stale_exemption_makes_main_fail(tmp_path, monkeypatch):
    """A stale exemption is part of the gate verdict, not just its report."""
    source = (
        "import pytest\n"
        "pytestmark = pytest.mark.unit\n"
        "def test_basic_fit(): pass\n"
        "def test_param_invalid(): pass\n"
        "def test_edge_empty(): pass\n"
        "def test_determ_seed(): pass\n"
    )
    (tmp_path / "test_ready.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(gate, "TESTS_DIR", tmp_path)
    monkeypatch.setattr(gate, "UNLIMITED_FILES", {"test_ready.py": "temporary exemption"})
    monkeypatch.setattr(gate, "CONTRACT_COVERED_FILES", {})
    monkeypatch.setattr(gate, "PARTIAL_COVERAGE", {})
    assert gate.main() == 1


@pytest.mark.unit
def test_deterministic_list_printout_with_reasons(tmp_path, capsys):
    """Every run prints the exemption lists and the reason for each entry."""
    report = _exempt_report(tmp_path, "test_builtin_methods.py", ["test_basic_fit"])
    review_exemptions([report])
    out = capsys.readouterr().out
    assert "─ Exemption review ─" in out
    assert "UNLIMITED_FILES" in out
    assert "CONTRACT_COVERED_FILES" in out
    assert "test_kldce_math.py — MATH formula verification" in out
    assert "test_import.py — smoke imports" in out


@pytest.mark.unit
def test_basic_partial_coverage_deducts_declared_categories(tmp_path):
    """Declared PARTIAL_COVERAGE categories are deducted from the missing set."""
    # test_history.py declares param + determ, so a basic+edge-only report
    # must have an empty effective missing set.
    report = _exempt_report(
        tmp_path,
        "test_history.py",
        ["test_basic_fit", "test_edge_empty"],
    )
    assert report.categories_missing == {"param", "determ"}
    assert _effective_missing(report) == set()


@pytest.mark.unit
def test_param_partial_coverage_printout_with_reasons(capsys):
    """Every run prints PARTIAL_COVERAGE with file, category, and reason."""
    review_exemptions([])
    out = capsys.readouterr().out
    assert "PARTIAL_COVERAGE (declared missing categories with reasons)" in out
    assert "test_history.py [param]" in out
    assert "no input validation path" in out


@pytest.mark.unit
def test_edge_declared_category_now_covered_hints_removal(tmp_path, capsys):
    """A declared category that is now covered is flagged as removable."""
    report = _exempt_report(
        tmp_path,
        "test_history.py",
        ["test_basic_fit", "test_edge_empty", "test_param_invalid"],
    )
    stale = review_exemptions([report])
    out = capsys.readouterr().out
    assert "must drop declared ['param'] from PARTIAL_COVERAGE (now covered)" in out
    assert len(stale) == 1


@pytest.mark.unit
def test_basic_module_level_pytestmark_detected(tmp_path):
    """Module-level ``pytestmark = pytest.mark.unit`` marks every test."""
    source = "import pytest\npytestmark = pytest.mark.unit\n\ndef test_smoke():\n    pass\n"
    fp = tmp_path / "test_module_marked.py"
    fp.write_text(source, encoding="utf-8")
    report = analyse_file(fp)
    assert report.has_marker_violations == []


@pytest.mark.unit
def test_param_pytestmark_list_form_detected(tmp_path):
    """List-form module-level pytestmark marks every test too."""
    source = (
        "import pytest\n"
        "pytestmark = [pytest.mark.unit, pytest.mark.paper]\n"
        "\n"
        "def test_smoke():\n"
        "    pass\n"
    )
    fp = tmp_path / "test_module_list_marked.py"
    fp.write_text(source, encoding="utf-8")
    report = analyse_file(fp)
    assert report.has_marker_violations == []


@pytest.mark.unit
def test_edge_unregistered_module_marker_still_flags(tmp_path):
    """A module-level marker outside REGISTERED_MARKERS still flags."""
    source = (
        "import pytest\npytestmark = pytest.mark.not_registered\n\ndef test_smoke():\n    pass\n"
    )
    fp = tmp_path / "test_module_unregistered.py"
    fp.write_text(source, encoding="utf-8")
    report = analyse_file(fp)
    assert [m.name for m in report.has_marker_violations] == ["test_smoke"]


@pytest.mark.unit
def test_edge_class_methods_inherit_module_marker(tmp_path):
    """Class methods inherit the module-level pytestmark as well."""
    source = (
        "import pytest\n"
        "pytestmark = pytest.mark.unit\n"
        "\n"
        "class TestFoo:\n"
        "    def test_smoke(self):\n"
        "        pass\n"
    )
    fp = tmp_path / "test_module_class_marked.py"
    fp.write_text(source, encoding="utf-8")
    report = analyse_file(fp)
    assert report.has_marker_violations == []
