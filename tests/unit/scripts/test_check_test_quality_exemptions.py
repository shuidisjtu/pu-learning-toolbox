"""Tests for the test-quality gate's exemption-review (exit mechanism).

The gate maintains two hand-grown exemption lists (``UNLIMITED_FILES``,
``CONTRACT_COVERED_FILES``).  Every run, the exemption-review section
reprints the lists with their reasons and flags any listed file whose
current category coverage no longer needs the exemption, so the lists
can be audited and shrunk instead of growing forever.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from check_test_quality import analyse_file, review_exemptions  # noqa: E402


def _exempt_report(tmp_path, name: str, test_names: list[str]):
    """Build a ModuleReport for an exemption-listed file in tmp_path."""
    source = "".join(f"def {t}():\n    pass\n\n" for t in test_names)
    fp = tmp_path / name
    fp.write_text(source, encoding="utf-8")
    return analyse_file(fp)


@pytest.mark.unit
def test_exemption_reporting(tmp_path, capsys):
    """豁免名单文件若已满足覆盖规则,输出中应出现可移出名单的提示."""
    report = _exempt_report(
        tmp_path,
        "test_builtin_methods.py",  # member of UNLIMITED_FILES
        ["test_basic_fit", "test_param_invalid", "test_edge_empty", "test_determ_seed"],
    )
    review_exemptions([report])
    out = capsys.readouterr().out
    assert "may be removable from UNLIMITED_FILES (covers 4/4 categories)" in out


@pytest.mark.unit
def test_basic_contract_list_member_flagged(tmp_path, capsys):
    """A CONTRACT_COVERED_FILES member satisfying coverage is flagged too."""
    report = _exempt_report(
        tmp_path,
        "test_bias_aware.py",  # member of CONTRACT_COVERED_FILES
        ["test_basic_fit", "test_param_invalid", "test_edge_empty", "test_determ_seed"],
    )
    review_exemptions([report])
    out = capsys.readouterr().out
    assert "may be removable from CONTRACT_COVERED_FILES (covers 4/4 categories)" in out


@pytest.mark.unit
def test_edge_missing_one_category_still_flagged(tmp_path, capsys):
    """Missing only one of four categories (covers 3/4) still hints removal."""
    report = _exempt_report(
        tmp_path,
        "test_builtin_methods.py",
        ["test_basic_fit", "test_param_invalid", "test_edge_empty"],
    )
    review_exemptions([report])
    out = capsys.readouterr().out
    assert "may be removable from UNLIMITED_FILES (covers 3/4 categories)" in out


@pytest.mark.unit
def test_edge_high_test_count_keeps_unlimited_hint_quiet(tmp_path, capsys):
    """A UNLIMITED_FILES member is only flagged when within the count limit.

    Full coverage alone must not suggest de-listing a file that still
    needs the exemption for the ≤15-test rule (e.g. a 20-test file would
    immediately break the gate if removed from UNLIMITED_FILES).
    """
    names = ["test_basic_fit", "test_param_invalid", "test_edge_empty", "test_determ_seed"]
    report = _exempt_report(tmp_path, "test_builtin_methods.py", names * 5)  # 20 tests
    review_exemptions([report])
    out = capsys.readouterr().out
    assert "may be removable from UNLIMITED_FILES" not in out


@pytest.mark.unit
def test_param_low_coverage_raises_no_removal_hint(tmp_path, capsys):
    """Covering fewer than three categories keeps the exemption quiet."""
    report = _exempt_report(
        tmp_path,
        "test_builtin_methods.py",
        ["test_basic_fit", "test_param_invalid"],
    )
    review_exemptions([report])
    out = capsys.readouterr().out
    assert "may be removable" not in out


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
