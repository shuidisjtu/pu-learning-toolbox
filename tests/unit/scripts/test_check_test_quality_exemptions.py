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
    assert "test_pen_l1.py — algorithm covered by contract tests" in out
