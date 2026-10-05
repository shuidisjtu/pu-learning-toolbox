# ruff: noqa: N802, N803, N806, E501

"""Tests for rule-1 path references found inside Python source text."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_doc_links as d  # noqa: E402


def _write_under_root(tmp_path: Path, files: dict[str, str]) -> None:
    """Write *files* ({relative_path: content}) directly under tmp_path."""
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _use_source_tree(tmp_path, monkeypatch, files: dict[str, str]) -> None:
    """Write *files* under tmp_path and point the gate at that root."""
    _write_under_root(tmp_path, files)
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(d, "SCRIPTS_DIR", tmp_path / "scripts")


@pytest.mark.unit
def test_basic_source_comment_path_passes(tmp_path, monkeypatch):
    """A backtick .md path inside a source comment resolves from the root."""
    _use_source_tree(
        tmp_path,
        monkeypatch,
        {"docs/index.md": "# index\n", "pu_toolbox/mod.py": '"""See `docs/index.md`."""\n'},
    )

    assert d.check_path_references(d._find_source_files()) == []


@pytest.mark.unit
@pytest.mark.parametrize("root", ["pu_toolbox", "scripts"])
def test_param_source_comment_missing_path_reported(tmp_path, monkeypatch, root):
    """A dangling backtick path in either corpus root is a rule-1 error."""
    _use_source_tree(tmp_path, monkeypatch, {f"{root}/mod.py": "# See `docs/missing.md`.\n"})

    issues = d.check_path_references(d._find_source_files())

    assert len(issues) == 1
    assert issues[0].rule == "rule-1"
    assert issues[0].severity == "error"
    assert issues[0].file == f"{root}/mod.py"


@pytest.mark.unit
def test_edge_source_scan_excludes_tests_and_pycache(tmp_path, monkeypatch):
    """tests/ fixtures and __pycache__ artifacts stay out of the corpus."""
    _use_source_tree(
        tmp_path,
        monkeypatch,
        {
            "pu_toolbox/__pycache__/stale.py": "# `docs/missing.md`\n",
            "scripts/__pycache__/stale.py": "# `docs/missing.md`\n",
            "tests/fixture.py": "# `docs/also-missing.md`\n",
        },
    )

    assert d._find_source_files() == []


@pytest.mark.unit
def test_determ_source_scan_order_is_stable(tmp_path, monkeypatch):
    """Repeated discovery returns the same sorted corpus."""
    _use_source_tree(
        tmp_path,
        monkeypatch,
        {
            "pu_toolbox/b.py": "value = 1\n",
            "pu_toolbox/a.py": "value = 1\n",
            "scripts/c.py": "value = 1\n",
        },
    )

    assert d._find_source_files() == d._find_source_files()
    assert [p.name for p in d._find_source_files()] == ["a.py", "b.py", "c.py"]


@pytest.mark.unit
def test_edge_string_literal_path_is_reported(tmp_path, monkeypatch):
    """A dangling backtick path inside a plain string literal is reported too.

    The gate reads whole-file text rather than parsed comments, so this
    exposure is deliberate -- pin it so a future refactor to comment-only
    scanning cannot silently narrow the gate.
    """
    _use_source_tree(
        tmp_path,
        monkeypatch,
        {"pu_toolbox/mod.py": 'HELP = "see `docs/missing.md`"\n'},
    )

    issues = d.check_path_references(d._find_source_files())

    assert len(issues) == 1
    assert issues[0].rule == "rule-1"
