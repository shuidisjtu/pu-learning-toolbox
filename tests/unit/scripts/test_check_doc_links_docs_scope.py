# ruff: noqa: N802, N803, N806, E501

"""Tests for rule-2's widened scope: ``docs/`` now has a reverse gate.

Rule-2 delegates the bidirectional tree check to ``generate_structure``, so
adding ``docs`` to the generator's roots widens this rule for free.  These
tests pin that inheritance -- an in-repo doc the block does not list is an
error, and a grouped subtree's unlisted files are not.

Separate file because ``test_check_doc_links.py`` sits exactly at the
per-file test budget (``scripts/check_test_quality.py``, max 15); rule-5 /
rule-2 core tests stay there, the docs-scope ones live here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_doc_links as d  # noqa: E402
import generate_structure as g  # noqa: E402

pytestmark = pytest.mark.unit

DOC = """\
```text
docs/
  README.md                    # 首页
  adr/                         # 决策记录(索引)
```
"""

DOC_WITH_ADR_INDEX = """\
```text
docs/
  README.md                    # 首页
  adr/                         # 决策记录(索引)
    README.md                  # 决策索引
```
"""

DOC_WITH_GHOST = """\
```text
docs/
  README.md                    # 首页
  user/
    ghost.md                   # 幽灵条目
```
"""


def _run_rule2(tmp_path, monkeypatch, doc_text: str, tracked: list[str]) -> list:
    """Run check_planned_consistency against a scratch document + tracked list."""
    md = tmp_path / "project_structure.md"
    md.write_text(doc_text, encoding="utf-8")
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(g, "tracked_files", lambda: tracked)
    return d.check_planned_consistency(md)


def test_basic_generatable_prefixes_include_docs():
    """Rule-2's scope derives from the generator, so it gained docs/ too."""
    derived = tuple(r + "/" for r in g.GENERATABLE_ROOTS)
    assert derived == d._GENERATABLE_PREFIXES
    assert "docs/" in d._GENERATABLE_PREFIXES


def test_basic_unlisted_in_repo_doc_reports_error(tmp_path, monkeypatch):
    """A docs file on disk that the block never lists is now a rule-2 error."""
    issues = _run_rule2(
        tmp_path,
        monkeypatch,
        DOC,
        tracked=["docs/README.md", "docs/user/quickstart.md"],
    )
    messages = [i.message for i in issues]
    assert any(
        "missing from project_structure.md" in m and "docs/user/quickstart.md" in m
        for m in messages
    )


def test_param_listed_docs_entry_gone_from_disk_reports_error(tmp_path, monkeypatch):
    """The forward direction covers docs too: a listed file that is gone.

    The legacy per-line check skips generator-managed prefixes, so this
    issue can only come from the generator's stale list.
    """
    issues = _run_rule2(tmp_path, monkeypatch, DOC_WITH_GHOST, tracked=["docs/README.md"])
    messages = [i.message for i in issues]
    assert any("does not exist on disk" in m and "docs/user/ghost.md" in m for m in messages)


def test_edge_grouped_subtree_files_are_exempt_from_rule2(tmp_path, monkeypatch):
    """adr/ is registered as a group: its unlisted files raise no rule-2 issue."""
    issues = _run_rule2(
        tmp_path,
        monkeypatch,
        DOC,
        tracked=["docs/README.md", "docs/adr/0001-governance.md", "docs/adr/README.md"],
    )
    assert [i for i in issues if "docs/" in i.message] == []


def test_determ_repeated_docs_rule2_checks_agree(tmp_path, monkeypatch):
    """Rule-2 over a docs block is a pure function of the document and the tree."""

    def run() -> list:
        return _run_rule2(
            tmp_path,
            monkeypatch,
            DOC_WITH_ADR_INDEX,
            tracked=["docs/README.md"],
        )

    first = run()
    assert [i.message for i in first] == [i.message for i in run()]
