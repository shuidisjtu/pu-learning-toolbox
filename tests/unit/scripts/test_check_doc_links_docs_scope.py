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

# Both declared grouped subtrees are registered in every fixture: the
# declaration is global, so a block that omits one is a (correct) problem.
SURVEY_LINES = """\
  research/
    pu_survey/                 # 调研协议与交付(索引)
"""

DOC = (
    """\
```text
docs/
  README.md                    # 首页
  adr/                         # 决策记录(索引)
"""
    + SURVEY_LINES
    + "```\n"
)

DOC_WITH_ADR_INDEX = (
    """\
```text
docs/
  README.md                    # 首页
  adr/                         # 决策记录(索引)
    README.md                  # 决策索引
"""
    + SURVEY_LINES
    + "```\n"
)

DOC_WITH_GHOST = (
    """\
```text
docs/
  README.md                    # 首页
  user/
    ghost.md                   # 幽灵条目
  adr/                         # 决策记录(索引)
"""
    + SURVEY_LINES
    + "```\n"
)

# A listed entry under docs/ whose suffix the docs block does not cover: the
# generator never parses it, so only the legacy existence check can see it.
DOC_WITH_OUT_OF_SCOPE = (
    """\
```text
docs/
  README.md                    # 首页
  legacy_script.py             # 早年的脚本
  adr/                         # 决策记录(索引)
"""
    + SURVEY_LINES
    + "```\n"
)

# The docs block with the `adr/` anchor line deleted.
DOC_WITHOUT_ADR_ANCHOR = (
    """\
```text
docs/
  README.md                    # 首页
"""
    + SURVEY_LINES
    + "```\n"
)


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


def test_param_out_of_scope_docs_entry_keeps_its_existence_check(tmp_path, monkeypatch):
    """Regression: a listed `docs/*.py` must still be existence-checked.

    The generator owns in-scope entries (root x suffix) and the legacy
    per-line check covers the rest.  Guarding that split by root prefix
    alone let every out-of-scope suffix under docs/ fall between the two.
    """
    issues = _run_rule2(tmp_path, monkeypatch, DOC_WITH_OUT_OF_SCOPE, tracked=["docs/README.md"])
    messages = [i.message for i in issues]
    assert any("does not exist on disk" in m and "docs/legacy_script.py" in m for m in messages)


def test_param_unregistered_grouped_subtree_is_reported(tmp_path, monkeypatch):
    """The generator's anchor problem surfaces as a rule-2 error here too."""
    issues = _run_rule2(
        tmp_path,
        monkeypatch,
        DOC_WITHOUT_ADR_ANCHOR,
        tracked=["docs/README.md", "docs/adr/README.md"],
    )
    messages = [i.message for i in issues]
    assert any("docs/adr" in m and "not registered" in m for m in messages)


def test_edge_excluded_doc_subtree_is_exempt_in_both_gates(tmp_path, monkeypatch):
    """A tracked ``docs/figures`` file is neither required nor complained about.

    The exclusion list has a single source (``generate_structure``) and this
    gate derives its own set from it, so the two cannot disagree about what
    is out of scope.  Rule-2 (delegated to the generator) must not report the
    figure as missing, and rule-4 must not demand it in the docs index.
    """
    derived = {subtree.removeprefix("docs/") for subtree in g.EXCLUDED_DOC_SUBTREES}
    assert derived == d._EXCLUDED_DOC_DIRS

    tracked = ["docs/README.md", "docs/figures/x.png"]
    new_text, missing, stale = g.generate(DOC, tracked)
    assert missing == []
    assert stale == []
    assert "x.png" not in new_text  # not emitted, not even as a placeholder
    assert "figures" not in new_text
    assert g.PLACEHOLDER not in new_text

    issues = _run_rule2(tmp_path, monkeypatch, DOC, tracked)
    assert [i for i in issues if "figures" in i.message] == []

    docs_dir = tmp_path / "docs"
    (docs_dir / "figures").mkdir(parents=True, exist_ok=True)
    (docs_dir / "figures" / "note.md").write_text("# note\n", encoding="utf-8")
    (docs_dir / "README.md").write_text("# index\n", encoding="utf-8")
    monkeypatch.setattr(d, "SCRIPTS_DIR", tmp_path / "scripts")  # 4b skipped
    index_issues = d.check_index_completeness(docs_dir / "README.md", tmp_path / "README.md", None)
    assert [i for i in index_issues if "figures" in i.message] == []


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
