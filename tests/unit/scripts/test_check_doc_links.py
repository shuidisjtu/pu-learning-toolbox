# ruff: noqa: N802, N803, N806, E501

"""Tests for the doc-links gate extension (orphan + md-link detection).

Covers the three coverage holes closed by the architecture-decay fix:
rule-1 ``.md`` path references, rule-5 markdown link targets, and
rule-4 index completeness over the whole docs tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_doc_links as d  # noqa: E402
import generate_structure as g  # noqa: E402


def _make_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """Write *files* ({relative_path: content}) under tmp_path/docs."""
    root = tmp_path / "docs"
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


def _use_tmp_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> Path:
    """Materialize *files* under tmp_path/docs and point the gate at it."""
    root = _make_tree(tmp_path, files)
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(d, "DOCS_DIR", root)
    monkeypatch.setattr(d, "SCRIPTS_DIR", tmp_path / "scripts")  # empty -> rule-4b skipped
    return root


# --------------------------------------------------------------------
# Rule 5: markdown link targets
# --------------------------------------------------------------------


@pytest.mark.unit
def test_basic_md_link_passes(tmp_path, monkeypatch):
    """A markdown link to an existing file yields no rule-5 issue."""
    root = _make_tree(
        tmp_path,
        {"index.md": "- [other](sub/other.md)\n", "sub/other.md": "# other\n"},
    )
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    assert d.check_md_links([root / "index.md"]) == []


@pytest.mark.unit
def test_param_dangling_md_link_reports_error(tmp_path, monkeypatch):
    """A link to a nonexistent .md file is a rule-5 error."""
    root = _make_tree(tmp_path, {"index.md": "- [missing](missing.md)\n"})
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    issues = d.check_md_links([root / "index.md"])
    assert len(issues) == 1
    assert issues[0].rule == "rule-5"
    assert issues[0].severity == "error"
    assert issues[0].line == 1
    assert "missing.md" in issues[0].message


@pytest.mark.unit
def test_param_external_and_anchor_links_skipped(tmp_path, monkeypatch):
    """External URLs and #anchors are not file references (no rule-5 issue)."""
    root = _make_tree(
        tmp_path,
        {
            "index.md": (
                "- [paper](https://arxiv.org/abs/1234)\n"
                "- [mail](mailto:x@example.com)\n"
                "- [sec](#section)\n"
                "- [file-sec](sub/other.md#sec)\n"
            ),
            "sub/other.md": "# other\n",
        },
    )
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    assert d.check_md_links([root / "index.md"]) == []


# --------------------------------------------------------------------
# Rule 1: backtick path references now cover .md too
# --------------------------------------------------------------------


@pytest.mark.unit
def test_basic_backtick_md_path_passes(tmp_path, monkeypatch):
    """A backtick-quoted .md path that exists yields no rule-1 issue."""
    root = _make_tree(tmp_path, {"index.md": "See `docs/index.md`.\n"})
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    assert d.check_path_references([root / "index.md"]) == []


@pytest.mark.unit
def test_param_backtick_md_path_reported(tmp_path, monkeypatch):
    """A backtick-quoted `docs/.../file.md` that does not exist is rule-1."""
    root = _make_tree(tmp_path, {"index.md": "See `docs/missing.md`.\n"})
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    issues = d.check_path_references([root / "index.md"])
    assert len(issues) == 1
    assert issues[0].rule == "rule-1"
    assert issues[0].severity == "error"
    assert "missing.md" in issues[0].message


@pytest.mark.unit
def test_edge_glob_token_skipped(tmp_path, monkeypatch):
    """Brace/star tokens (`docs/user/{a,b}.py`) are shorthand, not paths."""
    root = _make_tree(tmp_path, {"index.md": "See `docs/user/{a,b}.py`.\n"})
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)
    assert d.check_path_references([root / "index.md"]) == []


# --------------------------------------------------------------------
# Rule 4: index completeness over the whole docs tree
# --------------------------------------------------------------------


@pytest.mark.unit
def test_orphan_md_reported(tmp_path, monkeypatch):
    """A docs/md file listed in no index is reported by rule-4."""
    root = _use_tmp_tree(
        tmp_path,
        monkeypatch,
        {
            "README.md": "- [index](sub/index.md)\n",
            "sub/index.md": "- [orphan](orphan.md)\n",
            "sub/orphan.md": "# orphan\n",
        },
    )
    issues = d.check_index_completeness(root / "README.md", tmp_path / "README.md", None)
    orphans = [i for i in issues if i.rule == "rule-4" and "orphan.md" in i.message]
    assert len(orphans) == 1
    assert orphans[0].severity == "error"


@pytest.mark.unit
def test_param_directory_entry_covers_subtree(tmp_path, monkeypatch):
    """An index entry for a directory covers every doc below it."""
    root = _use_tmp_tree(
        tmp_path,
        monkeypatch,
        {
            "README.md": "- [cards](cards/)\n",
            "cards/a.md": "# a\n",
            "cards/sub/b.md": "# b\n",
        },
    )
    issues = d.check_index_completeness(root / "README.md", tmp_path / "README.md", None)
    assert issues == []


# --------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------


@pytest.mark.unit
def test_determ_results_reproducible(tmp_path, monkeypatch):
    """check_* must be deterministic: repeated calls give identical issues."""
    root = _use_tmp_tree(
        tmp_path,
        monkeypatch,
        {
            "README.md": "- [index](sub/index.md)\n",
            "sub/index.md": "- [missing](missing.md)\n",
            "sub/orphan.md": "# orphan\n",
        },
    )
    md_files = [root / "sub" / "index.md"]
    docs_readme = root / "README.md"
    root_readme = tmp_path / "README.md"

    def run() -> tuple:
        return (
            d.check_path_references(md_files),
            d.check_md_links(md_files),
            d.check_index_completeness(docs_readme, root_readme, None),
        )

    assert run() == run()


# --------------------------------------------------------------------
# Rule 2: bidirectional tree consistency (generator-backed)
# --------------------------------------------------------------------


STRUCTURE_SYNCED = """\
```text
pu_toolbox/
  core/
    base.py                  (core base)
```
"""


def _run_rule2(tmp_path, monkeypatch, doc_text: str, tracked: list[str]) -> list:
    """Run check_planned_consistency with a scratch structure_md + tracked files."""
    md = tmp_path / "project_structure.md"
    md.write_text(doc_text, encoding="utf-8")
    monkeypatch.setattr(d, "PROJECT_ROOT", tmp_path)  # legacy planned checks
    monkeypatch.setattr(g, "tracked_py_files", lambda: tracked)
    return d.check_planned_consistency(md)


@pytest.mark.unit
def test_rule2_missing_file_reports_error(tmp_path, monkeypatch):
    issues = _run_rule2(
        tmp_path,
        monkeypatch,
        STRUCTURE_SYNCED,
        tracked=["pu_toolbox/core/base.py", "pu_toolbox/core/new.py"],
    )
    messages = [i.message for i in issues]
    assert any("missing from project_structure.md" in m and "core/new.py" in m for m in messages)


@pytest.mark.unit
def test_rule2_stale_file_reports_error(tmp_path, monkeypatch):
    doc = """\
```text
pu_toolbox/
  core/
    base.py                  (core base)
    gone.py
```
"""
    issues = _run_rule2(tmp_path, monkeypatch, doc, tracked=["pu_toolbox/core/base.py"])
    messages = [i.message for i in issues]
    assert any("does not exist on disk" in m and "core/gone.py" in m for m in messages)


@pytest.mark.unit
def test_rule2_planned_missing_file_ok(tmp_path, monkeypatch):
    doc = """\
```text
pu_toolbox/
  core/
    base.py                  (core base)
    future.py                (planned)
```
"""
    issues = _run_rule2(tmp_path, monkeypatch, doc, tracked=["pu_toolbox/core/base.py"])
    messages = [i.message for i in issues]
    assert not any("future.py" in m for m in messages)


@pytest.mark.unit
def test_rule2_synced_document_passes(tmp_path, monkeypatch):
    issues = _run_rule2(
        tmp_path,
        monkeypatch,
        STRUCTURE_SYNCED,
        tracked=["pu_toolbox/core/base.py"],
    )
    assert issues == []
