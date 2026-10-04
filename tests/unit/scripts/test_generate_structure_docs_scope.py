# ruff: noqa: N802, N803, N806, E501

"""Tests for the ``docs/`` scope of the structure generator.

Covers the two concepts the docs block needed to join the generated roots:
the per-root suffix set (``GENERATABLE_SUFFIXES``) and grouped subtrees
(``GROUPED_SUBTREES``) -- a subtree registered by index instead of file by
file, so its unlisted files are never "missing" while its listed entries
are still verified.

Separate file because ``test_generate_structure.py`` sits over the per-file
budget (``scripts/check_test_quality.py``, max 15) under an exemption that
covers the pre-existing generator units; new scope tests go here instead of
growing that file further.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import generate_structure as g  # noqa: E402

pytestmark = pytest.mark.unit

# Both declared grouped subtrees are registered in every fixture: the
# declaration is global, so a block that omits one is a (correct) problem.
DOC = """\
# Project Directory Structure

## 5. 文档（`docs/`）

```text
docs/
  README.md                    # 首页

  adr/                         # 决策记录(索引)
  research/
    pu_survey/                 # 调研协议与交付(索引)
```
"""

DOC_LISTED_ADR = """\
```text
docs/
  README.md                    # 首页
  adr/                         # 决策记录(索引)
    README.md                  # 决策索引
  research/
    pu_survey/                 # 调研协议与交付(索引)
```
"""


def _block(text: str) -> str:
    """Return the docs block body (fence contents) of *text*."""
    lines = text.splitlines()
    for start, end, root in g.find_blocks(lines):
        if root == "docs":
            return "\n".join(lines[start + 1 : end])
    raise AssertionError("no docs block")


def _drop_entry(text: str, entry: str) -> str:
    """Drop the one tree-block line whose first token is *entry*."""
    lines = text.splitlines()
    kept = [ln for ln in lines if not (ln.strip() and ln.strip().split()[0] == entry)]
    assert len(kept) == len(lines) - 1, (entry, len(lines), len(kept))
    return "\n".join(kept) + "\n"


DOC_LEGACY = """\
```text
docs/
  README.md                    # 首页
  legacy_script.py             # 早年的脚本
  adr/                         # 决策记录(索引)
  research/
    pu_survey/                 # 调研协议与交付(索引)
```
"""


def test_basic_docs_root_is_generatable_and_suffixes_derive_the_root_list():
    assert [r for _, _, r in g.find_blocks(DOC.splitlines())] == ["docs"]
    assert g.GENERATABLE_SUFFIXES["docs"] == (".md", ".png", ".json")
    # The root list is derived from the mapping, never restated.
    derived = tuple(g.GENERATABLE_SUFFIXES)
    assert derived == g.GENERATABLE_ROOTS


def test_basic_grouped_subtree_unlisted_files_are_not_missing():
    disk = ["docs/README.md", "docs/adr/README.md", "docs/adr/0001-governance.md"]
    new_text, missing, stale = g.generate(DOC, disk)
    assert missing == []
    assert stale == []
    assert "0001-governance.md" not in new_text
    assert g.PLACEHOLDER not in new_text


def test_basic_unlisted_docs_file_outside_grouped_subtree_is_missing():
    disk = ["docs/README.md", "docs/user/quickstart.md"]
    new_text, missing, stale = g.generate(DOC, disk)
    assert missing == ["docs/user/quickstart.md"]
    assert stale == []
    assert "quickstart.md" in new_text and g.PLACEHOLDER in new_text


def test_param_scope_rejects_a_suffix_outside_its_root():
    assert g.in_scope("docs/user/quickstart.md")
    assert not g.in_scope("docs/data/table.csv")
    assert not g.in_scope("tests/data/fixture.json")
    assert not g.in_scope("README.md")


def test_param_listed_entry_under_grouped_subtree_gone_from_disk_is_stale():
    _, missing, stale = g.generate(DOC_LISTED_ADR, ["docs/README.md"])
    assert missing == []
    assert stale == ["docs/adr/README.md"]


def test_edge_grouped_subtree_new_directory_is_not_added():
    disk = ["docs/README.md", "docs/adr/README.md", "docs/adr/sub/deep.md"]
    new_text, missing, stale = g.generate(DOC, disk)
    body = _block(new_text)
    assert "sub/" not in body
    assert "deep.md" not in body
    assert missing == []
    assert stale == []


def test_determ_generate_with_grouped_subtrees_is_idempotent():
    disk = ["docs/README.md", "docs/adr/README.md", "docs/adr/0001-x.md", "docs/user/a.md"]
    once, _, _ = g.generate(DOC, disk)
    twice, _, _ = g.generate(once, disk)
    assert once == twice


def test_param_missing_group_anchor_is_reported():
    """A declared grouped subtree that left the block is a problem, not silence.

    Without the anchor the "registered by index" declaration and "the whole
    subtree was deleted" look identical to the gate: the files are exempt
    either way.
    """
    assert g.block_problems(DOC) == []

    problems = g.block_problems(_drop_entry(DOC, "adr/"))

    assert len(problems) == 1
    assert "docs/adr" in problems[0]
    assert "not registered" in problems[0]


def test_edge_out_of_scope_entry_is_reported_and_not_dropped_in_silence(
    tmp_path, monkeypatch, capsys
):
    """A listed entry whose suffix the block does not cover must not vanish quietly.

    ``legacy_script.py`` under ``docs/`` is never parsed (suffix out of
    scope), so the generator cannot check it and ``--update`` would drop the
    line; both the report and the update have to say so.
    """
    problems = g.block_problems(DOC_LEGACY)
    assert len(problems) == 1
    assert "docs/legacy_script.py" in problems[0]
    assert "outside that block's scope" in problems[0]

    md = tmp_path / "project_structure.md"
    md.write_text(DOC_LEGACY, encoding="utf-8")
    monkeypatch.setattr(g, "STRUCTURE_MD", md)
    monkeypatch.setattr(g, "tracked_files", lambda: ["docs/README.md"])

    assert g.main(["--update"]) == 0
    assert "outside that block's scope" in capsys.readouterr().out  # loud, not silent
    assert "legacy_script.py" not in md.read_text(encoding="utf-8")  # and it really went


def test_param_main_check_fails_on_unregistered_grouped_subtree(tmp_path, monkeypatch, capsys):
    """--check goes red (naming the subtree) instead of passing on a deletion."""
    md = tmp_path / "project_structure.md"
    md.write_text(_drop_entry(DOC, "adr/"), encoding="utf-8")
    monkeypatch.setattr(g, "STRUCTURE_MD", md)
    monkeypatch.setattr(g, "tracked_files", lambda: ["docs/README.md", "docs/adr/README.md"])

    assert g.main(["--check"]) == 1
    assert "docs/adr" in capsys.readouterr().err
