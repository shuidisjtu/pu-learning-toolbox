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

DOC = """\
# Project Directory Structure

## 5. 文档（`docs/`）

```text
docs/
  README.md                    # 首页

  adr/                         # 决策记录(索引)
```
"""

DOC_LISTED_ADR = """\
```text
docs/
  README.md                    # 首页
  adr/                         # 决策记录(索引)
    README.md                  # 决策索引
```
"""


def _block(text: str) -> str:
    """Return the docs block body (fence contents) of *text*."""
    lines = text.splitlines()
    for start, end, root in g.find_blocks(lines):
        if root == "docs":
            return "\n".join(lines[start + 1 : end])
    raise AssertionError("no docs block")


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
