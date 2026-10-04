# ruff: noqa: S101

"""Guard the §4.8 stable-module classification against drift.

``docs/dev/architecture_principles.md`` §4 item 8 classifies every module
as either **stable** (frozen against structural / contract migration and
refactoring) or **active**.  The classification itself is a human
judgement, and only its *completeness* is mechanically checked here: the
other half — "is this module *still* stable", i.e. was it touched after
the baseline release tag — has no mechanical guard, because no shallow
clone can answer it (see the item's own 如实登记 paragraph).

Scope note (口径): the tracked set is the top-level **directories** under
``pu_toolbox/``.  The three tracked root-level modules (``__init__.py``,
``progress.py``, ``run_config.py``) are deliberately outside the table:
the criterion command is per-directory
(``git log -- pu_toolbox/<模块>``), so a flat root file has no row.

The tests must not shell out to git history: CI checks out with
``fetch-depth: 1`` and fetches no tags, so ``git tag`` and ``git log``
against a tag are unusable there.  Only ``git ls-files`` (the index) is
read, which a shallow clone does provide.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs" / "dev" / "architecture_principles.md"

SECTION_HEADING = "## 4. 维护实践清单"
ITEM_HEADING = "8. **稳定模块标识**"
STABLE_LABEL = "稳定（冻结迁移与重构）"
ACTIVE_LABEL = "活跃"
CATEGORY_LABELS = (STABLE_LABEL, ACTIVE_LABEL)

_MODULE_RE = re.compile(r"`([^`]+)`")
_ITEM_RE = re.compile(r"^\d+\. ")


class StabilityTableError(RuntimeError):
    """Raised when §4.8's machine-readable block cannot be read."""


def tracked_top_level_modules(listing: str) -> set[str]:
    """Return the top-level ``pu_toolbox/`` directories in *listing*.

    *listing* is the text output of ``git ls-files pu_toolbox/``.  Only
    paths of the form ``pu_toolbox/<dir>/...`` contribute; the tracked
    root-level modules (``pu_toolbox/<name>.py``) are dropped by design.
    """
    modules: set[str] = set()
    for line in listing.splitlines():
        parts = line.strip().split("/")
        if len(parts) >= 3 and parts[0] == "pu_toolbox":
            modules.add(parts[1])
    return modules


def _item_8_block(text: str) -> list[str]:
    """Return the lines of §4 item 8, or raise ``StabilityTableError``."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(SECTION_HEADING)), None)
    if start is None:
        raise StabilityTableError(f"section heading not found: {SECTION_HEADING!r}")
    end = next(
        (j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")),
        len(lines),
    )
    section = lines[start:end]
    item_start = next((i for i, line in enumerate(section) if line.startswith(ITEM_HEADING)), None)
    if item_start is None:
        raise StabilityTableError(f"item heading not found in section: {ITEM_HEADING!r}")
    item_end = next(
        (j for j in range(item_start + 1, len(section)) if _ITEM_RE.match(section[j])),
        len(section),
    )
    return section[item_start:item_end]


def parse_stability_table(text: str) -> dict[str, set[str]]:
    """Return ``{category label: {module, ...}}`` from §4.8's table.

    Raises ``StabilityTableError`` — never a silent empty result — when
    the item, a category row, or a row's module cell cannot be read.
    """
    categories: dict[str, set[str]] = {}
    for line in _item_8_block(text):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2 or cells[0] not in CATEGORY_LABELS:
            continue  # header / separator / unrelated row
        modules = set(_MODULE_RE.findall(cells[1]))
        if not modules:
            raise StabilityTableError(f"category row {cells[0]!r} lists no module")
        if cells[0] in categories:
            raise StabilityTableError(f"category row {cells[0]!r} appears twice")
        categories[cells[0]] = modules
    missing = [label for label in CATEGORY_LABELS if label not in categories]
    if missing:
        raise StabilityTableError(f"missing category row(s): {missing}")
    return categories


def find_stability_problems(text: str, tracked: Iterable[str]) -> list[str]:
    """Return the coverage problems of §4.8 against *tracked*, in stable order.

    *tracked* is any iterable of module directory names; it is normalised
    to a set, so neither its order nor its type affects the result.  An
    unreadable table propagates ``StabilityTableError``: an unparsable
    document must never be reported as passing.
    """
    tracked = set(tracked)
    categories = parse_stability_table(text)
    stable = categories[STABLE_LABEL]
    active = categories[ACTIVE_LABEL]
    problems: list[str] = []
    unclassified = sorted(tracked - (stable | active))
    if unclassified:
        problems.append(f"tracked but unclassified: {unclassified}")
    stale = sorted((stable | active) - tracked)
    if stale:
        problems.append(f"classified but not tracked (stale): {stale}")
    overlap = sorted(stable & active)
    if overlap:
        problems.append(f"listed as both stable and active: {overlap}")
    return problems


def _read_doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def _tracked_listing() -> str:
    completed = subprocess.run(
        ["git", "ls-files", "pu_toolbox/"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def _row(label: str, modules: list[str]) -> str:
    return f"| {label} | " + "、".join(f"`{module}`" for module in modules) + " |"


def _synthetic_doc(rows: list[str]) -> str:
    return "\n".join(
        [
            "# 架构原则",
            "",
            SECTION_HEADING,
            "",
            "7. **进化式架构**：…",
            "",
            f"{ITEM_HEADING}：长期无提交的模块视为稳定。",
            "",
            *rows,
            "",
            "## 5. 审计历史",
            "",
        ]
    )


def _complete_rows(stable: list[str], active: list[str]) -> list[str]:
    return ["| 分类 | 模块 |", "|---|---|", _row(STABLE_LABEL, stable), _row(ACTIVE_LABEL, active)]


def test_basic_parses_both_categories_from_the_live_document():
    categories = parse_stability_table(_read_doc())

    assert set(categories) == set(CATEGORY_LABELS)
    assert categories[STABLE_LABEL]
    assert categories[ACTIVE_LABEL]


def test_basic_live_document_classifies_every_tracked_module():
    tracked = tracked_top_level_modules(_tracked_listing())
    categories = parse_stability_table(_read_doc())

    assert categories[STABLE_LABEL] | categories[ACTIVE_LABEL] == tracked
    assert find_stability_problems(_read_doc(), tracked) == []


def test_basic_tracked_scan_keeps_only_top_level_directories():
    listing = "\n".join(
        [
            "pu_toolbox/__init__.py",
            "pu_toolbox/run_config.py",
            "pu_toolbox/losses/__init__.py",
            "pu_toolbox/losses/llsvm.py",
            "pu_toolbox/ui/app.py",
            "pu_toolbox/ui/app.py",
        ]
    )

    assert tracked_top_level_modules(listing) == {"losses", "ui"}


def test_param_blank_text_raises():
    with pytest.raises(StabilityTableError, match="section heading"):
        parse_stability_table("")


def test_param_missing_item_heading_raises():
    text = "\n".join([SECTION_HEADING, "", "7. **进化式架构**：…", "", "## 5. 审计历史", ""])

    with pytest.raises(StabilityTableError, match="item heading"):
        parse_stability_table(text)


def test_param_missing_category_row_raises():
    text = _synthetic_doc(["| 分类 | 模块 |", "|---|---|", _row(STABLE_LABEL, ["losses"])])

    with pytest.raises(StabilityTableError, match="missing category row"):
        parse_stability_table(text)


def test_param_empty_module_cell_raises():
    text = _synthetic_doc(
        _complete_rows(["losses"], [])  # renders an empty active cell
    )

    with pytest.raises(StabilityTableError, match="lists no module"):
        parse_stability_table(text)


def test_edge_unclassified_tracked_module_is_reported():
    text = _synthetic_doc(_complete_rows(["losses"], ["ui"]))
    problems = find_stability_problems(text, {"losses", "ui", "newmod"})

    assert len(problems) == 1
    assert "unclassified" in problems[0] and "newmod" in problems[0]


def test_edge_stale_classified_module_is_reported():
    text = _synthetic_doc(_complete_rows(["losses", "gone"], ["ui"]))
    problems = find_stability_problems(text, {"losses", "ui"})

    assert len(problems) == 1
    assert "stale" in problems[0] and "gone" in problems[0]


def test_edge_module_in_both_rows_is_reported():
    text = _synthetic_doc(_complete_rows(["losses", "ui"], ["ui", "cli"]))
    problems = find_stability_problems(text, {"losses", "ui", "cli"})

    assert len(problems) == 1
    assert "both" in problems[0] and "ui" in problems[0]


def test_determ_parse_is_independent_of_table_row_order():
    """Row order is presentation: a positional parser must not pass.

    The two rows are swapped between the documents, so any implementation
    that reads "row 1 = stable, row 2 = active" assigns the wrong labels
    here.  The expected mapping is pinned explicitly, not just compared
    to a second parse of the same text.
    """
    stable_row = _row(STABLE_LABEL, ["losses", "prior"])
    active_row = _row(ACTIVE_LABEL, ["ui", "cli"])
    header = ["| 分类 | 模块 |", "|---|---|"]
    rows = [*header, stable_row, active_row]
    swapped = [*header, active_row, stable_row]

    original = parse_stability_table(_synthetic_doc(rows))

    assert original == parse_stability_table(_synthetic_doc(swapped))
    assert original == {STABLE_LABEL: {"losses", "prior"}, ACTIVE_LABEL: {"ui", "cli"}}


def test_determ_problems_ignore_the_order_of_the_tracked_collection():
    """The verdict and its message order come from the data, not the input order.

    Both the module names inside each message and the messages themselves
    are pinned to their sorted form, so an implementation that iterates
    the caller's collection (or drops ``sorted``) is caught here.
    """
    text = _synthetic_doc(_complete_rows(["losses", "gone"], ["ui"]))
    expected = [
        "tracked but unclassified: ['amod', 'zmod']",
        "classified but not tracked (stale): ['gone']",
    ]

    forward = find_stability_problems(text, ["ui", "losses", "zmod", "amod"])
    backward = find_stability_problems(text, ["amod", "zmod", "losses", "ui"])

    assert forward == backward == expected
