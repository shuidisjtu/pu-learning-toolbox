# ruff: noqa: N802, N803, N806, E501

"""Tests for the source comment-quality gate."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from check_comment_quality import scan_files  # noqa: E402


@pytest.mark.unit
def test_basic_clean_comments_pass(tmp_path):
    """Intent comments and tooling comments do not fail the default gate."""
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n# Keep the public value stable.\n", encoding="utf-8")

    report = scan_files([source])

    assert report.errors == ()
    assert report.comments_scanned == 1


@pytest.mark.unit
@pytest.mark.parametrize("marker", ["TODO", "FIXME", "XXX", "HACK"])
def test_param_legacy_marker_is_rejected(tmp_path, marker):
    """Legacy unfinished-work markers fail regardless of their spelling."""
    source = tmp_path / "marked.py"
    source.write_text(f"# {marker}: replace this\n", encoding="utf-8")

    report = scan_files([source])

    assert len(report.errors) == 1
    assert report.errors[0].kind == "legacy-marker"


@pytest.mark.unit
def test_edge_tooling_comments_are_not_trailing_advisories(tmp_path):
    """Ruff, coverage, and typing metadata stay exempt from style advisories."""
    source = tmp_path / "metadata.py"
    source.write_text(
        "value = 1  # noqa: E501\n"
        "other = 2  # pragma: no cover\n"
        "typed = 3  # type: ignore[assignment]\n"
        "documented = 4  # explain this branch\n",
        encoding="utf-8",
    )

    report = scan_files([source])

    assert len(report.findings) == 1
    assert report.findings[0].kind == "trailing-comment"
    assert report.findings[0].severity == "warning"


@pytest.mark.unit
def test_edge_strict_inline_mode_promotes_advisories(tmp_path):
    """Selected paths can opt into strict trailing-comment cleanup."""
    source = tmp_path / "strict.py"
    source.write_text("value = 1  # explain this branch\n", encoding="utf-8")

    report = scan_files([source], strict_inline=True)

    assert len(report.errors) == 1
    assert report.errors[0].kind == "trailing-comment"


@pytest.mark.unit
def test_determ_results_are_reproducible(tmp_path):
    """Repeated scans produce identical reports and finding order."""
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("a = 1  # explain a\n", encoding="utf-8")
    second.write_text("# stable intent\nb = 2\n", encoding="utf-8")

    assert scan_files([second, first]) == scan_files([second, first])


@pytest.mark.unit
@pytest.mark.parametrize("body", ["TBD", "TBD then decide", "left TBD here"])
def test_param_bare_tbd_is_rejected(tmp_path, body):
    """A TBD without a colon or parenthesis carries no traceable context."""
    source = tmp_path / "pending.py"
    source.write_text(f"# {body}\n", encoding="utf-8")

    report = scan_files([source])

    assert len(report.errors) == 1
    assert report.errors[0].kind == "bare-tbd"


@pytest.mark.unit
@pytest.mark.parametrize("body", ["TBD: wire the retry policy", "TBD(#123) revisit later"])
def test_basic_tbd_with_context_passes(tmp_path, body):
    """A TBD followed by a colon or a tracker reference is accepted."""
    source = tmp_path / "pending.py"
    source.write_text(f"# {body}\n", encoding="utf-8")

    assert scan_files([source]).errors == ()


@pytest.mark.unit
def test_edge_tbd_substring_is_not_a_marker(tmp_path):
    """Only a standalone TBD is a marker claim, not TBDs or TBDX."""
    source = tmp_path / "words.py"
    source.write_text("# the TBDs are listed elsewhere\n# TBDX is unrelated\n", encoding="utf-8")

    assert scan_files([source]).errors == ()


@pytest.mark.unit
def test_edge_unparsable_file_reports_scan_error(tmp_path):
    """A file that cannot be tokenized yields a scan-error, never a silent pass."""
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n", encoding="utf-8")

    report = scan_files([source])

    assert len(report.errors) == 1
    assert report.errors[0].kind == "scan-error"
    assert report.errors[0].line == 1


# --------------------------------------------------------------------
# §3's kept trailing comments: the ruling carries a machine-checkable anchor
# --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOVERNANCE_MD = PROJECT_ROOT / "docs" / "dev" / "comment_governance.md"
# The anchor paragraph under §3's table, e.g.
# "§3 保留项合计 25 行(...)：`pu_toolbox/core/config.py` 4、..." -- the counts
# are the part a human ruling cannot otherwise be checked against.
_ANCHOR_TOTAL_RE = re.compile(r"保留项合计\s+(\d+)\s+行")
_ANCHOR_PAIR_RE = re.compile(r"`(pu_toolbox/[^`]+\.py)`\s+(\d+)")


def _section3(text: str) -> str:
    """Return the §3 body of *text* (the ruling that keeps these comments)."""
    start = text.index("## 3.")
    return text[start : text.index("## 4.", start)]


def _anchor_counts(section: str) -> tuple[int, dict[str, int]]:
    """Parse §3's anchor into (stated total, {file: kept trailing comments})."""
    total = int(_ANCHOR_TOTAL_RE.search(section).group(1))
    counts = {path: int(n) for path, n in _ANCHOR_PAIR_RE.findall(section)}
    return total, counts


def _trailing_comments(path: Path) -> int:
    """Count the trailing comments the gate sees in *path*."""
    return sum(finding.kind == "trailing-comment" for finding in scan_files([path]).findings)


def _anchor_problems(section: str, actual: dict[str, int]) -> list[str]:
    """Return how §3's anchor disagrees with *actual* comment counts."""
    total, counts = _anchor_counts(section)
    problems: list[str] = []
    if total != sum(counts.values()):
        problems.append(f"锚点自相矛盾：合计 {total}，逐项之和 {sum(counts.values())}")
    for rel, expected in counts.items():
        if f"`{rel}`" not in section:
            problems.append(f"锚点点名的 `{rel}` 不在 §3 表格中")
        if actual.get(rel) != expected:
            problems.append(
                f"§3 的保留裁定与现实脱节：`{rel}` 锚点称 {expected} 条行尾注释，"
                f"实测 {actual.get(rel)} 条"
            )
    return problems


def _drop_one_trailing_comment(path: Path) -> str:
    """Return *path*'s source with its first gate-visible trailing comment cut."""
    finding = next(f for f in scan_files([path]).findings if f.kind == "trailing-comment")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    line = lines[finding.line - 1]
    match = re.search(r"\s+#.*$", line.rstrip("\n"))
    assert match is not None, line
    lines[finding.line - 1] = line[: match.start()] + ("\n" if line.endswith("\n") else "")
    return "".join(lines)


@pytest.mark.unit
def test_basic_section3_kept_comment_counts_match_reality():
    """§3 keeps 25 named trailing comments; they must still be there.

    Without this the ruling and the code drift apart in silence: clean one
    away and both §3 and the gate's pending reason keep claiming it exists.
    """
    section = _section3(GOVERNANCE_MD.read_text(encoding="utf-8"))
    total, counts = _anchor_counts(section)

    assert total == 25
    assert counts == {
        "pu_toolbox/core/config.py": 4,
        "pu_toolbox/core/tags.py": 8,
        "pu_toolbox/losses/llsvm.py": 13,
    }
    actual = {rel: _trailing_comments(PROJECT_ROOT / rel) for rel in counts}

    assert _anchor_problems(section, actual) == []


@pytest.mark.unit
def test_param_section3_anchor_detects_a_wrong_number_and_a_removed_comment(tmp_path):
    """Negative controls: a wrong anchor number, or a removed comment, goes red."""
    section = _section3(GOVERNANCE_MD.read_text(encoding="utf-8"))
    _, counts = _anchor_counts(section)
    actual = {rel: _trailing_comments(PROJECT_ROOT / rel) for rel in counts}

    # (a) one anchor number edited -> the mismatch is named
    wrong_number = section.replace("`pu_toolbox/core/tags.py` 8", "`pu_toolbox/core/tags.py` 9")
    assert wrong_number != section
    assert any("tags.py" in p for p in _anchor_problems(wrong_number, actual))

    # (b) one kept comment deleted from a copy of the file -> the mismatch is named
    stripped = tmp_path / "llsvm.py"
    stripped.write_text(
        _drop_one_trailing_comment(PROJECT_ROOT / "pu_toolbox/losses/llsvm.py"),
        encoding="utf-8",
    )
    shrunk = dict(actual)
    shrunk["pu_toolbox/losses/llsvm.py"] = _trailing_comments(stripped)

    assert shrunk["pu_toolbox/losses/llsvm.py"] == actual["pu_toolbox/losses/llsvm.py"] - 1
    assert any("llsvm.py" in p for p in _anchor_problems(section, shrunk))
