# ruff: noqa: N802, N803, N806, E501

"""Tests for the source comment-quality gate."""

from __future__ import annotations

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
