# ruff: noqa: N802, N803, N806, E501

"""Tests for the skill-sync consistency gate (5th quality gate)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from check_skill_sync import check_sync  # noqa: E402

_GOOD_FRONTMATTER = (
    "---\n"
    "name: pu-workflow\n"
    "description: End-to-end PU learning workflow\n"
    "license: MIT\n"
    "compatibility: Python >=3.10, uv\n"
    "metadata:\n"
    "  author: shuidisjtu\n"
    "  version: 0.1.0\n"
    "---\n"
)


def _write_pair(tmp_path: Path, text_a: str, text_b: str | None = None) -> tuple[Path, Path]:
    """Write two distinct SKILL.md copies under *tmp_path* (a/ and b/)."""
    a = tmp_path / "a" / "pu-workflow" / "SKILL.md"
    b = tmp_path / "b" / "pu-workflow" / "SKILL.md"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_text(text_a, encoding="utf-8")
    b.write_text(text_b if text_b is not None else text_a, encoding="utf-8")
    return a, b


@pytest.mark.unit
def test_basic_identical_pair_passes(tmp_path):
    a, b = _write_pair(tmp_path, _GOOD_FRONTMATTER + "# pu-workflow\n\nbody\n")
    assert check_sync(a, b) == []


@pytest.mark.unit
def test_param_crlf_tolerated(tmp_path):
    """CRLF/LF differences from git autocrlf must not fail the gate."""
    a, b = _write_pair(tmp_path, _GOOD_FRONTMATTER + "# pu-workflow\n\nbody\n")
    crlf_text = _GOOD_FRONTMATTER.replace("\n", "\r\n") + "# pu-workflow\r\n\r\nbody\r\n"
    b.write_bytes(
        crlf_text.encode("utf-8")
    )  # literal CRLF; write_text would double-translate on Windows
    assert check_sync(a, b) == []


@pytest.mark.unit
def test_edge_content_drift_reported(tmp_path):
    a, b = _write_pair(tmp_path, "line1\nline2\n", "line1\nline2 CHANGED\n")
    issues = check_sync(a, b)
    assert any("content mismatch" in i for i in issues)


@pytest.mark.unit
def test_edge_non_portable_frontmatter_reported(tmp_path):
    a, b = _write_pair(
        tmp_path, "---\nname: pu-workflow\ndescription: x\nuser-invocable: true\n---\n"
    )
    issues = check_sync(a, b)
    assert any("user-invocable" in i for i in issues)


@pytest.mark.unit
def test_edge_name_mismatch_reported(tmp_path):
    a, b = _write_pair(tmp_path, "---\nname: other-name\ndescription: x\n---\n")
    issues = check_sync(a, b)
    assert any("directory" in i for i in issues)


@pytest.mark.unit
def test_edge_missing_description_reported(tmp_path):
    a, b = _write_pair(tmp_path, "---\nname: pu-workflow\n---\n")
    issues = check_sync(a, b)
    assert any("description" in i for i in issues)


@pytest.mark.unit
def test_edge_empty_value_key_does_not_swallow_next_line(tmp_path):
    """An empty-value key must not swallow the next line into its value."""
    a, b = _write_pair(
        tmp_path,
        "---\nname: pu-workflow\ndescription: x\nmetadata:\nuser-invocable: true\n---\n",
    )
    issues = check_sync(a, b)
    assert any("user-invocable" in i for i in issues)
