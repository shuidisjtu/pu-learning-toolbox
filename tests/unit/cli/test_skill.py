# ruff: noqa: N802, N803, N806, E501

"""Tests for the skill subcommand (bundled pu-workflow skill installer)."""

from __future__ import annotations

import pytest

from pu_toolbox.cli import build_parser
from pu_toolbox.core.exceptions import PULearningError

_TARGETS = (".claude/skills/pu-workflow", ".agents/skills/pu-workflow")


def _install(tmp_path, *extra):
    args = build_parser().parse_args(["skill", "install", "--dest", str(tmp_path), *extra])
    args.func(args)
    return args


@pytest.mark.unit
def test_basic_installs_to_claude_and_agents(tmp_path):
    _install(tmp_path)
    for target in _TARGETS:
        skill_file = tmp_path / target / "SKILL.md"
        assert skill_file.is_file()
        assert skill_file.read_text(encoding="utf-8").strip()


@pytest.mark.unit
def test_basic_reports_destination(tmp_path, capsys):
    _install(tmp_path)
    out = capsys.readouterr().out
    assert "installed pu-workflow skill" in out
    assert str(tmp_path / ".claude") in out


@pytest.mark.unit
def test_param_custom_dest_respected(tmp_path):
    custom = tmp_path / "agents-root"
    _install(custom)
    assert (custom / ".claude/skills/pu-workflow/SKILL.md").is_file()


@pytest.mark.unit
def test_param_force_overwrites_existing(tmp_path):
    target = tmp_path / ".claude/skills/pu-workflow"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("stale", encoding="utf-8")
    _install(tmp_path, "--force")
    content = (target / "SKILL.md").read_text(encoding="utf-8")
    assert content != "stale" and content.strip()


@pytest.mark.unit
def test_edge_existing_without_force_skips(tmp_path, capsys):
    target = tmp_path / ".claude/skills/pu-workflow"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("keep-me", encoding="utf-8")
    _install(tmp_path)
    assert "already installed" in capsys.readouterr().out
    assert (target / "SKILL.md").read_text(encoding="utf-8") == "keep-me"


@pytest.mark.unit
def test_edge_missing_source_raises(tmp_path, monkeypatch):
    from pu_toolbox.cli import skill as skill_cli

    monkeypatch.setattr(skill_cli, "_skill_source", lambda: tmp_path / "missing")
    with pytest.raises(PULearningError, match="skill assets not found"):
        _install(tmp_path)


@pytest.mark.unit
def test_determ_repeat_install_stable(tmp_path):
    _install(tmp_path)
    first = (tmp_path / ".claude/skills/pu-workflow/SKILL.md").read_bytes()
    _install(tmp_path, "--force")
    second = (tmp_path / ".claude/skills/pu-workflow/SKILL.md").read_bytes()
    assert first == second
