# ruff: noqa: N802, N803, N806, E501

"""Tests for the project-metadata gate (Python / CI / extras consistency).

The gate shipped with no test at all, so nothing proved a drifted
pyproject/CI pair could actually turn it red.  Each case below runs
``main()`` against a throwaway repo assembled from this repository's own
metadata files: the healthy copy must pass, and a single mutation must
flip the exit code.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_project_metadata as m  # noqa: E402

_REAL_ROOT = Path(__file__).resolve().parents[3]
_COPIED_FILES = (
    "pyproject.toml",
    ".python-version",
    ".gitignore",
    ".github/workflows/tests.yml",
    ".github/workflows/nightly.yml",
    "pu_toolbox/__init__.py",
)


def _fake_repo(tmp_path: Path) -> Path:
    """Assemble a minimal repo that passes the gate, ready to be broken."""
    root = tmp_path / "repo"
    for relative in _COPIED_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REAL_ROOT / relative, target)
    (root / "uv.lock").write_text("", encoding="utf-8")
    return root


def _mutate(root: Path, relative: str, old: str, new: str) -> None:
    """Replace every occurrence of *old*, failing loudly if the anchor is gone."""
    target = root / relative
    text = target.read_text(encoding="utf-8")
    mutated = text.replace(old, new)
    assert mutated != text, f"anchor {old!r} is gone from {relative}"
    target.write_text(mutated, encoding="utf-8")


@pytest.mark.unit
def test_basic_assembled_repo_passes(tmp_path, monkeypatch):
    """The assembled baseline is green, so each mutation below is the only cause."""
    monkeypatch.setattr(m, "ROOT", _fake_repo(tmp_path))
    assert m.main() == 0


@pytest.mark.unit
def test_param_version_drift_fails_main(tmp_path, monkeypatch, capsys):
    """__version__ drifting from pyproject project.version fails the gate."""
    root = _fake_repo(tmp_path)
    (root / "pu_toolbox" / "__init__.py").write_text('__version__ = "0.0.0"\n', encoding="utf-8")
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert "__version__" in capsys.readouterr().out


@pytest.mark.unit
def test_edge_missing_uv_lock_fails_main(tmp_path, monkeypatch, capsys):
    """A missing uv.lock is reported instead of silently passing."""
    root = _fake_repo(tmp_path)
    (root / "uv.lock").unlink()
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert "uv.lock" in capsys.readouterr().out


@pytest.mark.unit
def test_param_dev_tool_in_all_extra_fails_main(tmp_path, monkeypatch, capsys):
    """A developer tool leaking into the runtime ``all`` extra fails the gate."""
    root = _fake_repo(tmp_path)
    _mutate(root, "pyproject.toml", "all = [\n", 'all = [\n    "ruff>=0.3",\n')
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert "all extra" in capsys.readouterr().out


@pytest.mark.unit
def test_edge_gitignored_uv_lock_fails_main(tmp_path, monkeypatch, capsys):
    """uv.lock present but gitignored is still a failure (the lock must ship)."""
    root = _fake_repo(tmp_path)
    gitignore = root / ".gitignore"
    gitignore.write_text(gitignore.read_text(encoding="utf-8") + "uv.lock\n", encoding="utf-8")
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert "uv.lock" in capsys.readouterr().out


@pytest.mark.unit
def test_edge_manifest_in_present_fails_main(tmp_path, monkeypatch, capsys):
    """A MANIFEST.in is not authoritative under Hatchling and must not exist."""
    root = _fake_repo(tmp_path)
    (root / "MANIFEST.in").write_text("include README.md\n", encoding="utf-8")
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert "MANIFEST.in" in capsys.readouterr().out


@pytest.mark.unit
def test_param_ci_matrix_version_missing_fails_main(tmp_path, monkeypatch, capsys):
    """Dropping a supported interpreter from the CI matrix fails the gate."""
    root = _fake_repo(tmp_path)
    _mutate(root, ".github/workflows/tests.yml", '"3.11"', '"3.13"')
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert "CI matrix is missing Python 3.11" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("uv sync --upgrade", "uv sync --no-lock", "unsupported uv sync --no-lock"),
        ("uv sync --upgrade", "uv sync", "re-resolve dependencies"),
        (
            'uv run --python "${{ matrix.python-version }}"',
            "uv run",
            "nightly must pass the matrix interpreter",
        ),
    ],
)
def test_param_invalid_nightly_commands_fail_main(tmp_path, monkeypatch, capsys, old, new, message):
    """Reject broken nightly dependency resolution or interpreter routing."""
    root = _fake_repo(tmp_path)
    _mutate(root, ".github/workflows/nightly.yml", old, new)
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert message in capsys.readouterr().out


@pytest.mark.unit
def test_param_wheel_packages_drift_fails_main(tmp_path, monkeypatch, capsys):
    """Widening the wheel target beyond pu_toolbox fails the gate."""
    root = _fake_repo(tmp_path)
    _mutate(
        root,
        "pyproject.toml",
        'packages = ["pu_toolbox"]',
        'packages = ["pu_toolbox", "scripts"]',
    )
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert "wheel target" in capsys.readouterr().out


@pytest.mark.unit
def test_edge_sdist_missing_contributing_fails_main(tmp_path, monkeypatch, capsys):
    """Dropping CONTRIBUTING.md from the sdist manifest fails the gate."""
    root = _fake_repo(tmp_path)
    _mutate(root, "pyproject.toml", '    "/CONTRIBUTING.md",\n', "")
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert "sdist must include CONTRIBUTING.md" in capsys.readouterr().out


@pytest.mark.unit
def test_determ_repeated_runs_identical(tmp_path, monkeypatch, capsys):
    """main() is a pure read: repeated runs give the same verdict and message."""
    monkeypatch.setattr(m, "ROOT", _fake_repo(tmp_path))
    first = (m.main(), capsys.readouterr().out)
    second = (m.main(), capsys.readouterr().out)
    assert first == second


def _run_ci_summary(tmp_path: Path):
    workflow = (_REAL_ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "--junitxml=pytest-results.xml" in workflow
    assert "fail-fast: false" in workflow
    match = re.search(r"python - <<'PY'\n(.*?)\n\s+PY\n", workflow, re.DOTALL)
    assert match is not None
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(match.group(1))],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return summary.read_text(encoding="utf-8") if summary.exists() else None


@pytest.mark.unit
def test_edge_ci_summary_missing_report_preserves_dependency_failure(tmp_path):
    """An installation failure must not be replaced by a summary parsing error."""
    assert _run_ci_summary(tmp_path) is None


@pytest.mark.unit
@pytest.mark.parametrize("problem", ["failure", "error"])
def test_basic_ci_summary_exposes_failure_traceback_and_is_repeatable(tmp_path, problem):
    report = tmp_path / "pytest-results.xml"
    report.write_text(
        '<testsuites><testsuite><testcase classname="tests.example" name="test_case">'
        f'<{problem} message="failed">AssertionError: expected 2, got 1\n```</{problem}>'
        "</testcase></testsuite></testsuites>",
        encoding="utf-8",
    )
    summary = _run_ci_summary(tmp_path)
    assert "tests.example::test_case" in summary
    assert "AssertionError: expected 2, got 1" in summary
    assert summary.count("```") == 2
    (tmp_path / "summary.md").unlink()
    assert _run_ci_summary(tmp_path) == summary


@pytest.mark.unit
def test_basic_ci_summary_passing_report_is_explicit(tmp_path):
    (tmp_path / "pytest-results.xml").write_text(
        '<testsuites><testsuite><testcase name="test_ok"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    assert "No test failures recorded" in _run_ci_summary(tmp_path)
