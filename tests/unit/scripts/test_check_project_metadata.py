# ruff: noqa: N802, N803, N806, E501

"""Tests for the project-metadata gate (Python / CI / extras consistency).

The gate shipped with no test at all, so nothing proved a drifted
pyproject/CI pair could actually turn it red.  Each case below runs
``main()`` against a throwaway repo assembled from this repository's own
metadata files: the healthy copy must pass, and a single mutation must
flip the exit code.
"""

from __future__ import annotations

import shutil
import sys
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
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    mutated = text.replace("all = [\n", 'all = [\n    "ruff>=0.3",\n', 1)
    assert mutated != text  # canary: the anchor still exists in pyproject.toml
    pyproject.write_text(mutated, encoding="utf-8")
    monkeypatch.setattr(m, "ROOT", root)
    assert m.main() == 1
    assert "all extra" in capsys.readouterr().out


@pytest.mark.unit
def test_determ_repeated_runs_identical(tmp_path, monkeypatch, capsys):
    """main() is a pure read: repeated runs give the same verdict and message."""
    monkeypatch.setattr(m, "ROOT", _fake_repo(tmp_path))
    first = (m.main(), capsys.readouterr().out)
    second = (m.main(), capsys.readouterr().out)
    assert first == second
