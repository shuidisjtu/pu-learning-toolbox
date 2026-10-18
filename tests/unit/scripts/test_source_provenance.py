"""Exploratory source identity must never come from an unrelated caller repo."""

import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from scripts import _source_provenance as provenance

pytestmark = pytest.mark.unit


def fixture_files(tmp_path):
    root = tmp_path / "checkout"
    script = root / "scripts/run_example.py"
    implementation = root / "pu_toolbox/model.py"
    script.parent.mkdir(parents=True)
    implementation.parent.mkdir(parents=True)
    script.write_text("# script\n", encoding="utf-8")
    implementation.write_text("# implementation\n", encoding="utf-8")
    return root, script, implementation


def test_basic_determ_git_identity_is_bound_to_script_not_cwd(tmp_path, monkeypatch):
    root, script, implementation = fixture_files(tmp_path)
    caller = tmp_path / "other_checkout"
    caller.mkdir()
    monkeypatch.chdir(caller)
    calls = []

    def git(command, *, cwd, **kwargs):
        calls.append(command)
        assert cwd == root
        assert kwargs["check"] and kwargs["timeout"] == 10
        if command[-1] == "--show-toplevel":
            return SimpleNamespace(stdout=str(root) + "\n")
        if command[-1] == "HEAD":
            return SimpleNamespace(stdout="a" * 40 + "\n")
        return SimpleNamespace(stdout="?? unpublished-file\n")

    monkeypatch.setattr(provenance.subprocess, "run", git)
    result = provenance.source_provenance(script, {"model": implementation})
    assert result == provenance.source_provenance(script, {"model": implementation})
    assert len(calls) == 6
    assert result["script_checkout_commit"] == "a" * 40
    assert result["script_checkout_dirty"] is True
    assert result["git_identity_status"] == "script_checkout"
    assert (
        result["loaded_implementations"]["model"]["sha256"]
        == hashlib.sha256(implementation.read_bytes()).hexdigest()
    )
    assert result["all_loaded_implementations_inside_script_checkout"] is True
    assert result["formal_admission"] is False
    assert "unpublished-file" not in str(result)


@pytest.mark.parametrize(
    "failure", [FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired]
)
def test_edge_unavailable_git_is_unknown_not_a_clean_worktree(tmp_path, monkeypatch, failure):
    _, script, implementation = fixture_files(tmp_path)

    def fail(*args, **kwargs):
        if failure is subprocess.CalledProcessError:
            raise failure(128, "git")
        if failure is subprocess.TimeoutExpired:
            raise failure("git", 10)
        raise failure("git unavailable")

    monkeypatch.setattr(provenance.subprocess, "run", fail)
    result = provenance.source_provenance(script, {"model": implementation})
    assert result["script_checkout_commit"] is None
    assert result["script_checkout_dirty"] is None
    assert result["git_identity_status"] == "unavailable"
    assert result["loaded_implementations"]["model"]["sha256"]


def test_edge_parent_repository_commit_is_not_accepted(tmp_path, monkeypatch):
    root, script, implementation = fixture_files(tmp_path)
    calls = []

    def git(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=str(root.parent) + "\n")

    monkeypatch.setattr(provenance.subprocess, "run", git)
    result = provenance.source_provenance(script, {"model": implementation})
    assert len(calls) == 1
    assert result["git_identity_status"] == "not_script_checkout"
    assert result["script_checkout_commit"] is None and result["script_checkout_dirty"] is None


def test_edge_loaded_wheel_is_distinct_from_script_checkout(tmp_path, monkeypatch):
    _, script, implementation = fixture_files(tmp_path)
    wheel_file = tmp_path / "installed/model.py"
    wheel_file.parent.mkdir()
    wheel_file.write_bytes(implementation.read_bytes())
    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
    )
    result = provenance.source_provenance(script, {"source": implementation, "wheel": wheel_file})
    assert result["loaded_implementations"]["source"]["inside_script_checkout"] is True
    assert result["loaded_implementations"]["wheel"]["inside_script_checkout"] is False
    assert result["all_loaded_implementations_inside_script_checkout"] is False
    assert result["git_commit_scope"] == "script_checkout_only_not_loaded_distribution"


@pytest.mark.parametrize("implementations", [None, {}, []])
def test_param_empty_implementation_mapping_does_not_vacuously_verify(tmp_path, implementations):
    _, script, _ = fixture_files(tmp_path)
    with pytest.raises(ValueError, match="nonempty mapping"):
        provenance.source_provenance(script, implementations)


def test_param_missing_loaded_source_cannot_get_provenance(tmp_path, monkeypatch):
    _, script, _ = fixture_files(tmp_path)
    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
    )
    with pytest.raises(FileNotFoundError):
        provenance.source_provenance(script, {"model": tmp_path / "missing.py"})
