"""Import/prepare on all CI platforms; host locks still fail closed."""

import builtins
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts._survey_host import lock_controller
from scripts.run_candidate_readiness_campaign import sha

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("size", [0, 17, 1024 * 1024 + 3])
def test_basic_determ_sha_matches_hashlib_without_python311_api(tmp_path, monkeypatch, size):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    payload = b"x" * size
    path = tmp_path / "payload"
    path.write_bytes(payload)
    assert sha(path) == sha(path) == hashlib.sha256(payload).hexdigest()


def test_param_missing_host_lock_fails_before_creating_file(tmp_path, monkeypatch):
    original = builtins.__import__

    def without_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("simulated Windows")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_fcntl)
    with pytest.raises(RuntimeError, match="POSIX host"):
        lock_controller(tmp_path / "controller.lock")
    assert not (tmp_path / "controller.lock").exists()


def test_edge_real_host_lock_excludes_second_controller_and_releases_on_close(tmp_path):
    pytest.importorskip("fcntl")
    path = tmp_path / "controller.lock"
    with lock_controller(path), pytest.raises(BlockingIOError):
        lock_controller(path)
    with lock_controller(path) as handle:
        assert not handle.closed


def test_basic_all_planning_modules_import_without_fcntl():
    code = """
import builtins,importlib
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name == 'fcntl': raise ImportError('simulated Windows')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
for name in ['run_candidate_readiness_campaign','run_candidate_five_seed_grid',
             'run_survey_backlog_campaign','recover_survey_backlog_canary',
             'run_survey_parallel_queue','monitor_survey_campaign']:
    importlib.import_module('scripts.'+name)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
