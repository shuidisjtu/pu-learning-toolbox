# tests/unit/experiment/_survey_pilot_helpers.py

# ruff: noqa: N803, N806, S101

"""Shared fixtures/helpers for the ``scripts/run_survey_pilot.py`` tests.

Not collected by pytest (``python_files = ["test_*.py"]``): importing
``driver``/``unit_calls``/``splits_tree`` into a test module registers the
fixtures there, so every driver test module exercises the same script source
without duplicating the loader.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"

#: The population priors the pilot datasets record.  A test that wants the
#: *missing* prior path passes a splits tree that records none instead.
ALL_PRIORS = ["--class-prior", "spambase=0.39,imdb=0.5,cifar10=0.1"]


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location(
        "run_survey_pilot", _SCRIPTS / "run_survey_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def unit_calls(driver, monkeypatch):
    """Capture what the driver would have executed instead of executing it."""
    recorded: list[list[str]] = []

    def _record(argv, check=False):
        recorded.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(driver.subprocess, "run", _record)
    return recorded


def splits_tree(
    root: Path, *, prior: float | None = None, datasets=("cifar10", "imdb", "spambase")
):
    """A minimal splits tree: enough for the driver's digest and prior reads."""
    for dataset in datasets:
        path = root / dataset / "split_0" / "split_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dataset": dataset, "seed": 0, "indices_sha256": "a" * 64}
        if prior is not None:
            payload["class_prior"] = {"population": prior}
        path.write_text(json.dumps(payload), encoding="utf-8")
    return root
