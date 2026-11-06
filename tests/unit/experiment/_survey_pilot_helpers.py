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
SMOKE_DATASETS = ("cifar10", "imdb", "spambase")
SMOKE_METHODS = ("upu", "nnpu", "pusb_kernel", "dist_pu", "lbe", "self_pu", "pn_oracle")


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "run_survey_pilot", _SCRIPTS / "run_survey_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw_driver():
    """The script exactly as shipped, so its own ``--protocol`` default is what runs."""
    return _load_driver()


@pytest.fixture
def driver():
    """Exercise the current protocol within an explicit three-dataset smoke scope.

    Requests keep their explicit filters and protocol. Synthetic protocol stubs
    keep their own scope, while real protocols use the common methods whose
    four-role split fixtures this suite provides. Default-version behavior is
    tested separately through ``raw_driver``.
    """
    module = _load_driver()
    shipped_main = module.main

    def _main_in_smoke_scope(argv=None):
        args = list(argv or [])
        if "--protocol" not in args:
            args += ["--protocol", "survey-v3.2"]
        try:
            protocol_path = module.resolve_protocol_path(args[args.index("--protocol") + 1])
            protocol = module.load_protocol(protocol_path)
        except (OSError, ValueError):
            return shipped_main(args)
        if "profile_identity_schema_version" in protocol:
            units = protocol["execution_units"]
            if "--datasets" not in args:
                datasets = [
                    name
                    for name in SMOKE_DATASETS
                    if any(unit["dataset"] == name for unit in units)
                ]
                args += ["--datasets", ",".join(datasets)]
            if "--methods" not in args:
                args += ["--methods", ",".join(SMOKE_METHODS)]
        return shipped_main(args)

    module.main = _main_in_smoke_scope
    return module


@pytest.fixture
def unit_calls(driver, monkeypatch):
    """Capture what the driver would have executed instead of executing it.

    Patching ``subprocess.run`` is process-wide, so the stand-in also answers the
    driver's provenance probes (``git rev-parse``, and ``nvidia-smi`` when a GPU
    is visible).  Those get an empty stdout rather than an error: the probes
    already treat an unreadable identity as ``unavailable``, and recording them
    would make ``unit_calls`` mean something other than the unit script's argv.
    """
    recorded: list[list[str]] = []
    unit_script = str(_SCRIPTS / "run_survey_experiment.py")

    def _record(argv, check=False, **kwargs):
        arguments = [str(argument) for argument in argv]
        if unit_script in arguments:
            recorded.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

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
