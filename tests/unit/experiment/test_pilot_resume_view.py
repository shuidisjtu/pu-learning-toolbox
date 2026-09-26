# tests/unit/experiment/test_pilot_resume_view.py

# ruff: noqa: N803, N806, S101

"""Resuming a pilot must match the view each unit is going to run under.

P2.0e lets one method be run under two views, so the driver's completion scan
has to tell them apart.  A scan keyed on identity alone folds the OS and TS
manifests of one unit into a single entry -- whichever path happens to sort
first -- and then reports that unit done under *either* request.  The hole it
leaves is silent: the view that was never run stays unrun, and no log line says
so.  Re-running a finished unit wastes compute; a hole in the matrix is not
recoverable after the fact.

The unit script resolves each run's default view from the method ledger and the
estimator's ``fit`` signature.  The driver has to resolve it the same way, per
unit, instead of holding one global expectation.
"""

import json
from pathlib import Path

import pytest

from pu_toolbox.experiment.pilot_plan import (
    PN_ORACLE,
    completed_runs,
    pending_runs,
    planned_runs,
)

pytestmark = pytest.mark.unit


def _protocol() -> dict:
    """The smallest matrix with a counting unit and the oracle.

    ``nnpu`` is native to TS and wired for calibration, so its default view is
    the calibrated one.  The oracle is not a PU row at all and never has a
    calibrated view.
    """
    return {
        "seeds": [0],
        "c_tokens": {"scar": ["0.1"]},
        "candidate_pool": [{}],
        "budgets": {"minibatch": {"epochs": 200}},
        "backbone_specs": {"mlp128": {"hidden_dims": [128], "activation": "relu"}},
        "execution_units": [
            {
                "dataset": "spambase",
                "method": "nnpu",
                "training_path": "native_2d",
                "budget": "minibatch",
                "backbone": "mlp128",
                "runnable": True,
            },
            {
                "dataset": "spambase",
                "method": PN_ORACLE,
                "training_path": "native_2d",
                "budget": "minibatch",
                "backbone": "mlp128",
                "runnable": True,
            },
        ],
    }


def _manifest(*, method: str, seed: int = 0, run_view: str = "os-compatible") -> dict:
    return {
        "execution_mode": "versioned_pilot",
        "seed": seed,
        "execution_unit": {
            "dataset": "spambase",
            "method": method,
            "training_path": "native_2d",
        },
        "generation": {"train": {"mechanism": "scar"}},
        "c_requested_token": None if method == PN_ORACLE else "0.1",
        "selection": {"OA": {"candidate_index": 0}},
        "run_view": run_view,
        "calibration_applied": run_view == "ts-compatible",
    }


def _write(root: Path, relative: str, payload: dict) -> Path:
    path = root / relative / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- one identity, two views --------------------------------------------------


def test_basic_the_os_and_ts_results_of_one_unit_are_both_kept(tmp_path):
    """Keying the scan on identity alone keeps only one of the two manifests."""
    _write(tmp_path, "os", _manifest(method="nnpu", run_view="os-compatible"))
    _write(tmp_path, "ts", _manifest(method="nnpu", run_view="ts-compatible"))

    done = completed_runs(tmp_path)

    assert len(done) == 2


# --- what a resumed unit is held to ------------------------------------------


def _expected_views(protocol: dict, *, nnpu: str = "ts-compatible") -> dict:
    """Per-unit expectations: ``nnpu`` calibrated, the oracle always on OS."""
    return {
        run.key: ("os-compatible" if run.is_oracle else nnpu) for run in planned_runs(protocol)
    }


def test_basic_an_os_result_does_not_satisfy_a_default_ts_unit(tmp_path):
    """The default path is the one the driver used to leave blind.

    ``nnpu`` resolves to the calibrated view when no ``--os-or-ts`` is given, so
    a manifest with the same identity but the OS view is not evidence that the
    unit ran -- and treating it as evidence is what leaves the hole.
    """
    protocol = _protocol()
    _write(tmp_path, "os", _manifest(method="nnpu", run_view="os-compatible"))
    nnpu_run = next(run for run in planned_runs(protocol) if run.method == "nnpu")

    pending, _ = pending_runs(protocol, tmp_path, expected_views=_expected_views(protocol))

    assert nnpu_run in pending
