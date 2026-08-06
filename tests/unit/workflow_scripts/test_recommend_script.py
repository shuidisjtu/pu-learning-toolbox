# ruff: noqa: N802, N803, N806, E501

"""Tests for the pu-workflow recommend step script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_scar_data


def _write_inputs(tmp_path, rng):
    X, y_pu, _ = make_scar_data(rng, n=120, separation=4.0)
    # Non-numeric header row required: the CLI's CSV contract rejects
    # all-numeric first rows (indistinguishable from data) as headerless.
    pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]).to_csv(
        tmp_path / "X.csv", index=False
    )
    pd.DataFrame({"label": y_pu}).to_csv(tmp_path / "y_pu.csv", index=False)
    return tmp_path / "X.csv", tmp_path / "y_pu.csv"


def _make_profile(tmp_path, rng):
    """Run the profile script (Task 2) to produce profile.json in tmp_path."""
    X, y_pu = _write_inputs(tmp_path, rng)
    subprocess.run(
        [
            sys.executable,
            "scripts/pu_workflow/profile.py",
            "--data",
            str(X),
            "--labels",
            str(y_pu),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        check=True,
    )
    return tmp_path / "profile.json"


def _run_script(tmp_path, profile, *extra):
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/pu_workflow/recommend.py",
            "--profile",
            str(profile),
            *extra,
            "--out-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    return proc.returncode, proc.stderr, out


@pytest.mark.unit
def test_basic_recommend_writes_contract_json(tmp_path, rng):
    """Output recommendation.json parses, with ranked candidates."""
    profile = _make_profile(tmp_path, rng)
    code, _, out = _run_script(tmp_path, profile)
    assert code == 0
    payload = json.loads((out / "recommendation.json").read_text(encoding="utf-8"))
    assert isinstance(payload["candidates"], list)
    assert payload["candidates"], "at least one candidate on SCAR data"


@pytest.mark.unit
def test_param_class_prior_user_and_estimated(tmp_path, rng):
    """--class-prior records source 'user'; km1 estimation records 'estimated'."""
    X, y_pu = _write_inputs(tmp_path, rng)
    profile = _make_profile(tmp_path, rng)
    code, _, out = _run_script(tmp_path, profile, "--class-prior", "0.3")
    assert code == 0
    assert (
        "class_prior_source"
        not in json.loads((out / "recommendation.json").read_text(encoding="utf-8"))["provenance"]
    )  # script does not invent provenance fields it cannot verify

    code, stderr, out = _run_script(
        tmp_path, profile, "--prior-estimator", "km1", "--data", str(X), "--labels", str(y_pu)
    )
    assert code == 0, stderr


@pytest.mark.unit
def test_edge_prior_estimator_without_data_errors(tmp_path, rng):
    """--prior-estimator != none without --data/--labels -> exit 1."""
    profile = _make_profile(tmp_path, rng)
    code, stderr, out = _run_script(tmp_path, profile, "--prior-estimator", "recpe")
    assert code == 1
    assert "error:" in stderr


@pytest.mark.unit
def test_edge_unknown_prior_estimator_reports_error(tmp_path, rng):
    """Unknown --prior-estimator -> exit 1 with error: message, no traceback."""
    X, y_pu = _write_inputs(tmp_path, rng)
    profile = _make_profile(tmp_path, rng)
    code, stderr, out = _run_script(
        tmp_path, profile, "--prior-estimator", "nope", "--data", str(X), "--labels", str(y_pu)
    )
    assert code == 1
    assert "error:" in stderr
    assert "Traceback" not in stderr


@pytest.mark.unit
def test_deterministic_recommend_same_input(tmp_path, rng):
    """Same profile twice -> identical recommendation.json bytes."""
    profile = _make_profile(tmp_path, rng)
    _, _, out1 = _run_script(tmp_path, profile)
    _, _, out2 = _run_script(tmp_path, profile)
    assert (out1 / "recommendation.json").read_bytes() == (
        out2 / "recommendation.json"
    ).read_bytes()
