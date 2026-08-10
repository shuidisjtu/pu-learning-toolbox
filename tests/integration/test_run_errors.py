# ruff: noqa: N802, N803, N806, E501
"""CLI run error-path tests: friendly failures for bad user input.
Kept separate from test_run.py (success paths) so neither file exceeds
the test-quality per-file limit.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from pu_toolbox.cli import main
from pu_toolbox.cli.demo import run_demo


def _write_demo(tmp_path, rng=None):
    """Write SCAR demo CSVs via the real make-demo-data implementation."""
    run_demo(
        argparse.Namespace(
            out_dir=str(tmp_path),
            n=30,
            c=0.5,
            n_features=5,
            separation=4.0,
            seed=42,
        )
    )
    return tmp_path / "X.csv", tmp_path / "y_pu.csv", tmp_path / "y_true.csv"


@pytest.mark.integration
def test_basic_non_numeric_prior_param_exits_one(tmp_path, rng, capsys):
    """A non-numeric --prior-param value exits 1 instead of silently degrading."""
    data, labels, _ = _write_demo(tmp_path, rng)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data",
                str(data),
                "--labels",
                str(labels),
                "--out-dir",
                str(out),
                "--classifier",
                "upu",
                "--prior-param",
                "sigma=abc",
            ]
        )
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "not a number" in err
    assert not (out / "report.json").exists()


@pytest.mark.integration
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_param_non_finite_features_friendly_error(tmp_path, rng, capsys, bad_value):
    """NaN/Inf in the feature CSV exits 1 with a user-facing message,
    not sklearn internals."""
    data, labels, _ = _write_demo(tmp_path, rng)
    frame = pd.read_csv(data)
    frame.iloc[0, 0] = bad_value
    bad = tmp_path / "X_bad.csv"
    frame.to_csv(bad, index=False)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data",
                str(bad),
                "--labels",
                str(labels),
                "--out-dir",
                str(tmp_path / "out"),
                "--classifier",
                "upu",
            ]
        )
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "NaN or Inf" in err
    assert "LogisticRegression" not in err


@pytest.mark.integration
def test_edge_bad_input_writes_no_report(tmp_path, rng, capsys):
    """A rejected input must leave no report artifacts behind."""
    data, labels, _ = _write_demo(tmp_path, rng)
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--data",
                str(data / "nope.csv"),
                "--labels",
                str(labels),
                "--out-dir",
                str(out),
                "--classifier",
                "upu",
            ]
        )
    assert not out.exists()


@pytest.mark.integration
def test_determ_same_bad_input_same_error(tmp_path, rng, capsys):
    """Repeated bad input produces byte-identical stderr (deterministic errors)."""
    data, labels, _ = _write_demo(tmp_path, rng)
    frame = pd.read_csv(data)
    frame.iloc[0, 0] = np.nan
    bad = tmp_path / "X_nan.csv"
    frame.to_csv(bad, index=False)
    first_err = None
    for _ in range(2):
        with pytest.raises(SystemExit):
            main(
                [
                    "run",
                    "--data",
                    str(bad),
                    "--labels",
                    str(labels),
                    "--out-dir",
                    str(tmp_path / "out2"),
                    "--classifier",
                    "upu",
                ]
            )
        err = capsys.readouterr().err
        if first_err is None:
            first_err = err
        else:
            assert err == first_err
