# ruff: noqa: N802, N803, N806, E501

"""Tests for the run subcommand: end-to-end, switches, error paths."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from pu_toolbox.cli import main
from pu_toolbox.preprocessing.pu_labeling import make_scar_dataset


def _write_demo(tmp_path, rng):
    """Write SCAR demo CSVs into tmp_path; return (data, labels, truth) paths."""
    X, y_pu, y_true = make_scar_dataset(
        n=30, c=0.5, n_features=5, separation=4.0, random_state=rng
    )
    data = tmp_path / "X.csv"
    labels = tmp_path / "y_pu.csv"
    truth = tmp_path / "y_true.csv"
    pd.DataFrame(X).to_csv(data, index=False)
    pd.DataFrame({"label": y_pu}).to_csv(labels, index=False)
    pd.DataFrame({"label": y_true}).to_csv(truth, index=False)
    return data, labels, truth


def _run(tmp_path, rng, *extra):
    """Run the CLI against demo data; return the parsed report payload."""
    data, labels, truth = _write_demo(tmp_path, rng)
    out = tmp_path / "out"
    main(
        [
            "run",
            "--data",
            str(data),
            "--labels",
            str(labels),
            "--out-dir",
            str(out),
            "--cv",
            "3",
            "--seed",
            "42",
            *extra,
        ]
    )
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()
    return out, json.loads((out / "report.json").read_text(encoding="utf-8"))


@pytest.mark.unit
def test_run_end_to_end(tmp_path, rng, capsys):
    """Full run produces a parseable strict-JSON report and prints a summary."""
    _, payload = _run(tmp_path, rng)
    assert payload["schema_version"] == "1.0"
    assert set(payload["cv_metrics"]) == {
        "pu_zero_one_risk",
        "pu_recall",
        "pu_estimated_precision",
        "pu_auc_roc",
    }
    assert payload["provenance"]["classifier_mode"] in {"auto", "name", "instance"}
    assert "PU Pipeline Report" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Class prior" in capsys.readouterr().out


@pytest.mark.unit
def test_run_with_true_labels_enables_oracle_metrics(tmp_path, rng):
    out, payload = _run(tmp_path, rng, "--true-labels", str(tmp_path / "y_true.csv"))
    auc = payload["cv_metrics"]["pu_auc_roc"]
    assert auc["available"] is True
    assert auc["mean"] is not None


@pytest.mark.unit
def test_run_explicit_class_prior_source_user(tmp_path, rng):
    out, payload = _run(tmp_path, rng, "--class-prior", "0.3")
    assert payload["prior"]["source"] == "user"
    assert payload["prior"]["value"] == 0.3


@pytest.mark.unit
def test_run_save_model_writes_picklable_model(tmp_path, rng):
    out, _ = _run(tmp_path, rng, "--save-model")
    model_path = out / "model.pkl"
    assert model_path.exists()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    preds = model.predict(np.zeros((3, 5)))
    assert preds.shape == (3,)


@pytest.mark.unit
def test_run_quiet_suppresses_summary(tmp_path, rng, capsys):
    _run(tmp_path, rng, "--quiet")
    assert "PU Pipeline Report" not in capsys.readouterr().out


@pytest.mark.unit
def test_run_metrics_comma_separated(tmp_path, rng):
    out, payload = _run(tmp_path, rng, "--metrics", "pu_risk,recall")
    assert set(payload["cv_metrics"]) == {"pu_zero_one_risk", "pu_recall"}


@pytest.mark.unit
def test_run_deterministic_same_seed(tmp_path, rng):
    """Same seed twice → identical metrics and prior in both reports."""
    _, first = _run(tmp_path, rng)
    # NOTE: brief's original `_write_demo(tmp_path, rng)` advanced the shared
    # fixture RNG, producing *different* data for the second run.  A fresh
    # RandomState(42) regenerates byte-identical data, matching the intent
    # "same seed twice -> identical metrics".
    data, labels, _ = _write_demo(tmp_path, np.random.RandomState(42))
    out2 = tmp_path / "out2"
    main(
        [
            "run",
            "--data",
            str(data),
            "--labels",
            str(labels),
            "--out-dir",
            str(out2),
            "--cv",
            "3",
            "--seed",
            "42",
        ]
    )
    second = json.loads((out2 / "report.json").read_text(encoding="utf-8"))
    assert first["cv_metrics"] == second["cv_metrics"]
    assert first["prior"] == second["prior"]


@pytest.mark.unit
def test_run_missing_input_file_exits_one(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data",
                str(tmp_path / "nope.csv"),
                "--labels",
                str(tmp_path / "y.csv"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1


@pytest.mark.unit
def test_run_invalid_labels_exits_one(tmp_path, rng, capsys):
    data, _, _ = _write_demo(tmp_path, rng)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"label": [0, 1, 2, 1, 0]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data",
                str(data),
                "--labels",
                str(bad),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.unit
def test_run_invalid_classifier_exits_one(tmp_path, rng, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, rng, "--classifier", "not_a_method")
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.unit
def test_basic_prior_estimator_none_allowed(tmp_path, rng):
    """--prior-estimator none works for methods that need no class prior."""
    _, payload = _run(tmp_path, rng, "--classifier", "elkan_noto", "--prior-estimator", "none")
    assert payload["prior"]["source"] == "none"


@pytest.mark.unit
def test_param_prior_estimator_none_missing_prior_exits_one(tmp_path, rng, capsys):
    """--prior-estimator none with a prior-requiring method and no prior → exit 1."""
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, rng, "--classifier", "nnpu", "--prior-estimator", "none")
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err
