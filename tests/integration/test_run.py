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


def _write_demo(tmp_path, rng=None):
    """Write SCAR demo CSVs via the real make-demo-data implementation.

    The CSV contract (string feature names, three files, row counts) has a
    single source of truth in cli/demo.py; this helper calls it instead of
    replicating the generation logic.  *rng* is accepted for compatibility
    with call sites (the demo uses a fixed seed).
    """
    import argparse

    from pu_toolbox.cli.demo import run_demo

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


def _run(tmp_path, rng, *extra):
    """Run the CLI against demo data; return the parsed report payload.

    ``--classifier upu`` keeps the suite fast: auto mode selects LLSVM
    (3000 fixed epochs, ~15s/run); uPU finishes in ~0.2s and exercises the
    same CSV-IO / report / exit-code surface.  Auto mode is covered by
    test_basic_auto_mode_without_prior_estimator and
    test_demo_output_consumable_by_run.
    """
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
            "--classifier",
            "upu",
            *extra,
        ]
    )
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()
    return out, json.loads((out / "report.json").read_text(encoding="utf-8"))


@pytest.mark.integration
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
    # The _run helper passes an explicit --classifier upu, so the mode is
    # pinned to "name" (a vacuous set-membership assertion would not catch
    # a mode-selection regression).
    assert payload["provenance"]["classifier_mode"] == "name"
    assert "PU Pipeline Report" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Class prior" in capsys.readouterr().out


@pytest.mark.integration
def test_run_with_true_labels_enables_oracle_metrics(tmp_path, rng):
    out, payload = _run(tmp_path, rng, "--true-labels", str(tmp_path / "y_true.csv"))
    auc = payload["cv_metrics"]["pu_auc_roc"]
    assert auc["available"] is True
    assert auc["mean"] is not None


@pytest.mark.integration
def test_run_explicit_class_prior_source_user(tmp_path, rng):
    out, payload = _run(tmp_path, rng, "--class-prior", "0.3")
    assert payload["prior"]["source"] == "user"
    assert payload["prior"]["value"] == 0.3


@pytest.mark.integration
def test_run_save_model_writes_picklable_model(tmp_path, rng):
    out, _ = _run(tmp_path, rng, "--save-model")
    model_path = out / "model.pkl"
    assert model_path.exists()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    preds = model.predict(np.zeros((3, 5)))
    assert preds.shape == (3,)


@pytest.mark.integration
def test_run_quiet_suppresses_summary(tmp_path, rng, capsys):
    _run(tmp_path, rng, "--quiet")
    assert "PU Pipeline Report" not in capsys.readouterr().out


@pytest.mark.integration
def test_run_metrics_comma_separated(tmp_path, rng):
    """Comma-separated metrics work with stray whitespace after commas."""
    out, payload = _run(tmp_path, rng, "--metrics", "pu_risk, recall")
    assert set(payload["cv_metrics"]) == {"pu_zero_one_risk", "pu_recall"}


@pytest.mark.integration
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
            "--classifier",
            "upu",
        ]
    )
    second = json.loads((out2 / "report.json").read_text(encoding="utf-8"))
    assert first["cv_metrics"] == second["cv_metrics"]
    assert first["prior"] == second["prior"]


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
def test_run_invalid_classifier_exits_one(tmp_path, rng, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, rng, "--classifier", "not_a_method")
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.integration
def test_run_invalid_prior_param_value_exits_one(tmp_path, rng, capsys):
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
def test_run_nan_features_reports_friendly_error(tmp_path, rng, capsys):
    """NaN in the feature CSV exits 1 with a user-facing message, not sklearn internals."""
    data, labels, _ = _write_demo(tmp_path, rng)
    frame = pd.read_csv(data)
    frame.iloc[0, 0] = np.nan
    bad = tmp_path / "X_nan.csv"
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
    assert "NaN" in err
    assert "LogisticRegression" not in err


@pytest.mark.integration
def test_run_inf_features_reports_friendly_error(tmp_path, rng, capsys):
    """Inf in the feature CSV exits 1 with a user-facing message."""
    data, labels, _ = _write_demo(tmp_path, rng)
    frame = pd.read_csv(data)
    frame.iloc[0, 0] = np.inf
    bad = tmp_path / "X_inf.csv"
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
    assert "Inf" in capsys.readouterr().err


@pytest.mark.integration
def test_basic_prior_estimator_none_allowed(tmp_path, rng):
    """--prior-estimator none works for methods that need no class prior."""
    _, payload = _run(tmp_path, rng, "--classifier", "elkan_noto", "--prior-estimator", "none")
    assert payload["prior"]["source"] == "none"


@pytest.mark.integration
def test_param_prior_estimator_none_missing_prior_exits_one(tmp_path, rng, capsys):
    """--prior-estimator none with a prior-requiring method and no prior → exit 1."""
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, rng, "--classifier", "nnpu", "--prior-estimator", "none")
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.integration
def test_basic_auto_mode_without_prior_estimator(tmp_path, rng):
    """auto + --prior-estimator none degrades to a no-prior recommendation.

    Regression guard: auto mode used to hardcode needs_prior=True, so the
    documented 'none disables estimation' option always exited 1 with the
    default classifier.
    """
    data, labels, _ = _write_demo(tmp_path, rng)
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
            "--prior-estimator",
            "none",
            "--quiet",
        ]
    )
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["prior"]["source"] == "none"


@pytest.mark.integration
def test_edge_headerless_csv_reports_error(tmp_path, capsys):
    """A headerless numeric CSV is rejected, not silently truncated."""
    rng = np.random.RandomState(42)
    X, y_pu, _ = make_scar_dataset(n=10, c=0.5, n_features=5, separation=4.0, random_state=rng)
    np.savetxt(tmp_path / "x.csv", X, delimiter=",")
    np.savetxt(tmp_path / "y.csv", y_pu, delimiter=",", fmt="%d")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data",
                str(tmp_path / "x.csv"),
                "--labels",
                str(tmp_path / "y.csv"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "header" in capsys.readouterr().err


@pytest.mark.integration
def test_edge_multi_column_labels_reports_error(tmp_path, rng, capsys):
    """A multi-column labels CSV is rejected instead of silently using col 0."""
    data, _, _ = _write_demo(tmp_path, rng)
    wide = tmp_path / "wide.csv"
    pd.DataFrame({"label": [1, 0, 1, 0, 1], "extra": [0, 0, 1, 1, 0]}).to_csv(wide, index=False)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data",
                str(data),
                "--labels",
                str(wide),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 1
    assert "single column" in capsys.readouterr().err
