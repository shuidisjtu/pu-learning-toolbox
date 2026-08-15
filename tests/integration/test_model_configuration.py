# ruff: noqa: N803, N806, S101

"""Integration coverage for named classifier configuration and tuning."""

import json

import pandas as pd
import pytest

from pu_toolbox.cli import main
from pu_toolbox.model_selection import PUModelComparator, PUTuner
from pu_toolbox.workflows import PipelineError, PUPipeline
from tests.helpers import make_scar_data

pytestmark = pytest.mark.integration


def test_basic_named_classifier_parameters_are_applied(rng):
    X, y_pu, _ = make_scar_data(rng, n=100, separation=4.0)
    report = PUPipeline(
        classifier="upu",
        classifier_params={"reg_lambda": 0.25, "max_iter": 50},
        cv=2,
    ).fit_evaluate(X, y_pu, class_prior=0.4)
    assert report.final_model.reg_lambda == pytest.approx(0.25)
    assert report.provenance["classifier_params"] == {
        "reg_lambda": 0.25,
        "max_iter": 50,
    }
    cv_only = PUPipeline(classifier="upu", cv=2).fit_evaluate(X, y_pu, class_prior=0.4, refit=False)
    assert cv_only.final_model is None
    assert cv_only.diagnostic is None
    assert cv_only.to_dict()["final_model"]["class"] is None


@pytest.mark.parametrize(
    ("tuning_grid", "comparison_classifiers", "classifier_params", "artifact"),
    [
        ({}, [], {"max_iter": 30}, None),
        ({"reg_lambda": [0.001, 0.1]}, [], {"max_iter": 30}, "tuning.json"),
        ({}, ["upu", "pusb"], {}, "comparison.json"),
    ],
)
def test_basic_cli_imports_ui_configuration(
    tmp_path, rng, tuning_grid, comparison_classifiers, classifier_params, artifact
):
    X, y_pu, _ = make_scar_data(rng, n=40, separation=4.0)
    data = tmp_path / "X.csv"
    labels = tmp_path / "y.csv"
    pd.DataFrame(X, columns=[f"x{index}" for index in range(X.shape[1])]).to_csv(data, index=False)
    pd.DataFrame({"label": y_pu}).to_csv(labels, index=False)
    config = tmp_path / "run.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "classifier": "upu",
                "classifier_params": classifier_params,
                "prior_estimator": "pen_l1",
                "class_prior": 0.4,
                "cv": 2,
                "metrics": ["pu_zero_one_risk"],
                "random_state": 9,
                "architecture": "mlp",
                "backbone": "cnn13",
                "device": "auto",
                "tuning_grid": tuning_grid,
                "scoring": "pu_zero_one_risk",
                "comparison_classifiers": comparison_classifiers,
            }
        ),
        encoding="utf-8",
    )
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
            "--config",
            str(config),
            "--quiet",
        ]
    )
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert set(payload["cv_metrics"]) == {"pu_zero_one_risk"}
    if classifier_params:
        assert payload["provenance"]["classifier_params"]["max_iter"] == 30
    expected = {artifact} if artifact else set()
    assert {
        name for name in ("tuning.json", "comparison.json") if (out / name).exists()
    } == expected


def test_param_required_constructor_value_can_be_supplied():
    pipe = PUPipeline(
        classifier="ldce",
        classifier_params={"flip_probability": 0.2},
    )
    assert pipe.classifier_params["flip_probability"] == pytest.approx(0.2)


def test_edge_classifier_parameters_reject_unknown_and_managed_names():
    with pytest.raises(PipelineError, match="unknown"):
        PUPipeline(classifier="upu", classifier_params={"not_a_parameter": 1})
    with pytest.raises(ValueError, match="pipeline-managed"):
        PUPipeline(classifier="upu", classifier_params={"class_prior": 0.4})
    with pytest.raises(ValueError, match="explicit classifier"):
        PUPipeline(classifier="auto", classifier_params={"max_iter": 10})


def test_deterministic_tuning_grid_selects_same_result(rng):
    X, y_pu, _ = make_scar_data(rng, n=80, separation=4.0)
    kwargs = {
        "classifier": "upu",
        "param_grid": {"reg_lambda": [0.001, 0.1], "max_iter": [100]},
        "cv": 2,
        "random_state": 42,
    }
    first = PUTuner(**kwargs).fit(X, y_pu, class_prior=0.4)
    second = PUTuner(**kwargs).fit(X, y_pu, class_prior=0.4)
    assert first.best_params == second.best_params
    assert first.best_score == pytest.approx(second.best_score)


def test_deterministic_model_comparison_returns_fitted_best_report(rng):
    X, y_pu, _ = make_scar_data(rng, n=60, separation=4.0)
    comparison = PUModelComparator(classifiers=["upu", "pusb"], cv=2, random_state=42).fit(
        X, y_pu, class_prior=0.4
    )
    assert comparison.best_classifier in {"upu", "pusb"}
    assert all(trial.status == "ok" for trial in comparison.trials)
    assert comparison.best_report.final_model is not None
