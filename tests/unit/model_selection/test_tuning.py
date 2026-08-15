# ruff: noqa: N803, N806, S101

"""Tests for the PU-aware hyperparameter search."""

from types import SimpleNamespace

import pytest

from pu_toolbox.model_selection import PUTuner
from pu_toolbox.workflows import PipelineError
from tests.helpers import make_scar_data

pytestmark = pytest.mark.unit


def test_basic_tuner_returns_fitted_best_report(rng):
    X, y_pu, _ = make_scar_data(rng, n=100, separation=4.0)
    result = PUTuner(
        classifier="upu",
        param_grid={"reg_lambda": [0.001, 0.1], "max_iter": [50]},
        scoring="pu_risk",
        cv=2,
        random_state=42,
    ).fit(X, y_pu, class_prior=0.4)

    assert result.scoring == "pu_zero_one_risk"
    assert result.higher_is_better is False
    assert len(result.trials) == 2
    assert all(trial.status == "ok" for trial in result.trials)
    assert result.best_params in (
        {"max_iter": 50, "reg_lambda": 0.001},
        {"max_iter": 50, "reg_lambda": 0.1},
    )
    assert result.best_report.final_model.predict(X[:3]).shape == (3,)


def test_param_tuner_rejects_auto_classifier_and_empty_grid():
    with pytest.raises(ValueError, match="explicit"):
        PUTuner(classifier="auto", param_grid={"x": [1]})
    with pytest.raises(ValueError, match="param_grid"):
        PUTuner(classifier="upu", param_grid={"reg_lambda": []})


def test_edge_tuner_rejects_unavailable_scoring_and_isolates_runtime_failure(rng, monkeypatch):
    X, y_pu, _ = make_scar_data(rng, n=80, separation=4.0)
    tuner = PUTuner(
        classifier="upu",
        param_grid={"reg_lambda": [0.01]},
        scoring="pu_auc_roc",
        cv=2,
    )
    with pytest.raises(PipelineError, match="No tuning candidate"):
        tuner.fit(X, y_pu, class_prior=0.4)

    from pu_toolbox.model_selection import tuning as tuning_module

    class FakePipeline:
        def __init__(self, *, classifier_params, metrics, **kwargs):
            self.params = classifier_params
            self.metrics = ["pu_zero_one_risk"]

        def fit_evaluate(self, X, y_pu, *, y_true=None, class_prior=None):
            if self.params["backend_fails"]:
                raise RuntimeError("simulated backend failure")
            metric = SimpleNamespace(available=True, mean=0.25, reason=None)
            return SimpleNamespace(cv_metrics={"pu_zero_one_risk": metric})

    monkeypatch.setattr(tuning_module, "PUPipeline", FakePipeline)
    result = PUTuner(
        classifier="upu",
        param_grid={"backend_fails": [True, False]},
    ).fit(X, y_pu, class_prior=0.4)
    assert [trial.status for trial in result.trials] == ["failed", "ok"]
    assert "simulated backend failure" in result.trials[0].error


def test_deterministic_parameter_grid_order():
    kwargs = {
        "classifier": "upu",
        "param_grid": {"reg_lambda": [0.1, 0.01], "max_iter": [50, 100]},
    }
    assert PUTuner(**kwargs).param_grid == PUTuner(**kwargs).param_grid
