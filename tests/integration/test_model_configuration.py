# ruff: noqa: N803, N806, S101

"""Integration coverage for named classifier configuration and tuning."""

import pytest

from pu_toolbox.model_selection import PUTuner
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
