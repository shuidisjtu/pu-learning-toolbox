"""Regression tests for explicit population and evaluation priors."""

import numpy as np
import pytest

from pu_toolbox.diagnostics import analyze_pu_sensitivity, build_diagnostic_report
from pu_toolbox.metrics import pu_estimated_precision, pu_zero_one_risk
from pu_toolbox.preprocessing import make_gaussian_case_control_data, profile_pu_data

pytestmark = pytest.mark.unit


def test_basic_gaussian_population_design_and_audit():
    _, labels, truth, metadata = make_gaussian_case_control_data(
        n_p=5, n_u=20, population_prior=0.4, random_state=13
    )
    assert metadata["design_population_prior"] == 0.4
    assert metadata["empirical_unlabeled_prior"] == pytest.approx(truth[labels == 0].mean())
    assert metadata["empirical_training_table_prior"] == pytest.approx(truth.mean())


def test_basic_evaluation_prior_metrics_and_report():
    labels = np.array([1, 1, 0, 0, 0, 0])
    predictions = np.array([1, 0, 1, 0, 1, 0])
    features = np.arange(12).reshape(6, 2)
    risk = pu_zero_one_risk(labels, predictions, evaluation_prior=0.3, input_sampling="os")
    precision = pu_estimated_precision(labels, predictions, evaluation_prior=0.3)
    report = build_diagnostic_report(features, labels, y_pred=predictions, evaluation_prior=0.3)
    assert risk == pytest.approx(0.5)
    assert precision == pytest.approx(0.3)
    assert report.metrics["pu_zero_one_risk"].value == pytest.approx(risk)


def test_basic_determ_profile_and_sensitivity_population_priors():
    labels = np.array([1, 1, 0, 0, 0, 0])
    predictions = np.array([1, 0, 1, 0, 1, 0])
    features = np.arange(12).reshape(6, 2)
    profile = profile_pu_data(features, labels, evaluation_prior=0.4)
    assert profile.summary["evaluation_prior"] == 0.4
    analysis = analyze_pu_sensitivity(labels, predictions, population_priors=[0.4])
    assert analysis.points[0].population_prior == 0.4
    assert analysis.points[0].axis == "population_prior"
    assert not analysis.provenance["model_refit"]
    repeated = analyze_pu_sensitivity(labels, predictions, population_priors=[0.4])
    assert repeated.to_dict() == analysis.to_dict()


def test_basic_sensitivity_cli_accepts_population_grid():
    from pu_toolbox.cli import build_parser

    args = build_parser().parse_args(
        [
            "sensitivity",
            "--data",
            "x.csv",
            "--labels",
            "y.csv",
            "--population-priors",
            "0.3,0.4",
        ]
    )
    assert args.population_priors == "0.3,0.4"


def test_edge_old_ambiguous_keyword_is_not_an_alias():
    labels = np.array([1, 0])
    predictions = np.array([1, 0])
    with pytest.raises(TypeError):
        make_gaussian_case_control_data(class_prior=0.3)
    with pytest.raises(TypeError):
        pu_zero_one_risk(labels, predictions, class_prior=0.3)
    with pytest.raises(TypeError):
        profile_pu_data(np.ones((2, 1)), labels, class_prior=0.3)
    with pytest.raises(TypeError):
        analyze_pu_sensitivity(labels, predictions, class_priors=[0.3])


def test_param_evaluation_prior_targets_selected_pool():
    labels = np.array([1, 1, 0, 0, 0, 0])
    predictions = np.array([1, 0, 1, 0, 0, 0])
    os_risk = pu_zero_one_risk(labels, predictions, 0.3, input_sampling="os")
    ts_risk = pu_zero_one_risk(labels, predictions, 0.3, input_sampling="ts")
    assert os_risk == pytest.approx(1 / 3)
    assert ts_risk == pytest.approx(1 / 4)


def test_basic_new_report_schemas_do_not_claim_legacy_version():
    labels = np.array([1, 0, 0, 0])
    predictions = np.array([1, 0, 1, 0])
    report = build_diagnostic_report(np.arange(8).reshape(4, 2), labels, y_pred=predictions)
    sensitivity = analyze_pu_sensitivity(labels, predictions, population_priors=[0.5])
    assert report.to_dict()["schema_version"] == "2.0"
    assert sensitivity.to_dict()["schema_version"] == "2.0"
