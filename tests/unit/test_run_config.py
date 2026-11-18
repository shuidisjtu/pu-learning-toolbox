# ruff: noqa: S101

"""Portable run-configuration contract tests."""

import json

import pytest

from pu_toolbox.run_config import RunConfiguration

pytestmark = pytest.mark.unit


def test_basic_configuration_json_round_trip():
    config = RunConfiguration.from_mapping(
        {
            "schema_version": 1,
            "classifier": "upu",
            "classifier_params": {"loss": "logistic", "max_iter": 50},
            "prior_estimator": "pen_l1",
            "class_prior": 0.4,
            "evaluation_cv": 3,
            "metrics": ["pu_zero_one_risk"],
            "random_state": 7,
            "tuning_grid": {"reg_lambda": [0.01, 0.1]},
        }
    )
    restored = RunConfiguration.from_json(config.to_json())
    assert restored == config
    assert json.loads(config.to_json())["schema_version"] == 1


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"unknown": True}, "unknown"),
        ({"class_prior": 1.0}, "class_prior"),
        ({"evaluation_cv": 1}, "evaluation_cv"),
        ({"metrics": []}, "metrics"),
        ({"tuning_grid": {"x": []}}, "non-empty list"),
    ],
)
def test_edge_configuration_rejects_invalid_fields(patch, message):
    with pytest.raises(ValueError, match=message):
        RunConfiguration.from_mapping(patch)


def test_deterministic_configuration_serialization():
    config = RunConfiguration(classifier="upu", classifier_params={"z": 1, "a": 2})
    assert config.to_json() == config.to_json()
    assert config.to_json().index('"a"') < config.to_json().index('"z"')


def test_param_comparison_configuration_round_trip():
    config = RunConfiguration.from_mapping(
        {
            "comparison_classifiers": ["upu", "pnu"],
            "metrics": ["pu_zero_one_risk"],
        }
    )
    assert config.classifier is None
    assert config.comparison_classifiers == ("upu", "pnu")
    assert RunConfiguration.from_json(config.to_json()) == config
    with pytest.raises(ValueError, match="cannot run together"):
        RunConfiguration.from_mapping(
            {
                "comparison_classifiers": ["upu", "pnu"],
                "tuning_grid": {"reg_lambda": [0.1]},
            }
        )


@pytest.mark.parametrize(
    "value",
    [{}, {"classifier": "auto"}, {"classifier": "auto", "comparison_classifiers": ["upu", "nnpu"]}],
)
def test_edge_implicit_or_auto_classifier_rejected(value):
    with pytest.raises(ValueError, match="classifier"):
        RunConfiguration.from_mapping(value)


def test_edge_direct_configuration_rejects_auto_comparison():
    with pytest.raises(ValueError, match="comparison_classifiers"):
        RunConfiguration(comparison_classifiers=("upu", "auto"))


@pytest.mark.parametrize("name", ["AUTO", " Auto ", "\tauto\n"])
@pytest.mark.parametrize("comparison", [False, True])
def test_param_auto_variants_rejected_in_classifier_and_candidates(name, comparison):
    values = {"comparison_classifiers": ["upu", "nnpu"]} if comparison else {}
    with pytest.raises(ValueError, match="classifier"):
        RunConfiguration.from_mapping({**values, "classifier": name})
    with pytest.raises(ValueError, match="classifier"):
        RunConfiguration(classifier=name, **values)
    with pytest.raises(ValueError, match="comparison_classifiers"):
        RunConfiguration.from_mapping({"comparison_classifiers": ["upu", name]})
    with pytest.raises(ValueError, match="comparison_classifiers"):
        RunConfiguration(comparison_classifiers=("upu", name))
