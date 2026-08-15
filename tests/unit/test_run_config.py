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
            "cv": 3,
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
        ({"cv": 1}, "cv"),
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
