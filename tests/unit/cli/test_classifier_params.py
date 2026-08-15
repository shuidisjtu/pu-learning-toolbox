# ruff: noqa: S101

"""CLI parser coverage for classifier constructor parameters."""

import pytest

from pu_toolbox.cli.run import _parse_classifier_params

pytestmark = pytest.mark.unit


def test_basic_classifier_param_parsing():
    assert _parse_classifier_params(["reg_lambda=0.25"]) == {"reg_lambda": 0.25}


def test_param_classifier_values_accept_json_literals():
    assert _parse_classifier_params(["enabled=true", "layers=[64,32]"]) == {
        "enabled": True,
        "layers": [64, 32],
    }


def test_edge_classifier_param_requires_key_value_syntax():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _parse_classifier_params(["reg_lambda"])
    with pytest.raises(ValueError, match="non-empty key"):
        _parse_classifier_params(["=1"])


def test_deterministic_duplicate_classifier_param_uses_last_value():
    assert _parse_classifier_params(["max_iter=10", "max_iter=20"]) == {"max_iter": 20}
