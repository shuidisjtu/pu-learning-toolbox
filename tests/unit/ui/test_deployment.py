# ruff: noqa: N803, N806

"""Dependency-light tests for UI deployment orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.ui import analyze_deployment_window

pytestmark = pytest.mark.unit


class _ProbabilityModel:
    def predict_proba(self, X):
        probability = 1.0 / (1.0 + np.exp(-np.asarray(X)[:, 0]))
        return np.column_stack([1.0 - probability, probability])


def _data(seed=5):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(60, 3))
    target = source + rng.normal(0.2, 0.05, source.shape)
    labels = np.zeros(60, dtype=int)
    labels[:15] = 1
    return source, target, labels


def test_basic_deployment_window_combines_monitor_and_review():
    source, target, labels = _data()
    result = analyze_deployment_window(
        reference_X=source,
        reference_y_pu=labels,
        target_X=target,
        target_y_pu=labels,
        model=_ProbabilityModel(),
        window_id="w1",
        cv=2,
        query_budget=3,
    )
    assert result.history["n_windows"] == 1
    assert result.review.summary["n_queries"] == 3
    assert result.window.window_id == "w1"


def test_param_previous_history_is_resumed_in_ui_helper():
    source, target, labels = _data()
    first = analyze_deployment_window(
        reference_X=source,
        reference_y_pu=labels,
        target_X=target,
        target_y_pu=labels,
        model=_ProbabilityModel(),
        window_id="w1",
        cv=2,
    )
    second = analyze_deployment_window(
        reference_X=source,
        reference_y_pu=labels,
        target_X=target,
        target_y_pu=labels,
        model=_ProbabilityModel(),
        window_id="w2",
        history_payload=first.history,
        cv=2,
    )
    assert second.history["n_windows"] == 2


def test_edge_incompatible_history_configuration_fails():
    source, target, labels = _data()
    first = analyze_deployment_window(
        reference_X=source,
        reference_y_pu=labels,
        target_X=target,
        target_y_pu=labels,
        model=_ProbabilityModel(),
        window_id="w1",
        cv=2,
    )
    with pytest.raises(ValueError, match="configuration"):
        analyze_deployment_window(
            reference_X=source,
            reference_y_pu=labels,
            target_X=target,
            target_y_pu=labels,
            model=_ProbabilityModel(),
            window_id="w2",
            history_payload=first.history,
            cv=3,
        )


def test_determ_same_inputs_produce_same_scores_and_alerts():
    source, target, labels = _data()
    values = []
    for _ in range(2):
        result = analyze_deployment_window(
            reference_X=source,
            reference_y_pu=labels,
            target_X=target,
            target_y_pu=labels,
            model=_ProbabilityModel(),
            window_id="w1",
            cv=2,
            query_budget=4,
        )
        values.append(
            (result.window.domain_auc, result.window.alert_codes, result.review.query_indices)
        )
    assert values[0][0:2] == values[1][0:2]
    np.testing.assert_array_equal(values[0][2], values[1][2])
