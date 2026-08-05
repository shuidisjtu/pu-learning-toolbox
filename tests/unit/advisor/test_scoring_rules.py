# ruff: noqa: N802, N803, N806, E501

"""Unit tests for the training-cost scoring dimension (rules._score_training_cost)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pu_toolbox.advisor.rules import DEFAULT_CONFIG, _score_training_cost, global_warnings
from pu_toolbox.core.tags import TrainingCost
from pu_toolbox.registry.metadata import AlgorithmMetadata


def _stub_meta(cost: TrainingCost) -> AlgorithmMetadata:
    return AlgorithmMetadata(name="stub", paper="stub", training_cost=cost)


def _stub_profile(n_samples: int | None) -> SimpleNamespace:
    summary = {} if n_samples is None else {"n_samples": n_samples}
    return SimpleNamespace(summary=summary)


def _stub_warning_profile() -> SimpleNamespace:
    """Profile shape global_warnings touches (selection diagnostic + issues)."""
    return SimpleNamespace(
        summary={"n_samples": 100},
        selection_diagnostic={"status": "inconclusive"},
        issues=[],
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "cost, expected_fraction",
    [
        (TrainingCost.LOW, 1.0),
        (TrainingCost.MEDIUM, 0.7),
        (TrainingCost.HIGH, 0.2),
        (TrainingCost.UNKNOWN, 0.6),
    ],
)
def test_basic_cost_fractions_small_data(cost, expected_fraction):
    """Small data (< 1000): low scores full cap, high only 0.2x."""
    score, _ = _score_training_cost(_stub_meta(cost), _stub_profile(100), DEFAULT_CONFIG)
    assert score == pytest.approx(expected_fraction * DEFAULT_CONFIG.cost_max)


@pytest.mark.unit
def test_basic_cost_fractions_large_data():
    """Large data (> 10000): high-cost penalty shrinks (0.7x vs 0.2x)."""
    small_score, _ = _score_training_cost(
        _stub_meta(TrainingCost.HIGH), _stub_profile(100), DEFAULT_CONFIG
    )
    large_score, _ = _score_training_cost(
        _stub_meta(TrainingCost.HIGH), _stub_profile(20000), DEFAULT_CONFIG
    )
    assert large_score > small_score


@pytest.mark.unit
def test_edge_cost_unknown_neutral_and_missing_n():
    """UNKNOWN scores neutrally at every band; missing n_samples is small."""
    mid = _score_training_cost(
        _stub_meta(TrainingCost.UNKNOWN), _stub_profile(5000), DEFAULT_CONFIG
    )[0]
    missing = _score_training_cost(
        _stub_meta(TrainingCost.UNKNOWN), _stub_profile(None), DEFAULT_CONFIG
    )[0]
    assert mid == pytest.approx(0.6 * DEFAULT_CONFIG.cost_max)
    assert missing == pytest.approx(0.6 * DEFAULT_CONFIG.cost_max)

    _, reason_low = _score_training_cost(
        _stub_meta(TrainingCost.LOW), _stub_profile(100), DEFAULT_CONFIG
    )
    _, reason_high = _score_training_cost(
        _stub_meta(TrainingCost.HIGH), _stub_profile(100), DEFAULT_CONFIG
    )
    assert reason_low == "Fast on small datasets"
    assert reason_high == "High training cost on small datasets"


@pytest.mark.unit
def test_deterministic_cost_score_stable():
    """Same input twice → identical score and reason (pure function)."""
    meta = _stub_meta(TrainingCost.MEDIUM)
    profile = _stub_profile(100)
    s1, r1 = _score_training_cost(meta, profile, DEFAULT_CONFIG)
    s2, r2 = _score_training_cost(meta, profile, DEFAULT_CONFIG)
    assert s1 == s2
    assert r1 == r2


@pytest.mark.unit
def test_basic_prior_warning_only_for_user_supplied():
    """'user-supplied' warning must fire only for explicitly provided priors.

    Regression guard: pipeline passes the auto-estimated prior through the
    recommender, which used to mislabel it as user-supplied.
    """
    profile = _stub_warning_profile()
    estimated = global_warnings(profile, 0.5, class_prior_source="estimated")
    assert not any("user-supplied" in w for w in estimated)

    user = global_warnings(profile, 0.5, class_prior_source="user")
    assert any("user-supplied" in w for w in user)

    default = global_warnings(profile, 0.5)
    assert any("user-supplied" in w for w in default)
