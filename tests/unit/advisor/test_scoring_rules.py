# ruff: noqa: N802, N803, N806, E501

"""Unit tests for recommendation scoring rules and recommender edge
behaviors (training cost, assumption bands, sparse input, prior warnings)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pu_toolbox.advisor.rules import (
    DEFAULT_CONFIG,
    ScoringConfig,
    _score_training_cost,
    global_warnings,
    score_method,
)
from pu_toolbox.core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    Maturity,
    SourceStatus,
    TrainingCost,
)
from pu_toolbox.preprocessing import profile_pu_data
from pu_toolbox.registry import recommend_from_profile
from pu_toolbox.registry.metadata import AlgorithmMetadata


def _stub_meta(cost: TrainingCost) -> AlgorithmMetadata:
    return AlgorithmMetadata(name="stub", paper="stub", training_cost=cost)


def _assumption_stub(
    assumption: Assumption,
    family: AlgorithmFamily = AlgorithmFamily.RISK_ESTIMATION,
    cost: TrainingCost = TrainingCost.LOW,
) -> AlgorithmMetadata:
    """Metadata identical except for the assumption claim (isolates the
    assumption dimension when scoring)."""
    return AlgorithmMetadata(
        name="stub",
        paper="stub",
        assumption=[assumption],
        family=family,
        backend=Backend.NUMPY,
        maturity=Maturity.RESEARCH,
        source_status=SourceStatus.OFFICIAL_EXACT,
        requires_class_prior=False,
        training_cost=cost,
    )


def _diagnostic_profile(status: str) -> SimpleNamespace:
    return SimpleNamespace(
        summary={"n_samples": 100},
        selection_diagnostic={"status": status},
        issues=[],
    )


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"maturity_scores": {"stable": 100.0}},
        {"source_scores": {"official_exact": 50.0}},
    ],
)
def test_param_score_tables_capped_by_anchors(kwargs):
    """Table values above their dimension anchor break the 0-100 output
    contract (raw can exceed max_raw_score, e.g. a 170.9 score).

    Regression guard: the ``*_max`` anchors used to be inert knobs with
    no validation linking them to the ``*_scores`` tables.
    """
    with pytest.raises(ValueError, match="must not exceed"):
        ScoringConfig(**kwargs)


@pytest.mark.unit
def test_basic_at_risk_boosts_sar_over_scar():
    """The assumption dimension must actually reorder methods: under an
    at-risk diagnostic a SAR-aware method outscores an identical SCAR-only
    one, and the reverse under a plausible diagnostic."""
    at_risk = _diagnostic_profile("at_risk")
    s_sar, _ = score_method(_assumption_stub(Assumption.SAR), at_risk, None, False, DEFAULT_CONFIG)
    s_scar, _ = score_method(
        _assumption_stub(Assumption.SCAR), at_risk, None, False, DEFAULT_CONFIG
    )
    assert s_sar > s_scar

    plausible = _diagnostic_profile("plausible")
    p_sar, _ = score_method(
        _assumption_stub(Assumption.SAR), plausible, None, False, DEFAULT_CONFIG
    )
    p_scar, _ = score_method(
        _assumption_stub(Assumption.SCAR), plausible, None, False, DEFAULT_CONFIG
    )
    assert p_scar > p_sar


@pytest.mark.unit
def test_edge_sparse_input_warns_explicitly():
    """No registered method supports sparse input, so a sparse profile
    must yield an explicit global warning instead of silently returning
    zero candidates."""
    from scipy.sparse import csr_matrix

    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y_pu = np.array([1] * 20 + [0] * 80)
    profile = profile_pu_data(csr_matrix(X), y_pu, random_state=42)
    result = recommend_from_profile(profile, top_k=15)
    assert len(result.candidates) == 0
    assert any("sparse" in w for w in result.global_warnings)
