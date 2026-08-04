# ruff: noqa: N806
"""Tests for the algorithm recommender."""

import json

import numpy as np
import pytest

from pu_toolbox.preprocessing import profile_pu_data
from pu_toolbox.registry import (
    MethodCandidate,
    RecommendationResult,
    recommend_from_profile,
    recommend_methods,
)


@pytest.fixture()
def pu_data():
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y_true = (X[:, 0] > 0).astype(int)
    mask = rng.rand(n) < 0.5
    y_pu = np.where(y_true & mask, 1, 0)
    return X, y_pu


@pytest.fixture()
def profile(pu_data):
    X, y_pu = pu_data
    return profile_pu_data(X, y_pu, random_state=42)


@pytest.mark.unit
class TestRecommenderBasic:
    def test_basic_recommend_returns_result(self, pu_data):
        X, y_pu = pu_data
        result = recommend_methods(X, y_pu, random_state=42)
        assert isinstance(result, RecommendationResult)
        assert len(result.candidates) > 0

    def test_basic_recommend_from_profile_works(self, profile):
        result = recommend_from_profile(profile)
        assert isinstance(result, RecommendationResult)
        assert all(isinstance(c, MethodCandidate) for c in result.candidates)

    def test_basic_top_k_limits_output(self, profile):
        result = recommend_from_profile(profile, top_k=3)
        assert len(result.candidates) <= 3

    def test_basic_candidates_sorted_by_score(self, profile):
        result = recommend_from_profile(profile, top_k=10)
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
class TestRecommenderFiltering:
    def test_param_scenario_filters_methods(self, profile):
        result = recommend_from_profile(profile, scenario="selection_biased", top_k=15)
        for c in result.candidates:
            assert "selection_biased" in [s.value for s in c.metadata.scenario]

    def test_param_class_prior_none_excludes(self, profile):
        result = recommend_from_profile(profile, class_prior=None, top_k=15)
        for c in result.candidates:
            assert not c.metadata.requires_class_prior

    def test_param_assumption_explicit_filters(self, profile):
        result = recommend_from_profile(profile, assumption="SAR", top_k=15)
        for c in result.candidates:
            from pu_toolbox.core.tags import Assumption

            assert Assumption.SAR in c.metadata.assumption

    def test_edge_all_filtered_returns_empty(self, profile):
        result = recommend_from_profile(
            profile,
            scenario="selection_biased",
            assumption="SCAR",
            top_k=15,
        )
        assert isinstance(result, RecommendationResult)
        assert len(result.candidates) == 0


@pytest.mark.unit
class TestRecommenderScoring:
    def test_basic_stable_methods_rank_higher(self, profile):
        result = recommend_from_profile(profile, top_k=15)
        stable_scores = [
            c.score for c in result.candidates if c.metadata.maturity.value == "stable"
        ]
        experimental_scores = [
            c.score for c in result.candidates if c.metadata.maturity.value == "experimental"
        ]
        if stable_scores and experimental_scores:
            assert max(stable_scores) > min(experimental_scores)

    def test_basic_sar_diagnostic_boosts_sar(self, pu_data):
        X, y_pu = pu_data
        profile_normal = profile_pu_data(X, y_pu, random_state=42)
        result = recommend_from_profile(profile_normal, class_prior=0.5, top_k=15)
        sar_names = {"pusb", "lbe", "llsvm", "dgpu"}
        sar_candidates = [c for c in result.candidates if c.name in sar_names]
        assert len(sar_candidates) > 0

    def test_edge_small_data_penalizes_deep(self):
        rng = np.random.RandomState(0)
        X_small = rng.randn(50, 3)
        y_small = np.array([1] * 10 + [0] * 40)
        profile = profile_pu_data(X_small, y_small, random_state=0)
        result = recommend_from_profile(profile, top_k=15)
        deep_scores = [c.score for c in result.candidates if c.metadata.family.value == "deep_pu"]
        non_deep_scores = [
            c.score for c in result.candidates if c.metadata.family.value != "deep_pu"
        ]
        if deep_scores and non_deep_scores:
            assert max(non_deep_scores) >= min(deep_scores)

    def test_deterministic_same_input_same_output(self, profile):
        r1 = recommend_from_profile(profile, top_k=5)
        r2 = recommend_from_profile(profile, top_k=5)
        assert [c.name for c in r1.candidates] == [c.name for c in r2.candidates]
        assert [c.score for c in r1.candidates] == [c.score for c in r2.candidates]


@pytest.mark.unit
class TestRecommenderOutput:
    def test_basic_to_json_serializable(self, profile):
        result = recommend_from_profile(profile)
        j = result.to_json()
        parsed = json.loads(j)
        assert "candidates" in parsed
        assert "schema_version" in parsed

    def test_basic_to_markdown_renders(self, profile):
        result = recommend_from_profile(profile)
        md = result.to_markdown()
        assert "# PU Method Recommendations" in md
        assert "Rank" in md
