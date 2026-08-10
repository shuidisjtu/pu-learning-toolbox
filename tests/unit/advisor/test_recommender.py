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

    def test_basic_sar_diagnostic_boosts_sar(self):
        """A SAR-like (at_risk) diagnostic must flip the ranking between
        SAR-aware and SCAR-only methods, while a plausible (SCAR)
        diagnostic keeps SCAR methods ahead.

        Regression guard: the old test only asserted that SAR methods
        appear in top-15 (true regardless of the assumption dimension),
        so a dead SAR boost would go unnoticed.
        """
        from pu_toolbox.preprocessing import make_sar_dataset

        scar_X, scar_y, scar_yt, _ = make_sar_dataset(
            n_samples=2000,
            n_features=5,
            class_prior=0.5,
            separation=2.0,
            mechanism="scar",
            label_frequency=0.5,
            random_state=7,
        )
        sar_X, sar_y, sar_yt, _ = make_sar_dataset(
            n_samples=2000,
            n_features=5,
            class_prior=0.5,
            separation=2.0,
            mechanism="linear",
            label_frequency=0.5,
            strength=3.0,
            random_state=7,
        )
        scar_scores = {
            c.name: c.score
            for c in recommend_from_profile(
                profile_pu_data(scar_X, scar_y, y_true=scar_yt, random_state=42),
                class_prior=0.5,
                top_k=15,
            ).candidates
        }
        sar_scores = {
            c.name: c.score
            for c in recommend_from_profile(
                profile_pu_data(sar_X, sar_y, y_true=sar_yt, random_state=42),
                class_prior=0.5,
                top_k=15,
            ).candidates
        }
        assert sar_scores["pusb"] > sar_scores["upu"]  # at_risk: SAR boosted
        assert scar_scores["pusb"] < scar_scores["upu"]  # plausible: SCAR ahead

    def test_at_risk_observed_mixture_does_not_boost_sar(self):
        """Without y_true the at-risk signal is non-identifying: strong
        class separation on SCAR data must not steer users to SAR methods.

        Regression guard: PUSB/LBE used to top the list on plain SCAR
        demo data (observed-mixture AUC ~1.0 > 0.65 -> at_risk -> SAR cap),
        so a SCAR user's first recommendation was a SAR-only method.
        """
        from pu_toolbox.preprocessing import make_sar_dataset

        scar_X, scar_y, _, _ = make_sar_dataset(
            n_samples=2000,
            n_features=5,
            class_prior=0.5,
            separation=2.0,
            mechanism="scar",
            label_frequency=0.5,
            random_state=7,
        )
        # No y_true: the pipeline cannot audit positives, so the diagnostic
        # stays in the non-identifying observed-mixture evidence mode.
        scores = {
            c.name: c.score
            for c in recommend_from_profile(
                profile_pu_data(scar_X, scar_y, random_state=42),
                class_prior=0.5,
                top_k=15,
            ).candidates
        }
        assert scores["pusb"] <= scores["upu"]  # non-identifying at_risk: no SAR boost
        pusb = next(
            c for c in recommend_from_profile(
                profile_pu_data(scar_X, scar_y, random_state=42),
                class_prior=0.5,
                top_k=15,
            ).candidates
            if c.name == "pusb"
        )
        assert not any("Strong assumption match" in r for r in pusb.reasons)

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

    def test_basic_small_data_high_cost_ranks_below_low_cost(self):
        """On small SCAR data, the 3000-epoch LLSVM must rank below fast methods.

        Regression guard for the training-cost dimension: auto mode used to
        pick LLSVM (rank 1, ~30s per run); uPU / Elkan-Noto are closed-form
        and must now outrank it on a SCAR/plausible profile.
        """
        from pu_toolbox.preprocessing import make_scar_dataset

        X, y_pu, y_true = make_scar_dataset(
            n=100, c=0.5, n_features=5, separation=4.0, random_state=42
        )
        # y_true makes the SCAR diagnostic plausible (without it the
        # profile is at_risk and SCAR-only methods like uPU are penalized
        # by the assumption dimension, masking the cost effect).
        result = recommend_from_profile(
            profile_pu_data(X, y_pu, y_true=y_true, random_state=42),
            class_prior=0.5,
            top_k=15,
        )
        scores = {c.name: c.score for c in result.candidates}
        reasons = {c.name: c.reasons for c in result.candidates}
        assert scores["llsvm"] < scores["upu"]
        assert scores["llsvm"] < scores["elkan_noto"]
        assert "Fast on small datasets" in reasons["upu"]
        assert "High training cost on small datasets" in reasons["llsvm"]


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
