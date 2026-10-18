"""Independent PU-only RP arithmetic and actual nested-fit role evidence."""

# Match the public feature naming convention.
# ruff: noqa: N803, N806

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.estimators.classic.rank_pruning import (
    RankPruningClassifier,
    confident_positive_noise_rate,
    rank_pruning_pu_mask,
)
from pu_toolbox.experiment.method_ledger import load_ledger
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
REVIEW_REF = "docs/research/pu_survey/data/rp_source_review_20261018.json"


def scalar_rate(labels, probabilities):
    positive = [p for s, p in zip(labels, probabilities, strict=True) if s == 1]
    threshold = sum(positive) / len(positive)
    selected = [s for s, p in zip(labels, probabilities, strict=True) if p >= threshold]
    return min(selected.count(0) / len(selected), 0.9999)


def scalar_prune(labels, probabilities, rho, minimum):
    observed = sum(labels) / len(labels)
    prior = observed / (1 - rho)
    inverse_noise = min(max(rho * prior / (1 - observed), 0), 0.9999)
    u = sorted([p for s, p in zip(labels, probabilities, strict=True) if s == 0], reverse=True)
    count = max(0, min(int(len(u) * inverse_noise), len(u) - minimum))
    removed = [
        s == 0 and count > 0 and p > u[count] for s, p in zip(labels, probabilities, strict=True)
    ]
    return np.asarray(removed), prior, inverse_noise


class ProbeClassifier(BaseEstimator):
    """Cloneable score lookup; logs actual fit/query identities, not RP metadata."""

    calls = []
    fail_call = None

    def __init__(self, class_order=(0, 1)):
        self.class_order = class_order

    def fit(self, X, y, sample_weight=None):
        assert not hasattr(self, "rows_"), "a previously fitted classifier was reused"
        type(self).calls.append(self)
        if type(self).fail_call == len(type(self).calls):
            raise RuntimeError("synthetic base fit failed")
        self.rows_ = X[:, 0].astype(int).copy()
        self.labels_ = y.copy()
        self.weights_ = None if sample_weight is None else sample_weight.copy()
        self.queries_ = []
        self.classes_ = np.asarray(self.class_order)
        return self

    def predict_proba(self, X):
        rows = X[:, 0].astype(int)
        if self.weights_ is None:
            assert not set(rows) & set(self.rows_), "OOF probabilities used training identities"
        self.queries_.append(rows.copy())
        positive = X[:, 1]
        return np.column_stack(
            [positive if label == 1 else 1 - positive for label in self.classes_]
        )


def data():
    scores = np.random.RandomState(3).randint(0, 9, 36) / 8
    return np.column_stack((np.arange(36), scores)), np.r_[np.ones(12, int), np.zeros(24, int)]


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.9, 0.9999])
@pytest.mark.parametrize("minimum", [1, 10, 30])
@pytest.mark.parametrize("ties", [False, True])
def test_basic_param_pruning_matches_independent_full_sort_and_inverse_noise(rho, minimum, ties):
    labels = np.r_[np.ones(8, int), np.zeros(24, int)]
    scores = np.r_[np.linspace(0.2, 1, 8), np.linspace(1, 0, 24)]
    if ties:
        scores = np.round(scores * 4) / 4
    expected_mask, expected_prior, expected_inverse = scalar_prune(labels, scores, rho, minimum)
    mask, prior, inverse = rank_pruning_pu_mask(labels, scores, rho, min_retained=minimum)
    np.testing.assert_array_equal(mask, expected_mask)
    assert prior == pytest.approx(expected_prior) and inverse == pytest.approx(expected_inverse)
    assert not mask[labels == 1].any()
    assert np.count_nonzero((labels == 0) & ~mask) >= min(minimum, 24)


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_edge_confident_count_boundary_is_inclusive_and_caps_near_one(score):
    labels = np.r_[1, np.zeros(20000, int)]
    probabilities = np.full(len(labels), score)
    assert confident_positive_noise_rate(labels, probabilities) == scalar_rate(
        labels, probabilities
    )
    assert confident_positive_noise_rate(labels, probabilities) == 0.9999


@pytest.mark.parametrize("folds", [2, 3, 4])
@pytest.mark.parametrize("known_noise", [None, 0.0, 0.3])
@pytest.mark.parametrize("class_order", [(0, 1), (1, 0)])
def test_determ_actual_crossfits_are_fresh_heldout_only_then_one_exact_weighted_refit(
    monkeypatch, folds, known_noise, class_order
):
    monkeypatch.setattr(ProbeClassifier, "calls", [])
    X, s = data()
    before = X.copy(), s.copy()
    template = ProbeClassifier(class_order)
    template.rows_ = np.array([-1])  # Fitted template state must not reach any clone.
    fitted = RankPruningClassifier(
        base_estimator=template,
        n_cv_folds=folds,
        frac_pos2neg=known_noise,
        min_retained=2,
        random_state=7,
    ).fit(X, s, class_prior=0.4)
    calls = ProbeClassifier.calls
    assert len(calls) == len({id(model) for model in calls}) == folds + 1
    heldout_all, rates = [], []
    for model, recorded in zip(calls[:-1], fitted.cv_fold_indices_, strict=True):
        assert model.weights_ is None and len(model.queries_) == 1
        heldout = model.queries_[0]
        assert set(model.rows_) == set(range(len(s))) - set(heldout)
        np.testing.assert_array_equal(recorded["train"], model.rows_)
        np.testing.assert_array_equal(recorded["heldout"], heldout)
        np.testing.assert_array_equal(model.labels_, s[model.rows_])
        heldout_all.extend(heldout)
        rates.append(scalar_rate(s[heldout].tolist(), X[heldout, 1].tolist()))
    assert sorted(heldout_all) == list(range(len(s)))
    np.testing.assert_array_equal(fitted.oof_probabilities_, X[:, 1])
    np.testing.assert_allclose(fitted.fold_noise_rates_, rates, rtol=0, atol=0)
    rho = sum(rates) / len(rates) if known_noise is None else known_noise
    assert fitted.rho1_ == pytest.approx(rho)
    if folds == 3 and known_noise is None:
        assert rates == pytest.approx([2 / 3, 1 / 3, 4 / 7])
        assert abs(rho - scalar_rate(s.tolist(), X[:, 1].tolist())) > 0.05
    mask, prior, inverse = scalar_prune(s.tolist(), X[:, 1].tolist(), rho, 2)
    np.testing.assert_array_equal(fitted.pruned_mask_, mask)
    final = calls[-1]
    np.testing.assert_array_equal(final.rows_, np.flatnonzero(~mask))
    np.testing.assert_array_equal(final.labels_, s[~mask])
    expected_weights = [1 / (1 - rho) if label == 1 else 1 for label in s[~mask]]
    np.testing.assert_allclose(final.weights_, expected_weights, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(fitted.final_sample_weights_, final.weights_)
    assert fitted.estimator_ is final and fitted.rho0_ == 0
    assert fitted.estimated_prior_ == pytest.approx(prior) and fitted.pi0_ == pytest.approx(inverse)
    np.testing.assert_array_equal(fitted.predict_proba(X)[:, 1], X[:, 1])
    np.testing.assert_array_equal(fitted.predict(X), X[:, 1] >= 0.5)
    assert fitted._class_prior is None and fitted.calibration_applied_ is False
    np.testing.assert_array_equal(template.rows_, [-1])
    np.testing.assert_array_equal(X, before[0])
    np.testing.assert_array_equal(s, before[1])


@pytest.mark.parametrize("known_noise", [None, 0.6, 0.9999])
def test_edge_degenerate_prior_is_diagnostic_not_clipped_to_a_valid_prior(monkeypatch, known_noise):
    monkeypatch.setattr(ProbeClassifier, "calls", [])
    X = np.column_stack((np.arange(24), np.full(24, 0.5)))
    s = np.r_[np.ones(12, int), np.zeros(12, int)]
    fitted = RankPruningClassifier(
        base_estimator=ProbeClassifier(), frac_pos2neg=known_noise, random_state=2
    ).fit(X, s)
    assert fitted.estimated_prior_ >= 1
    assert fitted.prior_diagnostic_ == "incompatible_noise_model_estimate_not_a_valid_prior"
    assert fitted.pi0_ == 0.9999 and not fitted.pruned_mask_.any()  # Strict ties retain all.
    assert fitted._class_prior is None and fitted.rho0_ == 0
    assert len(ProbeClassifier.calls) == 4


@pytest.mark.parametrize("fault", ["rows", "columns", "classes", "nan", "range", "sum"])
def test_param_invalid_base_probability_contract_fails_before_use(fault):
    p, classes = np.full((4, 2), 0.5), np.array([0, 1])
    if fault == "rows":
        p = p[:3]
    elif fault == "columns":
        p = p[:, :1]
    elif fault == "classes":
        classes = np.array([0, 2])
    elif fault == "nan":
        p[0, 0] = np.nan
    elif fault == "range":
        p[0] = [-0.1, 1.1]
    else:
        p[0] = [0.4, 0.4]
    estimator = SimpleNamespace(classes_=classes, predict_proba=lambda X: p)
    with pytest.raises(ValueError, match="base_estimator"):
        RankPruningClassifier._positive_proba(estimator, np.zeros((4, 2)))


@pytest.mark.parametrize("fail_call", [2, 4])
def test_edge_base_fit_failure_is_not_retried_and_cannot_expose_previous_model(
    monkeypatch, fail_call
):
    monkeypatch.setattr(ProbeClassifier, "calls", [])
    X, s = data()
    fitted = RankPruningClassifier(base_estimator=ProbeClassifier(), min_retained=2).fit(X, s)
    monkeypatch.setattr(ProbeClassifier, "calls", [])
    monkeypatch.setattr(ProbeClassifier, "fail_call", fail_call)
    with pytest.raises(RuntimeError, match="synthetic base fit failed"):
        fitted.fit(X, s)
    assert len(ProbeClassifier.calls) == fail_call
    for api in ("predict", "predict_proba", "decision_function", "score_samples"):
        with pytest.raises(NotFittedError):
            getattr(fitted, api)(X)


def test_basic_determ_default_logistic_path_and_clone_repeat_without_external_prior():
    X = np.random.RandomState(9).normal(size=(48, 3))
    s = np.r_[np.ones(16, int), np.zeros(32, int)]
    first = RankPruningClassifier(random_state=7).fit(X, s)
    second = clone(first).fit(X, s, class_prior=0.9)
    for field in ("oof_probabilities_", "pruned_mask_", "final_sample_weights_"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
    assert first.rho1_ == second.rho1_ and first._class_prior is second._class_prior is None


@pytest.mark.parametrize("score", [0.0, 0.1, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("positive_count", [1, 3, 9])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_edge_constant_score_roundoff_cannot_empty_the_confident_positive_set(
    score, positive_count, dtype
):
    s = np.r_[np.ones(positive_count, int), np.zeros(2 * positive_count, int)]
    probabilities = np.full(len(s), score, dtype=dtype)
    before = probabilities.copy()
    with np.errstate(invalid="raise", divide="raise"):
        rho = confident_positive_noise_rate(s, probabilities)
    assert rho == pytest.approx(2 / 3)
    np.testing.assert_array_equal(probabilities, before)


@pytest.mark.parametrize("known_noise", [None, 0.0, 0.3])
def test_edge_real_dummy_prior_cv_handles_nonbinary_constant_probability(known_noise):
    X = np.zeros((90, 2))
    s = np.r_[np.ones(9, int), np.zeros(81, int)]
    with np.errstate(invalid="raise", divide="raise"):
        fitted = RankPruningClassifier(
            base_estimator=DummyClassifier(strategy="prior"),
            n_cv_folds=3,
            frac_pos2neg=known_noise,
            random_state=7,
        ).fit(X, s)
    np.testing.assert_array_equal(fitted.oof_probabilities_, np.full(90, 0.1))
    np.testing.assert_allclose(fitted.fold_noise_rates_, [0.9, 0.9, 0.9])
    rho = 0.9 if known_noise is None else known_noise
    assert fitted.rho1_ == pytest.approx(rho)
    assert not fitted.pruned_mask_.any()  # All OOF scores tie at the pruning boundary.
    expected_positive = (9 / (1 - rho)) / (9 / (1 - rho) + 81)
    np.testing.assert_allclose(fitted.predict_proba(X)[:, 1], expected_positive)
    assert fitted._class_prior is None and fitted.rho0_ == 0.0


@pytest.mark.parametrize("seed", [0, 5, 91])
def test_determ_nonconstant_counts_remain_bitwise_equal_to_previous_mean_formula(seed):
    X, s = data()
    probabilities = np.random.RandomState(seed).uniform(size=len(s))
    threshold = probabilities[s == 1].mean()
    confident = probabilities >= threshold
    assert confident.sum() > 0
    old_rate = min(float(np.sum(confident & (s == 0)) / confident.sum()), 0.9999)
    assert confident_positive_noise_rate(s, probabilities) == old_rate


def test_basic_source_review_binds_code_ledger_and_pu_only_non_admission():
    review = json.loads((ROOT / REVIEW_REF).read_text(encoding="utf-8"))
    entry = load_ledger()["methods"]["rp"]
    assert entry["source_behavior_review_ref"] == REVIEW_REF
    assert review["method"] == "rp" and review["formal_admission"] is False
    assert (
        review["owner_signatures"] == []
        and review["review_status"] == "pending_method_owner_review"
    )
    assert "rp" not in load_protocol()["method_profiles"]
    source = review["source"]
    assert (
        source["commit"]
        == entry["code_version"]["upstream_commit"]
        == "40ae17cc50fcfeec8d7a345a1c5426036d2bdb1a"
    )
    assert source["repository"] == entry["code_version"]["upstream_url"]
    assert (
        source["source_status"] == "official_related" and source["executed_upstream_code"] is False
    )
    assert {record["path"]: record["sha256"] for record in source["files"]} == entry[
        "code_version"
    ]["source_files_sha256"]
    assert {record["path"] for record in source["files"]} == {
        "rankpruning/rankpruning.py",
        "LICENSE",
        "README.md",
        "setup.py",
    }
    implementation = ROOT / review["implementation_ref"]
    assert (
        hashlib.sha256(implementation.read_text(encoding="utf-8").encode()).hexdigest()
        == review["implementation_sha256"]
    )
    assert review["calibration"] == {
        "run_view": entry["run_view"],
        "calibration_applied": entry["calibration_applied"],
        "fixed_rho0": 0.0,
        "external_population_prior_used": False,
        "source_equivalence_claim": False,
    }
    assert review["engineering_scope"]["mathematical_algorithm_or_defaults_changed"] is False
    assert review["engineering_scope"]["mean_roundoff_guard_added"] is True
    assert review["engineering_scope"]["new_real_data_or_GPU_run"] is False
    assert type(review["calibration"]["fixed_rho0"]) is float
    assert source["license_observation"]["declared_license"] == "MIT"
    assert source["license_observation"]["author_code_vendored_executed_or_redistributed"] is False
    assert source["normalized_text_digest_policy"] == "utf8_text_universal_newlines_sha256"
    assert set(source["normalized_text_sha256"]) == {row["path"] for row in source["files"]}
    assert review["cost_contract"]["successful_fit_calls"] == "n_cv_folds+1"
    assert review["cost_contract"]["known_rho1_skips_oof_fits"] is False
    assert review["cost_contract"]["formal_budget_approved"] is False
    for ref in review["behavior_test_refs"]:
        file, name = ref.split("::")
        nodes = ast.parse((ROOT / file).read_text(encoding="utf-8")).body
        assert any(isinstance(n, ast.FunctionDef) and n.name == name for n in nodes)
