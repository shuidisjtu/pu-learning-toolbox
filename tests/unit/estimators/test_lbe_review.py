"""LBE source equations and estimator safety, without executing uploaded code."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from pu_toolbox.core.exceptions import NotFittedError
from pu_toolbox.estimators.bias_aware.lbe import LBEClassifier

pytestmark = [pytest.mark.unit, pytest.mark.math]


@pytest.fixture
def observations():
    features = np.random.default_rng(7).normal(size=(40, 3))
    labels = np.r_[np.ones(12), np.zeros(28)]
    return features, labels


def test_basic_estep_matches_independent_joint_normalization():
    # Uploaded em.py normalizes [P(Y=0,S|X), P(Y=1,S|X)].
    observed = np.array([1, 0, 0, 1])
    posterior = np.array([0.7, 0.2, 0.9, 0.1])
    propensity = np.array([0.3, 0.8, 0.2, 0.9])
    expected = []
    for s, p, c in zip(observed, posterior, propensity, strict=True):
        joint_positive = p * (c if s else 1 - c)
        joint_negative = (1 - p) * (1 - s)
        expected.append(joint_positive / (joint_positive + joint_negative))
    np.testing.assert_allclose(
        LBEClassifier._positive_posterior(observed, posterior, propensity), expected
    )


def test_edge_estep_near_probability_bounds_is_finite():
    p = np.array([1e-5, 1 - 1e-5, 1 - 1e-5])
    c = np.array([1 - 1e-5, 1 - 1e-5, 1e-5])
    q = LBEClassifier._positive_posterior(np.zeros(3), p, c)
    assert np.isfinite(q).all()
    assert ((q >= 0) & (q <= 1)).all()


def test_basic_weighted_mstep_matches_soft_target_loss(observations):
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, labels = observations
    q = np.where(labels == 1, 1.0, 0.3)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=2.0, max_iter=1000))
    LBEClassifier._weighted_binary_fit(model, features, q)
    scaled = model[0].transform(features)
    fitted = model[1]
    residual = expit(fitted.decision_function(scaled)) - q
    # sklearn minimizes weighted summed CE + ||coef||²/(2*C).
    np.testing.assert_allclose(scaled.T @ residual + fitted.coef_[0] / 2, 0, atol=0.01)
    assert abs(residual.sum()) < 0.01


@pytest.mark.parametrize(
    "method", ["predict", "predict_proba", "predict_label_proba", "decision_function"]
)
def test_param_unfitted_public_prediction_raises_domain_error(method):
    with pytest.raises(NotFittedError):
        getattr(LBEClassifier(), method)(np.zeros((2, 3)))


@pytest.mark.parametrize(
    "params",
    [
        {"n_em_iter": 0},
        {"n_em_iter": 1.5},
        {"n_em_iter": True},
        {"max_iter": 0},
        {"max_iter": 2.5},
        {"max_iter": True},
        {"C": 0},
        {"C": float("nan")},
        {"C": float("inf")},
        {"C": "1"},
        {"C": True},
    ],
)
def test_param_invalid_optimizer_settings_rejected(observations, params):
    with pytest.raises(ValueError):
        LBEClassifier(**params).fit(*observations)


@pytest.mark.parametrize("prior", [0, 1, -0.1, np.nan, np.inf, True, [0.3], "0.3"])
def test_param_invalid_initialization_prior_rejected(observations, prior):
    with pytest.raises(ValueError):
        LBEClassifier().fit(*observations, class_prior=prior)


def test_basic_diagnostics_and_label_probability_use_final_models(observations):
    features, labels = observations
    model = LBEClassifier(n_em_iter=3).fit(features, labels, class_prior=0.3)
    p = np.clip(model.predict_proba(features)[:, 1], 1e-5, 1 - 1e-5)
    c = np.clip(model.propensity_model_.predict_proba(features)[:, 1], 1e-5, 1 - 1e-5)
    np.testing.assert_allclose(
        model._latent_positive_probability_, model._positive_posterior(labels, p, c)
    )
    np.testing.assert_allclose(
        model.predict_label_proba(features),
        model.predict_proba(features)[:, 1] * model.propensity_model_.predict_proba(features)[:, 1],
    )
    assert model.n_iter_ == 3
    assert len(model.observed_log_likelihood_history_) == model.n_iter_
    expected = np.sum(labels * np.log(p * c) + (1 - labels) * np.log1p(-p * c))
    assert model.observed_log_likelihood_history_[-1] == pytest.approx(expected)
    assert model._class_prior == pytest.approx(p.mean())


def test_edge_failed_refit_invalidates_old_fitted_state(observations):
    model = LBEClassifier(n_em_iter=1).fit(*observations)
    model.set_params(C=np.nan)
    with pytest.raises(ValueError):
        model.fit(*observations)
    with pytest.raises(NotFittedError):
        model.predict_proba(observations[0])


def test_edge_fixed_posterior_reports_early_stop(observations, monkeypatch):
    monkeypatch.setattr(
        LBEClassifier,
        "_positive_posterior",
        staticmethod(lambda s, p, c: np.where(s == 1, 1.0, 0.3)),
    )
    model = LBEClassifier(n_em_iter=5).fit(*observations, class_prior=0.3)
    assert model.n_iter_ == 1
    assert model.converged_ is True
    assert len(model.observed_log_likelihood_history_) == 1


def test_determ_clone_pickle_and_label_conventions(observations):
    features, labels = observations
    model = LBEClassifier(n_em_iter=3).fit(features, labels)
    second = clone(model).fit(features, np.where(labels == 1, 1, -1))
    np.testing.assert_array_equal(model.predict_proba(features), second.predict_proba(features))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        model.predict_label_proba(features), restored.predict_label_proba(features)
    )


def test_basic_source_variants_remain_distinct_and_preflight_uses_actual_dataset_directory():
    root = Path(__file__).resolve().parents[3]
    lock = json.loads(
        (root / "benchmarks/assigned_methods/configs/official_sources.lock.json").read_text(
            encoding="utf-8"
        )
    )["sources"]["lbe"]
    receipt = json.loads(
        (root / "docs/research/pu_survey/data/lbe_source_review_20261012.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock["sha256"] == receipt["comparison_to_official_lock"]["official_sha256"]
    assert lock["sha256"] != lock["user_supplied_variant"]["sha256"]
    assert lock["user_supplied_variant"]["sha256"] == receipt["archive"]["sha256"]
    upstream = receipt["official_upstream_verification_20261019"]
    assert upstream["download_sha256"] == lock["sha256"]
    assert upstream["matches_locked_official_sha256"] is True
    assert upstream["matches_user_supplied_archive"] is False
    assert upstream["license_interpretation"].startswith("The author's publication page says")
    assert lock["user_supplied_variant"]["mlp_recipe"]["pretrain_steps_per_model"] == 20
    lf_adam = receipt["files_sha256"]["LBE_LF/toydata/MStep_Adam.m"]
    assert len(lf_adam) == 64
    finding = receipt["source_behavior_findings_20261019"][0]
    assert finding["status"] == "confirmed_source_control_flow_not_executed"
    assert "overwrites its incoming Theta1/Theta2" in finding["claim"]
    assert "LBE_LF/breast/MStep_Adam.m:5-7,14" in finding["locations"]
    assert "LBE_LF/toydata/MStep_Adam.m:5-7,14-15" in finding["locations"]
    ledger = json.loads(
        (root / "pu_toolbox/experiment/method_ledger.json").read_text(encoding="utf-8")
    )
    neural = ledger["methods"]["lbe_neural"]
    assert (
        "2026-10-19 对作者发表页当前链接"
        in ledger["methods"]["lbe"]["code_version"]["source_status"]
    )
    assert neural["source_behavior_review_ref"] == (
        "docs/research/pu_survey/data/lbe_source_review_20261012.json"
    )
    assert any("覆盖传入的 Theta 参数" in item for item in neural["uncertainty"])
    config = json.loads(
        (root / "benchmarks/assigned_methods/configs/official/lbe.json").read_text(encoding="utf-8")
    )
    assert config["runner"]["pretrain_steps_per_model"] == 100
    for source in config["execution_requirements"]["required_source_files"]:
        assert source in receipt["files_sha256"]


def test_param_lbe_os_likelihood_is_not_ts_calibrated():
    from pu_toolbox.experiment.method_ledger import load_ledger
    from pu_toolbox.experiment.training_views import resolve_training_view

    ledger = load_ledger()
    assert ledger["methods"]["lbe"]["calibration_applied"] is False
    assert (
        resolve_training_view(ledger, "lbe", None, is_oracle=False, estimator_class=LBEClassifier)
        == "os"
    )
    with pytest.raises(ValueError, match="native to 'os'"):
        resolve_training_view(ledger, "lbe", "ts", is_oracle=False, estimator_class=LBEClassifier)
