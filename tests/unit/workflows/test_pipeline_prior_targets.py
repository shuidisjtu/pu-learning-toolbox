"""Pipeline prior targets through estimation, training and reporting."""

# ruff: noqa: N803, N806

import json

import numpy as np
import pytest

from pu_toolbox.core.base import BasePriorEstimator, BasePUClassifier
from pu_toolbox.diagnostics import build_diagnostic_report
from pu_toolbox.metrics import pu_estimated_precision, pu_zero_one_risk
from pu_toolbox.preprocessing import profile_pu_data
from pu_toolbox.workflows import PUPipeline
from pu_toolbox.workflows._evaluation import compute_metric


class NativeUPrior(BasePriorEstimator):
    estimate_target = "native_unlabeled"
    estimation_pool = "native_unlabeled"

    def fit(self, X, y_pu):
        return self

    def estimate(self):
        return 0.25


class RecordingClassifier(BasePUClassifier):
    requires_class_prior = True
    class_prior_target = "native_unlabeled"

    def __init__(self, class_prior=None):
        super().__init__()
        self.class_prior = class_prior

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        self.received_prior_ = class_prior
        self._is_fitted = True
        return self

    def _predict(self, X):
        return (X[:, 0] > 0).astype(int)

    def _decision_function(self, X):
        return X[:, 0]


class PopulationPrior(NativeUPrior):
    estimate_target = BasePriorEstimator.estimate_target


class PopulationClassifier(RecordingClassifier):
    class_prior_target = BasePUClassifier.class_prior_target


class FailPrior(NativeUPrior):
    def fit(self, X, y_pu):
        raise AssertionError("explicit and constructor priors precede estimation")


@pytest.mark.unit
class TestPriorTargets:
    def test_edge_classifier_sampling_declaration_must_match_pipeline(self):
        from pu_toolbox.estimators.risk import CVIRClassifier

        with pytest.raises(ValueError, match="input_sampling"):
            PUPipeline(classifier=CVIRClassifier(input_sampling="os"), input_sampling="ts")

    def test_basic_named_classifier_receives_pipeline_sampling(self):
        pipeline = PUPipeline(classifier="cvir", input_sampling="ts")
        model = pipeline._fresh_estimator(pipeline._classifier_cls, None, None)
        assert model.input_sampling == "ts"

    @pytest.mark.parametrize("invalid", ["cc", "OS", None, [], True])
    def test_edge_invalid_input_sampling_raises(self, invalid):
        with pytest.raises(ValueError, match="input_sampling"):
            PUPipeline(input_sampling=invalid)

    def test_basic_native_u_estimate_resolves_population(self):
        X = np.random.RandomState(7).normal(size=(25, 2))
        y = np.r_[np.ones(5, dtype=int), np.zeros(20, dtype=int)]
        report = PUPipeline(
            classifier=RecordingClassifier(), prior_estimator=NativeUPrior(), cv=2
        ).fit_evaluate(X, y)
        assert report.prior.value == pytest.approx(0.4)
        assert report.final_model.received_prior_ == pytest.approx(0.25)
        payload = json.loads(report.to_json())
        assert payload["prior"]["input_value"] == 0.25
        assert payload["prior"]["input_target"] == "native_unlabeled"
        assert payload["prior"]["target"] == "population"
        assert payload["prior"]["input_sampling"] == "os"
        assert len(payload["prior"]["training_consumption"]) == 3
        for record in payload["prior"]["training_consumption"]:
            q = record["labeled_fraction"]
            assert record["value"] == pytest.approx((0.4 - q) / (1 - q))
            assert record["target"] == "native_unlabeled"

    @pytest.mark.parametrize("input_sampling, expected", [("os", 0.4), ("ts", 0.25)])
    def test_param_input_sampling_controls_estimate_conversion(self, input_sampling, expected):
        X = np.random.RandomState(7).normal(size=(25, 2))
        y = np.r_[np.ones(5, dtype=int), np.zeros(20, dtype=int)]
        report = PUPipeline(
            classifier=RecordingClassifier(),
            prior_estimator=NativeUPrior(),
            input_sampling=input_sampling,
            cv=2,
        ).fit_evaluate(X, y)
        assert report.prior.value == pytest.approx(expected)
        assert report.final_model.received_prior_ == pytest.approx(0.25)

    def test_edge_ts_allows_extra_positives_above_population_prior(self):
        X = np.random.RandomState(1).normal(size=(20, 2))
        y = np.r_[np.ones(16, dtype=int), np.zeros(4, dtype=int)]
        profile = profile_pu_data(X, y, class_prior=0.4, input_sampling="ts")
        assert profile.summary["implied_label_frequency"] is None
        assert profile.summary["input_sampling"] == "ts"
        assert "inconsistent_class_prior" not in {issue.code for issue in profile.issues}
        assert profile.format_text()

    def test_basic_legacy_population_contract_is_preserved(self):
        X = np.random.RandomState(7).normal(size=(25, 2))
        y = np.r_[np.ones(5, dtype=int), np.zeros(20, dtype=int)]
        report = PUPipeline(
            classifier=PopulationClassifier(), prior_estimator=PopulationPrior(), cv=2
        ).fit_evaluate(X, y)
        assert report.prior.value == 0.25
        assert report.final_model.received_prior_ == 0.25
        assert report.prior.conversion == "identity"

    @pytest.mark.parametrize(
        "explicit, source, population", [(None, "constructor", 0.4), (0.5, "user", 0.5)]
    )
    def test_param_prior_precedence_and_constructor_target(self, explicit, source, population):
        X = np.random.RandomState(7).normal(size=(25, 2))
        y = np.r_[np.ones(5, dtype=int), np.zeros(20, dtype=int)]
        classifier = RecordingClassifier(class_prior=0.25)
        report = PUPipeline(classifier=classifier, prior_estimator=FailPrior(), cv=2).fit_evaluate(
            X, y, class_prior=explicit
        )
        assert report.prior.value == pytest.approx(population)
        assert report.prior.source == source
        assert report.final_model.received_prior_ == pytest.approx((population - 0.2) / 0.8)
        assert classifier.class_prior == 0.25
        assert report.final_model.class_prior == report.final_model.received_prior_

    @pytest.mark.parametrize(
        "input_sampling, risk, precision", [("os", 0.2, 0.6), ("ts", -0.05, 1.0)]
    )
    def test_param_metrics_use_the_population_pool(self, input_sampling, risk, precision):
        y = np.array([1, 1, 0, 0, 0, 0])
        pred = np.array([1, 1, 1, 0, 0, 0])
        assert pu_zero_one_risk(y, pred, 0.3, input_sampling=input_sampling) == pytest.approx(risk)
        assert pu_estimated_precision(y, pred, 0.3, input_sampling=input_sampling) == pytest.approx(
            precision
        )
        report = build_diagnostic_report(
            np.arange(12).reshape(6, 2),
            y,
            y_pred=pred,
            class_prior=0.3,
            input_sampling=input_sampling,
        )
        assert report.metrics["pu_zero_one_risk"].value == pytest.approx(risk)
        assert report.metrics["pu_estimated_precision"].value == pytest.approx(precision)

    def test_deterministic_prior_provenance(self):
        X = np.random.RandomState(7).normal(size=(25, 2))
        y = np.r_[np.ones(5, dtype=int), np.zeros(20, dtype=int)]
        pipeline = PUPipeline(
            classifier=RecordingClassifier(), prior_estimator=NativeUPrior(), cv=2
        )
        first = pipeline.fit_evaluate(X, y).prior.to_dict()
        second = pipeline.fit_evaluate(X, y).prior.to_dict()
        assert first == second

    def test_basic_ts_oracle_metrics_use_native_population_rows(self):
        y = np.array([1, 1, 0, 0, 0, 0])
        truth = np.array([1, 1, 1, 0, 0, 0])
        pred = np.array([0, 0, 1, 0, 0, 0])
        value, reason = compute_metric(
            "pu_accuracy",
            y,
            pred,
            pred.astype(float),
            truth,
            0.3,
            input_sampling="ts",
        )
        assert value == 1.0
        assert reason is None
        diagnostic = build_diagnostic_report(
            np.arange(12).reshape(6, 2),
            y,
            y_pred=pred,
            scores=pred.astype(float),
            y_true=truth,
            class_prior=0.3,
            input_sampling="ts",
        )
        assert diagnostic.metrics["accuracy"].value == 1.0
        assert diagnostic.metrics["roc_auc"].value == 1.0

    @pytest.mark.parametrize("method", ["upu", "nnpu", "robust_pu"])
    def test_param_real_classifiers_consume_native_u(self, method, monkeypatch):
        from pu_toolbox.registry import get_algorithm

        if method != "upu":
            pytest.importorskip("torch")
        params = {
            "upu": {"fit_intercept": False},
            "nnpu": {"max_epochs": 1, "batch_size": 16, "device": "cpu"},
            "robust_pu": {
                "pretrain_epochs": 1,
                "episodes": 1,
                "hidden_dim": 4,
                "batch_size": 16,
                "device": "cpu",
            },
        }[method]
        cls = get_algorithm(method)
        original_fit = cls.fit
        consumed = []

        def record_fit(model, X, y_pu, **kwargs):
            result = original_fit(model, X, y_pu, **kwargs)
            consumed.append((float(np.mean(y_pu == 1)), model._class_prior))
            return result

        monkeypatch.setattr(cls, "fit", record_fit)
        X = np.random.RandomState(7).normal(size=(60, 2))
        y = np.r_[np.ones(12, dtype=int), np.zeros(48, dtype=int)]
        report = PUPipeline(
            classifier=method, classifier_params=params, prior_estimator=None, cv=2
        ).fit_evaluate(X, y, class_prior=0.4)
        assert len(consumed) == 3
        for q, value in consumed:
            assert value == pytest.approx((0.4 - q) / (1 - q))
        assert report.prior.value == 0.4

    @pytest.mark.parametrize(
        "input_sampling, table_prior, risk_prior", [("os", 0.4, 0.2), ("ts", 0.55, 0.4)]
    )
    def test_param_pusb_receives_both_consumption_targets(
        self, input_sampling, table_prior, risk_prior
    ):
        X = np.random.RandomState(7).normal(size=(80, 2))
        y = np.r_[np.ones(20, dtype=int), np.zeros(60, dtype=int)]
        report = PUPipeline(
            classifier="pusb_kernel",
            prior_estimator=None,
            input_sampling=input_sampling,
            cv=2,
            classifier_params={"n_basis": 8, "cv": 2, "sigma_grid": (1.0,), "reg_grid": (0.1,)},
        ).fit_evaluate(X, y, class_prior=0.4)
        assert report.final_model._class_prior == pytest.approx(table_prior)
        assert report.final_model.risk_class_prior_ == pytest.approx(risk_prior)
        consumed = report.prior.training_consumption[-1]["parameters"]
        assert consumed["class_prior"]["target"] == "training_table"
        assert consumed["risk_class_prior"]["target"] == "native_unlabeled"

    def test_basic_tuning_and_comparison_preserve_ts_sampling(self):
        from pu_toolbox.model_selection import PUModelComparator, PUTuner

        X = np.r_[np.ones((76, 1)), -np.ones((24, 1))]
        y = np.r_[np.ones(60, dtype=int), np.zeros(40, dtype=int)]
        tuner = PUTuner(
            classifier="upu",
            param_grid={"fit_intercept": [False]},
            input_sampling="ts",
            cv=2,
        ).fit(X, y, class_prior=0.4)
        assert tuner.best_report.prior.input_sampling == "ts"
        assert tuner.best_report.final_model._class_prior == pytest.approx(0.4)
        comparison = PUModelComparator(
            classifiers=["upu", "elkan_noto"],
            classifier_params={"upu": {"fit_intercept": False}},
            input_sampling="ts",
            cv=2,
        ).fit(X, y, class_prior=0.4)
        assert comparison.best_report.prior.input_sampling == "ts"
