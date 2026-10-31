"""Actual forest/nested-fit work without changing frozen estimator math."""

import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone

from pu_toolbox.estimators.classic.rank_pruning import RankPruningClassifier
from pu_toolbox.estimators.risk.puet import PUExtraTreesClassifier
from pu_toolbox.experiment.classical_candidate import ClassicalCandidateTrainer

pytestmark = pytest.mark.unit


def data():
    features = np.random.RandomState(31).normal(size=(36, 4))
    return features, np.r_[np.ones(12, dtype=int), np.zeros(24, dtype=int)]


@pytest.mark.parametrize("view", ["os", "ts"])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_param_basic_determ_forest_records_real_trees_and_keeps_predictions(view, bootstrap):
    features, labels = data()
    model = PUExtraTreesClassifier(
        0.4, n_estimators=4, max_depth=3, bootstrap=bootstrap, random_state=3
    )
    direct = clone(model).fit(features, labels, os_or_ts=view)
    trainer = ClassicalCandidateTrainer()
    trajectory = trainer.fit(model, features, labels, os_or_ts=view)
    assert trajectory.epochs == [] and trajectory.best_epoch is None
    assert trajectory.checkpoints == [] and trajectory.model is model
    receipt = model.training_provenance_
    assert receipt["trees_completed"] == receipt["work_completed"] == 4
    assert receipt["neural_epoch_count"] == 0 and not receipt["formal_admission"]
    for event in receipt["events"]:
        assert event["positive_rows"] == 12
        assert event["unlabeled_rows"] == (36 if view == "ts" else 24)
        assert event["positive_rows"] * event["positive_weight"] == pytest.approx(0.4)
        assert event["unlabeled_rows"] * event["unlabeled_weight"] == pytest.approx(1)
        assert event["elapsed_seconds"] >= 0
    assert receipt["calibration_applied"] is (view == "ts")
    assert "_build_tree" not in model.__dict__
    np.testing.assert_array_equal(
        model.decision_function(features), direct.decision_function(features)
    )
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        restored.decision_function(features), direct.decision_function(features)
    )


def test_basic_determ_pruning_counts_every_cross_fit_and_unwraps_final_model():
    features, labels = data()
    model = RankPruningClassifier(n_cv_folds=3, min_retained=2, random_state=3)
    direct = clone(model).fit(features, labels)
    original = features.copy(), labels.copy()
    trainer = ClassicalCandidateTrainer()
    trajectory = trainer.fit(model, features, labels, class_prior=0.4, os_or_ts="os")
    receipt = model.training_provenance_
    assert trajectory.epochs == [] and receipt["work_completed"] == 4
    assert receipt["cross_fits_completed"] == 3 and receipt["final_refits_completed"] == 1
    for event in receipt["events"][:3]:
        assert event["training_rows"] == 24 and event["heldout_prediction_rows"] == 12
        assert not event["weighted"]
    assert receipt["events"][-1]["weighted"]
    assert receipt["events"][-1]["training_rows"] == len(model.retained_indices_)
    assert not receipt["external_class_prior_used"] and not receipt["calibration_applied"]
    assert model.base_estimator is None and type(model.estimator_) is type(direct.estimator_)
    np.testing.assert_array_equal(model.oof_probabilities_, direct.oof_probabilities_)
    np.testing.assert_array_equal(model.pruned_mask_, direct.pruned_mask_)
    np.testing.assert_array_equal(model.predict_proba(features), direct.predict_proba(features))
    np.testing.assert_array_equal(features, original[0])
    np.testing.assert_array_equal(labels, original[1])
    assert pickle.loads(pickle.dumps(model)).training_provenance_ == receipt


class FailingBase(BaseEstimator):
    calls = 0

    def fit(self, X, y, sample_weight=None):  # noqa: N803
        type(self).calls += 1
        if type(self).calls == 2:
            raise TypeError("internal failure, not an unsupported prior")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):  # noqa: N803
        return np.tile([0.6, 0.4], (len(X), 1))


def test_edge_internal_typeerror_is_not_retried_or_hidden_and_partial_work_survives():
    features, labels = data()
    base = FailingBase()
    model = RankPruningClassifier(base_estimator=base)
    trainer = ClassicalCandidateTrainer()
    FailingBase.calls = 0
    with pytest.raises(TypeError, match="internal failure"):
        trainer.fit(model, features, labels, class_prior=0.4)
    assert FailingBase.calls == 2 and model.base_estimator is base
    assert not model._is_fitted and not hasattr(model, "training_provenance_")
    assert trainer.last_receipt["status"] == "failed"
    assert trainer.last_receipt["work_started"] == 2
    assert trainer.last_receipt["work_completed"] == 1
    assert trainer.last_receipt["events"][-1]["status"] == "failed"


def test_edge_forest_failure_restores_tree_builder_and_invalidates_old_receipt(monkeypatch):
    features, labels = data()
    model = PUExtraTreesClassifier(0.4, n_estimators=3, random_state=3)
    trainer = ClassicalCandidateTrainer()
    trainer.fit(model, features, labels)
    original = PUExtraTreesClassifier._build_tree
    calls = []

    def fail_second(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("tree failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PUExtraTreesClassifier, "_build_tree", fail_second)
    with pytest.raises(RuntimeError, match="tree failure"):
        trainer.fit(model, features, labels)
    assert "_build_tree" not in model.__dict__ and not model._is_fitted
    assert not hasattr(model, "training_provenance_")
    assert trainer.last_receipt["work_completed"] == 1


class ConstructorRNGBase(BaseEstimator):
    constructions = 0

    def __init__(self):
        type(self).constructions += 1
        self.offset = np.random.uniform(-0.2, 0.2)

    def fit(self, X, y, sample_weight=None):  # noqa: N803
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):  # noqa: N803
        positive = 1 / (1 + np.exp(-(X[:, 0] + self.offset)))
        return np.column_stack((1 - positive, positive))


def test_basic_determ_pruning_does_not_add_constructor_clones_or_consume_extra_rng():
    features, labels = data()
    model = RankPruningClassifier(base_estimator=ConstructorRNGBase(), min_retained=2)
    direct = clone(model)
    state = np.random.get_state()
    try:
        np.random.seed(83)
        ConstructorRNGBase.constructions = 0
        direct.fit(features, labels)
        direct_draw = np.random.random()
        assert ConstructorRNGBase.constructions == 4
        np.random.seed(83)
        ConstructorRNGBase.constructions = 0
        ClassicalCandidateTrainer().fit(model, features, labels)
        assert ConstructorRNGBase.constructions == 4
        assert np.random.random() == direct_draw
        np.testing.assert_array_equal(model.oof_probabilities_, direct.oof_probabilities_)
        np.testing.assert_array_equal(model.predict_proba(features), direct.predict_proba(features))
    finally:
        np.random.set_state(state)
