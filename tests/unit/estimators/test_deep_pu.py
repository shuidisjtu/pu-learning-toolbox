# ruff: noqa: E402, N803, N806

"""Focused unit tests for InfoMax PU, WConPU and DGPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep import (
    DGPUClassifier,
    InfoMaxPUClassifier,
    InfoMaxPURepresentation,
    WeightedContrastivePUClassifier,
)
from pu_toolbox.estimators.deep.infomax_pu import build_purl_mlp, pu_smi_objective
from pu_toolbox.estimators.deep.weighted_contrastive_pu import (
    embedding_dissimilarity,
)


def _data():
    rng = np.random.RandomState(7)
    X = np.vstack(
        [
            rng.normal(1.0, 0.3, size=(10, 4)),
            rng.normal(-1.0, 0.3, size=(20, 4)),
        ]
    ).astype(np.float32)
    y_pu = np.concatenate([np.ones(6, dtype=int), np.zeros(24, dtype=int)])
    return X, y_pu


class MockConditionalGenerator:
    def fit(self, X, y, *, warm_start=True):
        self.fit_calls_ = getattr(self, "fit_calls_", 0) + 1
        self.means_ = {label: X[y == label].mean(axis=0) for label in np.unique(y)}
        self.n_features_in_ = X.shape[1]
        return self

    def sample(self, n_samples, *, class_label, random_state=None):
        rng = np.random.RandomState(random_state)
        mean = self.means_.get(class_label, np.zeros(self.n_features_in_))
        return mean + 0.02 * rng.randn(n_samples, self.n_features_in_)


@pytest.mark.math
def test_pu_smi_objective_matches_paper_equation():
    positive = np.array([2.0, 4.0])
    unlabeled = np.array([1.0, 3.0])
    expected = 0.5 * (1.0 + 9.0) / 2.0 - (2.0 + 4.0) / 2.0
    assert pu_smi_objective(positive, unlabeled) == pytest.approx(expected)


@pytest.mark.math
def test_embedding_dissimilarity_bounds_and_values():
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    result = embedding_dissimilarity(query, keys)
    np.testing.assert_allclose(result.numpy(), [0.0, 1.0, 0.5], atol=1e-7)


@pytest.mark.unit
def test_infomax_representation_fit_transform_is_deterministic():
    X, y_pu = _data()
    kwargs = {
        "representation_dim": 2,
        "hidden_dim": 6,
        "max_epochs": 2,
        "random_state": 3,
    }
    first = InfoMaxPURepresentation(**kwargs).fit_transform(X, y_pu)
    second = InfoMaxPURepresentation(**kwargs).fit_transform(X, y_pu)
    assert first.shape == (len(X), 2)
    np.testing.assert_allclose(first, second)


@pytest.mark.unit
def test_basic_infomax_paper_networks_have_expected_layers_and_fit():
    X, y_pu = _data()
    model = build_purl_mlp(2, (5, 4, 3), batch_norm=True)
    assert [type(layer).__name__ for layer in model] == [
        "Linear",
        "BatchNorm1d",
        "ReLU",
        "Linear",
        "BatchNorm1d",
        "ReLU",
        "Linear",
        "BatchNorm1d",
        "ReLU",
        "Linear",
    ]

    estimator = InfoMaxPUClassifier(
        class_prior=0.3,
        representation_dim=2,
        hidden_dim=6,
        representation_epochs=1,
        classifier_epochs=1,
        representation_batch_norm=True,
        representation_activation=True,
        representation_batch_size=4,
        representation_gradient_noise=0.01,
        classifier_hidden_dims=(5, 4, 3),
        classifier_batch_norm=True,
        classifier_optimizer="adagrad",
        classifier_batch_size=4,
        random_state=3,
    ).fit(X, y_pu)
    assert estimator.decision_function(X).shape == (len(X),)
    assert (
        sum(isinstance(layer, torch.nn.BatchNorm1d) for layer in estimator.representation_.encoder_)
        == 2
    )
    assert isinstance(estimator.classifier_.model_[0], torch.nn.Linear)


@pytest.mark.unit
def test_determ_infomax_minibatch_gradient_noise_is_seeded():
    X, y_pu = _data()
    kwargs = {
        "representation_dim": 2,
        "hidden_dim": 6,
        "max_epochs": 2,
        "batch_norm": True,
        "representation_activation": True,
        "batch_size": 4,
        "gradient_noise": 0.01,
        "random_state": 13,
    }
    first = InfoMaxPURepresentation(**kwargs).fit_transform(X, y_pu)
    second = InfoMaxPURepresentation(**kwargs).fit_transform(X, y_pu)
    np.testing.assert_allclose(first, second)


@pytest.mark.unit
def test_param_infomax_rejects_invalid_paper_parameters():
    X, y_pu = _data()
    with pytest.raises(ValueError, match="gradient_noise"):
        InfoMaxPURepresentation(gradient_noise=-0.1).fit(X, y_pu)
    with pytest.raises(ValueError, match="classifier_optimizer"):
        InfoMaxPUClassifier(
            class_prior=0.3,
            representation_epochs=1,
            classifier_epochs=1,
            classifier_optimizer="sgd",
        ).fit(X, y_pu)


@pytest.mark.unit
def test_edge_wconpu_preserves_pseudo_label_and_queue_invariants():
    X, y_pu = _data()
    estimator = WeightedContrastivePUClassifier(
        0.3,
        hidden_dim=8,
        embedding_dim=4,
        queue_size=12,
        batch_size=10,
        max_epochs=2,
        random_state=5,
    ).fit(X, y_pu)
    np.testing.assert_allclose(estimator.pseudo_labels_.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        estimator.pseudo_labels_[y_pu == 1],
        np.tile([0.0, 1.0], (np.sum(y_pu == 1), 1)),
    )
    assert len(estimator.queue_embeddings_) <= 12
    np.testing.assert_allclose(
        np.linalg.norm(estimator.queue_embeddings_, axis=1),
        1.0,
        atol=1e-5,
    )
    assert np.isfinite(estimator.predict_proba(X)).all()


@pytest.mark.unit
def test_dgpu_generator_protocol_and_class_prior_counts():
    X, y_pu = _data()
    estimator = DGPUClassifier(
        0.3,
        MockConditionalGenerator(),
        hidden_dim=8,
        rounds=2,
        initialization_epochs=2,
        annotation_epochs=1,
        generated_samples=10,
        random_state=11,
    ).fit(X, y_pu)
    assert estimator.generator_.fit_calls_ == 2
    assert estimator.generated_counts_ == [
        {"negative": 7, "positive": 3},
        {"negative": 7, "positive": 3},
    ]
    assert estimator.predicted_distribution_.sum() == pytest.approx(1.0)
    assert np.all(estimator.predicted_distribution_ > 0)
    assert estimator.predict_proba(X).shape == (len(X), 2)


@pytest.mark.unit
def test_param_dgpu_missing_generator_raises():
    X, y_pu = _data()
    with pytest.raises(ValueError, match="generator is required"):
        DGPUClassifier(
            0.3,
            None,
            rounds=1,
            initialization_epochs=1,
            annotation_epochs=1,
            generated_samples=2,
        ).fit(X, y_pu)
