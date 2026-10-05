# ruff: noqa: N806
"""PULNS reward equations and extra-clean-label budget cannot be bypassed."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.advisor import recommend_methods  # noqa: E402
from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.deep.pulns import (  # noqa: E402
    PULNSClassifier,
    pulns_discounted_returns,
    pulns_intermediate_reward,
)
from pu_toolbox.registry import (  # noqa: E402
    get_algorithm,
    get_metadata,
    register_all_builtin_methods,
)

pytestmark = pytest.mark.unit


def data():
    rng = np.random.RandomState(7)
    X = rng.normal(size=(24, 3)).astype("float32")
    y = np.r_[np.ones(8, int), np.zeros(16, int)]
    support = (rng.normal(size=(6, 3)).astype("float32"), np.array([0, 1] * 3))
    return X, y, support


def model(**kwargs):
    params = dict(
        hidden_dim=4,
        pretrain_epochs=1,
        episodes=2,
        classifier_epochs=1,
        batch_size=8,
        random_state=2,
        device="cpu",
    )
    params.update(kwargs)
    return PULNSClassifier(**params)


@pytest.mark.math
def test_intermediate_rewards_and_discounted_return_terminal_not_discounted():
    logits = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    actions = torch.tensor([1, 1, 0, 0])
    rewards = pulns_intermediate_reward(logits, actions)
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.5, 0.5, 1.0]))
    assert not rewards.requires_grad
    returns = pulns_discounted_returns(
        torch.tensor([1.0, 2.0, 3.0]), 0.25, discount=0.5, terminal_weight=2.0
    )
    torch.testing.assert_close(returns, torch.tensor([3.25, 4.0, 3.5]))


def test_missing_support_bad_labels_and_identity_overlap_fail_closed():
    X, y, support = data()
    with pytest.raises(ValueError, match="clean support_data"):
        model().fit(X, y)
    with pytest.raises(ValueError, match="both clean classes"):
        model().fit(X, y, support_data=(support[0], np.ones(6)))
    with pytest.raises(ValueError, match="overlap"):
        model().fit(
            X, y, support_data=support, train_indices=np.arange(24), support_indices=np.arange(6)
        )
    with pytest.raises(ValueError, match="both train_indices"):
        model().fit(X, y, support_data=support, train_indices=np.arange(24))
    with pytest.raises(ValueError, match="ts risk substitution"):
        model().fit(X, y, support_data=support, os_or_ts="ts")


def test_fit_cloning_pickle_seed_registry_and_auto_recommendation_exclusion():
    X, y, support = data()
    fitted = clone(model())
    with pytest.raises(NotFittedError):
        fitted.predict(X)
    fitted.fit(
        X, y, support_data=support, train_indices=np.arange(24), support_indices=np.arange(24, 30)
    )
    repeated = model().fit(X, y, support_data=support, class_prior=0.4)
    np.testing.assert_array_equal(fitted.decision_function(X), repeated.decision_function(X))
    assert fitted.history_ == repeated.history_
    assert fitted.support_isolation_status_ == "train_support_ids_disjoint"
    assert repeated.support_isolation_status_ == "unverified_caller_responsibility"
    assert len(fitted.history_["episode"]) == 2 and fitted.n_support_ == 6
    assert set(fitted.selected_negative_indices_) <= set(np.flatnonzero(y == 0))
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.predict_proba(X), fitted.predict_proba(X))
    register_all_builtin_methods()
    assert get_algorithm("pulns") is PULNSClassifier
    assert get_metadata("pulns").requires_clean_support
    recommendation = recommend_methods(X, y, class_prior=0.4, top_k=100)
    assert "pulns" not in [m.name for m in recommendation.candidates]


def test_clean_support_never_enters_classifier_minibatches(monkeypatch):
    X, y, support = data()
    original = PULNSClassifier._train_network
    seen = []

    def capture(self, network, tensors, labels, rng, epochs):
        rows = tensors.cpu().numpy()
        assert all(any(np.array_equal(row, source) for source in X) for row in rows)
        seen.append(len(rows))
        return original(self, network, tensors, labels, rng, epochs)

    monkeypatch.setattr(PULNSClassifier, "_train_network", capture)
    model().fit(X, y, support_data=support)
    assert seen[0] == len(X) and len(seen) >= 3


def test_probe_reward_baseline_is_separate_from_returned_best_classifier(monkeypatch):
    X, y, support = data()
    scores = iter([0.2, 0.4, 0.3, 0.6, 0.5])  # initial; probe, updated; probe, updated
    monkeypatch.setattr(PULNSClassifier, "_support_accuracy", lambda *args: next(scores))
    fitted = model().fit(X, y, support_data=support)
    np.testing.assert_allclose(fitted.history_["terminal_reward"], [0.2, 0.2])
    assert fitted.reward_baseline_ == 0.6
    assert fitted.best_support_accuracy_ == 0.5


def test_empty_selection_does_not_invent_negatives_or_train_on_support(monkeypatch):
    X, y, support = data()
    original = PULNSClassifier._select

    def select_none(self, *args, **kwargs):
        _, logs, rewards = original(self, *args, **kwargs)
        return [], logs, rewards

    monkeypatch.setattr(PULNSClassifier, "_select", select_none)
    fitted = model().fit(X, y, support_data=support)
    assert fitted.empty_negative_episodes_ == 2
    assert fitted.selected_negative_indices_.size == 0
    assert fitted.history_["selected_negatives"] == [0, 0]
    # Three pretraining minibatches plus one policy update per episode.
    assert fitted.optimizer_steps_ == 5
    assert np.isfinite(fitted.predict_proba(X)).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"episodes": 0}, {"discount": 1.1}, {"terminal_weight": -1}, {"learning_rate": np.nan}],
)
def test_invalid_parameters(kwargs):
    X, y, support = data()
    with pytest.raises(ValueError):
        model(**kwargs).fit(X, y, support_data=support)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_short_cuda_and_pickle_prediction_roundtrip():
    X, y, support = data()
    fitted = model(episodes=1, device="cuda").fit(X, y, support_data=support)
    assert next(fitted.model_.parameters()).is_cuda
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.predict(X), fitted.predict(X))
