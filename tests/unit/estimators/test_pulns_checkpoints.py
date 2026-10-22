"""Completed-episode callbacks must not replace or retrain PULNS's classifier."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.deep.pulns import PULNSClassifier  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def data():
    rng = np.random.RandomState(7)
    return (
        rng.normal(size=(24, 3)).astype("float32"),
        np.r_[np.ones(8, int), np.zeros(16, int)],
        (rng.normal(size=(6, 3)).astype("float32"), np.array([0, 1] * 3)),
    )


def model(seed=2, episodes=2):
    return PULNSClassifier(
        hidden_dim=4,
        pretrain_epochs=1,
        episodes=episodes,
        classifier_epochs=1,
        batch_size=8,
        random_state=seed,
        device="cpu",
    )


@pytest.mark.parametrize("seed", [0, 2, 7])
def test_determ_callback_keeps_default_weights_policy_and_history(seed):
    features, labels, support = data()
    plain = model(seed).fit(features, labels, support_data=support)
    events = []

    def capture(episode, fitted):
        assert fitted._is_fitted
        assert fitted.checkpoint_stage_ == "episode"
        assert fitted.checkpoint_stage_epoch_ == episode + 1
        assert fitted.checkpoint_round_ is None
        assert fitted.optimizer_steps_ == sum(fitted.stage_optimizer_steps_.values())
        assert len(fitted.history_["episode"]) == episode + 1
        assert np.isfinite(fitted.decision_function(features)).all()
        events.append((episode, fitted.optimizer_steps_))

    recorded = model(seed).fit(features, labels, support_data=support, epoch_callback=capture)
    assert [episode for episode, _ in events] == [0, 1]
    assert events[1][1] > events[0][1]
    assert plain.history_ == recorded.history_
    assert plain.stage_optimizer_steps_ == recorded.stage_optimizer_steps_
    np.testing.assert_array_equal(
        plain.decision_function(features), recorded.decision_function(features)
    )
    for name in ("model_", "selector_"):
        for key, value in getattr(plain, name).state_dict().items():
            torch.testing.assert_close(
                value, getattr(recorded, name).state_dict()[key], rtol=0, atol=0
            )


@pytest.mark.parametrize("callback", [False, 1, "callback", object()])
def test_param_noncallable_callback_refused_before_seed_or_training(callback, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("callback validation must precede seeding and training")

    monkeypatch.setattr(torch, "manual_seed", never)
    features, labels, support = data()
    fitted = model()
    with pytest.raises(ValueError, match="epoch_callback"):
        fitted.fit(features, labels, support_data=support, epoch_callback=callback)
    with pytest.raises(NotFittedError):
        fitted.predict(features)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_edge_callback_failure_revokes_previously_fitted_predictor(error):
    features, labels, support = data()
    fitted = model().fit(features, labels, support_data=support)
    calls = []

    def fail(episode, current):
        calls.append(episode)
        assert current._is_fitted
        raise error("synthetic callback interruption")

    with pytest.raises(error, match="synthetic callback"):
        fitted.fit(features, labels, support_data=support, epoch_callback=fail)
    assert calls == [0]
    with pytest.raises(NotFittedError):
        fitted.predict(features)


def test_basic_callback_is_current_episode_before_internal_best_restoration(monkeypatch):
    features, labels, support = data()
    scores = iter([1.0, 0.4, 0.2, 0.3, 0.1])
    monkeypatch.setattr(PULNSClassifier, "_support_accuracy", lambda *args: next(scores))
    original = PULNSClassifier._train_network
    initial, episodes = [], []

    def train(self, network, *args, **kwargs):
        result = original(self, network, *args, **kwargs)
        if self._training_stage_ == "pretrain":
            initial.append(copy.deepcopy(network.state_dict()))
        return result

    monkeypatch.setattr(PULNSClassifier, "_train_network", train)
    fitted = model().fit(
        features,
        labels,
        support_data=support,
        epoch_callback=lambda _, current: episodes.append(
            copy.deepcopy(current.model_.state_dict())
        ),
    )
    assert len(initial) == 1 and len(episodes) == 2
    for key, value in fitted.model_.state_dict().items():
        torch.testing.assert_close(value, initial[0][key], rtol=0, atol=0)
    assert any(not torch.equal(value, episodes[-1][key]) for key, value in initial[0].items())


def test_edge_single_episode_is_one_snapshot_not_pretrain_or_classifier_epochs():
    features, labels, support = data()
    events = []
    fitted = model(episodes=1).set_params(pretrain_epochs=2, classifier_epochs=3)
    fitted.fit(
        features, labels, support_data=support, epoch_callback=lambda epoch, _: events.append(epoch)
    )
    assert fitted.checkpoint_epoch_count == 1 and events == [0]
    assert fitted.checkpoint_prediction_batch_size == 8
    assert fitted.stage_optimizer_steps_["pretrain"] == 6


def test_edge_generic_checkpoint_writer_still_cannot_invent_clean_support():
    features, labels, _ = data()
    with pytest.raises(ValueError, match="independent clean support_data"):
        EpochCheckpointTrainer().fit(model(), features, labels)
