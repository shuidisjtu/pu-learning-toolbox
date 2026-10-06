"""Native PULNS CNN state/reward isolation, learning, budgets and restoration."""

import copy
import os
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.pulns import PULNSClassifier  # noqa: E402
from pu_toolbox.registry import get_metadata, register_all_builtin_methods  # noqa: E402

pytestmark = pytest.mark.unit


def problem():
    rng = np.random.RandomState(7)
    features = rng.normal(size=(25, 3, 4, 4)).astype("float32")
    labels = np.r_[np.ones(8, int), np.zeros(17, int)]
    support = (rng.normal(size=(6, 3, 4, 4)).astype("float32"), np.array([0, 1] * 3))
    torch.manual_seed(17)
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.BatchNorm1d(4),
        torch.nn.Dropout(0.2),
    ).requires_grad_(False)
    return features, labels, support, encoder


def model(encoder, **overrides):
    params = dict(
        encoder=encoder,
        hidden_dim=4,
        pretrain_epochs=1,
        episodes=2,
        classifier_epochs=1,
        batch_size=8,
        random_state=2,
        device="cpu",
    )
    params.update(overrides)
    return PULNSClassifier(**params)


def test_basic_encoder_trains_in_pretrain_probe_and_classifier_without_support_rows(monkeypatch):
    features, labels, support, encoder = problem()
    original_train, original_select = PULNSClassifier._train_network, PULNSClassifier._select
    stages = []
    template = copy.deepcopy(encoder.state_dict())
    pointers = []

    def train(self, network, data, targets, rng, epochs):
        assert data.device.type == "cpu"
        assert all(any(np.array_equal(row, x) for x in features) for row in data.numpy())
        before = copy.deepcopy(network[0].state_dict())
        pointers.append((self._training_stage_, next(network[0].parameters()).data_ptr()))
        original_train(self, network, data, targets, rng, epochs)
        assert not torch.equal(before["0.weight"], network[0].state_dict()["0.weight"])
        assert all(p.requires_grad for p in network[0].parameters())
        stages.append(self._training_stage_)

    def all_selected(self, *args, **kwargs):
        _, logs, rewards = original_select(self, *args, **kwargs)
        return np.flatnonzero(labels == 0).tolist(), logs, rewards

    monkeypatch.setattr(PULNSClassifier, "_train_network", train)
    monkeypatch.setattr(PULNSClassifier, "_select", all_selected)
    fitted = model(encoder).fit(features, labels, support_data=support)
    assert stages == ["pretrain", "reward_probe", "classifier", "reward_probe", "classifier"]
    assert pointers[0][1] == pointers[2][1] == pointers[4][1]
    assert all(pointers[i][1] != pointers[0][1] for i in (1, 3))
    assert fitted.encoder_ is fitted.model_[0] and fitted.encoder_ is not encoder
    assert fitted.stage_optimizer_steps_ == {
        "pretrain": 4,
        "reward_probe": 8,
        "policy": 2,
        "classifier": 8,
    }
    assert fitted.optimizer_steps_ == 22 and fitted.history_["optimizer_steps"] == [13, 22]
    assert set(fitted.selected_negative_indices_) == set(np.flatnonzero(labels == 0))
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, template[key], rtol=0, atol=0)
    assert not any(p.requires_grad for p in encoder.parameters())


def test_determ_clone_seed_refit_pickle_and_independent_encoder_copies():
    features, labels, support, encoder = problem()
    left = clone(model(encoder)).fit(features, labels, support_data=support)
    right = clone(model(encoder)).fit(features, labels, support_data=support)
    np.testing.assert_array_equal(
        left.decision_function(features), right.decision_function(features)
    )
    assert left.history_ == right.history_
    assert (
        next(left.encoder_.parameters()).data_ptr() != next(right.encoder_.parameters()).data_ptr()
    )
    restored = pickle.loads(pickle.dumps(left))
    np.testing.assert_array_equal(restored.predict_proba(features), left.predict_proba(features))
    left.fit(features, labels, support_data=support)
    np.testing.assert_array_equal(
        left.decision_function(features), right.decision_function(features)
    )
    assert left.stage_optimizer_steps_ == right.stage_optimizer_steps_
    assert sum(left.stage_optimizer_steps_.values()) == left.optimizer_steps_


def test_basic_returned_best_cnn_restores_actual_classifier_not_reward_probe(monkeypatch):
    features, labels, support, encoder = problem()
    original = PULNSClassifier._train_network
    actual_states = []

    def capture(self, network, *args):
        original(self, network, *args)
        if self._training_stage_ == "classifier":
            actual_states.append(copy.deepcopy(network.state_dict()))

    # initial; probe1, classifier1; probe2, classifier2
    scores = iter([0.2, 0.3, 0.9, 0.99, 0.4])
    monkeypatch.setattr(PULNSClassifier, "_train_network", capture)
    monkeypatch.setattr(PULNSClassifier, "_support_accuracy", lambda *args: next(scores))
    fitted = model(encoder).fit(features, labels, support_data=support)
    assert len(actual_states) == 2
    assert fitted.best_support_accuracy_ == 0.9 and fitted.reward_baseline_ == 0.99
    for key, value in fitted.model_.state_dict().items():
        torch.testing.assert_close(value, actual_states[0][key], rtol=0, atol=0)


@pytest.mark.math
def test_basic_batched_episode_state_matches_last_hidden_features_and_reward_row_order():
    features, labels, support, encoder = problem()
    fitted = model(encoder).fit(features, labels, support_data=support)
    unlabeled = np.flatnonzero(labels == 0)
    data = torch.as_tensor(features)
    before = copy.deepcopy(fitted.encoder_.state_dict())
    seen = []
    hook = fitted.encoder_.register_forward_pre_hook(lambda _, inputs: seen.append(len(inputs[0])))
    try:
        representation, raw_u = fitted._episode_features(data, unlabeled)
    finally:
        hook.remove()
    assert max(seen) <= fitted.batch_size
    assert representation.shape == (25, 4) and representation.device.type == "cpu"
    assert not representation.requires_grad and not raw_u.requires_grad
    with torch.no_grad():
        expected = torch.cat([fitted.model_[:-1](chunk) for chunk in data.split(8)])
    torch.testing.assert_close(representation, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(raw_u.numpy(), fitted.decision_function(features[unlabeled]))
    for key, value in fitted.encoder_.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "fault", ["missing_encoder", "bad_encoder", "spatial", "support_shape", "overlap", "ts"]
)
def test_param_image_support_and_encoder_boundaries_remain_fail_closed(fault):
    features, labels, support, encoder = problem()
    kwargs = {"support_data": support}
    pattern = "encoder|shape|overlap|ts risk"
    if fault == "missing_encoder":
        encoder = None
    elif fault == "bad_encoder":
        encoder = "flatten"
    elif fault == "spatial":
        encoder = torch.nn.Conv2d(3, 4, 1)
    elif fault == "support_shape":
        kwargs["support_data"] = (support[0][:, :, :, :3], support[1])
    elif fault == "overlap":
        kwargs.update(train_indices=np.arange(25), support_indices=np.arange(6))
    else:
        kwargs["os_or_ts"] = "ts"
    fitted = model(encoder)
    with pytest.raises((ValueError, TypeError), match=pattern):
        fitted.fit(features, labels, **kwargs)
    assert not fitted._is_fitted


def test_edge_empty_prediction_shape_and_eval_failure_restore_mixed_modes(monkeypatch):
    features, labels, support, encoder = problem()
    fitted = model(encoder).fit(features, labels, support_data=support)
    assert fitted.predict_proba(features[:0]).shape == (0, 2)
    with pytest.raises(ValueError, match="input shape"):
        fitted.predict(features[:, :, :, :3])
    fitted.model_.train()
    fitted.encoder_[1].eval()
    modes = [layer.training for layer in fitted.model_.modules()]
    before = copy.deepcopy(fitted.encoder_.state_dict())
    fitted.decision_function(features)
    assert [layer.training for layer in fitted.model_.modules()] == modes
    for key, value in fitted.encoder_.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def fail(*args):
        raise ValueError("synthetic forward failure")

    monkeypatch.setattr(fitted.model_[-1], "forward", fail)
    with pytest.raises(ValueError, match="synthetic forward"):
        fitted.decision_function(features)
    assert [layer.training for layer in fitted.model_.modules()] == modes


def test_edge_missing_reward_support_and_registry_keep_pu_only_exclusion():
    features, labels, _, encoder = problem()
    with pytest.raises(ValueError, match="PA-ineligible"):
        model(encoder).fit(features, labels)
    register_all_builtin_methods()
    meta = get_metadata("pulns")
    assert meta.requires_clean_support and meta.trains_encoder
    assert meta.native_architectures == frozenset({"mlp", "cnn"})


@pytest.mark.gpu
def test_cuda_cnn_state_cache_and_pickle():
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    features, labels, support, encoder = problem()
    fitted = model(encoder, device="cuda", episodes=1).fit(features, labels, support_data=support)
    assert next(fitted.encoder_.parameters()).is_cuda
    representation, raw_u = fitted._episode_features(
        torch.as_tensor(features), np.flatnonzero(labels == 0)
    )
    assert representation.device.type == raw_u.device.type == "cpu"
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored.predict(features), fitted.predict(features))
