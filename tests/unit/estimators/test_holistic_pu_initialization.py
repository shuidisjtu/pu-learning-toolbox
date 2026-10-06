"""Source-supported stage restart, optimizer isolation and legacy compatibility."""

import copy
import pickle

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.holistic_pu import HolisticPUClassifier  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def problem(architecture):
    shape = (27, 3, 4, 4) if architecture == "cnn" else (27, 3)
    features = np.random.RandomState(7).normal(size=shape).astype("float32")
    features[:10] += 1
    labels = np.r_[np.ones(10, int), np.zeros(17, int)]
    torch.manual_seed(17)
    encoder = None
    if architecture == "cnn":
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(4),
        )
    return features, labels, encoder


def model(encoder=None, **overrides):
    params = dict(
        encoder=encoder,
        hidden_dim=4,
        warmup_epochs=3,
        max_epochs=1,
        batch_size=8,
        random_state=3,
        device="cpu",
        pseudo_pn_initialization="reinitialize",
    )
    params.update(overrides)
    return HolisticPUClassifier(**params)


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
@pytest.mark.parametrize("initialization", ["continue", "reinitialize"])
def test_basic_real_optimizer_instances_moments_and_cumulative_stage_cost(
    monkeypatch, architecture, initialization
):
    features, labels, encoder = problem(architecture)
    optimizers, stages = [], {}
    original = torch.optim.Adam

    def capture(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        assert not optimizer.state
        optimizers.append(optimizer)
        return optimizer

    def callback(epoch, fitted):
        stages[fitted.checkpoint_stage_] = fitted.model_

    monkeypatch.setattr(torch.optim, "Adam", capture)
    fitted = model(encoder, pseudo_pn_initialization=initialization).fit(
        features, labels, epoch_callback=callback
    )
    restart = initialization == "reinitialize"
    assert len(optimizers) == (2 if restart else 1)
    assert (stages["warmup"] is not stages["pseudo_pn"]) == restart
    assert fitted.pseudo_pn_optimizer_reset_ is restart
    assert fitted.pseudo_pn_initialization_ == initialization
    assert fitted.stage_optimizer_steps_ == {"warmup": 9, "pseudo_pn": 4}
    assert fitted.optimizer_steps_ == 13
    assert fitted.history_["optimizer_steps"] == [3, 6, 9, 13]
    steps = [{int(state["step"].item()) for state in opt.state.values()} for opt in optimizers]
    assert steps == ([{9}, {4}] if restart else [{13}])
    if restart:
        old_ids = {id(param) for param in stages["warmup"].parameters()}
        new_ids = {id(param) for param in stages["pseudo_pn"].parameters()}
        assert not old_ids & new_ids


def test_basic_cnn_reinitializes_all_parameters_and_bn_not_just_new_head(monkeypatch):
    features, labels, encoder = problem("cnn")
    with torch.no_grad():
        encoder[0].weight.fill_(0.25)
        encoder[1].running_mean.fill_(3)
        encoder[5].running_mean.fill_(3)
    encoder.requires_grad_(False)
    before = copy.deepcopy(encoder.state_dict())
    original = HolisticPUClassifier._build_model
    constructed = []

    def capture(self, rows, **kwargs):
        network, own_encoder = original(self, rows, **kwargs)
        constructed.append((network, copy.deepcopy(own_encoder.state_dict())))
        return network, own_encoder

    monkeypatch.setattr(HolisticPUClassifier, "_build_model", capture)
    fitted = model(encoder).fit(features, labels)
    assert len(constructed) == 2
    torch.testing.assert_close(constructed[0][1]["0.weight"], before["0.weight"], rtol=0, atol=0)
    assert not torch.equal(constructed[1][1]["0.weight"], before["0.weight"])
    for prefix in ("1", "5"):
        fresh = constructed[1][1]
        torch.testing.assert_close(fresh[f"{prefix}.running_mean"], torch.zeros(4))
        torch.testing.assert_close(fresh[f"{prefix}.running_var"], torch.ones(4))
        torch.testing.assert_close(fresh[f"{prefix}.weight"], torch.ones(4))
        assert fresh[f"{prefix}.num_batches_tracked"].item() == 0
    assert fitted.encoder_[1].num_batches_tracked.item() == 4  # PN only, no warmup BN history
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not encoder[0].weight.requires_grad


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_determ_default_continue_and_restart_share_warmup_not_final_network(architecture):
    features, labels, encoder = problem(architecture)
    explicit = model(encoder, pseudo_pn_initialization="continue").fit(features, labels)
    params = explicit.get_params(deep=False)
    params.pop("pseudo_pn_initialization")
    legacy = HolisticPUClassifier(**params).fit(features, labels)
    restarted = model(encoder).fit(features, labels)
    np.testing.assert_array_equal(
        explicit.decision_function(features), legacy.decision_function(features)
    )
    np.testing.assert_array_equal(explicit.prediction_trajectory_, restarted.prediction_trajectory_)
    np.testing.assert_array_equal(explicit.pseudo_labels_, restarted.pseudo_labels_)
    assert not np.array_equal(
        explicit.decision_function(features), restarted.decision_function(features)
    )
    assert legacy.pseudo_pn_initialization_ == "continue" and not legacy.pseudo_pn_optimizer_reset_


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_determ_clone_refit_pickle_and_stage_snapshots_survive_network_restart(
    tmp_path, architecture
):
    features, labels, encoder = problem(architecture)
    fitted = clone(model(encoder))
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, features, labels)
    expected = fitted.decision_function(features)
    repeated = clone(model(encoder)).fit(features, labels)
    np.testing.assert_array_equal(repeated.decision_function(features), expected)
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(fitted)).decision_function(features), expected
    )
    for checkpoint in trajectory.checkpoints:
        restored = checkpoint.restore(device="cpu")
        assert np.isfinite(restored.decision_function(features)).all()
        assert checkpoint.reference()["training_resume_supported"] is False
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(features), expected
    )
    first = trajectory.checkpoints[0].restore().decision_function(features)
    fitted.fit(features, labels)
    np.testing.assert_array_equal(fitted.decision_function(features), expected)
    np.testing.assert_array_equal(
        trajectory.checkpoints[0].restore().decision_function(features), first
    )
    assert fitted.optimizer_steps_ == 13


@pytest.mark.parametrize("initialization", [None, [], "fresh_template", "", 1])
def test_param_invalid_initialization_fails_before_training(initialization):
    features, labels, _ = problem("mlp")
    fitted = model(pseudo_pn_initialization=initialization)
    with pytest.raises(ValueError, match="pseudo_pn_initialization"):
        fitted.fit(features, labels)
    assert not fitted._is_fitted


class NoResetEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))

    def forward(self, values):
        return values.mean(dim=(2, 3)) * self.weight


def test_edge_custom_parameter_owner_cannot_silently_copy_pretrained_weights():
    features, labels, _ = problem("cnn")
    source = NoResetEncoder()
    before = source.weight.detach().clone()
    fitted = model(source)
    with pytest.raises(ValueError, match="reset_parameters.*<root>"):
        fitted.fit(features, labels)
    assert not fitted._is_fitted
    torch.testing.assert_close(source.weight, before, rtol=0, atol=0)
    # A legacy adapter can still use an explicit pretrained/custom template.
    assert model(source, pseudo_pn_initialization="continue").fit(features, labels)._is_fitted


def test_edge_legacy_weight_norm_cannot_claim_its_computed_weight_reset_is_fresh():
    features, labels, _ = problem("mlp")
    with pytest.warns(FutureWarning, match="weight_norm"):
        source = torch.nn.utils.weight_norm(torch.nn.Linear(3, 4))
    before = copy.deepcopy(source.state_dict())
    fitted = model(source)
    with pytest.raises(ValueError, match="weight_g/weight_v"):
        fitted.fit(features, labels)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not fitted._is_fitted
