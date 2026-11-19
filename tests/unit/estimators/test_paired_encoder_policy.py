"""The frozen arm preserves the same encoder tree, weights and running buffers."""

import copy

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep._encoder_policy import (  # noqa: E402
    is_frozen_encoder,
    is_policy_encoder,
    set_encoder_training_policy,
)
from pu_toolbox.estimators.deep.holistic_pu import HolisticPUClassifier  # noqa: E402
from pu_toolbox.estimators.deep.robust_pu import RobustPUClassifier  # noqa: E402
from pu_toolbox.estimators.deep.split_pu import SplitPUClassifier  # noqa: E402
from pu_toolbox.estimators.risk.dist_pu import DistPUClassifier  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def _fixture():
    torch.manual_seed(11)
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 1),
        torch.nn.BatchNorm2d(3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )
    features = np.random.RandomState(5).normal(size=(24, 3, 4, 4)).astype("float32")
    return features, np.r_[np.ones(8, int), np.zeros(16, int)], encoder


@pytest.mark.parametrize("trainable", [False, True], ids=["frozen", "trainable"])
@pytest.mark.parametrize(
    "classifier,params,stages,view",
    [
        (
            RobustPUClassifier,
            {"class_prior": 0.4, "pretrain_epochs": 1, "episodes": 1, "inner_epochs": 1},
            ("pretrain", "self_paced"),
            "ts",
        ),
        (
            SplitPUClassifier,
            {
                "class_prior": 0.4,
                "encoder_feature_layer": "0",
                "teacher_epochs": 1,
                "split_epochs": 1,
                "student_epochs": 1,
                "rounds": 2,
            },
            ("teacher", "splitter", "student"),
            "ts",
        ),
        (
            HolisticPUClassifier,
            {"warmup_epochs": 2, "max_epochs": 1},
            ("warmup", "pseudo_pn"),
            "os",
        ),
        (
            DistPUClassifier,
            {
                "class_prior": 0.4,
                "training_recipe": "author_staged",
                "distribution_loss": "author_histogram",
                "warmup_epochs": 1,
                "pu_epochs": 1,
            },
            ("warmup", "dist_pu"),
            "ts",
        ),
    ],
)
def test_basic_each_stage_preserves_frozen_encoder_but_updates_trainable_encoder(
    classifier, params, stages, view, trainable
):
    features, labels, untagged = _fixture()
    initial = copy.deepcopy(untagged.state_dict())
    assert not is_policy_encoder(untagged)
    source = set_encoder_training_policy(copy.deepcopy(untagged), trainable=trainable)
    estimator = clone(
        classifier(
            encoder=source, hidden_dim=4, batch_size=8, random_state=3, device="cpu", **params
        )
    )
    seen = set()
    retained = []

    def capture(_epoch, fitted):
        encoder = fitted.encoder_
        assert is_policy_encoder(encoder)
        assert is_frozen_encoder(encoder) is not trainable
        assert all(param.requires_grad is trainable for param in encoder.parameters())
        state = encoder.state_dict()
        if trainable:
            assert not torch.equal(state["0.weight"], initial["0.weight"])
            assert state["1.num_batches_tracked"] > initial["1.num_batches_tracked"]
        else:
            assert all(torch.equal(value, initial[name]) for name, value in state.items())
            assert all(not module.training for module in encoder.modules())
        seen.add(fitted.checkpoint_stage_)
        retained.append(encoder)

    fitted = estimator.fit(features, labels, os_or_ts=view, epoch_callback=capture)
    assert seen == set(stages)
    assert fitted.calibration_applied_ is (view == "ts")
    assert fitted.decision_function(features).shape == labels.shape
    assert all(torch.equal(value, initial[name]) for name, value in source.state_dict().items())
    assert all(param.requires_grad for param in source.parameters())
    if classifier is SplitPUClassifier:
        assert len({id(encoder) for encoder in retained}) == 4
        assert is_policy_encoder(fitted.teacher_encoder_)
        assert is_policy_encoder(fitted.splitter_encoder_)
        assert all(not param.requires_grad for param in fitted.teacher_encoder_.parameters())
    if classifier is DistPUClassifier:
        assert fitted.training_provenance_["encoder_trained"] is trainable


def test_determ_frozen_seed_and_checkpoint_roundtrip_keep_encoder_identity(tmp_path):
    features, labels, template = _fixture()
    template = set_encoder_training_policy(template, trainable=False)
    model = RobustPUClassifier(
        encoder=template,
        class_prior=0.4,
        hidden_dim=4,
        batch_size=8,
        pretrain_epochs=1,
        episodes=1,
        random_state=3,
        device="cpu",
    )
    repeated = clone(model).fit(features, labels)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(model, features, labels)
    restored = trajectory.checkpoints[-1].restore()
    restored_encoder = restored.model_[0]
    np.testing.assert_array_equal(
        repeated.decision_function(features), model.decision_function(features)
    )
    assert is_frozen_encoder(restored_encoder)
    np.testing.assert_allclose(
        restored.decision_function(features), model.decision_function(features)
    )
    for key, value in restored_encoder.state_dict().items():
        torch.testing.assert_close(value, template.state_dict()[key], rtol=0, atol=0)


def test_edge_frozen_encoder_incompatible_with_holistic_reinitialization():
    features, labels, template = _fixture()
    model = HolisticPUClassifier(
        encoder=set_encoder_training_policy(template, trainable=False),
        pseudo_pn_initialization="reinitialize",
        device="cpu",
    )
    with pytest.raises(ValueError, match="fixed encoder cannot be reset"):
        model.fit(features, labels)


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_param_policy_requires_an_explicit_boolean(value):
    _, _, template = _fixture()
    with pytest.raises(TypeError, match="boolean"):
        set_encoder_training_policy(template, trainable=value)
    assert not is_policy_encoder(template)
