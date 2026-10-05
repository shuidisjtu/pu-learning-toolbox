# ruff: noqa: N806
"""End-to-end supervised oracle support does not change the frozen pilot."""

import copy

import numpy as np
import pytest
import torch
from sklearn.base import clone
from torch import nn

from pu_toolbox.experiment.survey_execution import PilotOracleCNN, assemble_model
from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_unit

pytestmark = pytest.mark.unit


def test_basic_encoder_is_trained_without_mutating_source_and_callbacks_are_safe():
    torch.manual_seed(9)
    encoder = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    encoder.requires_grad_(False)
    before = copy.deepcopy(encoder.state_dict())
    model = PilotOracleCNN(encoder=encoder, max_epochs=2, batch_size=4, random_state=3)
    model = clone(model)
    X = np.random.RandomState(3).normal(size=(8, 3, 4, 4)).astype("float32")
    y = np.array([0, 1] * 4)
    callbacks = []

    def callback(epoch, fitted):
        callbacks.append(epoch)
        assert fitted.model_.training
        assert fitted.decision_function(X).shape == (8,)
        assert fitted.model_.training

    model.fit(X, y, epoch_callback=callback)
    assert callbacks == [0, 1]
    assert not model.model_.training
    assert not torch.equal(model.model_[0][0].weight, before["0.weight"])
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, before[key])
    assert model.predict(X).shape == (8,)
    assert model.decision_function(X[:0]).shape == (0,)


def test_seed_determinism_for_end_to_end_supervised_encoder():
    torch.manual_seed(9)
    encoder = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    images = np.random.RandomState(3).normal(size=(8, 3, 4, 4)).astype("float32")
    labels = np.tile([0, 1], 4)
    scores = []
    for _ in range(2):
        fitted = PilotOracleCNN(
            encoder=encoder, max_epochs=2, batch_size=4, random_state=3, device="cpu"
        ).fit(images, labels)
        scores.append(fitted.decision_function(images))
    np.testing.assert_array_equal(scores[0], scores[1])


def test_param_edge_cnn_rejects_flattened_input_and_missing_encoder():
    X = np.ones((4, 3, 4, 4), dtype="float32")
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="encoder"):
        PilotOracleCNN(encoder=None, max_epochs=1).fit(X, y)
    with pytest.raises(ValueError, match="supported inputs"):
        PilotOracleCNN(encoder=nn.Identity(), max_epochs=1).fit(X.reshape(4, -1), y)


def test_assembly_support_is_separate_from_frozen_matrix_admission():
    protocol = load_protocol()
    with pytest.raises(ValueError):
        resolve_unit(protocol, "cifar10", "pn_oracle", "native_cnn")
    row = {"method": "pn_oracle", "training_path": "native_cnn"}
    model = assemble_model(
        protocol, row, 512, seed=0, params={}, class_prior=None, device="cpu", encoder=nn.Identity()
    )
    assert isinstance(model, PilotOracleCNN)
    assert "class_prior" not in model.get_params()


def test_runner_oa_checkpoint_recovery_and_reclaim(tmp_path):
    from pu_toolbox.experiment.bundle import DatasetBundle, DatasetPart
    from pu_toolbox.experiment.runner import ExperimentRunner
    from pu_toolbox.experiment.strategies import CleanLabelGenerator, ProtocolOA, SupervisedTrainer

    rng = np.random.RandomState(3)
    bundle = DatasetBundle(
        **{
            role: DatasetPart(
                X=rng.normal(size=(8, 3, 4, 4)).astype("float32"),
                labels=np.array([0, 1] * 4),
                indices=np.arange(i * 8, (i + 1) * 8),
                view="clean",
                for_selection=role != "test",
            )
            for i, role in enumerate(("train", "pu_val", "clean_val", "test"))
        }
    )
    encoder = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    result = ExperimentRunner(
        generator=CleanLabelGenerator(),
        protocols=[ProtocolOA()],
        manifest_path=str(tmp_path / "manifest.json"),
        config={"trainer": SupervisedTrainer(), "reclaim_unselected_checkpoints": True},
    ).fit(
        PilotOracleCNN(encoder=encoder, max_epochs=3, batch_size=4, random_state=0),
        bundle.train,
        bundle.pu_val,
        bundle.clean_val,
        bundle.test,
    )
    assert set(result.manifest["selection"]) == {"OA"}
    references = result.manifest["candidate_runs"][0]["epoch_checkpoints"]
    assert len(references) == 3
    assert sum(not reference["reclaimed"] for reference in references) == 1
    assert len(list(tmp_path.rglob("*.pt"))) == 1
