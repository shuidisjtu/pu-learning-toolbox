"""Synthetic neural-LBE CNN/runner evidence, not frozen Survey admission."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import NeuralLBEClassifier  # noqa: E402
from pu_toolbox.experiment import DatasetPart  # noqa: E402
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402
from pu_toolbox.experiment.runner import ExperimentRunner  # noqa: E402

pytestmark = pytest.mark.unit


def model(**kwargs):
    params = dict(
        hidden_dim=4,
        pretrain_steps=1,
        n_em_iter=2,
        m_steps=2,
        sample_fraction=0.5,
        batch_size=8,
        device="cpu",
        random_state=0,
    )
    params.update(kwargs)
    return NeuralLBEClassifier(**params)


def parts():
    rng = np.random.RandomState(4)
    labels = np.tile([0, 1], 16)
    return [
        DatasetPart(
            X=rng.normal(size=(32, 3)),
            labels=labels.copy(),
            indices=np.arange(role * 32, (role + 1) * 32),
            view="clean",
            for_selection=role != 3,
        )
        for role in range(4)
    ]


def test_basic_cnn_networks_are_independent_trainable_copies():
    x = np.random.RandomState(1).normal(size=(12, 1, 4, 4)).astype("float32")
    labels = np.r_[np.ones(4), np.zeros(8)]
    encoder = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.ReLU(), torch.nn.Flatten())
    before = copy.deepcopy(encoder.state_dict())
    fitted = model(encoder=encoder).fit(x, labels)
    assert fitted.model_[0] is not fitted.propensity_model_[0]
    assert (
        next(fitted.model_[0].parameters()).data_ptr()
        != next(fitted.propensity_model_[0].parameters()).data_ptr()
    )
    assert any(not torch.equal(before[key], fitted.model_[0].state_dict()[key]) for key in before)
    assert any(
        not torch.equal(before[key], fitted.propensity_model_[0].state_dict()[key])
        for key in before
    )
    for key in before:
        torch.testing.assert_close(encoder.state_dict()[key], before[key], rtol=0, atol=0)
    assert np.isfinite(fitted.predict_label_proba(x)).all()


def test_basic_checkpoint_classifier_recovery_and_full_pretrain_cost(tmp_path):
    bundle = parts()[0]
    labels = bundle.labels.copy()
    labels[np.flatnonzero(labels == 1)[::2]] = 0
    fitted = model()
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, bundle.X, labels)
    assert len(trajectory.checkpoints) == 2
    assert [item.training_context["optimizer_steps"] for item in trajectory.checkpoints] == [4, 6]
    np.testing.assert_allclose(
        trajectory.checkpoints[-1].restore().decision_function(bundle.X),
        fitted.decision_function(bundle.X),
        rtol=1e-5,
        atol=1e-6,
    )


def test_basic_four_role_runner_keeps_os_and_epoch_identity(tmp_path):
    path = tmp_path / "manifest.json"
    result = ExperimentRunner(
        seed=0,
        class_prior=0.5,
        manifest_path=str(path),
        config={
            "method": "lbe_neural",
            "c": 0.5,
            "os_or_ts": "os",
            "checkpoint_bytes_per_component": 4096,
        },
    ).fit(model(), *parts())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "technical_smoke"
    assert payload["run_view"] == "os-compatible" and not payload["calibration_applied"]
    assert set(result.test_metrics) == {"PA", "OA"}
    for metric in result.test_metrics.values():
        assert np.isfinite(metric["accuracy"]) and np.isfinite(metric["auc"])
    references = payload["candidate_runs"][0]["epoch_checkpoints"]
    assert len(references) == 2
    assert all(item["training_context"]["stage"] == "em" for item in references)


def test_determ_test_truth_and_features_cannot_change_training_or_selection(tmp_path):
    original = parts()
    changed = parts()
    changed[-1].X[:] += 100
    changed[-1].labels[:] = 1 - changed[-1].labels
    results = []
    for index, bundle in enumerate((original, changed)):
        result = ExperimentRunner(
            seed=0,
            class_prior=0.5,
            manifest_path=str(tmp_path / str(index) / "manifest.json"),
            config={"c": 0.5, "os_or_ts": "os", "checkpoint_bytes_per_component": 4096},
        ).fit(model(), *bundle)
        results.append(result)
    for name in ("PA", "OA"):
        np.testing.assert_array_equal(
            results[0].selected_models[name].decision_function(original[0].X),
            results[1].selected_models[name].decision_function(original[0].X),
        )


def test_param_ts_runner_fails_loud_without_relabeling_as_os(tmp_path):
    with pytest.raises(RuntimeError, match="all candidate runs failed"):
        ExperimentRunner(
            seed=0,
            class_prior=0.5,
            manifest_path=str(tmp_path / "manifest.json"),
            config={"c": 0.5, "os_or_ts": "ts"},
        ).fit(model(), *parts())
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["selection"] == {} and payload["test_results"] == {}
    assert "TS is not applicable" in json.dumps(payload["failures"])


def test_edge_bad_image_and_prediction_shapes_do_not_flatten_silently():
    images = np.ones((6, 1, 4, 4))
    with pytest.raises(ValueError, match="explicit encoder"):
        model().fit(images, np.array([1, 1, 0, 0, 0, 0]))
    bundle = parts()[0]
    fitted = model().fit(bundle.X, bundle.labels)
    with pytest.raises(ValueError, match="shape"):
        fitted.predict(np.zeros((2, 4)))
