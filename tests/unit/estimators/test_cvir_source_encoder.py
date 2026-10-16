# ruff: noqa: N803, N806
"""CVIR paper/source distinctions, CNN streaming and stage provenance."""

import copy
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from pu_toolbox.core.exceptions import NotFittedError, ValidationError  # noqa: E402
from pu_toolbox.estimators.risk.cvir import (  # noqa: E402
    CVIRClassifier,
    _cvir_objective,
    select_cvir_negatives,
)
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer  # noqa: E402

pytestmark = pytest.mark.unit


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, padding=1)
        self.shapes = []

    def forward(self, X):
        self.shapes.append(tuple(X.shape))
        return self.conv(X).tanh().mean((2, 3))


def data(images=True):
    shape = (19, 1, 4, 4) if images else (19, 3)
    return np.random.RandomState(4).normal(size=shape).astype("float32"), np.r_[
        np.ones(6), np.zeros(13)
    ]


def model(**kwargs):
    params = dict(
        unlabeled_positive_prior=0.3,
        hidden_dim=4,
        warm_start_epochs=1,
        max_epochs=2,
        batch_size=4,
        random_state=2,
        device="cpu",
    )
    params.update(kwargs)
    return CVIRClassifier(**params)


@pytest.mark.math
@pytest.mark.parametrize("reweight", [False, True])
def test_basic_paper_weighted_loss_and_gradients_not_released_balanced_loss(reweight):
    p = torch.tensor([-2.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    n = torch.tensor([-1.0, 2.0], dtype=torch.float64, requires_grad=True)
    alpha = 0.3
    actual = _cvir_objective(p, n, alpha, reweight=reweight)
    wp, wn = (alpha, 1 - alpha) if reweight else (1.0, 1.0)
    expected = (
        wp * np.logaddexp(0, -p.detach().numpy()).mean()
        + wn * np.logaddexp(0, n.detach().numpy()).mean()
    )
    assert actual.item() == pytest.approx(expected, abs=1e-14)
    actual.backward()
    np.testing.assert_allclose(
        p.grad.numpy(), wp * (torch.sigmoid(p).detach().numpy() - 1) / len(p), atol=1e-14
    )
    np.testing.assert_allclose(
        n.grad.numpy(), wn * torch.sigmoid(n).detach().numpy() / len(n), atol=1e-14
    )
    balanced = (_cvir_objective(p, n, alpha, reweight=False) / 2).item()
    if reweight:
        assert actual.item() != pytest.approx(balanced)


def test_edge_finite_count_differs_from_author_discard_floor_without_default_change():
    scores = np.arange(7, dtype=float)
    selected = select_cvir_negatives(scores, 0.3)
    assert len(selected) == 4
    assert len(scores) - int(0.3 * len(scores)) == 5
    np.testing.assert_array_equal(selected, [0, 1, 2, 3])
    assert len(select_cvir_negatives(scores, 0.999)) == 1


def test_basic_cnn_is_copied_jointly_trained_and_all_device_forwards_are_bounded():
    encoder = ImageEncoder()
    before = copy.deepcopy(encoder.state_dict())
    X, y = data()
    expected = None
    original_u = np.flatnonzero(y == 0)

    def check_rank_before_update(epoch, current):
        nonlocal expected
        if expected is not None:
            np.testing.assert_array_equal(current.selected_negative_indices_, expected)
        scores = current.decision_function(X[original_u])
        expected = original_u[np.argsort(scores, kind="stable")[:9]]

    fitted = model(encoder=encoder).fit(X, y, epoch_callback=check_rank_before_update)
    assert fitted.encoder_ is not encoder and fitted.model_[0] is fitted.encoder_
    assert fitted.data_device_policy_ == "cpu_stream" and fitted.training_data_device_ == "cpu"
    assert any(not torch.equal(before[k], fitted.encoder_.state_dict()[k]) for k in before)
    assert encoder.shapes == []
    for k, v in encoder.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert fitted.predict(X).shape == (19,)
    assert all(0 < shape[0] <= 4 and shape[1:] == (1, 4, 4) for shape in fitted.encoder_.shapes)
    assert set(fitted.selected_negative_indices_) <= set(np.flatnonzero(y == 0))
    with torch.no_grad():
        # Last-round selection was made before its updates; explicitly recompute
        # a current rank to exercise CPU-streamed prediction on the original U.
        selected = select_cvir_negatives(fitted.decision_function(X[y == 0]), 0.3)
    assert len(selected) == 9 and fitted.history_["selected_negative_count"] == [9, 9]


@pytest.mark.parametrize("warm_start", [0, 1, 2])
def test_basic_stage_snapshots_include_all_cost_and_exact_streamed_replay(tmp_path, warm_start):
    X, y = data()
    fitted = model(encoder=ImageEncoder(), warm_start_epochs=warm_start)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(fitted, X, y)
    assert fitted.checkpoint_epoch_count == len(trajectory.checkpoints) == warm_start + 2
    assert [p["stage"] for p in fitted.epoch_provenance_] == ["warm_start"] * warm_start + [
        "cvir"
    ] * 2
    # max(ceil(6/4),ceil(13/4))=4 warm steps; ceil(9/4)=3 retained steps.
    assert fitted.optimizer_steps_ == warm_start * 4 + 6
    assert fitted.epoch_provenance_[-1]["optimizer_steps"] == fitted.optimizer_steps_
    for checkpoint, context in zip(trajectory.checkpoints, fitted.epoch_provenance_, strict=True):
        ref = checkpoint.reference()
        assert ref["training_context"] == context and ref["prediction_batch_size"] == 4
        assert not ref["training_resume_supported"]
    replay = trajectory.checkpoints[-1].restore(device="cpu")
    replay.model_[0].shapes.clear()
    np.testing.assert_array_equal(replay.decision_function(X), fitted.decision_function(X))
    assert [shape[0] for shape in replay.model_[0].shapes] == [4, 4, 4, 4, 3]


def test_determ_cpu_stream_and_legacy_resident_training_match_on_features():
    X, y = data(images=False)
    resident = model().fit(X, y)
    explicit = model(data_device_policy="resident").fit(X, y)
    streamed = model(data_device_policy="cpu_stream").fit(X, y)
    assert resident.data_device_policy_ == "resident"
    assert resident.history_ == explicit.history_ == streamed.history_
    np.testing.assert_array_equal(
        resident.selected_negative_indices_, streamed.selected_negative_indices_
    )
    # Different GEMM batch shapes can change a few float32 ulps, not the risk rule.
    np.testing.assert_allclose(
        resident.decision_function(X), streamed.decision_function(X), atol=1e-7
    )


def test_determ_encoder_clone_pickle_and_refit_keep_template_and_scores():
    X, y = data()
    template = model(encoder=ImageEncoder())
    first, second = clone(template).fit(X, y), clone(template).fit(X, y)
    assert first.history_ == second.history_
    np.testing.assert_array_equal(first.decision_function(X), second.decision_function(X))
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(first.decision_function(X), restored.decision_function(X))


@pytest.mark.parametrize("policy", [None, [], "gpu_all"])
def test_param_invalid_data_policy_rejected(policy):
    with pytest.raises(ValueError, match="data_device_policy"):
        model(data_device_policy=policy).fit(*data(images=False))


def test_param_encoder_and_image_contracts_fail_loud():
    with pytest.raises(ValidationError, match="2-D"):
        model().fit(*data())
    with pytest.raises(TypeError, match="torch.nn.Module"):
        model(encoder="cnn").fit(*data())
    with pytest.raises(ValueError, match="CPU-streamed"):
        model(encoder=ImageEncoder(), data_device_policy="resident").fit(*data())
    with pytest.raises(ValueError, match="2-D"):
        model(encoder=torch.nn.Identity()).fit(*data())
    fitted = model(encoder=ImageEncoder()).fit(*data())
    with pytest.raises(ValueError, match="shape"):
        fitted.predict(np.zeros((1, 1, 3, 4)))
    assert fitted.decision_function(np.empty((0, 1, 4, 4))).shape == (0,)


def test_edge_late_callback_error_invalidates_fit_and_invalid_refit_drops_old_fit():
    X, y = data()
    fitted = model(encoder=ImageEncoder())
    completed = []

    def callback(epoch, current):
        assert current.predict(X).shape == (19,)
        completed.append(epoch)
        if epoch == 1:
            raise RuntimeError("failed epoch callback")

    with pytest.raises(RuntimeError, match="failed epoch callback"):
        fitted.fit(X, y, epoch_callback=callback)
    assert completed == [0, 1]
    with pytest.raises(NotFittedError):
        fitted.predict(X)
    fitted.fit(X, y).set_params(unlabeled_positive_prior=None)
    with pytest.raises(ValueError, match="unlabeled_positive_prior"):
        fitted.fit(X, y)
    with pytest.raises(NotFittedError):
        fitted.predict(X)


def test_basic_cnn_does_not_add_population_prior_or_ts_calibration():
    X, y = data()
    with pytest.raises(ValueError, match="population prior"):
        model(encoder=ImageEncoder()).fit(X, y, class_prior=0.3)
    assert "os_or_ts" not in inspect.signature(CVIRClassifier.fit).parameters
    from pu_toolbox.experiment.method_ledger import load_ledger
    from pu_toolbox.experiment.training_views import resolve_training_view

    with pytest.raises(ValueError, match="calibration_hooked=false"):
        resolve_training_view(
            load_ledger(), "cvir", "ts", is_oracle=False, estimator_class=CVIRClassifier
        )


def test_basic_source_receipt_pins_distinctions_not_source_replay_or_admission():
    root = Path(__file__).resolve().parents[3]
    receipt = json.loads(
        (root / "docs/research/pu_survey/data/cvir_source_review_20261015.json").read_text(
            encoding="utf-8"
        )
    )
    for name, value in receipt["toolbox_defaults"].items():
        assert inspect.signature(CVIRClassifier).parameters[name].default == value
    assert receipt["source"]["commit"] == "5e5e350dc0588de95a36eb952e3cf5382e786aec"
    assert receipt["chosen_objective"] == "paper_algorithm_2_alpha_weighted"
    assert not receipt["author_code_replay"] and not receipt["formal_admission"]
    assert not receipt["calibration_applied"] and receipt["candidate_protocol_ref"] is None
    assert receipt["owner_signatures"] == []


def test_basic_public_cnn_pipeline_routes_encoder_and_explicit_alpha(monkeypatch):
    from pu_toolbox.estimators.deep import vision
    from pu_toolbox.workflows import PUPipeline

    monkeypatch.setattr(vision, "build_encoder", lambda *args, **kwargs: ImageEncoder())
    X, y = data()
    report = PUPipeline(
        classifier="cvir",
        architecture="cnn",
        prior_estimator=None,
        device="cpu",
        cv=2,
        classifier_params=dict(
            unlabeled_positive_prior=0.3,
            hidden_dim=4,
            max_epochs=1,
            warm_start_epochs=0,
            batch_size=4,
        ),
    ).fit_evaluate(X, y)
    assert report.final_model.predict(X).shape == (19,)
    assert report.final_model.unlabeled_positive_prior_ == 0.3
    assert report.final_model.data_device_policy_ == "cpu_stream"
