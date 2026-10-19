"""P3Mix paper equations, full training path, and engineering boundaries."""

# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.estimators.deep.p3mix import (
    P3MixClassifier,
    _update_teacher,
    p3mix_batch_loss,
    p3mix_candidate_positions,
    p3mix_early_learning_regularizer,
    p3mix_heuristic_batch,
    p3mix_training_candidate_pool,
)

pytestmark = pytest.mark.unit


def test_edge_candidate_entropy_endpoints_and_stable_ties():
    np.testing.assert_array_equal(
        p3mix_candidate_positions([0, 0.5, 0.5, 1, 0.8], pool_size=3), [1, 2, 4]
    )
    assert len(p3mix_candidate_positions([0.2], pool_size=10)) == 1


def test_basic_training_positive_pool_excludes_unlabeled_rows_and_keeps_ids():
    features = np.arange(10.0).reshape(5, 2)
    labels = np.array([1, 0, 1, 0, 1])
    predictions = np.array([0.1, 0.5, 0.7, 0.5, 0.9])
    identities = np.array([15, 12, 19, 20, 18])
    original = features.copy()
    result = p3mix_training_candidate_pool(features, labels, predictions, identities, pool_size=2)
    np.testing.assert_array_equal(result["training_positions"], [2, 0])
    np.testing.assert_array_equal(result["sample_indices"], [19, 15])
    np.testing.assert_array_equal(result["features"], features[[2, 0]])
    assert np.all(labels[result["training_positions"]] == 1)
    result["features"][:] = -1
    np.testing.assert_array_equal(features, original)


@pytest.mark.parametrize(
    "labels,predictions,identities,error",
    [
        ([1, 0], [0.4, 0.5], [2, 2], "identities"),
        ([1, 0], [0.4, 0.5], [2.0, 3.0], "identities"),
        ([0, 0], [0.4, 0.5], [2, 3], "positives"),
        ([1, -1], [0.4, 0.5], [2, 3], "observed"),
        ([1, 0], [0.4, np.nan], [2, 3], "predictions"),
    ],
)
def test_training_pool_provenance_and_label_contract_fail_closed(
    labels, predictions, identities, error
):
    with pytest.raises(ValueError, match=error):
        p3mix_training_candidate_pool(
            [[1, 2], [3, 4]], labels, predictions, identities, pool_size=1
        )


def test_determ_marginal_candidates_modified_mixup_and_no_mutation():
    features = np.arange(14.0).reshape(7, 2)
    labels = np.array([1, 0, 0, 0, 0, 0, 0])
    probabilities = np.array([0.5, 0.25, 0.5, 0.75, 0.1, 0.9, 0.8])
    candidates = np.array([[100.0, 200.0]])
    original = features.copy(), labels.copy()
    result = p3mix_heuristic_batch(features, labels, probabilities, candidates, gamma=0.75)
    expected = np.array([False, True, True, True, False, False, False])
    np.testing.assert_array_equal(result["marginal_mask"], expected)
    assert np.all((result["lambdas"] >= 0.5) & (result["lambdas"] <= 1))
    lambdas = result["lambdas"][expected]
    np.testing.assert_allclose(result["soft_labels"][expected], 1 - lambdas)
    np.testing.assert_allclose(
        result["features"][expected],
        lambdas[:, None] * features[expected] + (1 - lambdas[:, None]) * candidates,
    )
    assert np.all(result["candidate_partner_positions"][expected] == 0)
    assert np.all(result["minibatch_partner_positions"][expected] == -1)
    np.testing.assert_array_equal(features, original[0])
    np.testing.assert_array_equal(labels, original[1])
    repeated = p3mix_heuristic_batch(features, labels, probabilities, candidates, gamma=0.75)
    for name in result:
        np.testing.assert_array_equal(result[name], repeated[name])


@pytest.mark.math
def test_separate_group_means_and_stopped_soft_target_gradients():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([1.0, -0.5, 0.3], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0.8, 0.1, 0.3], dtype=torch.float64, requires_grad=True)
    observed = torch.tensor([1, 0, 0])
    loss = p3mix_batch_loss(logits, targets, observed, unlabeled_weight=2)
    independent = (
        np.logaddexp(0, logits.detach().numpy())
        - targets.detach().numpy() * logits.detach().numpy()
    )
    np.testing.assert_allclose(loss.item(), independent[0] + 2 * independent[1:].mean())
    loss.backward()
    assert targets.grad is None and torch.isfinite(logits.grad).all()


def test_invalid_inputs_fail_closed():
    for probabilities in ([], [0.1, np.nan], [-1, 0.5], [[0.2]]):
        with pytest.raises(ValueError):
            p3mix_candidate_positions(probabilities, pool_size=1)
    with pytest.raises(ValueError, match="candidate_features"):
        p3mix_heuristic_batch([[1, 2]], [1], [0.5], [])
    with pytest.raises(ValueError, match="gamma"):
        p3mix_heuristic_batch([[1, 2]], [1], [0.5], [[1, 2]], gamma=0.4)


def test_basic_full_estimator_is_registered_with_unavailable_author_code_status():
    from pu_toolbox import P3MixClassifier as PublicP3MixClassifier
    from pu_toolbox.registry import get_algorithm, list_algorithms, register_all_builtin_methods

    register_all_builtin_methods()
    methods = {meta.name: meta for meta in list_algorithms()}
    assert "p3mix" in methods
    assert methods["p3mix"].source_status.value == "not_found"
    assert methods["p3mix"].implementation_status.value == "native"
    assert get_algorithm("p3mix") is PublicP3MixClassifier


@pytest.mark.parametrize("variant", ["base", "e", "c"])
def test_basic_full_estimator_smoke_variant_history_predictions_and_candidate_provenance(variant):
    pytest.importorskip("torch")

    rng = np.random.RandomState(23)
    X = rng.normal(size=(24, 3)).astype(np.float32)
    labels = np.r_[np.ones(8, dtype=int), np.zeros(16, dtype=int)]
    checkpoint_contexts = []

    def checkpoint_callback(epoch, fitted):
        checkpoint_contexts.append(
            (
                epoch,
                fitted.checkpoint_stage_,
                fitted.checkpoint_stage_epoch_,
                fitted.checkpoint_round_,
                fitted.optimizer_steps_,
            )
        )

    estimator = P3MixClassifier(
        variant=variant, max_epochs=2, batch_size=4, pool_size=3, random_state=7
    ).fit(X, labels, epoch_callback=checkpoint_callback)
    assert estimator.epochs_completed_ == 2
    assert estimator.optimizer_steps_ > 0
    assert len(estimator.history_["loss"]) == 2
    assert estimator.history_["candidate_count"] == [3, 3]
    assert [entry[:3] for entry in checkpoint_contexts] == [(1, "train", 1), (2, "train", 2)]
    assert all(entry[3] is None for entry in checkpoint_contexts)
    assert set(estimator.candidate_training_positions_).issubset(
        set(estimator.train_positive_indices_)
    )
    assert estimator.training_provenance_["clean_labels_read"] is False
    assert estimator.training_provenance_["formal_admission"] is False
    scores = estimator.decision_function(X[:5])
    assert scores.shape == (5,) and np.isfinite(scores).all()
    probabilities = estimator.predict_proba(X[:5])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert estimator.predict(X[:5]).shape == (5,)
    assert (estimator.teacher_ is not None) is (variant == "e")
    if variant == "c":
        assert estimator.training_provenance_["pseudo_negative_correction"] is True


def test_basic_epoch_pool_refresh_matches_top_entropy_over_training_positives():
    pytest.importorskip("torch")

    rng = np.random.RandomState(29)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    labels = np.r_[np.ones(7, dtype=int), np.zeros(13, dtype=int)]
    estimator = P3MixClassifier(
        max_epochs=1, batch_size=5, pool_size=3, random_state=13
    ).fit(X, labels)
    positive_positions = estimator.train_positive_indices_
    probabilities = estimator.predict_proba(X[positive_positions])[:, 1]
    expected = positive_positions[
        p3mix_candidate_positions(probabilities, pool_size=estimator.pool_size)
    ]
    np.testing.assert_array_equal(estimator.candidate_training_positions_, expected)


def test_edge_training_contract_rejects_ts_and_failed_refit_invalidates_old_model():
    pytest.importorskip("torch")
    from pu_toolbox.core.exceptions import NotFittedError

    X = np.arange(36, dtype=np.float32).reshape(12, 3) / 10
    labels = np.r_[np.ones(4, dtype=int), np.zeros(8, dtype=int)]
    estimator = P3MixClassifier(max_epochs=1, batch_size=4, pool_size=2, random_state=3)
    with pytest.raises(ValueError, match="TS"):
        estimator.fit(X, labels, os_or_ts="ts")
    estimator.fit(X, labels)
    with pytest.raises(NotImplementedError, match="sample_weight"):
        estimator.fit(X, labels, sample_weight=np.ones(len(labels)))
    with pytest.raises(NotFittedError):
        estimator.predict(X[:1])


def test_edge_failed_epoch_and_argument_binding_invalidate_previous_fit():
    pytest.importorskip("torch")
    from pu_toolbox.core.exceptions import NotFittedError

    X = np.arange(36, dtype=np.float32).reshape(12, 3) / 10
    labels = np.r_[np.ones(4, dtype=int), np.zeros(8, dtype=int)]
    estimator = P3MixClassifier(max_epochs=1, batch_size=4, pool_size=2, random_state=3).fit(
        X, labels
    )
    with pytest.raises(TypeError):
        estimator.fit(X, labels, unexpected_fit_role=True)
    with pytest.raises(NotFittedError):
        estimator.predict(X[:1])

    estimator.fit(X, labels)

    def interrupt_after_epoch(_epoch, _fitted):
        raise KeyboardInterrupt("test interruption")

    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        estimator.fit(X, labels, epoch_callback=interrupt_after_epoch)
    with pytest.raises(NotFittedError):
        estimator.decision_function(X[:1])


def test_basic_prediction_preserves_per_module_training_modes_and_empty_input():
    pytest.importorskip("torch")

    X = np.random.RandomState(5).normal(size=(16, 3)).astype(np.float32)
    labels = np.r_[np.ones(6, dtype=int), np.zeros(10, dtype=int)]
    estimator = P3MixClassifier(max_epochs=1, batch_size=4, pool_size=2, random_state=1).fit(
        X, labels
    )
    modules = list(estimator.model_.modules())
    if len(modules) > 1:
        modules[-1].training = True
    before = [module.training for module in modules]
    assert estimator.decision_function(np.empty((0, 3), dtype=np.float32)).shape == (0,)
    assert [module.training for module in modules] == before
    estimator.decision_function(X[:3])
    assert [module.training for module in modules] == before


@pytest.mark.math
def test_basic_early_learning_regularizer_matches_paper_equation_and_detaches_teacher():
    torch = pytest.importorskip("torch")

    student = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([0.8, -0.1], dtype=torch.float64, requires_grad=True)
    actual = p3mix_early_learning_regularizer(student, teacher)
    p, q = student.detach().sigmoid().numpy(), teacher.detach().sigmoid().numpy()
    expected = np.log(1 - ((1 - p) * (1 - q) + p * q)).mean()
    np.testing.assert_allclose(actual.item(), expected)
    actual.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


@pytest.mark.math
def test_edge_early_learning_regularizer_is_finite_for_identical_saturated_targets():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([1000.0, -1000.0])
    loss = p3mix_early_learning_regularizer(logits, logits)
    assert torch.isfinite(loss)
    assert -20 < loss.item() < 0


@pytest.mark.math
def test_basic_teacher_ema_updates_float_state_and_copies_integer_buffers():
    torch = pytest.importorskip("torch")
    from torch import nn

    student = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    teacher = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    with torch.no_grad():
        for value in student.state_dict().values():
            value.fill_(4)
        for value in teacher.state_dict().values():
            value.zero_()
    _update_teacher(student, teacher, decay=0.75)
    state = teacher.state_dict()
    torch.testing.assert_close(state["0.weight"], torch.ones_like(state["0.weight"]))
    torch.testing.assert_close(state["1.running_mean"], torch.ones_like(state["1.running_mean"]))
    torch.testing.assert_close(state["1.running_var"], torch.ones_like(state["1.running_var"]))
    assert torch.equal(
        state["1.num_batches_tracked"], student.state_dict()["1.num_batches_tracked"]
    )


def test_basic_p3mix_c_corrects_high_confidence_u_but_keeps_u_loss_role():
    torch = pytest.importorskip("torch")

    features = np.random.RandomState(12).normal(size=(12, 3)).astype(np.float32)
    labels = np.r_[np.ones(4, dtype=int), np.zeros(8, dtype=int)]
    model = torch.nn.Linear(3, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(3.0)
    estimator = P3MixClassifier(
        variant="c",
        model=model,
        gamma=0.9,
        max_epochs=1,
        batch_size=4,
        pool_size=2,
        random_state=2,
    ).fit(features, labels)
    assert estimator.history_["corrected_count"][0] == len(labels) - int(labels.sum())
    assert np.isfinite(estimator.history_["mixup_loss"][0])


def test_basic_epoch_checkpoints_bind_p3mix_stage_and_restore_predictions(tmp_path):
    pytest.importorskip("torch")
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    features = np.random.RandomState(15).normal(size=(16, 3)).astype(np.float32)
    labels = np.r_[np.ones(6, dtype=int), np.zeros(10, dtype=int)]
    estimator = P3MixClassifier(max_epochs=2, batch_size=4, pool_size=2, random_state=4)
    trajectory = EpochCheckpointTrainer(checkpoint_dir=tmp_path).fit(estimator, features, labels)
    assert len(trajectory.checkpoints) == 2
    assert [checkpoint.training_context["stage"] for checkpoint in trajectory.checkpoints] == [
        "train",
        "train",
    ]
    stage_epochs = [
        checkpoint.training_context["stage_epoch"] for checkpoint in trajectory.checkpoints
    ]
    assert stage_epochs == [1, 2]
    np.testing.assert_array_equal(
        trajectory.checkpoints[-1].restore().decision_function(features),
        trajectory.model.decision_function(features),
    )


def test_basic_explicit_cnn_encoder_trains_without_mutating_template():
    torch = pytest.importorskip("torch")

    encoder = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 6), torch.nn.ReLU()
    )
    before = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    features = np.random.RandomState(19).normal(size=(12, 3, 4, 4)).astype(np.float32)
    labels = np.r_[np.ones(4, dtype=int), np.zeros(8, dtype=int)]
    estimator = P3MixClassifier(
        encoder=encoder, max_epochs=1, batch_size=3, pool_size=2, random_state=5
    ).fit(features, labels)
    assert estimator.input_shape_ == (3, 4, 4)
    assert estimator.model_[0] is estimator.encoder_
    assert estimator.decision_function(features[:2]).shape == (2,)
    trained = estimator.encoder_.state_dict()
    assert any(not torch.equal(trained[name].cpu(), value) for name, value in before.items())
    for name, value in before.items():
        torch.testing.assert_close(encoder.state_dict()[name], value)
