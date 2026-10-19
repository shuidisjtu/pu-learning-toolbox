"""P3Mix paper equations and pure candidate/mixup helpers."""

# ruff: noqa: N803, N806

import numpy as np
import pytest

from pu_toolbox.estimators.deep.p3mix import (
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


def test_edge_gamma_half_has_empty_marginal_set_as_paper_states():
    features = np.arange(6.0).reshape(3, 2)
    result = p3mix_heuristic_batch(
        features,
        [1, 0, 0],
        [0.5, 0.5, 0.6],
        [[10.0, 20.0]],
        gamma=0.5,
    )
    assert not result["marginal_mask"].any()


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
    for gamma, alpha in ((True, 1.0), (0.9, True), ("0.9", 1.0), (0.9, "1.0")):
        with pytest.raises(ValueError):
            p3mix_heuristic_batch([[1, 2]], [1], [0.5], [[1, 2]], gamma=gamma, alpha=alpha)
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])
    observed = torch.tensor([1, 0])
    for weight in (True, "1.0"):
        with pytest.raises(ValueError, match="unlabeled_weight"):
            p3mix_batch_loss(logits, targets, observed, unlabeled_weight=weight)


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
