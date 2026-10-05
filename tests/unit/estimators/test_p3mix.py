"""Author slide equations tested without claiming a full P3Mix estimator."""

import numpy as np
import pytest

from pu_toolbox.estimators.deep.p3mix import (
    p3mix_batch_loss,
    p3mix_candidate_positions,
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


def test_unverified_components_are_not_registered_as_classifier():
    from pu_toolbox.registry import list_algorithms, register_all_builtin_methods

    register_all_builtin_methods()
    assert "p3mix" not in {meta.name for meta in list_algorithms()}
