"""Paper Eq. (4) uses unchanged raw inputs, even with in-place custom networks."""

# ruff: noqa: E402, N803, N806

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.p3mix import P3MixClassifier

pytestmark = pytest.mark.unit


def data(images=False):
    shape = (3, 4, 4) if images else (3,)
    features = np.random.RandomState(51).normal(size=(15, *shape)).astype(np.float32)
    return features, np.r_[np.ones(5, dtype=int), np.zeros(10, dtype=int)]


@pytest.mark.parametrize("variant", ["base", "e", "c"])
@pytest.mark.parametrize("images", [False, True])
def test_basic_determ_inplace_network_matches_out_of_place_training_and_repeated_scores(
    variant, images
):
    features, labels = data(images)
    originals = features.copy(), labels.copy()
    network = torch.nn.Sequential(
        torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Linear(int(np.prod(features.shape[1:])), 4)
    )
    if not images:
        network.append(torch.nn.Linear(4, 1))
    inplace = copy.deepcopy(network)
    inplace[0].inplace = True
    parameter = "encoder" if images else "model"
    fitted = []
    for template in (network, inplace):
        estimator = P3MixClassifier(
            **{parameter: template},
            variant=variant,
            hidden_dim=4,
            max_epochs=2,
            batch_size=4,
            pool_size=3,
            random_state=17,
            device="cpu",
        ).fit(features, labels)
        fitted.append(estimator)
        first = estimator.predict_proba(features)
        second = estimator.predict_proba(features)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(features, originals[0])
        np.testing.assert_array_equal(labels, originals[1])
    reference, actual = fitted
    assert reference.history_ == actual.history_
    np.testing.assert_array_equal(
        reference.candidate_training_positions_, actual.candidate_training_positions_
    )
    for name, value in reference.model_.state_dict().items():
        torch.testing.assert_close(actual.model_.state_dict()[name], value, rtol=0, atol=0)
    if variant == "e":
        for name, value in reference.teacher_.state_dict().items():
            torch.testing.assert_close(actual.teacher_.state_dict()[name], value, rtol=0, atol=0)
    np.testing.assert_array_equal(reference.predict_proba(features), actual.predict_proba(features))
    assert actual.training_provenance_["forward_input_isolation"] == "private_tensor_clone"


@pytest.mark.parametrize("images", [False, True])
def test_param_edge_mutation_then_failure_preserves_scoring_input_and_all_modes(images):
    features, labels = data(images)
    template = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(int(np.prod(features.shape[1:])), 4)
    )
    if not images:
        template.append(torch.nn.Linear(4, 1))
    estimator = P3MixClassifier(
        **{"encoder" if images else "model": template},
        max_epochs=1,
        batch_size=4,
        device="cpu",
    ).fit(features, labels)
    modes = [(module, module.training) for module in estimator.model_.modules()]
    modes[-1][0].training = True
    expected_modes = [module.training for module, _ in modes]

    def fail(inputs):
        inputs.add_(123)
        raise RuntimeError("injected forward failure")

    estimator.model_.forward = fail
    original = features.copy()
    with pytest.raises(RuntimeError, match="injected forward failure"):
        estimator.decision_function(features)
    np.testing.assert_array_equal(features, original)
    assert [module.training for module, _ in modes] == expected_modes
