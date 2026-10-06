"""Built-in CNN backbones integrate without silently borrowing selection labels."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox import build_encoder  # noqa: E402
from pu_toolbox.estimators.deep.pulns import PULNSClassifier  # noqa: E402
from pu_toolbox.workflows import PUPipeline  # noqa: E402
from pu_toolbox.workflows._models import cnn_capable_classifier_names  # noqa: E402

pytestmark = pytest.mark.integration


def images():
    rng = np.random.RandomState(3)
    features = rng.normal(size=(17, 3, 8, 8)).astype("float32")
    labels = np.r_[np.ones(6, int), np.zeros(11, int)]
    support = (rng.normal(size=(6, 3, 8, 8)).astype("float32"), np.array([0, 1] * 3))
    return features, labels, support


@pytest.mark.parametrize("backbone", ["cnn13", "resnet18"])
def test_basic_determ_native_backbone_support_training_and_independent_fold_copies(backbone):
    features, labels, support = images()
    torch.manual_seed(3)
    encoder = build_encoder("cnn", backbone=backbone, in_channels=3)
    source = copy.deepcopy(encoder.state_dict())
    params = dict(
        encoder=encoder,
        hidden_dim=4,
        pretrain_epochs=1,
        episodes=1,
        classifier_epochs=1,
        batch_size=8,
        random_state=3,
        device="cpu",
    )
    first = PULNSClassifier(**params).fit(
        features,
        labels,
        support_data=support,
        train_indices=np.arange(17),
        support_indices=np.arange(17, 23),
    )
    second = PULNSClassifier(**params).fit(features, labels, support_data=support)
    np.testing.assert_array_equal(
        first.decision_function(features), second.decision_function(features)
    )
    assert first.history_ == second.history_
    assert first.support_isolation_status_ == "train_support_ids_disjoint"
    assert first.training_view_ == "os" and not first.calibration_applied_
    assert (
        next(first.encoder_.parameters()).data_ptr()
        != next(second.encoder_.parameters()).data_ptr()
    )
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, source[key], rtol=0, atol=0)


def test_param_edge_pipeline_architecture_routing_does_not_supply_hidden_reward_labels():
    features, labels, support = images()
    pipe = PUPipeline(classifier="pulns", architecture="cnn", cv=2, max_epochs=1, device="cpu")
    assert "pulns" in cnn_capable_classifier_names()
    estimator = pipe._fresh_estimator(pipe._classifier_cls, None, None)
    with pytest.raises(ValueError, match="clean support_data; PA-ineligible"):
        pipe._fit_and_evaluate_one(
            estimator, features, labels, features, labels, np.zeros(len(labels), int), None, None
        )
    assert not estimator._is_fitted
    with pytest.raises(ValueError, match="ts risk substitution"):
        estimator.fit(features, labels, support_data=support, os_or_ts="ts")
    assert not estimator._is_fitted
