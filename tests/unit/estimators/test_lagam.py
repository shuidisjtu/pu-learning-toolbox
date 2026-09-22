"""LaGAM meta-gradient, latent-group loss and clean-support boundary."""

# ruff: noqa: N806

import copy
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.advisor import recommend_methods  # noqa: E402
from pu_toolbox.core.exceptions import NotFittedError  # noqa: E402
from pu_toolbox.estimators.deep.lagam import (  # noqa: E402
    LaGAMClassifier,
    _balanced_soft_bce,
    lagam_contrastive_loss,
    lagam_meta_labels,
)
from pu_toolbox.registry import (  # noqa: E402
    AlgorithmMetadata,
    get_algorithm,
    get_metadata,
    register_all_builtin_methods,
)

pytestmark = pytest.mark.unit


def _data():
    rng = np.random.RandomState(11)
    X = np.vstack([rng.normal(1, 0.4, (8, 3)), rng.normal(-0.5, 0.7, (16, 3))])
    y = np.r_[np.ones(8, int), np.zeros(16, int)]
    support_X = np.vstack([rng.normal(1, 0.4, (5, 3)), rng.normal(-1, 0.4, (5, 3))])
    support_y = np.r_[np.ones(5, int), np.zeros(5, int)]
    return X.astype(np.float32), y, (support_X.astype(np.float32), support_y)


def _model(**kwargs):
    args = dict(
        hidden_dim=8,
        warmup_epochs=1,
        max_epochs=2,
        batch_size=8,
        support_batch_size=10,
        num_clusters=3,
        random_state=5,
        device="cpu",
    )
    args.update(kwargs)
    return LaGAMClassifier(**args)


@pytest.mark.math
def test_balanced_bce_matches_per_class_normalization():
    logits = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    labels = torch.tensor([0.0, 0.0, 1.0])
    loss = _balanced_soft_bce(logits, labels)
    expected = torch.nn.functional.softplus(logits[:2]).mean() + torch.nn.functional.softplus(
        -logits[2]
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None


@pytest.mark.math
def test_meta_label_uses_clean_support_gradient_and_locks_observed_positive():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    features = torch.tensor([[1.0], [-1.0], [-1.0]])
    observed = torch.tensor([0.0, 0.0, 1.0])
    support_features = torch.tensor([[1.0], [-1.0]])
    support_labels = torch.tensor([1.0, 0.0])
    detected = lagam_meta_labels(
        model, features, observed, support_features, support_labels, meta_lr=1.0
    )
    torch.testing.assert_close(detected, torch.tensor([1.0, 0.0, 1.0]))


@pytest.mark.math
def test_group_contrastive_is_finite_and_shape_checked():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    k = q + 0.1
    groups = torch.tensor([0, 1, 0])
    instance = lagam_contrastive_loss(q, k)
    grouped = lagam_contrastive_loss(q, k, groups)
    assert torch.isfinite(instance) and torch.isfinite(grouped)
    assert grouped != instance
    grouped.backward()
    assert q.grad is not None
    with pytest.raises(ValueError, match="groups"):
        lagam_contrastive_loss(q, k, torch.tensor([0]))


def test_missing_or_malformed_clean_support_fails_closed():
    X, y, support = _data()
    with pytest.raises(ValueError, match="clean support_data"):
        _model().fit(X, y)
    with pytest.raises(ValueError, match="both clean classes"):
        _model().fit(X, y, support_data=(support[0], np.ones(len(support[0]))))
    with pytest.raises(ValueError, match="feature shape"):
        _model().fit(X, y, support_data=(support[0][:, :2], support[1]))
    with pytest.raises(NotImplementedError, match="sample_weight"):
        _model().fit(X, y, support_data=support, sample_weight=np.ones(len(y)))


def test_registry_declares_support_and_pu_only_recommender_excludes_lagam():
    X, y, _ = _data()
    register_all_builtin_methods()
    assert get_algorithm("lagam") is LaGAMClassifier
    assert get_algorithm("la_gam") is LaGAMClassifier
    assert get_metadata("lagam").requires_clean_support is True
    result = recommend_methods(X, y, class_prior=0.4, top_k=100)
    assert "lagam" not in {candidate.name for candidate in result.candidates}
    assert result.filters_applied["clean_support_required"] == "excluded (not provided)"
    with pytest.raises(ValueError, match="requires_clean_support"):
        AlgorithmMetadata(name="invalid", paper="test", requires_clean_support="yes")


def test_fit_callback_weights_determinism_and_positive_lock():
    X, y, support = _data()
    clf = _model()
    with pytest.raises(NotFittedError):
        clf.predict(X)
    snapshots = []
    clf.fit(
        X,
        y,
        support_data=support,
        epoch_callback=lambda epoch, fitted: snapshots.append(
            (epoch, copy.deepcopy(fitted.model_.state_dict()))
        ),
    )
    assert [epoch for epoch, _ in snapshots] == [0, 1]
    assert clf.n_support_ == len(support[0])
    assert len(clf.history_["train_loss"]) == 2
    assert clf.history_["contrastive_loss"][0] > 0
    assert bool(torch.all(clf.pseudo_labels_[y == 1] == 1))
    scores = clf.decision_function(X)
    assert np.isfinite(scores).all() and scores.shape == (len(X),)
    np.testing.assert_array_equal(
        _model().fit(X, y, support_data=support).decision_function(X), scores
    )
    restored = copy.deepcopy(clf.model_)
    restored.load_state_dict(snapshots[-1][1])
    with torch.no_grad():
        np.testing.assert_allclose(restored(torch.as_tensor(X)).reshape(-1).numpy(), scores)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"warmup_epochs": 2}, "warmup_epochs"),
        ({"num_clusters": 99}, "num_clusters"),
        ({"temperature": 0}, "temperature"),
        ({"rho_start": 1}, "rho_start"),
        ({"support_batch_size": 0}, "support_batch_size"),
    ],
)
def test_invalid_parameters_fail(params, message):
    X, y, support = _data()
    with pytest.raises(ValueError, match=message):
        _model(**params).fit(X, y, support_data=support)


@pytest.mark.gpu
def test_cuda_smoke():
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    X, y, support = _data()
    fitted = _model(device="cuda").fit(X, y, support_data=support)
    assert next(fitted.model_.parameters()).device.type == "cuda"
    assert np.isfinite(fitted.decision_function(X)).all()
