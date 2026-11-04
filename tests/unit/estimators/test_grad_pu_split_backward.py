"""Split backward and GP chunking preserve GradPU's joint objective."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.grad_pu import GradPUClassifier  # noqa: E402

pytestmark = pytest.mark.unit


def _data():
    rng = np.random.RandomState(9)
    positive = rng.normal(1.0, 0.4, (8, 3)).astype(np.float32)
    unlabeled = rng.normal(-0.5, 0.8, (16, 3)).astype(np.float32)
    return np.vstack([positive, unlabeled]), np.r_[np.ones(8, int), np.zeros(16, int)]


def _fit(**kwargs):
    return GradPUClassifier(
        hidden_dim=6,
        batch_size=5,
        max_epochs=2,
        random_state=3,
        device="cpu",
        **kwargs,
    ).fit(*_data(), os_or_ts="ts")


def _assert_close(reference, candidate, atol=1e-5, rtol=1e-5):
    assert reference.history_.keys() == candidate.history_.keys()
    for key in reference.history_:
        np.testing.assert_allclose(
            reference.history_[key], candidate.history_[key], atol=atol, rtol=rtol
        )
    for left, right in zip(
        reference.model_.parameters(), candidate.model_.parameters(), strict=False
    ):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def test_basic_split_backward_matches_joint_history_and_parameters():
    joint = _fit(backward_mode="joint")
    split = _fit(backward_mode="split")
    _assert_close(joint, split)
    assert split.training_provenance_["activation_memory"]["backward_mode"] == "split"


def test_determ_chunked_gp_matches_joint_history_and_parameters():
    joint = _fit(backward_mode="joint")
    chunked = _fit(backward_mode="split", gp_chunk_size=3)
    _assert_close(joint, chunked)
    assert chunked.training_provenance_["activation_memory"]["gp_chunk_size"] == 3


def test_edge_zero_alpha_split_matches_joint_without_gp_branch():
    joint = _fit(alpha=0.0, backward_mode="joint")
    split = _fit(alpha=0.0, backward_mode="split")
    _assert_close(joint, split, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("invalid", ["auto", "", None, True])
def test_param_invalid_backward_mode_is_rejected(invalid):
    estimator = GradPUClassifier(backward_mode=invalid)
    with pytest.raises(ValueError, match="backward_mode"):
        estimator.fit(*_data(), os_or_ts="ts")


@pytest.mark.parametrize("invalid", [0, -1, 2.5, "162", True])
def test_param_invalid_gp_chunk_size_is_rejected(invalid):
    estimator = GradPUClassifier(gp_chunk_size=invalid)
    with pytest.raises(ValueError, match="gp_chunk_size"):
        estimator.fit(*_data(), os_or_ts="ts")


def test_param_gp_chunk_size_requires_split_backward_mode():
    estimator = GradPUClassifier(backward_mode="joint", gp_chunk_size=3)
    with pytest.raises(ValueError, match="requires backward_mode"):
        estimator.fit(*_data(), os_or_ts="ts")


@pytest.mark.parametrize(
    "layer",
    [
        torch.nn.Dropout(0.2),
        torch.nn.Dropout1d(0.2),
        torch.nn.Dropout2d(0.2),
        torch.nn.Dropout3d(0.2),
        torch.nn.AlphaDropout(0.2),
        torch.nn.FeatureAlphaDropout(0.2),
        torch.nn.RReLU(),
        torch.nn.LSTM(3, 3, num_layers=2, dropout=0.2),
        torch.nn.MultiheadAttention(3, 1, dropout=0.2),
    ],
)
@pytest.mark.parametrize("in_encoder", [False, True])
def test_edge_stochastic_chunking_rejected_before_training(layer, in_encoder):
    component = torch.nn.Sequential(torch.nn.Linear(3, 3), layer)
    options = (
        {"encoder": component}
        if in_encoder
        else {"model": torch.nn.Sequential(component, torch.nn.Linear(3, 1))}
    )
    with pytest.raises(ValueError, match="unsupported stochastic layer"):
        _fit(backward_mode="split", gp_chunk_size=3, **options)


@pytest.mark.parametrize("alpha,chunk,p", [(0.0, 3, 0.2), (0.1, None, 0.2), (0.1, 3, 0.0)])
def test_basic_inactive_gp_unchunked_and_zero_dropout_remain_supported(alpha, chunk, p):
    model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(p), torch.nn.Linear(6, 1))
    estimator = _fit(model=model, alpha=alpha, backward_mode="split", gp_chunk_size=chunk)
    assert np.isfinite(estimator.history_["train_risk"]).all()


def test_determ_zero_dropout_preserves_joint_chunked_parameters():
    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(0.0), torch.nn.Linear(6, 1))
    _assert_close(_fit(model=model), _fit(model=model, backward_mode="split", gp_chunk_size=3))
