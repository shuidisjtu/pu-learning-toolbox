# ruff: noqa: N802, N803, N806, S101, E501

"""TS-OS training-view behaviour of NonNegativePUClassifier (protocol §2.3).

The calibrated view is `D_U^k <- D_U^k union D_P^k`: the labeled-positive batch
keeps feeding the positive loss **and** joins the unlabeled-loss input.  These
tests observe that union at the model boundary rather than re-deriving the
formula, and keep the OS default byte-identical to today's behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.config import POSITIVE_LABEL, UNLABELED_LABEL

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from pu_toolbox.estimators.risk.nnpu import NonNegativePUClassifier  # noqa: E402

pytestmark = pytest.mark.unit

# Sized so the two sides differ: batch_P = 6 (< batch_size), batch_U = 10.
N_P, N_U, N_FEATURES, BATCH_SIZE = 6, 20, 4, 10
#: Batch sizes the two loaders actually yield (both capped by ``batch_size``).
BATCH_P = min(BATCH_SIZE, N_P)
BATCH_U = min(BATCH_SIZE, N_U)


def _make_data(seed: int = 7):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.randn(N_P, N_FEATURES) + 1.0, rng.randn(N_U, N_FEATURES) - 1.0])
    y_pu = np.concatenate(
        [np.full(N_P, POSITIVE_LABEL, dtype=int), np.full(N_U, UNLABELED_LABEL, dtype=int)]
    )
    return X, y_pu


class _RecordingModel(torch.nn.Module):
    """Records the batch size of every forward pass it serves."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)
        self.batch_sizes: list[int] = []

    def forward(self, x):  # noqa: D102
        self.batch_sizes.append(int(x.shape[0]))
        return self.linear(x)


def _fit(os_or_ts=None, **kwargs):
    """Fit once with a fixed model init and return (classifier, trained model).

    The estimator deep-copies the injected model (``nnpu.py:300``), so the
    recording instance that actually served the forward passes is
    ``clf.model_``, not the object handed in.
    """
    X, y_pu = _make_data()
    torch.manual_seed(0)
    model = _RecordingModel(N_FEATURES)
    clf = NonNegativePUClassifier(model=model, max_epochs=1, batch_size=BATCH_SIZE, random_state=0)
    fit_kwargs = dict(class_prior=N_P / (N_P + N_U), **kwargs)
    if os_or_ts is not None:
        fit_kwargs["os_or_ts"] = os_or_ts
    clf.fit(X, y_pu, **fit_kwargs)
    return clf, clf.model_


class TestCalibratedUnlabeledInput:
    """The TS view must reach the model, not just the config."""

    def test_ts_view_feeds_positive_batch_into_unlabeled_loss(self):
        """A forward pass larger than batch_size proves P was appended to U."""
        _, model = _fit("ts")
        # No forward pass can exceed batch_size unless the union happened.
        assert max(model.batch_sizes) == BATCH_U + BATCH_P

    def test_os_view_never_exceeds_the_batch_size(self):
        """The uncalibrated path stays byte-identical to the pre-existing view."""
        _, model = _fit("os")
        assert max(model.batch_sizes) <= BATCH_SIZE

    def test_default_matches_explicit_os(self):
        """Omitting the argument must not change behaviour (OS is the default)."""
        default_clf, _ = _fit(None)
        os_clf, _ = _fit("os")
        assert default_clf.get_training_history() == os_clf.get_training_history()


class TestViewGates:
    """Fail-loud on combinations the protocol does not define."""

    def test_invalid_view_value_rejected(self):
        with pytest.raises(ValueError, match="os_or_ts"):
            _fit("ts-compatible")

    def test_ts_view_rejected_with_sample_weight(self):
        """Appended positives have no weight definition; refuse rather than guess."""
        X, y_pu = _make_data()
        torch.manual_seed(0)
        clf = NonNegativePUClassifier(
            model=_RecordingModel(N_FEATURES),
            max_epochs=1,
            batch_size=BATCH_SIZE,
            random_state=0,
        )
        with pytest.raises(ValueError, match="sample_weight"):
            clf.fit(
                X,
                y_pu,
                class_prior=N_P / (N_P + N_U),
                sample_weight=np.ones(len(y_pu)),
                os_or_ts="ts",
            )

    def test_ts_view_leaves_the_validation_role_uncalibrated(self):
        """Validation keeps the OS view; only the training batch is calibrated."""
        X, y_pu = _make_data()
        torch.manual_seed(0)
        clf = NonNegativePUClassifier(
            model=_RecordingModel(N_FEATURES),
            max_epochs=1,
            batch_size=BATCH_SIZE,
            random_state=0,
        )
        clf.fit(
            X,
            y_pu,
            class_prior=N_P / (N_P + N_U),
            validation_data=(X.copy(), y_pu.copy()),
            os_or_ts="ts",
        )
        # Validation scoring is one call over the full validation set.
        assert len(X) in clf.model_.batch_sizes
