# tests/unit/estimators/test_self_pu_view_invariants.py

# ruff: noqa: E402, N803, N806, S101

"""Structural invariants of the Self-PU training view.

The math suite pins the blend's value; this one pins *who is in the room*.  A
calibration that quietly widened the role set — counting trusted rows as
negative, letting validation drift with the training view, or handing the
manager a population that includes the labeled positives — would still produce
finite losses and plausible scores, so the defect is only visible structurally.

The role count is asserted exactly, not bounded: the frozen configuration runs
with the whole U pool in one batch, which makes "untrusted rows in this batch"
determined by the trusted history alone, with no RNG assumption.  Using the
batch size as the blend's denominator — the plausible mistake — then fails by
construction rather than by luck.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from pu_toolbox.estimators.deep import self_pu as self_pu_module  # noqa: E402
from pu_toolbox.estimators.deep.self_pu import (  # noqa: E402
    SelfPUClassifier,
    TrustedSetManager,
    calibrate_meta_weights,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings("ignore:validation_data was not supplied"),
]

# One batch covers the whole U pool (30 rows), so every trusted row is in it.
_CONF = dict(
    class_prior=1 / 3,
    hidden_dim=4,
    warmup_epochs=0,
    self_paced_start=0,
    self_paced_end=2,
    distill_start=2,
    max_epochs=3,
    max_trust_ratio=0.2,
    pace_1=0.1,
    pace_2=0.2,
    batch_size=64,
    random_state=5,
    device="cpu",
)

_N_UNLABELED = 30
_N_POSITIVE = 6


def _data():
    rng = np.random.RandomState(17)
    X = np.vstack([rng.normal(1.0, 0.4, (12, 4)), rng.normal(-1.0, 0.4, (24, 4))]).astype(
        np.float32
    )
    y_pu = np.r_[np.ones(6, dtype=int), np.zeros(30, dtype=int)]
    X_val = np.vstack([rng.normal(1.0, 0.4, (6, 4)), rng.normal(-1.0, 0.4, (6, 4))]).astype(
        np.float32
    )
    y_val = np.r_[np.ones(6, dtype=int), np.zeros(6, dtype=int)]
    y_pu_val = np.r_[np.ones(3, dtype=int), np.zeros(9, dtype=int)]
    return X, y_pu, X_val, y_val, y_pu_val


def _fit(view="os", overrides=None, validation="pu", **fit_kwargs):
    """``validation`` picks the branch: PU tracking, clean-label meta, or none."""
    X, y_pu, X_val, y_val, y_pu_val = _data()
    if not fit_kwargs:
        if validation == "pu":
            fit_kwargs = {"pu_validation_data": (X_val, y_pu_val)}
        elif validation == "clean":
            fit_kwargs = {"validation_data": (X_val, y_val)}
    model = SelfPUClassifier(**{**_CONF, **(overrides or {})})
    model.fit(X, y_pu, os_or_ts=view, **fit_kwargs)
    return model, X, y_pu


def _spy_on_blend(monkeypatch, calls):
    """Record every blend call as (U rows in role, P rows, calibrated flag)."""
    original = self_pu_module._marginal_negative_risk

    def spy(unlabeled_negative, positive_negative_losses, **options):
        calls.append(
            (
                int(options["n_unlabeled_role"]),
                int(positive_negative_losses.shape[0]),
                bool(options["include_positive_in_unlabeled"]),
            )
        )
        return original(unlabeled_negative, positive_negative_losses, **options)

    monkeypatch.setattr(self_pu_module, "_marginal_negative_risk", spy)
    return calls


@pytest.mark.parametrize("view", ["os", "ts"])
def test_basic_the_negative_role_counts_untrusted_u_rows_not_the_batch(monkeypatch, view):
    """Trusted rows are supervised by pseudo-labels; they are not negative rows.

    ``trusted_history_`` and the blend calls are appended in the same order, so
    the U-side count is pinned exactly: U pool minus that epoch's trusted set.
    """
    calls = _spy_on_blend(monkeypatch, [])
    model, _X, _y = _fit(view, overrides={"max_epochs": 3})

    assert len(calls) == len(model.trusted_history_) == 6  # 2 students x 3 epochs
    for (n_role, n_positive, _flag), record in zip(calls, model.trusted_history_, strict=True):
        assert n_role == _N_UNLABELED - record["actual_size"], record
        assert n_positive == _N_POSITIVE
    # The distinction is live, not theoretical: a batch-size denominator is wrong.
    assert min(n_role for n_role, _p, _f in calls) < _N_UNLABELED


def test_basic_only_the_training_loss_carries_the_calibrated_flag(monkeypatch):
    """Validation, selection and checkpoints must not blend the positive batch.

    The call count is the guard, not just the flag: the ablation teacher
    selection branch (no validation at all) computes its own nnPU risk, and
    routing *that* through the blend would show up here as extra calls.
    """
    calls = _spy_on_blend(monkeypatch, [])
    _fit("ts", overrides={"max_epochs": 3})

    assert len(calls) == 6
    assert all(flag for _u, _p, flag in calls)

    os_calls = _spy_on_blend(monkeypatch, [])
    _fit("os", overrides={"max_epochs": 3})
    assert len(os_calls) == 6
    assert not any(flag for _u, _p, flag in os_calls)

    ablation_calls = _spy_on_blend(monkeypatch, [])
    model, _X, _y = _fit("ts", overrides={"max_epochs": 3}, validation="none")
    assert model.teacher_selection_basis_ == "training_nnpu_risk_ablation"
    assert len(ablation_calls) == 6  # training only; the ablation risk is untouched


def test_basic_the_trusted_population_and_pace_ignore_the_view():
    """Capacity is a function of the original U pool, so the view cannot move it."""
    populations = []
    original_init = TrustedSetManager.__init__

    def spy_init(self, n_unlabeled):
        populations.append(int(n_unlabeled))
        return original_init(self, n_unlabeled)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(TrustedSetManager, "__init__", spy_init)
        os_model, _X, _y = _fit("os")
        ts_model, _X, _y = _fit("ts")

    assert populations == [_N_UNLABELED] * 4  # two managers per fit
    os_targets = [record["target_size"] for record in os_model.trusted_history_]
    assert os_targets == [record["target_size"] for record in ts_model.trusted_history_]
    assert max(os_targets) <= 0.2 * _N_UNLABELED  # pace ceiling, still n_U based


@pytest.mark.parametrize("view", ["os", "ts"])
def test_basic_known_positives_are_absent_from_every_trusted_set(view):
    model, _X, y_pu = _fit(view)
    positives = set(np.flatnonzero(y_pu == 1).tolist())

    for student, value in model.trusted_indices_.items():
        assert len(value["indices"]) > 0, student  # a non-empty set is the interesting case
        assert set(value["indices"].tolist()).isdisjoint(positives)


def test_determ_the_rng_stream_is_view_independent(monkeypatch):
    """The calibrated view re-assigns roles; it never re-draws the batches.

    ``RandomState`` is an extension type, so its ``choice`` cannot be replaced in
    place; a recording subclass reaches the module's own construction instead.
    """
    drawn = []

    class RecordingRandomState(np.random.RandomState):
        def choice(self, *args, **kwargs):
            result = super().choice(*args, **kwargs)
            drawn.append(np.array(result, dtype=int).tolist())
            return result

    monkeypatch.setattr(self_pu_module.np.random, "RandomState", RecordingRandomState)
    _fit("os")
    os_draws = list(drawn)
    drawn.clear()
    _fit("ts")

    assert len(os_draws) == 2 * _CONF["max_epochs"]  # positive batch + U batch per epoch
    assert os_draws == drawn


class _CountingMLP(nn.Module):
    """Real MLP that records how many rows it is asked to score.

    The counter lives on the class so it survives the ``deepcopy`` that builds
    the two students and two teachers from one backbone.
    """

    calls: list[int] = []

    def __init__(self, n_features, hidden=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, X):
        type(self).calls.append(int(X.shape[0]))
        return self.net(X)


def test_determ_the_forward_pass_count_is_view_independent():
    """No second forward and no extra draw: the positive logits are reused."""
    _CountingMLP.calls = []
    _fit("os", overrides={"backbone": _CountingMLP(4), "max_epochs": 2})
    os_counts = list(_CountingMLP.calls)
    _CountingMLP.calls = []
    _fit("ts", overrides={"backbone": _CountingMLP(4), "max_epochs": 2})
    ts_counts = list(_CountingMLP.calls)

    assert os_counts  # the counter really ran
    assert os_counts == ts_counts


def test_basic_the_meta_matrix_covers_only_untrusted_u_rows(monkeypatch):
    """The learned weights are sized by untrusted U rows, in both columns.

    Only the clean-validation configuration reaches this path at all, so a
    positive row appended here would be invisible to the Pilot-shaped suite.
    """
    shapes = []
    original = SelfPUClassifier._meta_weights

    def spy(self, model, positive_loss, ce_losses, pu_losses, X_val, y_val):
        shapes.append((int(ce_losses.shape[0]), int(pu_losses.shape[0])))
        return original(self, model, positive_loss, ce_losses, pu_losses, X_val, y_val)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(SelfPUClassifier, "_meta_weights", spy)
        model, _X, _y = _fit("ts", validation="clean")

    assert shapes  # the meta path really ran
    assert len(shapes) == len(model.trusted_history_) == 6
    for (ce_rows, pu_rows), record in zip(shapes, model.trusted_history_, strict=True):
        assert ce_rows == pu_rows == _N_UNLABELED - record["actual_size"], record


def test_basic_meta_weights_are_invariant_to_scaling_the_negative_column():
    """Why alpha_U need not enter the meta probe: per-column normalisation.

    Column 1 is normalised on its own, so a constant factor cancels and the
    calibrated blend can be applied when the loss is assembled without the
    influence matrix having to know about it.
    """
    rng = np.random.RandomState(11)
    influences = rng.normal(size=(12, 2))
    weights, stats = calibrate_meta_weights(influences, gamma=1 / 16)
    scaled, scaled_stats = calibrate_meta_weights(influences * 3.5, gamma=1 / 16)

    np.testing.assert_allclose(scaled[:, 1], weights[:, 1])
    np.testing.assert_allclose(scaled[:, 0], weights[:, 0])
    assert stats["pu_weight_sum"] == pytest.approx(1.0)
    assert scaled_stats["pu_weight_sum"] == pytest.approx(1.0)


@pytest.mark.parametrize("view", ["os", "ts"])
def test_param_sample_weight_is_refused_under_both_views(view):
    """The rejection stays whole-estimator, so the view cannot change its meaning."""
    X, y_pu, X_val, _y_val, y_pu_val = _data()
    model = SelfPUClassifier(**_CONF)

    with pytest.raises(NotImplementedError, match="sample_weight"):
        model.fit(
            X,
            y_pu,
            pu_validation_data=(X_val, y_pu_val),
            os_or_ts=view,
            sample_weight=np.ones(len(X)),
        )


def test_edge_a_single_epoch_ts_fit_uses_the_whole_u_batch(monkeypatch):
    """Boundary: with the pace held at zero, every U row carries the role.

    This is the configuration Gate A is stated in, reached through the real
    training path rather than by calling the blend directly.
    """
    calls = _spy_on_blend(monkeypatch, [])
    _fit(
        "ts",
        overrides={
            "max_epochs": 1,
            "self_paced_end": 1,
            "distill_start": 1,
            "pace_1": 0.0,
            "pace_2": 0.0,
        },
    )

    assert calls == [(_N_UNLABELED, _N_POSITIVE, True)] * 2
