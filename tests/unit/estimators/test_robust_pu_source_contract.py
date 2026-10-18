"""Independent scalar/gradient evidence for the retained Robust-PU subset.

No author module is imported or executed; formula locations are in the source
review receipt. Adapted batching/selection is not certified as source replay.
"""

# Public feature names follow the estimator API.
# ruff: noqa: N803, N806

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pu_toolbox.estimators.deep.robust_pu as robust  # noqa: E402
from pu_toolbox.losses.nnpu import _nnpu_train_step  # noqa: E402

pytestmark = pytest.mark.unit


def _scalar_weights(logits, roles, tp, tn, temper_p, temper_n, kind="welsch"):
    weights = []
    for z, role in zip(logits, roles, strict=True):
        threshold = tp if role == 1 else tn
        signed = -float(z) / temper_p if role == 1 else float(z) / temper_n
        hardness = float(np.logaddexp(0, signed))
        if kind == "hard":
            weights.append(float(hardness < threshold))
        elif kind == "linear":
            weights.append(max(0, 1 - hardness / threshold))
        else:
            weights.append(math.exp(-hardness / threshold**2))
    return np.array(weights)


def _data():
    return np.arange(8, dtype="float32")[:, None], np.r_[np.ones(3), np.zeros(5)]


def _head():
    head = torch.nn.Linear(1, 1)
    with torch.no_grad():
        head.weight.fill_(0.1)
        head.bias.fill_(-0.6)
    return head


def _model(**kwargs):
    params = dict(
        class_prior=0.4,
        model=_head(),
        pretrain_epochs=0,
        episodes=1,
        inner_epochs=1,
        batch_size=8,
        alpha_p=0.9,
        alpha_n=1.2,
        temper_p=0.7,
        temper_n=1.5,
        learning_rate=0.05,
        random_state=17,
        device="cpu",
    )
    params.update(kwargs)
    return robust.RobustPUClassifier(**params)


@pytest.mark.math
@pytest.mark.parametrize("kind", ["hard", "linear", "welsch"])
@pytest.mark.parametrize("temper_p,temper_n", [(0.5, 2.0), (1.0, 1.0), (2.0, 0.3)])
def test_basic_temperature_hardness_weights_match_independent_scalar_formula(
    kind, temper_p, temper_n
):
    logits = torch.tensor([-12, -2, 0, 0.4, 3, 12], dtype=torch.float64, requires_grad=True)
    roles = torch.tensor([1, 0, 1, 0, 1, 0])
    actual = robust._episode_weights(
        logits,
        roles,
        threshold_p=0.9,
        threshold_n=1.2,
        temper_p=temper_p,
        temper_n=temper_n,
        kind=kind,
    )
    expected = _scalar_weights(
        logits.detach().numpy(), roles.numpy(), 0.9, 1.2, temper_p, temper_n, kind
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-15, rtol=1e-13)
    assert not actual.requires_grad


@pytest.mark.math
@pytest.mark.parametrize(
    "p_values,u_values", [([2.0, 4.0], [-3.0, -2.0]), ([-2.0, 0.0], [1.0, 3.0])]
)
def test_edge_nnpu_optimization_branches_match_source_not_clipped_risk_gradients(
    p_values, u_values
):
    p = torch.tensor(p_values, dtype=torch.float64, requires_grad=True)
    u = torch.tensor(u_values, dtype=torch.float64, requires_grad=True)
    pi = 0.7
    loss, info = _nnpu_train_step(
        torch.sigmoid(-p).mean(), torch.sigmoid(p).mean(), torch.sigmoid(u).mean(), class_prior=pi
    )
    sp = 1 / (1 + np.exp(-np.array(p_values)))
    su = 1 / (1 + np.exp(-np.array(u_values)))
    positive = pi * (1 - sp).mean()
    negative = su.mean() - pi * sp.mean()
    corrected = negative < 0
    expected = -negative if corrected else positive + negative
    assert loss.item() == pytest.approx(expected, abs=1e-14)
    assert info["correction"] == corrected
    assert info["nnpu_risk"] == pytest.approx(positive + max(negative, 0), abs=1e-14)
    loss.backward()
    expected_p = (pi if corrected else -2 * pi) * sp * (1 - sp) / len(p_values)
    expected_u = (-1 if corrected else 1) * su * (1 - su) / len(u_values)
    np.testing.assert_allclose(p.grad.numpy(), expected_p, atol=1e-14)
    np.testing.assert_allclose(u.grad.numpy(), expected_u, atol=1e-14)


@pytest.mark.math
def test_basic_actual_episode_bce_gradient_is_row_mean_not_weight_normalized(monkeypatch):
    X, roles = _data()
    estimator = _model()
    z = 0.1 * X[:, 0].astype(float) - 0.6
    weights = _scalar_weights(z, roles, 0.9, 1.2, 0.7, 1.5)
    derivatives = (1 / (1 + np.exp(-z)) - roles) * weights
    expected = np.array([(derivatives * X[:, 0]).mean(), derivatives.mean()])
    observed = []
    original_step = torch.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        observed.append(
            np.array(
                [
                    estimator.model_.weight.grad.item(),
                    estimator.model_.bias.grad.item(),
                ]
            )
        )
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    fitted = estimator.fit(X, roles)
    assert len(observed) == fitted.optimizer_steps_ == 1
    np.testing.assert_allclose(observed[0], expected, atol=2e-7, rtol=1e-6)
    assert not np.allclose(observed[0], expected * len(X) / weights.sum(), atol=1e-5)
    expected_loss = np.mean((np.logaddexp(0, z) - roles * z) * weights)
    assert fitted.history_["episode_loss"][0] == pytest.approx(expected_loss, abs=1e-7)


class RecordingLinear(torch.nn.Linear):
    def forward(self, X):
        logits = super().forward(X)
        if self.training:
            self.last_train_inputs = X.detach().cpu().numpy().copy()
            self.last_train_logits = logits.detach().cpu().numpy().reshape(-1).copy()
        return logits


@pytest.mark.parametrize("phi", [0.0, 0.6])
def test_determ_moving_weights_follow_original_rows_through_shuffle_and_inner_epochs(
    monkeypatch, phi
):
    X, roles = _data()
    head = RecordingLinear(1, 1)
    with torch.no_grad():
        head.weight.fill_(0.1)
        head.bias.fill_(-0.6)
    estimator = _model(model=head, batch_size=3, episodes=2, inner_epochs=2, phi=phi)
    real_weights, original_step = robust._episode_weights, torch.optim.Adam.step
    moving = None
    receipts, gradients, optimizers = [], [], []

    def calculate(logits, labels, **kwargs):
        nonlocal moving
        actual = real_weights(logits, labels, **kwargs)
        expected = _scalar_weights(
            logits.numpy(),
            roles,
            kwargs["threshold_p"],
            kwargs["threshold_n"],
            kwargs["temper_p"],
            kwargs["temper_n"],
        )
        np.testing.assert_array_equal(labels.numpy(), roles)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=1e-8)
        previous = moving
        moving = expected if moving is None else phi * moving + (1 - phi) * expected
        receipts.append((previous, expected.copy(), moving.copy()))
        return actual

    def step(optimizer, *args, **kwargs):
        fitted = estimator.model_
        ids = fitted.last_train_inputs[:, 0].astype(int)
        z = fitted.last_train_logits.astype(float)
        dz = (1 / (1 + np.exp(-z)) - roles[ids]) * moving[ids]
        expected = [(dz * ids).mean(), dz.mean()]
        actual = [fitted.weight.grad.item(), fitted.bias.grad.item()]
        np.testing.assert_allclose(actual, expected, atol=4e-7, rtol=3e-6)
        gradients.append(ids.copy())
        # Retain optimizer identities to prove reset-at-episode, not resume.
        optimizers.append(optimizer)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(robust, "_episode_weights", calculate)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    estimator.fit(X, roles)
    assert len(receipts) == 2 and len(gradients) == estimator.optimizer_steps_ == 12
    for start in (0, 3, 6, 9):
        assert sorted(np.concatenate(gradients[start : start + 3])) == list(range(8))
    assert len({id(optimizer) for optimizer in optimizers}) == 2
    assert all(optimizer is optimizers[0] for optimizer in optimizers[:6])
    assert all(optimizer is optimizers[6] for optimizer in optimizers[6:])
    if phi:
        assert not np.allclose(receipts[1][1], receipts[1][2])
    assert head.weight.grad is None and head.bias.grad is None


@pytest.mark.parametrize("grow", [1, 3])
@pytest.mark.parametrize("limits", [(0.2, 0.9, 0.8, 1.5), (0.9, 0.2, 1.5, 0.8)])
def test_edge_actual_linear_schedule_starts_at_zero_and_caps_both_directions(grow, limits):
    X, roles = _data()
    ap, mp, an, mn = limits
    fitted = _model(
        alpha_p=ap, max_thresh_p=mp, alpha_n=an, max_thresh_n=mn, grow_steps=grow, episodes=6
    ).fit(X, roles)
    for key, initial, maximum in (("threshold_p", ap, mp), ("threshold_n", an, mn)):
        expected = []
        for t in range(6):
            value = initial + (maximum - initial) / grow * t
            expected.append(min(value, maximum) if initial < maximum else max(value, maximum))
        np.testing.assert_allclose(fitted.history_[key], expected, atol=1e-15)


@pytest.mark.parametrize(
    "params",
    [
        {"temper_p": 0},
        {"temper_n": np.nan},
        {"alpha_p": -1},
        {"max_thresh_n": 0},
        {"spl_type": "cauchy"},
        {"phi": 1},
    ],
)
def test_param_invalid_retained_subset_contract_is_not_silently_expanded(params):
    X, roles = _data()
    with pytest.raises(ValueError):
        _model(**params).fit(X, roles)
