# ruff: noqa: E402, N803, N806, S101, E501

"""DistPUClassifier 训练视图（协议 §2.3）**不被允许改变**的那些量。

校准只允许换掉承担无标签分布角色的集合。本文件锁定其余一切：正例监督项、
Mixup 物理池与 RNG 调用序列、callback、训练矩阵形状，以及公共接口。

本文件存在的直接原因是变异检验：`ts` 把 P **重复两次**计入 marginal 这类缺陷
注入在 `fit` 的**调用点**，而 test_dist_pu_ts_view.py 的数学测试直接调用 helper，
看不见它——角色集合必须在模型真正使用它的那一层被钉住。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.risk import DistPUClassifier
from pu_toolbox.estimators.risk import dist_pu as dist_pu_module

_EPOCHS = 5
_MIXUP = 0.3


def _data():
    rng = np.random.RandomState(5)
    positive = rng.normal(1.4, 0.5, size=(10, 3)).astype(np.float32)
    unlabeled = rng.normal(-0.3, 1.0, size=(24, 3)).astype(np.float32)
    X = np.vstack((positive, unlabeled))
    y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(unlabeled), dtype=int)]
    return X, y


def _fit(os_or_ts=None, sample_weight=None, **ctor_overrides):
    params = {
        "hidden_dim": 6,
        "epochs": _EPOCHS,
        "learning_rate": 0.01,
        "alignment_weight": 1.0,
        "entropy_weight": 0.05,
        "mixup_weight": 0.0,
        "random_state": 11,
        "device": "cpu",
    }
    params.update(ctor_overrides)
    X, y = _data()
    seen = []
    kwargs = {"epoch_callback": lambda epoch, _model: seen.append(epoch)}
    if os_or_ts is not None:
        kwargs["os_or_ts"] = os_or_ts
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    model = DistPUClassifier(0.4, **params)
    model.fit(X, y, **kwargs)
    return model, X, y, seen


@pytest.mark.unit
def test_basic_both_regularizer_weights_zero_makes_the_views_bit_identical():
    """The strongest statement of scope this suite can make.

    With ``alignment_weight = entropy_weight = 0`` the objective is the positive
    loss alone, so if the view touched anything but the two regularizers the two
    trajectories could not stay bit-identical.
    """
    os_model, _, _, _ = _fit(alignment_weight=0.0, entropy_weight=0.0)
    ts_model, _, _, _ = _fit(alignment_weight=0.0, entropy_weight=0.0, os_or_ts="ts")

    assert os_model.loss_history_ == ts_model.loss_history_
    for name, tensor in os_model.model_.state_dict().items():
        assert torch.equal(tensor, ts_model.model_.state_dict()[name])


@pytest.mark.unit
def test_edge_ts_roles_are_the_whole_training_matrix_without_adding_rows():
    """Pin the role set where the model actually uses it.

    A `ts` branch that concatenated the positives onto the regularizer input
    (rather than reusing the rows the forward pass already produced) would leave
    the helper-level math tests green while inflating ``n`` here.
    """
    calls = []
    original = dist_pu_module._distribution_regularizers

    def spy(probs, mask, class_prior, *, include_positive_in_unlabeled):
        calls.append(
            (
                int(probs.shape[0]),
                int(mask.sum()),
                bool(include_positive_in_unlabeled),
            )
        )
        return original(
            probs,
            mask,
            class_prior,
            include_positive_in_unlabeled=include_positive_in_unlabeled,
        )

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(dist_pu_module, "_distribution_regularizers", spy)
        model, X, y, _ = _fit(os_or_ts="ts")

    n_positive = int((y == 1).sum())
    assert len(calls) == _EPOCHS
    for n_rows, n_unlabeled, flattened in calls:
        assert flattened is True
        assert n_unlabeled == len(X) - n_positive
        # The whole matrix, and *not* the matrix plus another copy of P.
        assert n_rows == len(X)
        assert n_rows != len(X) + n_positive
    assert model._X_shape_ == X.shape


@pytest.mark.unit
def test_basic_mixup_pool_length_is_the_training_size_in_both_views():
    lengths = []
    original = torch.randperm

    def spy(n, **kwargs):
        lengths.append(int(n))
        return original(n, **kwargs)

    for path in ("os", "ts"):
        lengths.clear()
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(torch, "randperm", spy)
            _fit(mixup_weight=_MIXUP, os_or_ts=path)
        assert lengths == [len(_data()[0])] * _EPOCHS


@pytest.mark.unit
def test_determ_rng_call_order_and_shapes_are_view_independent():
    """Only the RNG *call sequence* is frozen.

    The losses themselves diverge after the first update, because the two views
    fit different parameters -- that divergence is the point of the ablation and
    is deliberately not asserted here.
    """

    def record(path):
        order = []
        perm, rand = torch.randperm, torch.rand

        def perm_spy(n, **kwargs):
            order.append(("randperm", int(n)))
            return perm(n, **kwargs)

        def rand_spy(*args, **kwargs):
            order.append(("rand", tuple(args), tuple(sorted(kwargs))))
            return rand(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(torch, "randperm", perm_spy)
            patcher.setattr(torch, "rand", rand_spy)
            _fit(mixup_weight=_MIXUP, os_or_ts=path)
        return order

    os_order = record("os")
    assert os_order == record("ts")
    assert len(os_order) == 2 * _EPOCHS  # randperm then rand, once per epoch


@pytest.mark.unit
def test_basic_epoch_callback_sequence_is_view_independent():
    _, _, _, os_seen = _fit(mixup_weight=_MIXUP)
    _, _, _, ts_seen = _fit(mixup_weight=_MIXUP, os_or_ts="ts")

    assert os_seen == ts_seen == list(range(_EPOCHS))


@pytest.mark.unit
def test_basic_training_matrix_shape_is_not_extended_by_the_view():
    for path in ("os", "ts"):
        model, X, _, _ = _fit(os_or_ts=path)
        assert model._X_shape_ == X.shape


@pytest.mark.parametrize("path", ["os", "ts"])
@pytest.mark.unit
def test_param_sample_weight_is_ignored_under_both_views(path):
    """``sample_weight_support = IGNORED``, so the union cannot add a row whose
    weight is undefined -- unlike `nnpu`, which needs a gate."""
    plain, X, _, _ = _fit(os_or_ts=path)
    weighted, _, _, _ = _fit(os_or_ts=path, sample_weight=np.ones(len(X)))

    assert weighted.loss_history_ == plain.loss_history_


@pytest.mark.unit
def test_edge_single_epoch_ts_fits():
    model, X, _, seen = _fit(epochs=1, mixup_weight=_MIXUP, os_or_ts="ts")

    assert len(model.loss_history_) == 1
    assert seen == [0]
    assert model.predict(X).shape == (len(X),)
