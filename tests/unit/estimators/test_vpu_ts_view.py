# ruff: noqa: N803, N806, S101

"""VPUClassifier 的 TS-OS 训练视图行为（协议 §2.3，决策 D21/D22）。

VPU 的变分目标要求边缘项在训练总体的边缘分布上求值，故其原生视图是 **ts**：边缘池
为 ``D_U ∪ D_P``（即完整训练分区），正例同时留在两个角色里。OS 对照把边缘池限制回
``D_U``，是协议定义的消融，不是 VPU 在原生假设下的运行。

本文件只观察**角色**——两个池收到的行数与来源——不重复验证变分公式；公式的数值正确
性由 ``test_vpu.py`` 负责。关键的验证价值在于：两个池都必须由核心层给出的
``positive_positions`` / ``loss_unlabeled_positions`` 产生。若估计器绕过角色直接端走
源数组，公共构造器漏掉或多加入样本时这些测试仍会通过。

**证据边界**：这里的断言是**行数级**的——构造器漏掉或多加入样本（两个方向都验证过）
会被抓住；把等长的位置整体替换成别的行则不会。后者不是本文件的目标：角色由掩码直接
生成，等长错位属核心层契约测试（``tests/unit/core/test_training_view_contract.py``）的
职责。
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.risk import vpu as vpu_module  # noqa: E402
from pu_toolbox.estimators.risk.vpu import VPUClassifier  # noqa: E402

pytestmark = pytest.mark.unit


def _data():
    """标注正例在前、未标记行在后，便于按位置核对池的来源。"""
    rng = np.random.RandomState(31)
    positive = rng.normal(1.0, 0.5, (9, 3)).astype(np.float32)
    unlabeled = rng.normal(-0.3, 0.8, (15, 3)).astype(np.float32)
    return np.vstack([positive, unlabeled]), np.r_[np.ones(9, int), np.zeros(15, int)]


def _model(**kwargs):
    params = {"hidden_dim": 8, "depth": 1, "max_epochs": 2, "batch_size": 5, "random_state": 7}
    params.update(kwargs)
    return VPUClassifier(**params)


def _spy_objective(monkeypatch):
    """记录每次损失调用看到的两个角色各有几行。"""
    seen: dict[str, list[int]] = {"positive": [], "marginal": []}
    real = vpu_module.vpu_objective

    def spy(log_phi_positive, log_phi_marginal, log_phi_mixed, mixing, *, regularization_weight):
        seen["positive"].append(int(log_phi_positive.numel()))
        seen["marginal"].append(int(log_phi_marginal.numel()))
        return real(
            log_phi_positive,
            log_phi_marginal,
            log_phi_mixed,
            mixing,
            regularization_weight=regularization_weight,
        )

    monkeypatch.setattr(vpu_module, "vpu_objective", spy)
    return seen


def _spy_view_requests(monkeypatch):
    """记录估计器向核心层提出的每一个视图请求（含角色）。"""
    requests: list[dict] = []
    real = vpu_module.build_training_view

    def spy(X, y_pu, *, requested_view="os", role="train", indices=None):
        requests.append({"requested_view": requested_view, "role": role, "indices": indices})
        return real(X, y_pu, requested_view=requested_view, role=role, indices=indices)

    monkeypatch.setattr(vpu_module, "build_training_view", spy, raising=False)
    return requests


def _narrowed_positions(monkeypatch, **positions):
    """把核心构造器的角色位置换成已知的子集，用来证明池确实跟着位置走。"""
    real = vpu_module.build_training_view

    def narrowed(X, y_pu, *, requested_view="os", role="train", indices=None):
        view = real(X, y_pu, requested_view=requested_view, role=role, indices=indices)
        return dataclasses.replace(view, **positions)

    monkeypatch.setattr(vpu_module, "build_training_view", narrowed, raising=False)


# --- 声明的接口 ---------------------------------------------------------------


def test_basic_fit_declares_os_or_ts_defaulting_to_the_calibrated_view():
    """VPU 的声明视图就是 ts，故裸 fit 的默认值必须与之一致。"""
    parameters = inspect.signature(VPUClassifier.fit).parameters

    assert "os_or_ts" in parameters
    assert parameters["os_or_ts"].default == "ts"


# --- 两个角色各收到什么 -------------------------------------------------------


def test_basic_the_calibrated_pool_is_the_whole_partition():
    X, y_pu = _data()
    fitted = _model().fit(X, y_pu, os_or_ts="ts")
    assert fitted.training_view_ == "ts"
    assert fitted.calibration_applied_ is True

    assert fitted.n_loss_unlabeled_ == len(X)
    assert fitted.n_unlabeled_ == int((y_pu == 0).sum())
    assert fitted.n_positive_ == int((y_pu == 1).sum())


def test_basic_the_os_pool_keeps_only_the_unlabeled_rows():
    X, y_pu = _data()
    os_fit = _model().fit(X, y_pu, os_or_ts="os")
    ts_fit = _model().fit(X, y_pu, os_or_ts="ts")
    assert os_fit.training_view_ == "os"
    assert os_fit.calibration_applied_ is False

    assert os_fit.n_loss_unlabeled_ == int((y_pu == 0).sum())
    assert os_fit.n_loss_unlabeled_ < ts_fit.n_loss_unlabeled_
    # 正例角色与视图无关，两个视图下都是同一批标注正例。
    assert os_fit.n_positive_ == ts_fit.n_positive_


def test_basic_both_pools_follow_the_role_positions(monkeypatch):
    """池由角色位置产生；位置被换掉，池就随之改变。"""
    X, y_pu = _data()
    _narrowed_positions(
        monkeypatch,
        positive_positions=np.arange(4),
        loss_unlabeled_positions=np.arange(3),
    )

    fitted = _model().fit(X, y_pu, os_or_ts="ts")

    assert fitted.n_positive_ == 4
    assert fitted.n_loss_unlabeled_ == 3


def test_basic_the_loss_sees_the_pool_the_roles_defined(monkeypatch):
    """损失输入级别：池小于 batch size 时，边缘批的行数就等于池的行数。

    位置若被绕过（比如直接端走源数组），这里看到的仍是整个分区。
    """
    X, y_pu = _data()
    seen = _spy_objective(monkeypatch)
    _narrowed_positions(monkeypatch, loss_unlabeled_positions=np.arange(3))

    fitted = _model().fit(X, y_pu, os_or_ts="ts")

    assert set(seen["marginal"]) == {3}
    assert fitted.n_loss_unlabeled_ == 3


# --- 兼容与对照 ---------------------------------------------------------------


def test_basic_the_bare_default_matches_explicit_ts_exactly():
    """裸默认必须与显式 ts 逐位相同——默认值改了就是行为破坏。"""
    X, y_pu = _data()
    bare = _model().fit(X, y_pu)
    explicit = _model().fit(X, y_pu, os_or_ts="ts")

    np.testing.assert_array_equal(bare.decision_function(X), explicit.decision_function(X))
    assert bare.history_ == explicit.history_
    assert bare.max_log_phi_ == explicit.max_log_phi_
    assert bare.n_loss_unlabeled_ == explicit.n_loss_unlabeled_


def test_basic_the_view_request_is_always_the_train_role(monkeypatch):
    """D22：验证路径不跟随训练视图，故估计器只应请求 train 角色。"""
    X, y_pu = _data()
    requests = _spy_view_requests(monkeypatch)

    _model().fit(X, y_pu, pu_validation_data=(X, y_pu), os_or_ts="ts")

    assert requests, "估计器没有向核心层请求任何视图"
    assert {item["role"] for item in requests} == {"train"}
    assert {item["requested_view"] for item in requests} == {"ts"}


# --- 拒绝与不变量 -------------------------------------------------------------


@pytest.mark.parametrize("bad", ["TS", "case_control", "", "ts ", "both"])
def test_param_a_view_outside_os_ts_is_refused(bad):
    X, y_pu = _data()
    with pytest.raises(ValueError, match="os_or_ts"):
        _model().fit(X, y_pu, os_or_ts=bad)


def test_edge_class_prior_does_not_enter_training():
    X, y_pu = _data()
    without = _model().fit(X, y_pu, os_or_ts="os")
    with_prior = _model().fit(X, y_pu, class_prior=0.4, os_or_ts="os")

    np.testing.assert_array_equal(without.decision_function(X), with_prior.decision_function(X))
    assert without.history_ == with_prior.history_
    assert with_prior.get_pu_metadata()["class_prior"] is None


def test_determ_float64_input_trains_the_same_as_float32():
    """视图在 float32 转换之后构造，故输入 dtype 不得改变训练结果。"""
    X, y_pu = _data()
    as32 = _model().fit(X.astype(np.float32), y_pu)
    as64 = _model().fit(X.astype(np.float64), y_pu)

    np.testing.assert_array_equal(as32.decision_function(X), as64.decision_function(X))
