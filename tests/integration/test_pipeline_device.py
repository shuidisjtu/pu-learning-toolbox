# ruff: noqa: N803, N806

"""PUPipeline / deep-estimator defaults (device auto-detection, epochs)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _table_data(n=24, n_features=4, seed=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y_pu = np.concatenate([np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)])
    rng.shuffle(y_pu)
    return X, y_pu


def test_basic_pipeline_device_default_none():
    from pu_toolbox import PUPipeline

    pipe = PUPipeline(classifier="wconpu", cv=2)
    assert pipe.device is None


def test_basic_wconpu_device_default_none():
    # Default device is None ("auto"): the estimator resolves CUDA/CPU at
    # fit time instead of hard-coding cpu.
    from pu_toolbox.estimators.deep.weighted_contrastive_pu import (
        WeightedContrastivePUClassifier,
    )

    estimator = WeightedContrastivePUClassifier(0.3, max_epochs=2)
    assert estimator.device is None


def test_basic_wconpu_max_epochs_default_100():
    # Default max_epochs contract: 800 was unusable (no early stopping, ~1.5h
    # on 20k samples); 50 epochs already converged in practice, 100 is the
    # new default. Guarded by both the instance value and the signature.
    import inspect

    from pu_toolbox.estimators.deep.weighted_contrastive_pu import (
        WeightedContrastivePUClassifier,
    )

    estimator = WeightedContrastivePUClassifier(0.3)
    assert estimator.max_epochs == 100
    params = inspect.signature(WeightedContrastivePUClassifier.__init__).parameters
    assert params["max_epochs"].default == 100


def test_basic_fresh_estimator_skips_device_injection_when_default():
    # 默认 device=None 时不注入 → 估计器保留 None 自行解析(自动检测)
    from pu_toolbox import PUPipeline

    X, y_pu = _table_data()
    pipe = PUPipeline(classifier="wconpu", cv=2)
    clf = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
    assert clf.device is None


def test_param_auto_has_gpu_follows_cuda_availability(monkeypatch):
    """默认 device=None 时 has_gpu 如实反映 resolve_device_name(默认 auto)。

    Regression guard: has_gpu used to be ``self.device != "cpu"``; with
    the default switched to None, ``None != "cpu"`` would always be True
    and wrongly report a GPU as available.
    """
    import torch

    from pu_toolbox import PUPipeline
    from pu_toolbox.advisor._types import MethodCandidate, RecommendationResult
    from pu_toolbox.registry.metadata import AlgorithmMetadata

    X, y_pu = _table_data(n=40)
    captured: dict = {}

    def fake_recommend(profile, **kwargs):
        captured["has_gpu"] = kwargs["has_gpu"]
        meta = AlgorithmMetadata(name="upu", paper="fake")
        cand = MethodCandidate(
            name="upu", score=90.0, rank=1, reasons=(), warnings=(), metadata=meta
        )
        return RecommendationResult(
            candidates=(cand,), filters_applied={}, global_warnings=(), provenance={}
        )

    monkeypatch.setattr("pu_toolbox.workflows.pipeline.recommend_from_profile", fake_recommend)

    class _StubClf:
        def fit(self, X, y, **kwargs):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def decision_function(self, X):
            return np.zeros(len(X))

    def stub_fresh(cls, instance, prior):
        return _StubClf()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pipe = PUPipeline(classifier="auto", cv=2)
    monkeypatch.setattr(pipe, "_fresh_estimator", stub_fresh)
    pipe.fit_evaluate(X, y_pu, class_prior=0.3)
    assert captured["has_gpu"] is False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pipe = PUPipeline(classifier="auto", cv=2)
    monkeypatch.setattr(pipe, "_fresh_estimator", stub_fresh)
    pipe.fit_evaluate(X, y_pu, class_prior=0.3)
    assert captured["has_gpu"] is True


def test_determ_repeated_default_construction_stable():
    from pu_toolbox import PUPipeline

    pipe1 = PUPipeline(classifier="wconpu", cv=2)
    pipe2 = PUPipeline(classifier="wconpu", cv=2)
    assert pipe1.device is None
    assert pipe2.device is None


def test_param_fresh_estimator_injects_max_epochs():
    # max_epochs 按签名注入：nnpu 构造签名含 max_epochs → 注入
    from pu_toolbox import PUPipeline

    X, y_pu = _table_data()
    pipe = PUPipeline(classifier="nnpu", cv=2, max_epochs=5)
    clf = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
    assert clf.max_epochs == 5


def test_edge_fresh_estimator_skips_max_epochs_without_signature():
    # upu 构造签名不含 max_epochs → 不注入、不崩溃
    from pu_toolbox import PUPipeline

    X, y_pu = _table_data()
    pipe = PUPipeline(classifier="upu", cv=2, max_epochs=5)
    clf = pipe._fresh_estimator(pipe._classifier_cls, None, 0.3)
    assert not hasattr(clf, "max_epochs")
