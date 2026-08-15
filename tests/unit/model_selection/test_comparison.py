# ruff: noqa: N803, S101

"""Tests for PU-aware multi-model comparison."""

from types import SimpleNamespace

import pytest

from pu_toolbox.model_selection import PUModelComparator
from pu_toolbox.workflows import PipelineError

pytestmark = pytest.mark.unit


def _fake_pipeline(monkeypatch, scores):
    from pu_toolbox.model_selection import comparison

    calls = []

    class FakePipeline:
        def __init__(self, *, classifier, metrics, **kwargs):
            self.classifier = classifier
            self.metrics = list(metrics)

        def fit_evaluate(self, X, y, *, refit=True, **kwargs):
            calls.append((self.classifier, refit))
            outcome = scores[self.classifier]
            if isinstance(outcome, Exception):
                raise outcome
            metric = SimpleNamespace(available=True, mean=outcome, reason=None)
            return SimpleNamespace(cv_metrics={name: metric for name in self.metrics})

    monkeypatch.setattr(comparison, "PUPipeline", FakePipeline)
    return calls


def test_basic_comparison_refits_only_best_classifier(monkeypatch):
    calls = _fake_pipeline(monkeypatch, {"upu": 0.3, "pnu": 0.2})
    result = PUModelComparator(classifiers=["upu", "pnu"], metrics=["pu_zero_one_risk"]).fit(
        [[0]], [1], class_prior=0.4
    )
    assert result.best_classifier == "pnu"
    assert result.best_score == pytest.approx(0.2)
    assert calls == [("upu", False), ("pnu", False), ("pnu", True)]


def test_param_comparison_can_maximize_score(monkeypatch):
    _fake_pipeline(monkeypatch, {"upu": 0.3, "pnu": 0.2})
    result = PUModelComparator(
        classifiers=["upu", "pnu"], scoring="pu_recall", higher_is_better=True
    ).fit([[0]], [1])
    assert result.best_classifier == "upu"


def test_edge_comparison_isolates_failure_and_rejects_invalid_inputs(monkeypatch):
    _fake_pipeline(monkeypatch, {"upu": RuntimeError("backend missing"), "pnu": 0.2})
    result = PUModelComparator(classifiers=["upu", "pnu"]).fit([[0]], [1])
    assert result.trials[0].status == "failed"
    assert result.best_classifier == "pnu"
    with pytest.raises(ValueError, match="at least two"):
        PUModelComparator(classifiers=["upu"])
    with pytest.raises(ValueError, match="unique"):
        PUModelComparator(classifiers=["upu", "upu"])

    _fake_pipeline(monkeypatch, {"upu": RuntimeError("bad"), "pnu": RuntimeError("bad")})
    with pytest.raises(PipelineError, match="No compared classifier"):
        PUModelComparator(classifiers=["upu", "pnu"]).fit([[0]], [1])


def test_deterministic_comparison_preserves_input_order(monkeypatch):
    _fake_pipeline(monkeypatch, {"upu": 0.2, "pnu": 0.2})
    result = PUModelComparator(classifiers=["upu", "pnu"]).fit([[0]], [1])
    assert [trial.classifier for trial in result.trials] == ["upu", "pnu"]
    assert result.best_classifier == "upu"
    assert result.to_dict() == result.to_dict()
