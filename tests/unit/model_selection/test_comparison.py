# ruff: noqa: N803, N806, S101

"""Tests for PU-aware multi-model comparison."""

from types import SimpleNamespace

import pytest

from pu_toolbox.model_selection import PUModelComparator
from pu_toolbox.workflows import PipelineError
from tests.helpers import make_scar_data

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
    updates = []
    result = PUModelComparator(classifiers=["upu", "pnu"], metrics=["pu_zero_one_risk"]).fit(
        [[0]], [1], class_prior=0.4, progress_callback=updates.append
    )
    assert result.best_classifier == "pnu"
    assert result.best_score == pytest.approx(0.2)
    assert calls == [("upu", False), ("pnu", False), ("pnu", True)]
    assert updates[-1].stage == "complete"


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


def test_pnu_trial_reports_ternary_label_hint(rng):
    X, y_pu, _ = make_scar_data(rng, n=100, separation=4.0)
    result = PUModelComparator(classifiers=["upu", "pnu"], cv=2, random_state=42).fit(
        X, y_pu, class_prior=0.4
    )

    pnu_trial = next(trial for trial in result.trials if trial.classifier == "pnu")
    assert pnu_trial.status == "failed"
    assert "{+1, -1, 0}" in pnu_trial.error
    assert "Label vector must contain all of" in pnu_trial.error
    assert next(trial for trial in result.trials if trial.classifier == "upu").status == "ok"


def test_non_label_error_is_not_rewritten(rng, monkeypatch):
    """A non-PNU failure must keep its original error text (no false labeling)."""
    from pu_toolbox.model_selection import PUModelComparator

    comparator = PUModelComparator(classifiers=["upu", "pnu"], cv=2, random_state=42)

    def broken_pipeline(name):
        if name == "pnu":

            def fit_evaluate(*args, **kwargs):
                raise ValueError("some unrelated optimizer failure")
        else:

            def fit_evaluate(*args, **kwargs):
                metric = SimpleNamespace(available=True, mean=0.5, reason=None)
                return SimpleNamespace(
                    cv_metrics={"pu_zero_one_risk": metric},
                    final_model=None,
                )

        return SimpleNamespace(fit_evaluate=fit_evaluate)

    comparator._pipeline = broken_pipeline  # type: ignore[method-assign]
    result = comparator.fit(*make_scar_data(rng, n=100, separation=4.0)[:2], class_prior=0.4)

    pnu_trial = next(trial for trial in result.trials if trial.classifier == "pnu")
    assert pnu_trial.status == "failed"
    assert pnu_trial.error == "some unrelated optimizer failure"
    assert "{+1, -1, 0}" not in pnu_trial.error
    upu_trial = next(trial for trial in result.trials if trial.classifier == "upu")
    assert upu_trial.status == "ok"
