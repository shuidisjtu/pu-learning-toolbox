"""Public PU workflows reject semantic mismatches before any classifier fit."""

from types import SimpleNamespace

import numpy as np
import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.model_selection import PUModelComparator, PUTuner
from pu_toolbox.workflows import PipelineError, PUPipeline, _models

pytestmark = pytest.mark.unit


class SpyClassifier(BasePUClassifier):
    calls = 0

    def fit(self, features, y_pu, *, class_prior=None, sample_weight=None):
        type(self).calls += 1
        self._is_fitted = True
        return self

    def _predict(self, features):
        return np.ones(len(features), dtype=int)

    def _decision_function(self, features):
        return np.zeros(len(features))


class PNSpy(SpyClassifier):
    label_semantics = "pn"


class PNUSpy(SpyClassifier):
    label_semantics = "pnu"


class ConstructorSpy(SpyClassifier):
    def __init__(self, meaning="pn"):
        super().__init__()
        self.meaning = meaning
        self.label_semantics = meaning


@pytest.fixture
def data():
    return np.arange(80).reshape(40, 2), np.tile([0, 1], 20)


@pytest.mark.parametrize("cls", [PNSpy, PNUSpy])
def test_param_wrong_instance_rejected_before_prior_or_classifier_training(cls, data, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("semantic preflight must precede prior estimation and classifier fit")

    monkeypatch.setattr(PUPipeline, "_resolve_prior", forbidden)
    monkeypatch.setattr(cls, "fit", forbidden)
    with pytest.raises(ValueError, match="label_semantics=.*requires 'pu'"):
        PUPipeline(classifier=cls(), cv=2).fit_evaluate(*data)


def test_basic_default_pu_subclass_remains_usable(data):
    SpyClassifier.calls = 0
    report = PUPipeline(
        classifier=SpyClassifier(),
        cv=2,
        metrics=["pu_recall"],
        prior_estimator=None,
    ).fit_evaluate(*data, refit=False)
    assert SpyClassifier.calls == 2
    assert report.cv_metrics["pu_recall"].available


def test_edge_mutated_instance_rechecked_at_fit_entry(data, monkeypatch):
    spy = SpyClassifier()
    pipeline = PUPipeline(classifier=spy, cv=2)
    spy.label_semantics = "pn"
    monkeypatch.setattr(SpyClassifier, "fit", lambda *a, **k: pytest.fail("fit reached"))
    with pytest.raises(ValueError, match="label_semantics='pn'"):
        pipeline.fit_evaluate(*data)


def test_edge_clone_cannot_change_primary_label_meaning():
    class ChangingClone(SpyClassifier):
        def __sklearn_clone__(self):
            return PNSpy()

    with pytest.raises(ValueError, match="PNSpy.*label_semantics='pn'"):
        _models.fresh_estimator(
            cls=None,
            instance=ChangingClone(),
            prior=None,
            classifier_params={},
            random_state=None,
            encoder=None,
            device=None,
            max_epochs=None,
            classifier_name="custom",
        )


def test_edge_equal_pu_and_true_values_do_not_imply_wrong_semantics(data):
    report = PUPipeline(
        classifier=SpyClassifier(),
        cv=2,
        metrics=["pu_recall"],
        prior_estimator=None,
    ).fit_evaluate(*data, y_true=data[1], refit=False)
    assert report.cv_metrics["pu_recall"].available


@pytest.mark.parametrize("meaning", ["pn", "pnu", "invalid", None])
def test_param_constructor_mutation_is_checked_on_fresh_model(meaning):
    with pytest.raises(ValueError, match="label_semantics"):
        _models.fresh_estimator(
            cls=ConstructorSpy,
            instance=None,
            prior=None,
            classifier_params={"meaning": meaning},
            random_state=None,
            encoder=None,
            device=None,
            max_epochs=None,
            classifier_name="custom",
        )


def test_determ_auto_candidates_skip_incompatible_semantics(monkeypatch):
    classes = {"bad": PNSpy, "ternary": PNUSpy, "good": SpyClassifier}
    monkeypatch.setattr(_models, "get_algorithm", classes.__getitem__)
    candidates = [SimpleNamespace(name=name) for name in classes]
    first = _models.pick_first_instantiable(candidates)
    assert first == _models.pick_first_instantiable(candidates)
    assert first[0] is SpyClassifier
    assert [item["name"] for item in first[1]] == ["bad", "ternary"]
    assert all("label_semantics" in item["reason"] for item in first[1])


def test_edge_comparison_pnu_fails_before_any_model_fit(monkeypatch):
    from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

    register_all_builtin_methods()
    for name in ("upu", "pnu"):
        monkeypatch.setattr(get_algorithm(name), "fit", lambda *a, **k: pytest.fail("fit reached"))
    with pytest.raises(ValueError, match="label_semantics='pnu'"):
        PUModelComparator(classifiers=["upu", "pnu"], cv=2)


def test_edge_tuning_pnu_never_trains_even_with_valid_binary_values(data, monkeypatch):
    from pu_toolbox.registry import get_algorithm, register_all_builtin_methods

    register_all_builtin_methods()
    monkeypatch.setattr(get_algorithm("pnu"), "fit", lambda *a, **k: pytest.fail("fit reached"))
    with pytest.raises(PipelineError, match="label_semantics='pnu'"):
        PUTuner(classifier="pnu", param_grid={}, cv=2).fit(*data, class_prior=0.4)
