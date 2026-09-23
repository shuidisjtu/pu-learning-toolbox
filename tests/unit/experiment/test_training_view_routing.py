# ruff: noqa: N803, N806, S101

"""Routing of the ``os_or_ts`` training view from a trainer to an estimator.

The calibrated view only exists if it reaches ``estimator.fit``.  A silent drop
would leave the run's manifest claiming a view the estimator never applied, so
every hop here is fail-loud rather than best-effort.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pu_toolbox.experiment.protocols import route_training_view
from pu_toolbox.experiment.strategies import DeepFitTrainer, FitTrainer, SupervisedTrainer

pytestmark = pytest.mark.unit

X = np.array([[0.0], [1.0], [2.0], [3.0]])
Y = np.array([1, 0, 0, 0])


class _DeclaringEstimator:
    """Estimator whose ``fit`` advertises the calibrated view."""

    def __init__(self):
        self.received = "unset"

    def fit(self, X, y, *, class_prior=None, os_or_ts=None):  # noqa: N803
        self.received = os_or_ts
        return self


class _PlainEstimator:
    """Estimator that never declares the view; training it must not happen."""

    def __init__(self):
        self.calls = 0

    def fit(self, X, y, *, class_prior=None):  # noqa: N803
        self.calls += 1
        return self


def _params(estimator):
    """The ``fit`` parameters the router inspects, taken from a real signature."""
    return inspect.signature(type(estimator).fit).parameters


class TestRouteTrainingView:
    def test_os_and_none_are_not_routed(self):
        """Only a calibrated request changes training, so only it is forwarded."""
        kwargs: dict = {}
        declaring = _DeclaringEstimator()
        route_training_view(kwargs, _params(declaring), "os", declaring)
        route_training_view(kwargs, _params(declaring), None, declaring)
        assert kwargs == {}

    def test_ts_is_routed_when_declared(self):
        declaring = _DeclaringEstimator()
        kwargs: dict = {}
        route_training_view(kwargs, _params(declaring), "ts", declaring)
        assert kwargs == {"os_or_ts": "ts"}

    def test_ts_is_routed_when_fit_takes_var_keyword(self):
        """Matches the repo's `_select_kwargs` rule for optional fit kwargs."""

        class _VarKeywordEstimator:
            def fit(self, X, y, **kwargs):  # noqa: N803
                self.kwargs = kwargs

        estimator = _VarKeywordEstimator()
        kwargs: dict = {}
        route_training_view(kwargs, _params(estimator), "ts", estimator)
        assert kwargs == {"os_or_ts": "ts"}

    def test_ts_is_refused_when_undeclared(self):
        plain = _PlainEstimator()
        with pytest.raises(ValueError, match="does not accept os_or_ts"):
            route_training_view({}, _params(plain), "ts", plain)


class TestTrainerForwarding:
    def test_fit_trainer_forwards_the_view(self):
        estimator = _DeclaringEstimator()
        FitTrainer().fit(estimator, X, Y, os_or_ts="ts")
        assert estimator.received == "ts"

    def test_supervised_trainer_forwards_the_view(self):
        estimator = _DeclaringEstimator()
        SupervisedTrainer().fit(estimator, X, Y, os_or_ts="ts")
        assert estimator.received == "ts"

    def test_deep_fit_trainer_forwards_the_view_without_validation(self):
        """No validation data means the bare-fit path; the view must survive it."""
        estimator = _DeclaringEstimator()
        DeepFitTrainer().fit(estimator, X, Y, os_or_ts="ts")
        assert estimator.received == "ts"

    @pytest.mark.parametrize("trainer", [FitTrainer(), SupervisedTrainer(), DeepFitTrainer()])
    def test_trainers_refuse_ts_for_a_plain_estimator(self, trainer):
        """A view the estimator cannot honour must abort before any training."""
        estimator = _PlainEstimator()
        with pytest.raises(ValueError, match="does not accept os_or_ts"):
            trainer.fit(estimator, X, Y, os_or_ts="ts")
        assert estimator.calls == 0

    @pytest.mark.parametrize("trainer", [FitTrainer(), SupervisedTrainer(), DeepFitTrainer()])
    def test_os_default_leaves_plain_estimators_untouched(self, trainer):
        """Existing OS runs keep calling ``fit`` exactly as before."""
        estimator = _PlainEstimator()
        trainer.fit(estimator, X, Y)
        assert estimator.calls == 1
