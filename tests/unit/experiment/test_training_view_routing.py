# F811: ``survey_script`` is a pytest fixture looked up by name.
# ruff: noqa: N803, N806, S101, F811

"""Routing of the ``os_or_ts`` training view from a trainer to an estimator.

The calibrated view only exists if it reaches ``estimator.fit``.  A silent drop
would leave the run's manifest claiming a view the estimator never applied, so
every hop here is fail-loud rather than best-effort.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from _survey_script_helpers import survey_script  # noqa: F401 - pytest fixture

from pu_toolbox.estimators.bias_aware import PUSBKernelClassifier
from pu_toolbox.estimators.deep import SelfPUClassifier
from pu_toolbox.estimators.risk import DistPUClassifier
from pu_toolbox.experiment.protocols import route_training_view
from pu_toolbox.experiment.strategies import DeepFitTrainer, FitTrainer, SupervisedTrainer
from pu_toolbox.experiment.training_views import resolve_training_view

pytestmark = pytest.mark.unit

X = np.array([[0.0], [1.0], [2.0], [3.0]])
Y = np.array([1, 0, 0, 0])


class _DeclaringEstimator:
    """Estimator whose ``fit`` advertises the calibrated view.

    Its own default is ``"ts"``, standing for a method native to the calibrated
    view (VPU is the first): for such a target, dropping an explicit ``os``
    request would run the opposite of what the operator asked for. That is why
    ``os`` is routed at all.
    """

    def __init__(self):
        self.received = "unset"

    def fit(self, X, y, *, class_prior=None, os_or_ts="ts"):  # noqa: N803
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
    def test_none_is_not_routed(self):
        """No request means no keyword: the estimator's own default decides."""
        kwargs: dict = {}
        declaring = _DeclaringEstimator()
        route_training_view(kwargs, _params(declaring), None, declaring)
        assert kwargs == {}

    def test_os_is_routed_when_the_target_declares_it(self):
        """An explicit ``os`` must reach a target whose own default is not os.

        Without this, a method native to the calibrated view runs the view the
        operator did not ask for, and the manifest records the request rather
        than what happened -- the silent reversal the module exists to prevent,
        in the other direction.
        """
        declaring = _DeclaringEstimator()
        kwargs: dict = {}
        route_training_view(kwargs, _params(declaring), "os", declaring)
        assert kwargs == {"os_or_ts": "os"}

    def test_os_is_not_routed_when_undeclared(self):
        """A target that never declares the view is OS by construction."""
        plain = _PlainEstimator()
        kwargs: dict = {}
        route_training_view(kwargs, _params(plain), "os", plain)
        assert kwargs == {}

    def test_os_is_not_routed_when_fit_only_takes_var_keyword(self):
        """``**kwargs`` is not evidence that the view was meant to be carried.

        The ``ts`` path keeps the repo's named-or-var-keyword rule (frozen
        behaviour); an ``os`` request has nothing to downgrade, so it is not
        injected into a signature that never named the parameter.
        """

        class _VarKeywordEstimator:
            def fit(self, X, y, **kwargs):  # noqa: N803
                self.kwargs = kwargs

        estimator = _VarKeywordEstimator()
        kwargs: dict = {}
        route_training_view(kwargs, _params(estimator), "os", estimator)
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


@pytest.mark.parametrize(
    ("method", "estimator_class"),
    [
        ("pusb_kernel", PUSBKernelClassifier),
        ("dist_pu", DistPUClassifier),
        ("self_pu", SelfPUClassifier),
    ],
)
class TestResolutionForARealMethod:
    """The resolver's answer for a real method, against the real ledger.

    The other suites pin the resolution *rule* with stand-in estimators; this
    one pins the wiring of an actual method, so a ledger entry the code cannot
    honour (or code the ledger does not declare) is caught instead of assumed.

    Parametrized rather than duplicated per method: this file sits on the
    test-quality gate's per-file method limit, and the methods are checked by
    exactly the same two properties.
    """

    def test_basic_the_method_is_declared_and_wired_to_the_calibrated_view(
        self, survey_script, method, estimator_class
    ):
        ledger = survey_script.load_ledger(survey_script.LEDGER_PATH)
        entry = ledger["methods"][method]

        assert entry["native_sampling_assumption"].startswith("ts")
        assert entry["run_view"] == "ts-compatible"
        assert entry["calibration_applied"] is True
        assert (
            resolve_training_view(
                ledger,
                method,
                None,
                is_oracle=False,
                estimator_class=estimator_class,
            )
            == "ts"
        )

    def test_basic_the_explicit_view_request_is_honoured(
        self, survey_script, method, estimator_class
    ):
        ledger = survey_script.load_ledger(survey_script.LEDGER_PATH)

        for requested, expected in (("os", "os"), ("ts", "ts")):
            assert (
                resolve_training_view(
                    ledger,
                    method,
                    requested,
                    is_oracle=False,
                    estimator_class=estimator_class,
                )
                == expected
            )
