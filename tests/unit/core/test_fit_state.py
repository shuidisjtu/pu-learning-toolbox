"""Opt-in lifecycle guard preserves training ownership and exception identity."""

import inspect

import pytest

from pu_toolbox.core.fit_state import invalidate_failed_fit

pytestmark = pytest.mark.unit


class Estimator:
    _is_fitted = True

    @invalidate_failed_fit
    def fit(self, value, *, success=True, failure=None):
        """Synthetic fit with observable entry state and partial diagnostics."""
        self.entry_state = self._is_fitted
        self.learned_value = value
        self._is_fitted = success
        if failure is not None:
            raise failure
        return self


def test_basic_determ_success_retains_return_signature_and_estimator_ownership():
    instance = Estimator()
    assert instance.fit(7) is instance
    assert instance.entry_state is False
    assert instance._is_fitted is True
    assert instance.learned_value == 7
    assert inspect.signature(Estimator.fit) == inspect.signature(Estimator.fit.__wrapped__)
    assert Estimator.fit.__doc__ == Estimator.fit.__wrapped__.__doc__
    assert Estimator.fit.__name__ == "fit"


def test_edge_guard_does_not_manufacture_success_or_rollback_diagnostics():
    instance = Estimator()
    assert instance.fit(3, success=False) is instance
    assert instance._is_fitted is False
    assert instance.learned_value == 3


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, KeyboardInterrupt, SystemExit])
def test_param_error_or_cancellation_invalidates_then_reraises_same_object(error_type):
    instance = Estimator()
    failure = error_type("stop")
    with pytest.raises(error_type) as caught:
        instance.fit(9, failure=failure)
    assert caught.value is failure
    assert instance._is_fitted is False
    assert instance.learned_value == 9
    assert instance.fit(10) is instance
    assert instance._is_fitted is True


@pytest.mark.parametrize("args,kwargs", [((), {}), ((1,), {"clean_val": [1]})])
def test_param_binding_failure_invalidates_before_original_body(args, kwargs):
    instance = Estimator()
    with pytest.raises(TypeError):
        instance.fit(*args, **kwargs)
    assert instance._is_fitted is False
    assert not hasattr(instance, "learned_value")
