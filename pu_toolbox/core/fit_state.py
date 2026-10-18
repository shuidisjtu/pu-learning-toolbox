"""Opt-in failed-fit invalidation without changing estimator training logic."""

from functools import wraps


def invalidate_failed_fit(fit):
    """Keep a failed or interrupted fit from exposing an old/partial model.

    The estimator still owns successful completion and callback visibility.
    Only the public fitted flag is invalidated: learned attributes, callbacks'
    external side effects and saved artifacts are not rolled back. Cancellation
    is re-raised unchanged, just like ordinary validation or training errors.
    """

    @wraps(fit)
    def guarded(self, *args, **kwargs):
        self._is_fitted = False
        try:
            return fit(self, *args, **kwargs)
        except BaseException:
            self._is_fitted = False
            raise

    return guarded
