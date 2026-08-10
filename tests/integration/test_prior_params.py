# ruff: noqa: N802, N803, N806, S101, S113, E501

"""Tests for PUPipeline prior_params forwarding (CLI --prior-param backend).

The pipeline accepts a dict of estimator constructor parameters alongside
a prior-estimator *name*; an estimator *instance* already carries its own
parameters, so combining the two is rejected.
"""

from __future__ import annotations

import pytest

from pu_toolbox.prior import ClassPriorEstimator
from pu_toolbox.workflows import PipelineError, PUPipeline
from tests.helpers import make_scar_data

pytestmark = [pytest.mark.integration]


@pytest.mark.integration
def test_basic_prior_params_forwarded_to_estimator(rng):
    """prior_params reach the estimator constructor (string name path)."""
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    via_params = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": 3.0})
    via_instance = PUPipeline(prior_estimator=ClassPriorEstimator(sigma=3.0))
    report_a = via_params.fit_evaluate(X, y_pu)
    report_b = via_instance.fit_evaluate(X, y_pu)
    assert report_a.prior.value == pytest.approx(report_b.prior.value)


@pytest.mark.integration
def test_param_prior_params_with_instance_raises():
    with pytest.raises(TypeError, match="prior_params cannot be combined"):
        PUPipeline(prior_estimator=ClassPriorEstimator(sigma=1.0), prior_params={"sigma": 2.0})


@pytest.mark.integration
def test_edge_invalid_prior_param_raises(rng):
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    pipe = PUPipeline(prior_estimator="pen_l1", prior_params={"not_a_param": 1})
    with pytest.raises(PipelineError, match="invalid prior parameters"):
        pipe.fit_evaluate(X, y_pu)


@pytest.mark.integration
def test_edge_non_numeric_value_for_numeric_param_raises(rng):
    """A string like 'abc' for a numeric parameter is rejected upfront.

    Without this check the value only fails inside the estimator at fit time
    and is swallowed by the auto-mode degradation path, leaving the user a
    no-prior report instead of a clear error.
    """
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    pipe = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": "abc"})
    with pytest.raises(PipelineError, match="not a number"):
        pipe.fit_evaluate(X, y_pu)


@pytest.mark.integration
def test_param_string_value_allowed_for_string_param(rng):
    """String values remain valid for string-typed parameters (e.g. variant)."""
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    pipe = PUPipeline(prior_estimator="km2", prior_params={"variant": "km1"})
    report = pipe.fit_evaluate(X, y_pu)
    assert report.prior.source == "estimated"


@pytest.mark.integration
def test_determ_prior_params_reproducible(rng):
    """Same params give the same estimate across runs."""
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    first = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": 2.0}).fit_evaluate(X, y_pu)
    second = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": 2.0}).fit_evaluate(X, y_pu)
    assert first.prior.value == pytest.approx(second.prior.value)
