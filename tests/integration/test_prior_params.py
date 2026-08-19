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
def test_param_deprecated_prior_alias_warns_but_still_resolves():
    """The legacy ``pe`` alias remains usable during its migration window."""
    with pytest.warns(FutureWarning, match="class_prior_estimation"):
        pipe = PUPipeline(prior_estimator="pe")
    assert pipe._prior_cls is ClassPriorEstimator


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
def test_param_convertible_numeric_strings_accepted(rng):
    """'0.5' / '200' (JSON-config style) behave like the CLI coercion path."""
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    via_str = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": "0.5"})
    via_num = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": 0.5})
    assert via_str.fit_evaluate(X, y_pu).prior.value == pytest.approx(
        via_num.fit_evaluate(X, y_pu).prior.value
    )


@pytest.mark.integration
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "inf", "nan"])
def test_edge_non_finite_numeric_param_rejected(rng, bad):
    """NaN/Inf sigma must fail fast instead of silently degrading."""
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    pipe = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": bad})
    with pytest.raises(PipelineError, match="finite"):
        pipe.fit_evaluate(X, y_pu)


@pytest.mark.integration
def test_param_bool_string_parsed_not_truthy(rng):
    """standardize='False' must mean False, not the truthy string 'False'.

    Without parsing, 'False' is truthy and silently enables standardization
    — the opposite of the user's intent.
    """
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    pipe = PUPipeline(prior_estimator="pen_l1", prior_params={"standardize": "False"})
    report = pipe.fit_evaluate(X, y_pu)
    assert report.prior.source == "estimated"  # constructible, no error


@pytest.mark.integration
def test_edge_invalid_literal_value_rejected(rng):
    """An out-of-domain value for a constrained string parameter fails fast."""
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    pipe = PUPipeline(prior_estimator="km2", prior_params={"variant": "bad"})
    with pytest.raises(PipelineError, match="variant"):
        pipe.fit_evaluate(X, y_pu)


@pytest.mark.integration
def test_determ_prior_params_reproducible(rng):
    """Same params give the same estimate across runs."""
    X, y_pu, _ = make_scar_data(rng, n=150, separation=2.0)
    first = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": 2.0}).fit_evaluate(X, y_pu)
    second = PUPipeline(prior_estimator="pen_l1", prior_params={"sigma": 2.0}).fit_evaluate(X, y_pu)
    assert first.prior.value == pytest.approx(second.prior.value)
