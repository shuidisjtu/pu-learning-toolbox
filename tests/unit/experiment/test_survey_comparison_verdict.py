# ruff: noqa: N803, N806
"""Comparison verdicts: the unit contract, the two SE branches and the threshold.

Split from the loader suite: that one establishes what a well-formed matrix
*is*, this one asks what a well-formed matrix *decides*.  The distinction
matters because a loader change should never be able to move a verdict, and
these two are the only place where the arithmetic is pinned.
"""

import pytest
from _survey_comparison_helpers import (
    payload as _payload,
)
from _survey_comparison_helpers import (
    rejected as _rejected,
)
from _survey_comparison_helpers import (
    resolved as _resolved,
)
from _survey_comparison_helpers import (
    result as _result,
)

from pu_toolbox.experiment.survey_comparison import evaluate_anchor

pytestmark = pytest.mark.unit


# --- 8/9/10. unit contract and the two uncertainty branches ------------------


def test_basic_fraction_result_is_converted_to_percentage_points():
    verdict = evaluate_anchor(_result(), _resolved(_payload()))
    assert verdict["mean_ours_pp"] == pytest.approx(85.3)
    assert verdict["spread_ours_pp"] == pytest.approx(2.1)


def test_basic_pooled_se_divides_for_std_and_not_for_sem():
    """The two branches must differ, or the implementation silently collapsed."""
    verdict_std = evaluate_anchor(_result(mean=0.893), _resolved(_payload()))
    # sqrt(2.63^2/10 + 2.1^2/5)
    assert verdict_std["se_pooled_pp"] == pytest.approx(1.254468, abs=1e-6)
    assert verdict_std["limit_pp"] == pytest.approx(3.0)

    sem_payload = _payload()
    sem_payload["anchors"][0]["uncertainty_kind"] = "sem"
    verdict_sem = evaluate_anchor(_result(mean=0.893), _resolved(sem_payload))
    # sqrt(2.63^2 + 2.1^2/5) -- the published spread already is the standard error
    assert verdict_sem["se_pooled_pp"] == pytest.approx(2.792651, abs=1e-6)
    assert verdict_sem["limit_pp"] == pytest.approx(5.585302, abs=1e-6)


def test_basic_threshold_outcome_differs_between_uncertainty_kinds():
    """delta_pp == 4.0 sits between the two limits; the branch must decide."""
    assert evaluate_anchor(_result(mean=0.893), _resolved(_payload()))["status"] == ("investigate")
    sem_payload = _payload()
    sem_payload["anchors"][0]["uncertainty_kind"] = "sem"
    assert evaluate_anchor(_result(mean=0.893), _resolved(sem_payload))["status"] == ("consistent")


def test_basic_error_rate_anchor_requires_an_explicit_conversion(tmp_path):
    converted = _payload()
    converted["anchors"][0].update(
        {
            "metric": "error_rate",
            "mean_percent": 19.0,
            "spread_percent": 1.4,
            "metric_conversion": {
                "to": "accuracy",
                "rule": "accuracy = 100 - error_rate",
                "mean_percent": 81.0,
                "spread_percent": 1.4,
            },
        }
    )
    verdict = evaluate_anchor(_result(mean=0.812), _resolved(converted))
    assert verdict["mean_anchor_pp"] == pytest.approx(81.0)

    unreported = _payload()
    unreported["anchors"][0]["metric"] = "error_rate"
    _rejected(tmp_path, unreported, "metric_conversion")


# --- 7/11/14. verdict discipline ---------------------------------------------


def test_edge_verdicts_stay_within_the_numeric_contract():
    payload = _payload()
    payload["mappings"][0]["eligibility"] = "magnitude_and_trend"
    with pytest.raises(ValueError, match="numeric"):
        evaluate_anchor(_result(), _resolved(payload))

    verdict = evaluate_anchor(_result(mean=0.70), _resolved(_payload()))
    assert verdict["status"] == "investigate"
    assert "implementation_error" not in verdict
    assert "conclusion" not in verdict


def test_edge_uncertain_provenance_cannot_be_numeric(tmp_path):
    payload = _payload()
    payload["anchors"][0]["protocol_provenance"] = "uncertain"
    _rejected(tmp_path, payload, "protocol_provenance")
