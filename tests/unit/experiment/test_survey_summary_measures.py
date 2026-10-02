"""How an absent value is read: a measurement, a peak, a metric, a status.

The four are not the same finding.  A cost the protocol requires a report to carry
is a reporting gap; a GPU peak on a run that never allocated GPU memory is the
complete answer; a secondary metric a run could not produce is an unavailable
observation rather than a missing seed; and a status that belongs to no report
table is a defect in the vocabulary.  Collapsing them either invents defects or
hides them, so each is asserted separately here.

The counterpart on the audit side -- which of these degrade a row's status -- is
``test_survey_audit_checks.py``.
"""

import pytest
from _survey_summary_helpers import manifest

from pu_toolbox.experiment import survey_comparison
from pu_toolbox.experiment.survey_summary import (
    COMPARISON_METRIC,
    REPORT_TIERS,
    STATUSES,
    SummaryError,
    aggregate_additional_metric,
    matrix_selector,
    tier_of,
    unrecorded_measurements,
)

pytestmark = pytest.mark.unit


def _resources(*, run_peak=None, candidate_peak=None, tuning=True, costs=True):
    """A ``resources`` block, with each measured field independently droppable."""
    block = {
        "peak_gpu_memory_bytes": run_peak,
        "single_configuration_costs": [
            {"elapsed_seconds": 10.0, "peak_gpu_memory_bytes": candidate_peak}
        ],
    }
    if tuning:
        block["tuning"] = {"elapsed_seconds": 12.0}
    if not costs:
        block["single_configuration_costs"] = []
    return block


def test_basic_a_run_records_the_two_costs_the_protocol_requires():
    assert unrecorded_measurements([_resources(run_peak=1, candidate_peak=1)]) == []


def test_basic_a_cpu_only_run_reports_no_peak_without_being_a_finding():
    # The measured shape of the B1 pilot: the classical methods are null at the run
    # level and at candidate level, because they never allocated GPU memory.
    assert unrecorded_measurements([_resources(run_peak=None, candidate_peak=None)]) == []


def test_param_missing_measurements_are_named_rather_than_counted():
    without_tuning = unrecorded_measurements([_resources(tuning=False)])
    without_costs = unrecorded_measurements([_resources(costs=False)])

    assert without_tuning == ["tuning_cost_seconds"]
    assert without_costs == ["single_configuration_cost_seconds"]


def test_edge_a_lost_run_level_peak_is_named_when_the_candidates_have_one():
    # The opposite shape from the CPU-only case: a peak existed to aggregate and the
    # run-level field did not receive it.
    assert unrecorded_measurements([_resources(run_peak=None, candidate_peak=7)]) == [
        "peak_gpu_memory_bytes"
    ]


def test_edge_an_empty_resources_block_names_both_required_costs():
    assert unrecorded_measurements([]) == [
        "single_configuration_cost_seconds",
        "tuning_cost_seconds",
    ]


def test_basic_an_additional_metric_aggregates_over_the_seeds_that_ran():
    block = aggregate_additional_metric(
        [(0, 0.5), (1, 0.6), (2, 0.7)], metric_name="auc", ran_seeds=[0, 1, 2]
    )

    assert block["mean"] == pytest.approx(0.6)
    assert block["std"] == pytest.approx(0.1)
    assert block["n_observed"] == 3
    assert block["n_seeds"] == 3
    assert block["unavailable_seeds"] == []


def test_edge_an_unavailable_observation_is_not_a_missing_seed():
    # Seed 1 ran -- the primary metric has it -- but this metric is NaN, which the
    # manifest's own auc_unavailable_reason documents.  Reporting it as a missing
    # seed would blame the run for a property of the data.
    block = aggregate_additional_metric(
        [(0, 0.5), (1, float("nan"))], metric_name="auc", ran_seeds=[0, 1]
    )

    assert block["n_observed"] == 1
    assert block["n_seeds"] == 2
    assert block["unavailable_seeds"] == [1]
    assert "missing_seeds" not in block


def test_param_an_additional_metric_that_repeats_a_seed_raises():
    with pytest.raises(SummaryError, match="repeats a seed"):
        aggregate_additional_metric([(0, 0.5), (0, 0.6)], metric_name="auc", ran_seeds=[0])


def test_param_an_additional_metric_observing_a_seed_outside_the_run_raises():
    # A seed the primary metric never saw would outvote the designed five, and it
    # would not even show as missing.
    with pytest.raises(SummaryError, match="outside the seeds that ran"):
        aggregate_additional_metric([(9, 0.5)], metric_name="auc", ran_seeds=[0, 1])


def test_basic_the_report_tiers_partition_the_closed_status_set():
    grouped = [status for statuses in REPORT_TIERS.values() for status in statuses]

    assert sorted(grouped) == sorted(STATUSES)
    assert len(grouped) == len(set(grouped))


def test_param_an_unknown_status_raises_rather_than_filing_silently():
    with pytest.raises(SummaryError, match="belongs to no report tier"):
        tier_of("blocked")


def test_determ_the_matrix_selector_is_spelled_the_way_the_matrix_reads():
    payload = manifest(seed=0, c=0.1)
    first = matrix_selector(payload, selection_protocol="OA")

    assert matrix_selector(payload, selection_protocol="OA") == first
    # The manifest stores results under upper-case keys; the matrix selects on
    # lower-case ones.
    assert first["selection_protocol"] == "oa"
    oracle = matrix_selector(manifest(c_independent=True), selection_protocol="OA")
    assert oracle["c_token"] == oracle["labeling_mechanism"] == "c_independent"


def test_determ_the_comparison_metric_is_the_matrix_own_constant():
    # Two modules name the same pre-registered fact; pinned so neither can move
    # without the other, and so a selector cannot be built on a metric the matrix
    # does not compare.
    assert COMPARISON_METRIC == survey_comparison._COMPARISON_METRIC
