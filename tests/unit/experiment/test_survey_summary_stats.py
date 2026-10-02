"""The summary's arithmetic: five repeats collapse to a mean and a spread.

Two choices here are silently wrong in the plausible alternative, so each gets a
test that names the alternative rather than only the chosen value:

* **Sample standard deviation, not population.**  ``statistics.pstdev`` differs
  by one denominator and reports a smaller spread.  The pilot reports the spread
  of the five runs it drew, so ``n - 1`` is the honest denominator.
* **A missing seed is reported, never filled.**  Filling an unrun seed with the
  mean of the others would make a four-seed partial row indistinguishable from a
  complete one, and protocol §5 clause 5 excludes an incomplete condition from
  the ranking rather than repairing it.

Cost lives here too, because protocol §5 clause 2 asks for the single-
configuration cost, the full tuning cost and the peak memory alongside every
mean it reports.  The peak is a maximum: clause 4 defines it over the whole
process, so summing peaks across runs would describe a machine nobody ran on.
"""

import math
import statistics

import pytest

from pu_toolbox.experiment.survey_summary import (
    SummaryError,
    aggregate_metric,
    sample_std,
    summarize_costs,
)

pytestmark = pytest.mark.unit


def test_basic_five_repeats_yield_a_mean_and_a_sample_std():
    values = [0.60, 0.70, 0.80, 0.90, 1.00]
    block = aggregate_metric(
        list(enumerate(values)), metric_name="accuracy", expected_seeds=[0, 1, 2, 3, 4]
    )

    assert block["n_expected"] == 5
    assert block["n_observed"] == 5
    assert block["seeds_observed"] == [0, 1, 2, 3, 4]
    assert block["missing_seeds"] == []
    assert block["mean"] == pytest.approx(0.8)
    # Independent implementation of the same statistic, to keep this from being
    # a restatement of the code under test.
    assert block["std"] == pytest.approx(statistics.stdev(values))


def test_determ_sample_std_uses_n_minus_one_not_the_population_form():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert sample_std(values) == pytest.approx(math.sqrt(2.5))
    assert statistics.pstdev(values) == pytest.approx(math.sqrt(2.0))
    assert sample_std(values) != pytest.approx(statistics.pstdev(values))


def test_basic_percent_pair_is_the_fraction_pair_scaled():
    block = aggregate_metric([(0, 0.5), (1, 0.7)], metric_name="accuracy", expected_seeds=[0, 1])

    assert block["metric_unit"] == "fraction"
    assert block["mean_percent"] == pytest.approx(block["mean"] * 100.0)
    assert block["std_percent"] == pytest.approx(block["std"] * 100.0)


def test_basic_a_missing_seed_is_reported_and_never_filled():
    block = aggregate_metric(
        [(0, 0.6), (1, 0.8), (3, 0.7), (4, 0.9)],
        metric_name="accuracy",
        expected_seeds=[0, 1, 2, 3, 4],
    )

    assert block["missing_seeds"] == [2]
    assert block["seeds_observed"] == [0, 1, 3, 4]
    # The mean is over the four that ran.  A fifth observation of any value would
    # move it, and no such observation exists.
    assert block["n_observed"] == 4
    assert block["mean"] == pytest.approx((0.6 + 0.8 + 0.7 + 0.9) / 4)
    assert block["std"] == pytest.approx(statistics.stdev([0.6, 0.8, 0.7, 0.9]))


def test_edge_a_single_observation_has_no_spread_to_report():
    block = aggregate_metric([(0, 0.9)], metric_name="accuracy", expected_seeds=[0, 1, 2])

    assert block["n_observed"] == 1
    assert block["mean"] == pytest.approx(0.9)
    # None rather than 0.0: one run is not evidence of a spread of zero.
    assert block["std"] is None
    assert block["std_percent"] is None
    assert sample_std([0.9]) is None


def test_edge_no_observations_at_all_reports_nothing_observed():
    block = aggregate_metric([], metric_name="accuracy", expected_seeds=[0, 1, 2, 3, 4])

    assert block["n_observed"] == 0
    assert block["mean"] is None
    assert block["std"] is None
    assert block["seeds_observed"] == []
    assert block["missing_seeds"] == [0, 1, 2, 3, 4]


def test_param_rejects_a_seed_observed_twice():
    with pytest.raises(SummaryError, match="repeats a seed"):
        aggregate_metric([(0, 0.5), (0, 0.6)], metric_name="accuracy", expected_seeds=[0])


def test_param_rejects_a_seed_outside_the_protocol_list():
    # The seed list is the protocol's, not the tree's: counting a sixth seed
    # would let one anomalous run outvote the five that were designed.
    with pytest.raises(SummaryError, match="outside the protocol"):
        aggregate_metric(
            [(0, 0.5), (5, 0.9)], metric_name="accuracy", expected_seeds=[0, 1, 2, 3, 4]
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_param_rejects_a_non_finite_observation(bad):
    with pytest.raises(SummaryError, match="finite"):
        aggregate_metric([(0, bad)], metric_name="accuracy", expected_seeds=[0])


def test_determ_seed_order_does_not_change_the_block():
    observations = [(0, 0.6), (1, 0.7), (2, 0.8)]

    forward = aggregate_metric(observations, metric_name="accuracy", expected_seeds=[0, 1, 2])
    reversed_ = aggregate_metric(
        list(reversed(observations)), metric_name="accuracy", expected_seeds=[0, 1, 2]
    )

    assert forward == reversed_


def test_basic_peak_gpu_memory_is_the_maximum_not_the_sum():
    resources = [
        {"peak_gpu_memory_bytes": 8_687_000_000, "tuning": {"elapsed_seconds": 100.0}},
        {"peak_gpu_memory_bytes": 9_000_000_000, "tuning": {"elapsed_seconds": 120.0}},
    ]
    block = summarize_costs(list(zip([0, 1], resources, strict=True)), expected_seeds=[0, 1])

    assert block["peak_gpu_memory_bytes"] == 9_000_000_000
    assert block["peak_gpu_memory_bytes"] != 8_687_000_000 + 9_000_000_000


def test_basic_tuning_cost_sums_over_seeds_while_single_config_keeps_its_mean():
    resources = [
        {
            "single_configuration_costs": [{"elapsed_seconds": 10.0}, {"elapsed_seconds": 20.0}],
            "tuning": {"elapsed_seconds": 30.0},
        },
        {
            "single_configuration_costs": [{"elapsed_seconds": 40.0}],
            "tuning": {"elapsed_seconds": 50.0},
        },
    ]
    block = summarize_costs(list(zip([0, 1], resources, strict=True)), expected_seeds=[0, 1])

    # The full tuning cost is the sum over the seeds that ran.
    assert block["tuning_cost_seconds"]["sum"] == pytest.approx(80.0)
    assert block["tuning_cost_seconds"]["n_observed"] == 2
    # A single configuration's cost is a per-candidate quantity, so it keeps both
    # the mean and the total -- one number with no scope would be unreadable.
    assert block["single_configuration_cost_seconds"]["sum"] == pytest.approx(70.0)
    assert block["single_configuration_cost_seconds"]["mean"] == pytest.approx(70.0 / 3)
    assert block["single_configuration_cost_seconds"]["n_observed"] == 3


def test_edge_unrecorded_costs_stay_none_rather_than_becoming_zero():
    block = summarize_costs([(0, {"peak_gpu_memory_bytes": None})], expected_seeds=[0, 1])

    assert block["peak_gpu_memory_bytes"] is None
    assert block["single_configuration_cost_seconds"] is None
    assert block["tuning_cost_seconds"] is None
    assert block["missing_seeds"] == [1]
