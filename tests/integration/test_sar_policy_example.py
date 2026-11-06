"""Check the labeling policies and provenance exposed by the SAR tutorial."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration
EXAMPLE = Path(__file__).resolve().parents[2] / "examples/minimal/17_sar_policies.py"


@pytest.mark.parametrize("frequency", [0.01, 0.4, 1.0])
def test_basic_param_edge_policies_preserve_one_sided_labels_and_count_identity(frequency):
    demo = runpy.run_path(str(EXAMPLE))
    features, truth = demo["make_pool"]()
    records = demo["compare_policies"](features, truth, label_frequency=frequency)
    assert len(records) == 4
    for labels, metadata in records:
        assert np.all(truth[labels == 1] == 1)
        assert metadata["n_labeled"] == int(labels.sum())
        assert metadata["c_realized"] == float(labels.sum() / truth.sum())
        assert metadata["input_sampling"] == "os"
        if metadata["sampling_policy"] == "independent_bernoulli":
            assert metadata["propensity_mean_positive"] == pytest.approx(frequency)
            assert metadata["ensure_labeled"] is False
        else:
            assert int(labels.sum()) == max(1, round(truth.sum() * frequency))
            assert metadata["posterior_fit_on"] == "real_labels"
        json.dumps(metadata, allow_nan=False)


def test_determ_repeated_seed_preserves_labels_and_source_records():
    demo = runpy.run_path(str(EXAMPLE))
    features, truth = demo["make_pool"]()
    first = demo["compare_policies"](features, truth)
    second = demo["compare_policies"](features, truth)
    for (labels_a, record_a), (labels_b, record_b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(labels_a, labels_b)
        assert record_a == record_b
    assert len({record["label_view_sha256"] for _, record in first}) > 1


@pytest.mark.parametrize("frequency", [0.0, -0.1, 1.1, float("nan")])
def test_param_invalid_requested_rate_uses_generator_validation(frequency):
    demo = runpy.run_path(str(EXAMPLE))
    features, truth = demo["make_pool"]()
    with pytest.raises(ValueError, match="label_frequency"):
        demo["compare_policies"](features, truth, label_frequency=frequency)
