"""Exercise the public tutorial's data flow and custom preprocessing."""

from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration
EXAMPLE = Path(__file__).resolve().parents[2] / "examples/minimal/14_diy_experiment.py"


@pytest.fixture(scope="module")
def demo():
    return runpy.run_path(str(EXAMPLE))


@pytest.mark.parametrize("custom", [False, True], ids=["built-in", "custom"])
def test_basic_param_selected_predictions_match_report_and_preserve_train_scaling(demo, custom):
    bundle = demo["make_bundle"]()
    result = demo["run_experiment"](bundle, custom=custom)

    assert not result.failures
    assert len(result.manifest["candidate_runs"]) == 3
    assert set(result.selections) == ({"PA-fixed", "OA"} if custom else {"PA", "OA"})
    for name in result.selections:
        predictions = demo["predict_selected"](result, name, bundle.test.X)
        assert np.mean(predictions == bundle.test.labels) == result.test_metrics[name]["accuracy"]
    if custom:
        for predictor in result.selected_models.values():
            np.testing.assert_allclose(predictor.named_steps["scale"].mean_, bundle.train.X.mean(0))
        for role in ("train", "pu_val"):
            assert result.manifest["generation"][role]["policy_id"] == "diy-scar-v1"
    assert result.manifest["calibration_applied"] is True


@pytest.mark.parametrize("custom", [False, True], ids=["built-in", "custom"])
def test_edge_determ_test_truth_changes_metrics_without_changing_selection(demo, custom):
    bundle = demo["make_bundle"]()
    first = demo["run_experiment"](bundle, custom=custom)
    changed = replace(bundle, test=replace(bundle.test, labels=1 - bundle.test.labels))
    second = demo["run_experiment"](changed, custom=custom)

    assert first.selections == second.selections
    for name in first.selections:
        np.testing.assert_array_equal(
            demo["predict_selected"](first, name, bundle.test.X),
            demo["predict_selected"](second, name, bundle.test.X),
        )
        assert second.test_metrics[name]["accuracy"] == pytest.approx(
            1 - first.test_metrics[name]["accuracy"]
        )


def test_basic_fixed_candidates_have_independent_hand_computed_optima(demo):
    records = demo["run_fixed_selection_case"]()
    assert records["PA"]["run_index"] == 0
    assert records["OA"]["run_index"] == 1
    assert records["PA"]["metrics"]["val_proxy_accuracy"] == 1.5
    assert records["OA"]["metrics"]["val_accuracy"] == 1.0
    for record in records.values():
        assert record["threshold"] == 0.5
        assert record["checkpoint_index"] is None
        assert record["epoch"] is None
    assert records["PA"]["test_accuracy"] == 0.0
    assert records["OA"]["test_accuracy"] == 1.0


def test_determ_edge_fixed_selection_is_independent_of_test_truth(demo):
    first = demo["run_fixed_selection_case"]()
    second = demo["run_fixed_selection_case"](test_labels=np.array([0, 1]))
    for name, record in first.items():
        assert second[name]["test_accuracy"] == 1 - record["test_accuracy"]
        assert {key: value for key, value in record.items() if key != "test_accuracy"} == {
            key: value for key, value in second[name].items() if key != "test_accuracy"
        }
