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
