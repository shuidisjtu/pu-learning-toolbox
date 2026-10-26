# tests/unit/experiment/test_pilot_plan_dimensions.py

# ruff: noqa: N803, N806, S101

"""Which rows the planner can size, and what it asks the caller for.

A dataset the protocol runs but the planner cannot size stops the whole plan, so
the shipped protocol is checked against the planner's own defaults.
"""

import importlib.util
from pathlib import Path

import pytest

from pu_toolbox.experiment.pilot_plan import estimate_checkpoint_bytes
from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_protocol_path

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def test_basic_every_dataset_with_a_native_2d_row_has_a_default_feature_dimension():
    """A dataset the protocol runs natively but the driver cannot size stops the plan."""
    spec = importlib.util.spec_from_file_location(
        "run_survey_pilot_dims", _SCRIPTS / "run_survey_pilot.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    protocol = load_protocol(resolve_protocol_path("survey-v3.0"))

    native = {
        row["dataset"]
        for row in protocol["execution_units"]
        if row["runnable"] and row["training_path"] == "native_2d"
    }

    assert native <= set(driver.DEFAULT_INPUT_DIMS)


@pytest.mark.parametrize("dataset", ["cifar10", "mnist", "fashion_mnist", "adni"])
def test_basic_every_image_row_is_sized_without_a_feature_dimension(dataset):
    """Only native_2d rows size a network from the input width.

    The guard used to key off a backbone name starting with ``resnet18``, so the
    GEN-PU/PULDA native CNNs and the three adapter profiles, which carry their own
    backbone names, were refused for a number the image path never reads.
    """
    protocol = load_protocol(resolve_protocol_path("survey-v3.0"))
    protocol["execution_units"] = [
        row
        for row in protocol["execution_units"]
        if row["dataset"] == dataset and row["training_path"] != "native_2d"
    ]

    estimate = estimate_checkpoint_bytes(protocol, input_dims={})

    assert estimate["units"] == sum(row["runnable"] for row in protocol["execution_units"])
