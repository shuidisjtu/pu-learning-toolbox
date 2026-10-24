"""Stage-aware diagnostic comparison cannot hide behind legacy placeholders."""

import copy

import pytest
from _episodic_budget_helpers import matrix

from pu_toolbox.experiment.survey_protocol import (
    budget_fairness_fields,
    validate_comparable_manifests,
)
from pu_toolbox.experiment.survey_staged_budget import FIELD
from pu_toolbox.experiment.survey_staged_budget import SCHEMA as STAGED_V1

pytestmark = pytest.mark.unit


def manifest():
    return {
        "protocol_version": "survey-v3.0",
        "protocol_sha256": "same-source",
        "seed": 0,
        "training_path": "native_2d",
        "comparability_group": "example",
        "backbone": {},
        "representation": {"split_sha256": "same", "feature_sha256": "same"},
        "budget": matrix("robust_pu")["budgets"]["native_stages"],
        "formal_eligible": False,
        "generation": {},
    }


def test_basic_nested_budget_reports_actual_cap_and_batch_not_placeholders():
    budget = matrix("robust_pu")["budgets"]["native_stages"]
    assert budget_fairness_fields(budget) == {"max_epochs": 3, "batch_size_candidates": (8,)}
    assert budget_fairness_fields({"epochs": 200, "batch_size": 256}) == {
        "max_epochs": 200,
        "batch_size_candidates": (256,),
    }


def test_basic_pulda_batch_pair_is_one_setting_not_two_candidates():
    left = manifest()
    left["budget"] = {
        FIELD: dict(
            schema_version=STAGED_V1,
            algorithm_id="pulda",
            parameter_values=dict(
                warmup_epochs=1,
                pu_epochs=2,
                positive_batch_size=2,
                unlabeled_batch_size=8,
                hidden_dim=4,
                depth=1,
                warmup_learning_rate=0.0001,
                learning_rate=0.001,
                warmup_weight_decay=0.0005,
                weight_decay=0.0001,
            ),
            checkpoint_epoch_count=3,
            checkpoint_bytes_per_component=2**20,
            eligible_stages=["warmup", "pu_mixup"],
        )
    }
    assert budget_fairness_fields(left["budget"]) == {
        "max_epochs": 3,
        "batch_size_candidates": (8,),
    }
    right = copy.deepcopy(left)
    right["budget"][FIELD]["parameter_values"]["positive_batch_size"] = 4
    assert budget_fairness_fields(left["budget"]) == budget_fairness_fields(right["budget"])
    with pytest.raises(ValueError, match="staged budget.parameter_values"):
        validate_comparable_manifests([left, right], require_formal=False)


@pytest.mark.parametrize("parameter,value", [("inner_epochs", 9), ("pretrain_lr", 0.02)])
def test_param_same_snapshot_cap_cannot_hide_different_paid_training(parameter, value):
    left = manifest()
    right = copy.deepcopy(left)
    right["budget"][FIELD]["parameter_values"][parameter] = value
    assert budget_fairness_fields(left["budget"]) == budget_fairness_fields(right["budget"])
    with pytest.raises(ValueError, match="staged budget.parameter_values"):
        validate_comparable_manifests([left, right], require_formal=False)


def test_edge_legacy_and_stage_budgets_do_not_compare_as_one_epoch_placeholders():
    left = manifest()
    right = copy.deepcopy(left)
    right["budget"] = {"epochs": 3, "batch_size": 8}
    with pytest.raises(ValueError, match="staged and legacy"):
        validate_comparable_manifests([left, right], require_formal=False)
    with pytest.raises(ValueError, match="formal"):
        validate_comparable_manifests([left])


def test_determ_resource_bound_does_not_change_training_budget_identity():
    left = manifest()
    original = copy.deepcopy(left)
    right = copy.deepcopy(left)
    right["budget"][FIELD]["checkpoint_bytes_per_component"] *= 2
    validate_comparable_manifests([left, right], require_formal=False)
    validate_comparable_manifests([left, right], require_formal=False)
    assert left == original
