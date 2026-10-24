"""Explicit expected cells remain visible even when no manifest was delivered."""

import copy

import pytest

from pu_toolbox.experiment.survey_result_coverage import (
    expand_expected_cells,
    reconcile_result_coverage,
)

pytestmark = pytest.mark.unit


def _row(**changes):
    identity = {
        "dataset": "cifar10",
        "method": "nnpu",
        "training_path": "native_cnn",
        "backbone": "resnet18",
        "comparability_group": "cifar10/native_cnn/minibatch",
        "run_view": "ts-compatible",
        "calibration_applied": True,
        "selection_path": "pure_pu",
        "selection_protocol": "oa",
        "labeling_mechanism": "scar",
        "c_token": "0.1",
        "metric": "accuracy",
        "budget_identity": "fixture-budget-v1",
    }
    identity.update(changes)
    return {"identity": identity}


def _observation(cell, **changes):
    manifest = {
        "selection": {"OA": {"epoch": 1}},
        "test_results": {"OA": {"accuracy": 0.75, "auc": 0.8}},
        "formal_eligible": True,
        "formal_blockers": [],
        "protocol_deviation": [],
    }
    manifest.update(changes)
    return {"identity": cell["identity"], "manifest": manifest, "source": "fixture.json"}


def test_basic_an_entire_absent_row_is_reported_for_every_expected_seed():
    cells = expand_expected_cells([_row()], seeds=[0, 1, 2, 3, 4])

    report = reconcile_result_coverage(cells, [])

    assert report["counts"] == {"missing": 5}
    assert [cell["identity"]["seed"] for cell in report["cells"]] == list(range(5))
    assert report["coverage_complete"] is False


def test_basic_missing_seed_is_not_silently_dropped_from_complete_results():
    cells = expand_expected_cells([_row()], seeds=[0, 1])
    report = reconcile_result_coverage(cells, [_observation(cells[0])])

    assert [cell["status"] for cell in report["cells"]] == ["complete", "missing"]
    assert report["cells"][0]["value"] == 0.75
    assert report["formal_readiness_assessed"] is False


@pytest.mark.parametrize(
    "manifest_changes,status",
    [
        ({"test_results": {}}, "incomplete"),
        ({"selection": {}}, "incomplete"),
        ({"test_results": {"OA": {"accuracy": None}}}, "metric_unavailable"),
        ({"test_results": {"OA": {}}}, "metric_unavailable"),
        ({"test_results": {"OA": {"accuracy": float("nan")}}}, "metric_unavailable"),
        ({"test_results": {"OA": {"accuracy": float("inf")}}}, "metric_unavailable"),
        ({"test_results": {"OA": {"accuracy": True}}}, "metric_unavailable"),
        ({"test_results": {"OA": {"accuracy": "0.75"}}}, "metric_unavailable"),
        ({"formal_eligible": False, "formal_blockers": ["pending_review"]}, "formal_blocked"),
        ({"protocol_deviation": ["seed_subset"]}, "protocol_deviation"),
    ],
)
def test_param_completion_metric_deviation_and_blockers_have_distinct_states(
    manifest_changes, status
):
    cells = expand_expected_cells([_row()], seeds=[0])
    report = reconcile_result_coverage(cells, [_observation(cells[0], **manifest_changes)])

    assert report["cells"][0]["status"] == status
    assert report["coverage_complete"] is (status in {"formal_blocked", "protocol_deviation"})


def test_edge_missing_metric_retains_its_reason_and_other_eligibility_issues():
    cells = expand_expected_cells([_row()], seeds=[0])
    observation = _observation(
        cells[0],
        test_results={"OA": {"accuracy": None, "accuracy_unavailable_reason": "not recorded"}},
        formal_eligible=False,
        formal_blockers=["pending_review"],
        protocol_deviation=["budget"],
        failures=["training interrupted"],
    )

    result = reconcile_result_coverage(cells, [observation])["cells"][0]

    assert result["status"] == "metric_unavailable"
    assert result["metric_unavailable_reason"] == "not recorded"
    assert result["formal_blockers"] == ["pending_review"]
    assert result["protocol_deviation"] == ["budget"]
    assert result["failures"] == ["training interrupted"]


def test_basic_additional_metric_gaps_do_not_make_completed_primary_results_missing():
    cells = expand_expected_cells([_row()], seeds=[0])
    result = reconcile_result_coverage(cells, [_observation(cells[0])])["cells"][0]

    assert result["status"] == "complete"
    assert result["additional_metric_gaps"] == {
        "f1": "Legacy manifest does not record this metric.",
        "precision": "Legacy manifest does not record this metric.",
        "recall": "Legacy manifest does not record this metric.",
    }


def test_basic_excluded_kldce_sar_cells_have_reasons_and_are_not_missing():
    scar = _row(method="kldce")
    sar = _row(method="kldce", labeling_mechanism="sar_lbe_a", c_token="0.05")
    sar["excluded_reason"] = "KLDCE is SCAR-only."
    cells = expand_expected_cells([scar, sar], seeds=[0])

    report = reconcile_result_coverage(cells, [_observation(cells[0])])

    assert report["counts"] == {"complete": 1, "excluded": 1}
    assert report["coverage_complete"] is True
    assert report["cells"][1]["excluded_reason"] == "KLDCE is SCAR-only."


def test_basic_pn_c_independent_results_are_expected_once_per_path_and_seed():
    rows = [
        _row(
            method="pn_oracle",
            training_path=path,
            labeling_mechanism="c_independent",
            c_token="c_independent",
            run_view="os-compatible",
            calibration_applied=False,
        )
        for path in ("native_cnn", "cnn_feature_adapter")
    ]

    cells = expand_expected_cells(rows, seeds=list(range(5)))
    report = reconcile_result_coverage(cells, [_observation(cell) for cell in cells])

    assert len(cells) == 10
    assert report["counts"] == {"complete": 10}
    assert {cell["identity"]["c_token"] for cell in cells} == {"c_independent"}


def test_edge_observation_for_an_excluded_cell_is_not_silently_accepted():
    row = _row()
    row["excluded_reason"] = "No approved support extraction."
    cells = expand_expected_cells([row], seeds=[0])
    report = reconcile_result_coverage(cells, [_observation(cells[0])])

    assert report["cells"][0]["status"] == "excluded"
    assert report["cells"][0]["unexpected_observation"] is True
    assert report["coverage_complete"] is False


def test_edge_unexpected_backbone_budget_or_support_variant_cannot_fill_a_cell():
    cells = expand_expected_cells([_row()], seeds=[0])
    observations = []
    for field in ("backbone", "budget_identity", "selection_path"):
        observation = _observation(copy.deepcopy(cells[0]))
        observation["identity"][field] = "another-variant"
        observations.append(observation)

    report = reconcile_result_coverage(cells, observations)

    assert report["counts"] == {"missing": 1}
    assert len(report["unexpected"]) == 3


def test_basic_lbe_variants_and_training_views_are_distinct_expected_cells():
    rows = [
        _row(method="lbe", labeling_mechanism=mechanism, run_view=view)
        for mechanism in ("sar_lbe_a", "sar_lbe_b")
        for view in ("os-compatible", "ts-compatible")
    ]
    cells = expand_expected_cells(rows, seeds=[0])

    report = reconcile_result_coverage(cells, [_observation(cells[0])])

    assert report["counts"] == {"complete": 1, "missing": 3}


def test_basic_clean_support_protocol_keeps_its_manifest_key_and_is_not_pure_pa():
    cells = expand_expected_cells(
        [_row(selection_protocol=name) for name in ("pa", "pa+clean-support", "oa")],
        seeds=[0],
    )
    block = {
        "PA": {"accuracy": 0.6},
        "PA+clean-support": {"accuracy": 0.7},
        "OA": {"accuracy": 0.8},
    }
    observations = [_observation(cell, selection=block, test_results=block) for cell in cells]
    report = reconcile_result_coverage(cells, observations)
    assert [cell["value"] for cell in report["cells"]] == [0.6, 0.7, 0.8]
    wrong_keys = {"PA": block["PA"], "PA+CLEAN-SUPPORT": block["PA+clean-support"]}
    report = reconcile_result_coverage(
        [cells[1]], [_observation(cells[1], selection=wrong_keys, test_results=wrong_keys)]
    )
    assert report["counts"] == {"incomplete": 1}


def test_edge_empty_matrix_is_not_reported_as_complete_and_exclusions_need_reasons():
    assert reconcile_result_coverage([], [])["coverage_complete"] is False
    cells = expand_expected_cells([_row()], seeds=[0])
    cells[0]["excluded_reason"] = " "
    with pytest.raises(ValueError, match="excluded_reason"):
        reconcile_result_coverage(cells, [])


@pytest.mark.parametrize("seeds", [[], [0, 0], [True], [-1], ["0"]])
def test_param_invalid_expected_seed_sets_are_rejected(seeds):
    with pytest.raises(ValueError, match="seed"):
        expand_expected_cells([_row()], seeds=seeds)


def test_edge_duplicate_or_incomplete_identities_are_rejected():
    cells = expand_expected_cells([_row()], seeds=[0])
    with pytest.raises(ValueError, match="duplicate expected"):
        reconcile_result_coverage(cells * 2, [])
    with pytest.raises(ValueError, match="duplicate observation"):
        reconcile_result_coverage(cells, [_observation(cells[0])] * 2)
    del cells[0]["identity"]["budget_identity"]
    with pytest.raises(ValueError, match="budget_identity"):
        reconcile_result_coverage(cells, [])


def test_determ_reconciliation_is_repeatable_and_does_not_mutate_inputs():
    cells = expand_expected_cells([_row()], seeds=[0])
    observations = [_observation(cells[0])]
    before = copy.deepcopy((cells, observations))

    first = reconcile_result_coverage(cells, observations)
    second = reconcile_result_coverage(cells, observations)

    assert first == second
    assert (cells, observations) == before
