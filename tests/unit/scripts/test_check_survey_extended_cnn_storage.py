"""Semantic/source refusal tests; fixtures are not measurement evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load_script("check_survey_extended_cnn_storage")
probe = load_script("profile_survey_cnn_storage")
driver = check.driver
pytestmark = pytest.mark.unit


def fixture_report(recipe):
    options = driver.RECIPES[recipe]
    row = probe.profile_cnn_storage(**options, seed=0, base_channels=2, image_size=8)
    row.update(probe_recipe=recipe, single_profile_isolated_process=True)
    return {
        "schema_version": 2,
        "status": "isolated_synthetic_extended_cnn_resource_evidence_not_formal_admission",
        "profile_order": "recipe_then_seed_sequential_fresh_process",
        "measurement_spec": {
            "recipes": [recipe],
            "seeds": [0],
            "base_channels": 2,
            "image_size": 8,
            "device": "cpu",
        },
        "driver_source_files_sha256": {
            reference: driver.source_digest(ROOT / reference)
            for reference in (
                "scripts/profile_survey_cnn_storage.py",
                "scripts/profile_survey_extended_cnn_storage.py",
            )
        },
        "formal_budget_approved": False,
        "candidate_protocol_ref": None,
        "owner_signatures": [],
        "profiles": [row],
    }


@pytest.fixture(scope="module")
def genpu_report():
    pytest.importorskip("torch")
    return fixture_report("genpu_identity")


@pytest.mark.parametrize("recipe", list(driver.RECIPES))
def test_basic_seven_recipe_scopes_stage_costs_and_source_identity(recipe):
    pytest.importorskip("torch")
    result = check.validate_report(fixture_report(recipe))
    assert result["ok"] and result["profiles_checked"] == 1
    assert len(result["source_files_checked"]) == 9
    assert (
        result["formal_admission"] is False and result["training_peak_upper_bound_proven"] is False
    )


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "missing",
        "isolation",
        "cost",
        "view",
        "formal",
        "sign",
        "geometry",
        "rollback",
        "bytes",
        "nan",
        "source",
        "width",
    ],
)
def test_param_edge_corrupted_receipts_fail_closed_without_repair(genpu_report, fault):
    report = copy.deepcopy(genpu_report)
    row = report["profiles"][0]
    if fault == "duplicate":
        report["profiles"].append(copy.deepcopy(row))
    elif fault == "missing":
        report["profiles"] = []
    elif fault == "isolation":
        row["single_profile_isolated_process"] = False
    elif fault == "cost":
        row["optimizer_steps"] = 0
    elif fault == "view":
        row["calibration_applied"] = False
    elif fault == "formal":
        row["formal_budget_approved"] = True
    elif fault == "sign":
        report["owner_signatures"] = ["forged"]
    elif fault == "geometry":
        row["input_shape"][-1] = 32
    elif fault == "rollback":
        row["checkpoint_training_contexts"][-1]["optimizer_steps"] = 1
    elif fault == "bytes":
        row["epoch_weights_bytes"]["total"] = 0
    elif fault == "nan":
        row["fit_elapsed_seconds"] = float("nan")
    elif fault == "source":
        row["source_files_sha256"]["pu_toolbox/estimators/deep/gen_pu.py"] = "0" * 64
    else:
        row["encoder"]["matches_toolbox_default_width"] = True
    before = copy.deepcopy(report)
    with pytest.raises(ValueError):
        check.validate_report(report)
    # NaN does not compare equal to itself; compare other structured failures.
    if fault != "nan":
        assert before == report


def test_basic_determ_checker_is_read_only_and_semantic_only_mode_discloses_missing_source_check(
    genpu_report,
):
    before = copy.deepcopy(genpu_report)
    first = check.validate_report(genpu_report)
    assert check.validate_report(genpu_report) == first
    assert genpu_report == before
    assert check.validate_report(genpu_report, verify_sources=False)["source_files_checked"] == []


@pytest.mark.parametrize("fault", ["support_overlap", "support_budget", "snapshot", "lzo_cost"])
def test_param_edge_support_and_lzo_roles_cannot_be_hidden_by_recipe_labels(fault):
    pytest.importorskip("torch")
    report = fixture_report(
        "holistic_lzo_continue" if fault == "lzo_cost" else "pulns_clean_reward"
    )
    row = report["profiles"][0]
    if fault == "support_overlap":
        row["clean_support"]["support_ids"][0] = 0
    elif fault == "support_budget":
        row["clean_support"]["selection_test_labels_provided"] = True
    elif fault == "snapshot":
        row["epoch_weights_bytes"] = {"count": 2}
    else:
        row["holistic_terminal_selection"]["lzo_evaluated_rows_"] = 0
    with pytest.raises(ValueError):
        check.validate_report(report)


def test_basic_real_isolated_child_receipt_passes_without_promoting_formal_budget():
    pytest.importorskip("torch")
    report = driver.profile_recipes(
        recipes=["holistic_lzo_reinitialize"], seeds=[1], base_channels=2, image_size=8
    )
    result = check.validate_report(report)
    assert result["ok"] and result["formal_admission"] is False
