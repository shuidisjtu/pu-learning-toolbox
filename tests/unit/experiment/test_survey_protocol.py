# ruff: noqa: N803, F811
"""Matrix, runner enforcement and honest eligibility for P2.0a."""

import copy
import json

import pytest
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_protocol import (
    PROTOCOL_PATH,
    digest,
    load_protocol,
    resolve_unit,
    validate_comparable_manifests,
    validate_parameters,
)

pytestmark = pytest.mark.unit


def _bound(survey_script, tmp_path):
    make_splits(tmp_path)
    parts = survey_script.load_split_parts(tmp_path)
    protocol = load_protocol()
    row = resolve_unit(protocol, "spambase", "upu")
    model = assemble_model(protocol, row, 3, seed=0, params={}, class_prior=0.3, device="cpu")
    config = {
        "architecture": "mlp",
        "c": 0.5,
        "c_requested_token": "0.5",
        "split_ref": {"dataset": "spambase", "seed": 0},
        "survey_protocol": {
            "path": str(PROTOCOL_PATH),
            "dataset": "spambase",
            "method": "upu",
            "mechanism": "scar",
            "seeds": [0],
            "c_tokens": ["0.5"],
        },
    }
    return model, parts, config


def test_basic_matrix_covers_seven_methods_and_oracle():
    protocol = load_protocol()
    assert len(protocol["execution_units"]) == 25
    assert "pusb_kernel" in protocol["method_profiles"]
    assert "pusb" not in protocol["method_profiles"]
    assert protocol["budgets"]["kernel_cv"]["internal_hyperparameter_pairs"] == 72


def test_determ_protocol_digest_is_key_order_independent():
    protocol = load_protocol()
    assert digest(protocol) == digest(dict(reversed(list(protocol.items()))))
    row = resolve_unit(protocol, "imdb", "nnpu")
    row["budget"] = "changed"
    assert resolve_unit(protocol, "imdb", "nnpu")["budget"] == "minibatch"


@pytest.mark.parametrize(
    "params", [{"max_epochs": 1}, {"model": None}, {"encoder__x": 1}, {"random_state": 4}]
)
def test_param_locked_candidates_rejected(params):
    with pytest.raises(ValueError, match="protocol-locked"):
        validate_parameters(params, load_protocol()["method_profiles"]["nnpu"])


def test_edge_empty_and_duplicate_matrices_rejected(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "matrix.json"
    protocol["execution_units"] = []
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="empty"):
        load_protocol(path)
    protocol = load_protocol()
    protocol["execution_units"].append(protocol["execution_units"][0])
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="duplicate"):
        load_protocol(path)


def test_param_cnn_oracle_never_falls_back_to_flattened_mlp():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_unit(load_protocol(), "cifar10", "pn_oracle")
    with pytest.raises(ValueError, match="Phase 2"):
        resolve_unit(load_protocol(), "cifar10", "pn_oracle", "native_cnn")


def test_basic_runner_consumes_matrix_and_records_blockers(survey_script, tmp_path):
    model, parts, config = _bound(survey_script, tmp_path)
    result = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts)
    manifest = result.manifest
    assert manifest["protocol_version"] == "survey-v1.1"
    assert manifest["budget"]["unit"] == "one_closed_form_fit"
    assert manifest["formal_eligible"] is False
    assert "seed_subset" in manifest["protocol_deviation"]
    assert manifest["selection_checkpoint_scope"] == "single_point_no_epoch"
    assert "per_epoch_independent_PA_OA_checkpoint_selection" not in manifest["formal_blockers"]
    assert len(manifest["representation"]["feature_sha256"]) == 4


def test_param_runner_preflight_rejects_budget_override_with_failure_manifest(
    survey_script, tmp_path
):
    model, parts, config = _bound(survey_script, tmp_path)
    config["candidates"] = [{"max_iter": 1}]
    path = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="protocol-locked"):
        ExperimentRunner(class_prior=0.3, config=config, manifest_path=str(path)).fit(model, *parts)
    payload = json.loads(path.read_text())
    assert payload["formal_eligible"] is False
    assert payload["failures"][0]["stage"] == "protocol_preflight"


def test_param_runner_rejects_estimator_identity_and_config_override(survey_script, tmp_path):
    model, parts, config = _bound(survey_script, tmp_path)
    config["survey_protocol"]["method"] = "lbe"
    with pytest.raises(ValueError, match="estimator class"):
        ExperimentRunner(config=config).fit(model, *parts)
    config["survey_protocol"]["method"] = "upu"
    config["backbone"] = "fake"
    with pytest.raises(ValueError, match="config override"):
        ExperimentRunner(config=config).fit(model, *parts)


def test_edge_custom_candidates_excluded_from_formal_results(survey_script, tmp_path):
    model, parts, config = _bound(survey_script, tmp_path)
    config["candidates"] = [{"reg_lambda": 0.01}]
    payload = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    assert "candidate_pool" in payload["protocol_deviation"]
    with pytest.raises(ValueError, match="formal aggregation"):
        validate_comparable_manifests([payload])


def test_param_paths_and_budgets_cannot_be_mixed(survey_script, tmp_path):
    model, parts, config = _bound(survey_script, tmp_path)
    left = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    right = copy.deepcopy(left)
    right["training_path"] = "native_cnn"
    with pytest.raises(ValueError, match="training_path"):
        validate_comparable_manifests([left, right], require_formal=False)
    right = copy.deepcopy(left)
    right["budget"]["unit"] = "epochs"
    with pytest.raises(ValueError, match="budget"):
        validate_comparable_manifests([left, right], require_formal=False)


def test_param_source_labels_and_features_must_match(survey_script, tmp_path):
    model, parts, config = _bound(survey_script, tmp_path)
    left = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    right = copy.deepcopy(left)
    right["generation"]["train"]["label_view_sha256"] = "changed"
    with pytest.raises(ValueError, match="label-view"):
        validate_comparable_manifests([left, right], require_formal=False)


def test_determ_bound_runs_reproduce_label_and_representation_hashes(survey_script, tmp_path):
    model, parts, config = _bound(survey_script, tmp_path)
    first = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    second = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    assert first["generation"] == second["generation"]
    assert first["representation"] == second["representation"]
    validate_comparable_manifests([first, second], require_formal=False)


def test_edge_diy_runner_is_explicitly_technical(survey_script, tmp_path):
    model, parts, config = _bound(survey_script, tmp_path)
    config.pop("survey_protocol")
    payload = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    assert payload["execution_mode"] == "technical_smoke"
    assert payload["formal_eligible"] is False


def test_param_changed_matrix_always_marks_deviation(survey_script, tmp_path):
    model, parts, config = _bound(survey_script, tmp_path)
    protocol = load_protocol()
    protocol["formal_blockers"] = []
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(protocol))
    config["survey_protocol"]["path"] = str(path)
    payload = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    assert "noncanonical_protocol_matrix" in payload["protocol_deviation"]
    assert payload["formal_eligible"] is False
