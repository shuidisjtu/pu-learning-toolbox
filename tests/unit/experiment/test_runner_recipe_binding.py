"""Recipe validation must happen before label generation, fit, or retry."""

# ruff: noqa: F811

import copy
import json

import pytest
from _survey_recipe_registry_helpers import locked_registry
from _survey_script_helpers import make_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import SCARGenerator
from pu_toolbox.experiment.survey_execution import assemble_model
from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATH, load_protocol, resolve_unit
from pu_toolbox.experiment.survey_recipe_registry import canonical_digest, validate_manifest_binding

pytestmark = pytest.mark.unit


def request(survey_script, tmp_path):
    make_splits(tmp_path)
    parts = survey_script.load_split_parts(tmp_path)
    protocol = load_protocol()
    row = resolve_unit(protocol, "spambase", "upu", "native_2d")
    model = assemble_model(protocol, row, 3, seed=0, params={}, class_prior=0.3, device="cpu")
    config = {
        "architecture": "mlp",
        "c": 0.5,
        "c_requested_token": "0.5",
        "split_ref": {"dataset": "spambase", "seed": 0},
        "recipe_registry": locked_registry(),
        "survey_protocol": {
            "path": str(PROTOCOL_PATH),
            "dataset": "spambase",
            "method": "upu",
            "training_path": "native_2d",
            "mechanism": "scar",
            "seeds": [0],
            "c_tokens": ["0.5"],
        },
    }
    return model, parts, config


def test_basic_determ_runner_records_one_identity_on_run_candidate_and_both_selections(
    survey_script, tmp_path
):
    model, parts, config = request(survey_script, tmp_path)
    manifest = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    assert (
        validate_manifest_binding(manifest, config["recipe_registry"], protocol=load_protocol())
        == []
    )
    for entry in [*manifest["candidate_runs"], *manifest["selection"].values()]:
        for field in ("recipe_registry_sha256", "recipe_candidate_id", "resolved_params_sha256"):
            assert entry[field] == manifest[field]
    again = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    assert again["resolved_params_sha256"] == manifest["resolved_params_sha256"]


@pytest.mark.parametrize("defect", ["draft", "digest", "override"])
def test_param_rejected_recipe_never_generates_labels_or_fits(
    survey_script, tmp_path, defect, monkeypatch
):
    model, parts, config = request(survey_script, tmp_path)
    if defect == "draft":
        config["recipe_registry"]["recipe_lifecycle"] = "draft"
    elif defect == "digest":
        config["expected_recipe_registry_sha256"] = "0" * 64
    else:
        config["survey_protocol"]["constructor_overrides"] = {"extra": 1}

    def generate(*args, **kwargs):
        pytest.fail("label generation happened before recipe rejection")

    monkeypatch.setattr(SCARGenerator, "generate", generate)

    path = tmp_path / "rejected.json"
    with pytest.raises(
        ValueError, match={"draft": "locked", "digest": "changed", "override": "override"}[defect]
    ):
        ExperimentRunner(class_prior=0.3, config=config, manifest_path=str(path)).fit(model, *parts)
    manifest = json.loads(path.read_text())
    assert manifest["execution_mode"] == "rejected_versioned_pilot"
    assert manifest["generation"] == {}
    assert "recipe_binding_status" not in manifest


def test_edge_unbound_runner_is_not_retroactively_labelled(survey_script, tmp_path):
    model, parts, config = request(survey_script, tmp_path)
    del config["recipe_registry"]
    manifest = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    assert "recipe_binding_status" not in manifest
    assert "recipe_registry_sha256" not in manifest


def test_edge_partial_pin_without_registry_is_refused(survey_script, tmp_path):
    model, parts, config = request(survey_script, tmp_path)
    del config["recipe_registry"]
    config["expected_recipe_registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="requires a recipe registry"):
        ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts)


def test_param_programmatic_estimator_cannot_hide_an_undeclared_tuned_parameter(
    survey_script, tmp_path
):
    model, parts, config = request(survey_script, tmp_path)
    model.set_params(reg_lambda=0.005)
    with pytest.raises(ValueError, match="outside the resolved recipe/defaults: reg_lambda"):
        ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts)


def test_edge_candidate_pool_is_frozen_before_a_generator_can_mutate_config(
    survey_script, tmp_path, monkeypatch
):
    model, parts, config = request(survey_script, tmp_path)
    expected = canonical_digest(config["recipe_registry"])

    original = SCARGenerator.generate

    def generate(generator, *args, **kwargs):
        config["candidates"] = [{"reg_lambda": 999}]
        model.set_params(reg_lambda=999)
        config["recipe_registry"] = copy.deepcopy(config["recipe_registry"])
        config["recipe_registry"]["recipe_registry_version"] = "changed-after-preflight"
        return original(generator, *args, **kwargs)

    monkeypatch.setattr(SCARGenerator, "generate", generate)

    manifest = ExperimentRunner(class_prior=0.3, config=config).fit(model, *parts).manifest
    assert manifest["recipe_registry_sha256"] == expected
    assert manifest["candidate_runs"][0]["params"] == {}
    assert manifest["estimator_parameters"]["reg_lambda"] != 999
