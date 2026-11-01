"""Version-isolated image additions, actual ResNet features and classical fits."""

# ruff: noqa: F811

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from _survey_script_helpers import survey_script  # noqa: F401

from pu_toolbox.experiment.survey_comparison import (
    comparison_for_protocol,
    validate_comparison_coverage,
)
from pu_toolbox.experiment.survey_profile_map import (
    load_profile_map,
    profile_map_path,
    validate_profile_map,
)
from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_protocol_path, resolve_unit
from pu_toolbox.experiment.survey_recipe_registry import validate_registry
from scripts.prepare_survey_classical_image_revision import build_artifacts
from scripts.run_survey_backlog_campaign import expand_jobs, validate_classical_image_receipt

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["puet", "rp"])
def test_param_native_cnn_and_historical_image_units_remain_rejected(method):
    old = load_protocol(resolve_protocol_path("survey-v3.1"))
    new = load_protocol(resolve_protocol_path("survey-v3.2"))
    with pytest.raises(ValueError):
        resolve_unit(old, "cifar10", method, "cnn_feature_adapter")
    with pytest.raises(ValueError):
        resolve_unit(new, "cifar10", method, "native_cnn")


def test_basic_new_revision_preserves_all_old_units_and_adds_exactly_280_image_fits():
    old = load_protocol(resolve_protocol_path("survey-v3.1"))
    new = load_protocol(resolve_protocol_path("survey-v3.2"))
    assert new["execution_units"][:171] == old["execution_units"]
    assert len(new["execution_units"]) == 179
    for name in ("backbone_specs", "budgets", "method_profiles", "seeds", "c_tokens"):
        assert new[name] == old[name]
    jobs = expand_jobs(
        new,
        old,
        {d: f"/data/{d}" for d in ("cifar10", "mnist", "fashion_mnist", "adni")},
        Path("/unused"),
    )
    assert len(jobs) == len({j["label"] for j in jobs}) == 280
    assert len({j["family"] for j in jobs}) == 24
    assert sum(j["canary"] for j in jobs) == 24
    assert {j["method"] for j in jobs} == {"puet", "rp"}
    assert all(j["lane"] == "cpu" and j["path"] == "cnn_feature_adapter" for j in jobs)
    assert {j["seed"] for j in jobs} == set(range(5))
    assert new["review_status"] == "pending_collaborator_review"


def test_determ_generated_registry_and_comparison_are_bound_without_new_signatures():
    artifacts = build_artifacts()
    root = resolve_protocol_path("survey-v3.2").parent
    for name, value in artifacts.items():
        assert json.loads((root / name).read_text()) == value
    p = artifacts["survey_protocol_v3_2.json"]
    assert not validate_registry(artifacts["survey_recipe_registry_v1_2.json"], p)
    mapping = load_profile_map(profile_map_path(p))
    comparison = comparison_for_protocol(p)
    validate_profile_map(mapping, protocol=p, comparison=comparison)
    assert validate_comparison_coverage(p, comparison, profile_map=mapping)["units"] == 1560
    additions = [m for m in comparison["mappings"] if m["mapping_id"].startswith("v3_2_")]
    assert len(additions) == 80
    assert all(
        m["review_state"] == "pending_review" and not m["verified_by"] and not m["anchor_ids"]
        for m in additions
    )


@pytest.mark.parametrize("method", ["puet", "rp"])
@pytest.mark.parametrize(
    "dataset,mechanism",
    [(d, "scar") for d in ("cifar10", "mnist", "fashion_mnist", "adni")]
    + [("cifar10", m) for m in ("sar_lbe_a", "sar_lbe_b")],
)
def test_basic_real_resnet_classical_cli_and_exact_budget(
    tmp_path, survey_script, method, dataset, mechanism, capsys
):
    data = tmp_path / "images"
    data.mkdir()
    rng = np.random.RandomState(18)
    shape = {
        "cifar10": (3, 32, 32),
        "mnist": (1, 28, 28),
        "fashion_mnist": (1, 28, 28),
        "adni": (1, 128, 128),
    }[dataset]
    for i, role in enumerate(("train", "pu_val", "clean_val", "test")):
        labels = np.arange(64) % 2
        images = rng.randint(0, 80, (64, *shape), dtype=np.uint8)
        images[labels == 1] += 150
        np.savez(data / f"{role}.npz", X=images, y=labels, indices=np.arange(64) + 64 * i)
    (data / "split_manifest.json").write_text(
        json.dumps({"dataset": dataset, "seed": 0, "class_prior": {"population": 0.5}})
    )
    output = tmp_path / "out"
    registry = resolve_protocol_path("survey-v3.2").with_name("survey_recipe_registry_v1_2.json")
    args = [
        str(data),
        "--method",
        method,
        "--protocol",
        "survey-v3.2",
        "--dataset",
        dataset,
        "--training-path",
        "cnn_feature_adapter",
        "--recipe-registry",
        str(registry),
        "--device",
        "cpu",
        "--c",
        "0.5",
        "--class-prior",
        "0.5",
        "--labeling-mechanism",
        mechanism,
        "--adapter-cache",
        str(tmp_path / "features"),
        "--out-dir",
        str(output),
    ]
    code = survey_script.main(args)
    captured = capsys.readouterr()
    assert code == 0, captured.err + captured.out
    m = json.loads(next(output.rglob("manifest.json")).read_text())
    assert m["protocol_version"] == "survey-v3.2"
    assert m["recipe_registry_version"] == "survey-recipes-v1.2"
    assert set(m["test_results"]) == ({"PA", "OA"} if mechanism == "scar" else {"OA"})
    assert not m["failures"] and not m["formal_eligible"]
    assert m["run_view"] == ("ts-compatible" if method == "puet" else "os-compatible")
    job = dict(method=method, path="cnn_feature_adapter", seed=0)
    validate_classical_image_receipt(m, job)
    bad = copy.deepcopy(m)
    receipt = bad["candidate_runs"][0]["training_provenance"]
    receipt["trees_completed" if method == "puet" else "cross_fits_completed"] = 0
    with pytest.raises(ValueError, match="budget"):
        validate_classical_image_receipt(bad, job)
    bad = copy.deepcopy(m)
    bad["representation"]["adapter"]["encoder_training"] = "finetuned"
    with pytest.raises(ValueError, match="representation"):
        validate_classical_image_receipt(bad, job)
