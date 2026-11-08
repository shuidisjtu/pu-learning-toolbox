"""New recipe revision: complete scope, historical isolation and real CLI fits."""

# ruff: noqa: F811

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from _survey_script_helpers import make_sar_splits, survey_script  # noqa: F401

from pu_toolbox.experiment.survey_backlog import DEEP_METHODS, VIEWS, SurveyBacklogTrainer
from pu_toolbox.experiment.survey_protocol import load_protocol, resolve_protocol_path, resolve_unit
from pu_toolbox.experiment.survey_recipe_registry import build_locked_registry, validate_registry

pytestmark = pytest.mark.unit


def protocol():
    return load_protocol(resolve_protocol_path("survey-v3.2"))


def test_basic_current_configuration_covers_all_project_recipe_cells():
    new = protocol()
    assert len(new["execution_units"]) == 188 and all(r["runnable"] for r in new["execution_units"])
    assert new["seeds"] == [0, 1, 2, 3, 4]
    for method in (*DEEP_METHODS, "kldce"):
        assert len({r["dataset"] for r in new["execution_units"] if r["method"] == method}) == 8
    for method in ("puet", "rp"):
        assert len({r["dataset"] for r in new["execution_units"] if r["method"] == method}) == 8
    for method in ("lagam", "pulns", "self_pu_full_meta"):
        assert len({r["dataset"] for r in new["execution_units"] if r["method"] == method}) == 8
        assert new["method_profiles"][method]["labeling_mechanisms"] == ["scar"]
    assert new["method_profiles"]["kldce"]["labeling_mechanisms"] == ["scar"]
    assert new["formal_blockers"] and new["review_status"] == "pending_collaborator_review"
    registry = build_locked_registry(new, recipe_registry_version="survey-recipes-v1.3")
    assert all(p["recipe_lifecycle"] == "locked" for p in registry["profiles"].values())
    assert not validate_registry(registry, new)
    stored = json.loads(
        resolve_protocol_path("survey-v3.2")
        .with_name("survey_recipe_registry_v1_3.json")
        .read_text()
    )
    assert registry == stored


def small_protocol(tmp_path, method):
    p = protocol()
    profile = p["method_profiles"][method]
    params = profile["params"]
    if method in DEEP_METHODS:
        params.update(max_epochs=2, batch_size=8)
        if method == "p3mix":
            params["pool_size"] = 4
    elif method == "puet":
        params.update(n_estimators=3, max_depth=3)
    elif method == "kldce":
        params.update(max_acs_iter=2, max_inner_iter=100)
    budget = p["budgets"][profile["budget"]]
    budget["parameter_values"] = copy.deepcopy(params)
    if method in DEEP_METHODS:
        budget.update(
            epochs=2, batch_size=8, checkpoint_epoch_count=2 + params.get("warm_start_epochs", 0)
        )
    elif method == "kldce":
        budget.update(max_acs_iter=2, max_inner_iter=100)
    path = tmp_path / "small_protocol.json"
    path.write_text(json.dumps(p))
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(build_locked_registry(p)))
    return path, registry


@pytest.mark.parametrize("method", [*DEEP_METHODS, "puet", "rp", "kldce"])
@pytest.mark.parametrize("mechanism", ["scar", "sar_lbe_a", "sar_lbe_b"])
def test_basic_cli_bound_recipe_trains_full_small_trajectory_and_correct_selection(
    tmp_path, survey_script, method, mechanism, capsys
):
    if method == "kldce" and mechanism != "scar":
        pytest.skip("KLDCE is explicitly SCAR-only")
    data = tmp_path / "data"
    data.mkdir()
    make_sar_splits(data)
    (data / "split_manifest.json").write_text(
        json.dumps({"dataset": "spambase", "seed": 0, "class_prior": {"population": 0.4}})
    )
    source, registry = small_protocol(tmp_path, method)
    output = tmp_path / "out"
    code = survey_script.main(
        [
            str(data),
            "--method",
            method,
            "--protocol",
            str(source),
            "--dataset",
            "spambase",
            "--recipe-registry",
            str(registry),
            "--device",
            "cpu",
            "--c",
            "0.5",
            "--class-prior",
            "0.4",
            "--labeling-mechanism",
            mechanism,
            "--out-dir",
            str(output),
            "--reclaim-unselected-checkpoints",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err + captured.out
    files = list(output.rglob("manifest.json"))
    assert len(files) == 1
    manifest = json.loads(files[0].read_text())
    assert manifest["protocol_version"] == "survey-v3.2"
    assert manifest["recipe_registry_version"] == "survey-recipes-v1.2"
    assert not manifest["failures"] and not manifest["formal_eligible"]
    assert set(manifest["test_results"]) == ({"PA", "OA"} if mechanism == "scar" else {"OA"})
    candidate = manifest["candidate_runs"][0]
    if method in DEEP_METHODS:
        assert len(candidate["epoch_checkpoints"]) == (3 if method == "cvir" else 2)
        assert candidate["training_provenance"]["external_validation_forwarded_to_fit"] is False
        assert manifest["run_view"] == f"{VIEWS[method]}-compatible"
        if method == "cvir":
            assert (
                candidate["training_provenance"]["population_prior_binding"][
                    "risk_pool_positive_prior"
                ]
                == 0.4
            )
    else:
        assert not candidate.get("epoch_checkpoints")
    assert "seed_subset" in manifest["protocol_deviation"]
    if method != "rp":
        assert "noncanonical_protocol_matrix" in manifest["protocol_deviation"]


def test_determ_gradpu_encoder_seed_does_not_depend_on_or_change_caller_rng():
    from types import SimpleNamespace

    import torch

    from pu_toolbox.experiment.feature_adapter import encoder_state_sha256
    from pu_toolbox.experiment.survey_execution import assemble_model

    p = protocol()
    row = resolve_unit(p, "cifar10", "gradpu")
    preprocessing = SimpleNamespace(
        in_channels=3, normalization_mean=[0.5] * 3, normalization_std=[0.2] * 3
    )

    def make(seed):
        return assemble_model(
            p,
            row,
            3,
            seed=seed,
            params={},
            class_prior=0.4,
            device="cpu",
            preprocessing=preprocessing,
        )

    torch.manual_seed(933)
    before = torch.random.get_rng_state().clone()
    one = make(7)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(23)
    two, other = make(7), make(8)
    assert encoder_state_sha256(one.encoder) == encoder_state_sha256(two.encoder)
    assert encoder_state_sha256(one.encoder) != encoder_state_sha256(other.encoder)


def test_edge_unvalidated_backlog_trainer_cannot_fit():
    with pytest.raises(ValueError, match="validated"):
        SurveyBacklogTrainer("cvir").fit(
            None, np.zeros((4, 2)), np.array([0, 1, 0, 1]), os_or_ts="ts"
        )


@pytest.mark.parametrize("method", DEEP_METHODS)
def test_basic_real_cnn_cli_retains_full_small_trajectory(tmp_path, survey_script, method, capsys):
    data = tmp_path / "images"
    data.mkdir()
    rng = np.random.RandomState(18)
    for i, role in enumerate(("train", "pu_val", "clean_val", "test")):
        np.savez(
            data / f"{role}.npz",
            X=rng.randint(0, 256, (12, 3, 32, 32), dtype=np.uint8),
            y=np.arange(12) % 2,
            indices=np.arange(12) + 12 * i,
        )
    (data / "split_manifest.json").write_text(
        json.dumps({"dataset": "cifar10", "seed": 0, "class_prior": {"population": 0.4}})
    )
    source, registry = small_protocol(tmp_path, method)
    output = tmp_path / "out"
    code = survey_script.main(
        [
            str(data),
            "--method",
            method,
            "--protocol",
            str(source),
            "--dataset",
            "cifar10",
            "--recipe-registry",
            str(registry),
            "--device",
            "cpu",
            "--c",
            "0.5",
            "--class-prior",
            "0.4",
            "--out-dir",
            str(output),
            "--reclaim-unselected-checkpoints",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err + captured.out
    manifest = json.loads(next(output.rglob("manifest.json")).read_text())
    candidate = manifest["candidate_runs"][0]
    assert len(candidate["epoch_checkpoints"]) == (3 if method == "cvir" else 2)
    assert manifest["training_path"] == "native_cnn"
    assert set(manifest["test_results"]) == {"PA", "OA"}
    assert not manifest["failures"]
    assert all(
        Path(c["path"]).stat().st_size <= 64 * 1024**2
        for c in candidate["epoch_checkpoints"]
        if Path(c["path"]).exists()
    )


def test_basic_cvir_disk_planning_counts_warm_start():
    from pu_toolbox.experiment.survey_staged_budget import checkpoint_epochs

    p = protocol()
    assert checkpoint_epochs(p["budgets"]["backlog_cvir"]) == 21


def test_edge_current_cvir_adapter_does_not_bypass_missing_binding():
    from pu_toolbox.experiment.survey_profile_identity import require_four_role_profile

    unbound = protocol()
    unbound["method_profiles"]["cvir"].pop("execution_adapter")
    with pytest.raises(ValueError, match="per-view"):
        require_four_role_profile(unbound, "cvir")
    require_four_role_profile(protocol(), "cvir")


@pytest.mark.parametrize("parameters", [{"max_epochs": 1}, {"encoder": "replacement"}])
def test_param_recipe_override_rejected_before_loading_data(
    tmp_path, survey_script, monkeypatch, parameters, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid recipe reached input loading")

    monkeypatch.setattr(survey_script, "load_split_parts", forbidden)
    code = survey_script.main(
        [
            str(tmp_path / "absent"),
            "--method",
            "pan",
            "--dataset",
            "spambase",
            "--protocol",
            "survey-v3.2",
            "--model-params",
            json.dumps(parameters),
        ]
    )
    assert code == 1 and "protocol-locked" in capsys.readouterr().err
