"""Actual meta training, independent PA/OA checkpoint choices and replay."""

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from _support_checkpoint_helpers import inputs, run  # noqa: E402

from pu_toolbox.experiment.bundle import DatasetPart  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_basic_real_meta_checkpoints_both_protocols_candidates_and_reclaim(method, tmp_path):
    result = run(method, tmp_path, candidates=[{"learning_rate": 0.01}, {"learning_rate": 0.02}])
    manifest = result.manifest
    assert set(result.selections) == {"PA+clean-support", "OA"}
    assert not result.failures
    assert len(manifest["candidate_runs"]) == 2
    disclosure = manifest["extra_supervision"]
    assert disclosure["pa_clean_support_eligible"] and not disclosure["pa_eligible"]
    assert not disclosure["formal_admission"] and not manifest["formal_eligible"]
    assert disclosure["comparison_group"] == "extra_clean_support"
    assert disclosure["protocol_pending_review_items"][0] == "v2_p4_clean_support_policy"
    expected_epochs = [1, 2, 3] if method == "self_pu" else [1, 2]
    components = {"teacher_1", "teacher_2"} if method == "self_pu" else {"model"}
    all_checkpoints = []
    for candidate in manifest["candidate_runs"]:
        provenance = candidate["training_provenance"]
        assert provenance["captured_meta_epochs"] == expected_epochs
        assert set(provenance["captured_components"]) == components
        assert provenance["total_observed_training_label_budget"] == 16
        if method == "self_pu":
            assert provenance["active_meta_reweight_records"] > 0
            assert provenance["distillation_scope"] == "paper_population"
        else:
            assert provenance["virtual_steps"] > 0
        checkpoints = candidate["epoch_checkpoints"]
        assert len(checkpoints) == len(expected_epochs) * len(components)
        assert {p["component"] for p in checkpoints} == components
        for checkpoint in checkpoints:
            assert set(checkpoint["validation_metrics"]) == {"PA+clean-support", "OA"}
            assert not checkpoint["training_resume_supported"]
            assert checkpoint["persistent"]
            assert Path(checkpoint["path"]).exists() is not checkpoint["reclaimed"]
        all_checkpoints.extend(checkpoints)
    kept = {item["checkpoint"]["path"] for item in manifest["selection"].values()}
    assert {p["path"] for p in all_checkpoints if not p["reclaimed"]} == kept
    assert len(list((tmp_path / "checkpoints").rglob("*.pt"))) == len(kept)
    disk = manifest["resources"]["checkpoint_disk_preflight"]
    assert disk["required_bytes"] == 65536 * 3 * len(components) * 2 * 2
    assert Path(disk["directory"]) == tmp_path / "checkpoints"
    assert json.loads((tmp_path / "manifest.json").read_text())["extra_supervision"] == disclosure


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_determ_selected_snapshots_pickle_replay_affine_threshold_and_metrics(method, tmp_path):
    parts, _ = inputs()
    result = run(method, tmp_path)
    for protocol, fitted in result.selected_models.items():
        restored = pickle.loads(pickle.dumps(fitted))
        np.testing.assert_array_equal(
            fitted.decision_function(parts[-1].X), restored.decision_function(parts[-1].X)
        )
        predictions = restored.predict(parts[-1].X)
        assert (
            float(np.mean(predictions == parts[-1].labels))
            == result.test_metrics[protocol]["accuracy"]
        )
        artifact = result.selections[protocol]
        assert artifact.checkpoint_index is not None and artifact.epoch is not None


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
@pytest.mark.parametrize("role", ["test", "clean_val", "pu_val"])
def test_determ_selection_roles_do_not_leak_into_training_or_other_protocol(method, role, tmp_path):
    parts, support = inputs()
    original = run(method, tmp_path / "original", parts=parts, support=support)
    changed = list(parts)
    index = {"pu_val": 1, "clean_val": 2, "test": 3}[role]
    part = parts[index]
    changed[index] = DatasetPart(
        part.X + 5, 1 - part.labels, "clean", part.indices, part.for_selection
    )
    second = run(method, tmp_path / "changed", parts=tuple(changed), support=support)
    unaffected = (
        ["PA+clean-support", "OA"]
        if role == "test"
        else ["PA+clean-support" if role == "clean_val" else "OA"]
    )
    for protocol in unaffected:
        assert asdict(original.selections[protocol]) == asdict(second.selections[protocol])
        np.testing.assert_array_equal(
            original.selected_models[protocol].decision_function(parts[0].X),
            second.selected_models[protocol].decision_function(parts[0].X),
        )
    first_provenance = original.manifest["candidate_runs"][0]["training_provenance"]
    assert first_provenance == second.manifest["candidate_runs"][0]["training_provenance"]


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_edge_meta_only_epoch_set_excludes_warmup(method, tmp_path):
    candidates = [{"warmup_epochs": 1, "self_paced_start": 1}] if method == "self_pu" else [{}]
    result = run(method, tmp_path, candidates=candidates)
    provenance = result.manifest["candidate_runs"][0]["training_provenance"]
    assert provenance["captured_meta_epochs"] == ([2, 3] if method == "self_pu" else [1, 2])


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
def test_determ_repeated_fixed_seed_keeps_exact_selection_and_scores(method, tmp_path):
    one, two = run(method, tmp_path / "one"), run(method, tmp_path / "two")
    for name in one.selections:
        assert asdict(one.selections[name]) == asdict(two.selections[name])
        assert one.test_metrics[name] == two.test_metrics[name]


@pytest.mark.parametrize("method", ["self_pu", "lagam"])
@pytest.mark.parametrize(
    "config",
    [
        {"checkpoint_bytes_per_component": None},
        {"checkpoint_bytes_per_component": True},
        {"capture_epoch_checkpoints": False},
        {"candidates": [{"max_epochs": 0}]},
        {"candidates": []},
    ],
)
def test_param_bad_checkpoint_or_candidate_budget_refused_before_training(method, config, tmp_path):
    with pytest.raises(ValueError):
        run(method, tmp_path, config=config)
    assert not (tmp_path / "checkpoints").exists()
