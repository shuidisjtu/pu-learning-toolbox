"""P3Mix opt-in image CLI covers explicit recipes, PA/OA replay and reclamation."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_candidate_image_experiment as script  # noqa: E402

pytestmark = pytest.mark.unit


def product(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()
    for i, role in enumerate(script.ROLES):
        np.savez(
            directory / f"{role}.npz",
            X=np.random.RandomState(i).randint(0, 256, (12, 3, 8, 8)).astype(np.uint8),
            y=np.arange(12) % 2,
            indices=np.arange(12) + i * 20,
        )
    return directory


def parameters(**overrides):
    return (
        dict(
            variant="base",
            hidden_dim=4,
            max_epochs=2,
            batch_size=4,
            batching_policy="role_balanced",
            learning_rate=0.001,
            beta=1.0,
            gamma=0.9,
            pool_size=2,
            mixup_alpha=1.0,
            early_learning_weight=1.0,
            teacher_decay=0.99,
            device="cpu",
        )
        | overrides
    )


def plan(directory, **overrides):
    return script.build_plan(
        directory,
        **(
            dict(
                method="p3mix",
                dataset="cifar10",
                prior=0.4,
                prior_source="synthetic population",
                seed=3,
                parameters=parameters(),
                checkpoint_bytes=64 * 2**20,
                c_token="0.5",
            )
            | overrides
        ),
    )


@pytest.mark.parametrize("variant", ["base", "e", "c"])
@pytest.mark.parametrize("policy", ["role_balanced", "merged_shuffle"])
def test_basic_determ_explicit_variant_dual_selection_and_replay(tmp_path, variant, policy):
    directory = product(tmp_path)
    planned = plan(directory, parameters=parameters(variant=variant, batching_policy=policy))
    assert planned == plan(
        directory, parameters=parameters(variant=variant, batching_policy=policy)
    )
    assert planned["training_view"] == "os" and planned["checkpoint_epoch_bound"] == 2
    assert not planned["formal_admission"] and not planned["formal_budget_approved"]
    summary = script.execute(planned, tmp_path / "results")
    assert summary["accepted_engineering_run"] and summary["selected_models_replayed"]
    assert summary["warning_count"] == summary["retry_count"] == 0
    assert 1 <= summary["retained_checkpoint_count"] <= 2
    assert summary["source_default_snapshot_projection"]["epochs"] == 100
    assert summary["source_default_snapshot_projection"]["origin"] == (
        "toolbox_constructor_not_author_or_frozen_recipe"
    )
    facts = summary["training_provenance"]
    assert facts["selection_prior_not_forwarded_to_P3Mix"]
    assert len(facts["epoch_cost_history"]) == 2
    assert facts["bound_fit_receipt"]["algorithm_training"]["variant"] == variant
    assert not facts["teacher_selection_eligible"]
    manifest = json.loads((tmp_path / "results/manifest.json").read_text())
    assert manifest["run_view"] == "os-compatible" and not manifest["formal_eligible"]
    assert set(summary["selection"]) == {"PA", "OA"}
    assert all((tmp_path / "results" / f"selected_{name}.pkl").is_file() for name in ("PA", "OA"))


@pytest.mark.parametrize("key", ["variant", "batching_policy", "beta", "gamma", "teacher_decay"])
def test_param_missing_native_recipe_field_refused_before_execution(tmp_path, key):
    requested = parameters()
    del requested[key]
    with pytest.raises(ValueError, match="explicit budget"):
        plan(product(tmp_path), parameters=requested)


def test_edge_changed_variant_or_input_source_does_not_create_output(tmp_path, monkeypatch):
    planned = plan(product(tmp_path))
    changed = copy.deepcopy(planned)
    changed["parameters_requested"]["variant"] = "e"
    with pytest.raises(ValueError, match="plan changed"):
        script.execute(changed, tmp_path / "bad")
    original = script.file_hash
    monkeypatch.setattr(
        script,
        "file_hash",
        lambda path: "changed"
        if str(path).endswith("estimators/deep/p3mix.py")
        else original(path),
    )
    with pytest.raises(ValueError, match="identity changed"):
        script.execute(planned, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
