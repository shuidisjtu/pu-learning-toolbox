"""ADNI-slot candidates use grayscale substitute data, not clinical admission."""

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


def product(tmp_path, channels=1):
    root = tmp_path / "input"
    root.mkdir()
    for position, role in enumerate(script.ROLES):
        np.savez(
            root / f"{role}.npz",
            X=np.random.RandomState(position)
            .randint(0, 256, (12, channels, 8, 8))
            .astype(np.uint8),
            y=np.arange(12) % 2,
            indices=np.arange(12) + position * 20,
        )
    return root


def plan(root, method="gradpu"):
    parameters = dict(device="cpu")
    parameters.update(
        dict(max_epochs=2, batch_size=4, lr_schedule="constant")
        if method == "gradpu"
        else dict(warmup_epochs=1, pu_epochs=1, positive_batch_size=2, unlabeled_batch_size=4)
    )
    return script.build_plan(
        root,
        method=method,
        dataset="adni",
        prior=0.5,
        prior_source="synthetic population",
        seed=3,
        parameters=parameters,
        checkpoint_bytes=64 * 2**20,
        c_token="0.5",
    )


@pytest.mark.parametrize("method", ["gradpu", "pulda"])
def test_basic_determ_grayscale_plan_identity_and_frozen_files_unchanged(tmp_path, method):
    root = product(tmp_path)
    protocol = script.ROOT / "pu_toolbox/experiment/survey_protocol_v3.json"
    before = script.file_hash(protocol)
    result = plan(root, method)
    assert result == plan(root, method)
    assert script.file_hash(protocol) == before
    assert result["image_setup"]["image_preprocessing"]["in_channels"] == 1
    assert not result["data_scope"]["clinical_adni"]
    assert not result["data_scope"]["raw_source_verified"]
    assert not result["data_scope"]["subject_level_isolation_verified"]
    assert script.ADNI_RECEIPT in result["source_files_sha256"]
    assert not result["formal_admission"] and not result["formal_budget_approved"]


@pytest.mark.parametrize("method", ["gradpu", "pulda"])
def test_basic_actual_grayscale_fit_dual_selection_replay_and_scope(tmp_path, method):
    result = script.execute(plan(product(tmp_path), method), tmp_path / "output")
    assert result["accepted_engineering_run"] and result["selected_models_replayed"]
    assert result["dataset"] == "adni" and not result["data_scope"]["clinical_adni"]
    assert result["warning_count"] == result["retry_count"] == 0
    assert set(result["selection"]) == {"PA", "OA"}
    assert not result["formal_admission"] and not result["paper_numerical_reproduction"]
    payload = json.loads((tmp_path / "output/manifest.json").read_text())
    assert [row["epoch_label"] for row in payload["candidate_runs"][0]["epoch_checkpoints"]] == [
        0,
        1,
    ]
    assert payload["split_ref"]["data_scope"] == result["data_scope"]
    if method == "gradpu":
        assert result["comparison_group"] == "method-private/no-BN CNN"


def test_param_rgb_input_cannot_masquerade_as_grayscale_substitute(tmp_path):
    with pytest.raises(ValueError, match="channel"):
        plan(product(tmp_path, channels=3))


def test_edge_data_scope_cannot_be_changed_to_clinical_adni(tmp_path):
    original = plan(product(tmp_path))
    changed = copy.deepcopy(original)
    changed["data_scope"]["clinical_adni"] = True
    changed["plan_sha256"] = script.strict_canonical_hash(
        {k: v for k, v in changed.items() if k != "plan_sha256"}
    )
    with pytest.raises(ValueError, match="identity changed"):
        script.execute(changed, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
