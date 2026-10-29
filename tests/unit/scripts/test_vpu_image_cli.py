"""Four-role VPU candidates keep train-only normalization and independent PA/OA."""

import copy
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_candidate_image_experiment as script  # noqa: E402

from pu_toolbox.estimators.risk.vpu import VPUClassifier  # noqa: E402

pytestmark = pytest.mark.unit


def product(tmp_path, dataset):
    directory = tmp_path / "input"
    directory.mkdir()
    for index, role in enumerate(script.ROLES):
        channels = 3 if dataset == "cifar10" else 1
        np.savez(
            directory / f"{role}.npz",
            X=np.random.RandomState(index).randint(0, 256, (12, channels, 8, 8)).astype(np.uint8),
            y=np.arange(12) % 2,
            indices=np.arange(12) + index * 20,
        )
    return directory


def parameters():
    return dict(
        device="cpu",
        hidden_dim=4,
        depth=1,
        max_epochs=3,
        batch_size=4,
        learning_rate=3e-4,
        regularization_weight=0.03,
        mixup_alpha=0.3,
        optimizer_schedule="constant",
        iterations_per_epoch=None,
        data_device_policy="cpu_stream",
    )


def plan(directory, dataset="cifar10", **kwargs):
    options = dict(
        method="vpu",
        dataset=dataset,
        prior=0.5,
        prior_source="synthetic population for selection only",
        seed=3,
        parameters=parameters(),
        checkpoint_bytes=64 * 2**20,
        c_token="0.5",
    )
    return script.build_plan(directory, **(options | kwargs))


@pytest.mark.parametrize("dataset", script.DATASETS)
def test_basic_determ_four_slot_plan_requires_explicit_ts_recipe(tmp_path, dataset):
    directory = product(tmp_path, dataset)
    before = torch.get_rng_state().clone()
    result = plan(directory, dataset)
    assert result == plan(directory, dataset) and torch.equal(before, torch.get_rng_state())
    assert result["training_view"] == "ts"
    assert result["image_setup"]["marginal_pool"] == "train_P_union_U"
    assert result["image_setup"]["prior_contract"]["source"] == "not_used"
    assert result["parameters_resolved"]["data_device_policy"] == "cpu_stream"
    assert (
        "docs/research/pu_survey/data/vpu_schedule_source_20261015.json"
        in result["source_files_sha256"]
    )
    assert not result["formal_admission"] and not result["formal_budget_approved"]


@pytest.mark.parametrize("dataset", ["cifar10", "adni"])
def test_basic_actual_training_only_fit_normalized_selection_and_replay(
    tmp_path, dataset, monkeypatch
):
    original, calls = VPUClassifier.fit, []

    @functools.wraps(original)
    def audit(self, features, observed, **kwargs):
        calls.append((len(features), int(observed.sum()), set(kwargs)))
        return original(self, features, observed, **kwargs)

    monkeypatch.setattr(VPUClassifier, "fit", audit)
    planned = plan(product(tmp_path, dataset), dataset)
    result = script.execute(planned, tmp_path / "output")
    assert calls == [(12, 3, {"epoch_callback", "os_or_ts"})]
    assert result["accepted_engineering_run"] and result["selected_models_replayed"]
    assert result["warning_count"] == result["retry_count"] == 0
    assert result["training_view"] == "ts" and not result["formal_admission"]
    assert not result["paper_numerical_reproduction"]
    assert set(result["selection"]) == {"PA", "OA"}
    assert 1 <= result["retained_checkpoint_count"] <= 2
    payload = json.loads((tmp_path / "output/manifest.json").read_text())
    assert payload["run_view"] == "ts-compatible" and payload["calibration_applied"]
    row = payload["candidate_runs"][0]
    facts = row["training_provenance"]
    assert [c["epoch_label"] for c in row["epoch_checkpoints"]] == [0, 1, 2]
    assert [c["training_context"]["optimizer_steps"] for c in row["epoch_checkpoints"]] == [3, 6, 9]
    assert facts["bound_fit_receipt"]["training_marginal_rows"] == 12
    assert facts["bound_fit_receipt"]["normalization_embedded_in_model_state"]
    assert not facts["source_validation_risk_used_for_selection"]
    assert facts["selection_prior_not_forwarded_to_VPU"]
    assert len(facts["normalization_history"]) == 3
    with pytest.raises(FileExistsError):
        script.execute(planned, tmp_path / "output")


@pytest.mark.parametrize("key", list(parameters()))
def test_param_missing_explicit_parameters_refused(tmp_path, key):
    supplied = parameters()
    supplied.pop(key)
    with pytest.raises(ValueError, match="explicit"):
        plan(product(tmp_path, "cifar10"), parameters=supplied)


@pytest.mark.parametrize("fault", ["view", "test_normalization", "source_validation_selection"])
def test_edge_tampered_or_relabelled_native_plan_refused(tmp_path, fault):
    planned = plan(product(tmp_path, "cifar10"))
    changed = copy.deepcopy(planned)
    if fault == "view":
        changed["training_view"] = "os"
    elif fault == "test_normalization":
        changed["image_setup"]["normalization_scope"] = "test"
    else:
        changed["image_setup"]["validation_diagnostic_selection_eligible"] = True
    changed["plan_sha256"] = script.strict_canonical_hash(
        {k: v for k, v in changed.items() if k != "plan_sha256"}
    )
    with pytest.raises(ValueError, match="identity changed"):
        script.execute(changed, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_basic_cli_plan_entrypoint_is_read_only(tmp_path, capsys):
    directory = product(tmp_path, "cifar10")
    parameter_path = tmp_path / "parameters.json"
    parameter_path.write_text(json.dumps(parameters()))
    assert (
        script.main(
            [
                "--method",
                "vpu",
                "--data-dir",
                str(directory),
                "--dataset",
                "cifar10",
                "--class-prior",
                "0.5",
                "--prior-source",
                "synthetic selection prior",
                "--seed",
                "3",
                "--parameters",
                str(parameter_path),
                "--checkpoint-bytes-per-component",
                str(64 * 2**20),
                "--c",
                "0.5",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["training_view"] == "ts" and not result["formal_admission"]
    assert not list(tmp_path.rglob("manifest.json"))
