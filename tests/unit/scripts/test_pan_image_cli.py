"""PAN four-role CLI plans and actual OS checkpoint selection, without admission."""

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

from pu_toolbox.estimators.deep.pan import PANClassifier  # noqa: E402

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


def plan(directory, dataset="cifar10", **kwargs):
    options = dict(
        method="pan",
        dataset=dataset,
        prior=0.5,
        prior_source="synthetic population for selection only",
        seed=3,
        parameters=dict(
            device="cpu", max_epochs=3, batch_size=4, learning_rate=1e-4, adversarial_weight=1.0
        ),
        checkpoint_bytes=64 * 2**20,
        c_token="0.5",
    )
    return script.build_plan(directory, **(options | kwargs))


@pytest.mark.parametrize("dataset", script.DATASETS)
def test_basic_determ_four_image_slot_plans_use_os_and_paper_target(tmp_path, dataset):
    directory = product(tmp_path, dataset)
    before = torch.get_rng_state().clone()
    result = plan(directory, dataset)
    assert result == plan(directory, dataset) and torch.equal(before, torch.get_rng_state())
    assert result["training_view"] == "os"
    assert result["image_setup"]["network_instances"] == 2
    assert result["comparison_group"] == "end-to-end CNN"
    assert "encoder" not in result["parameters_resolved"]
    assert result["image_setup"]["prior_contract"]["source"] == "not_used"
    assert (
        "docs/research/pu_survey/data/pan_source_review_20261018.json"
        in result["source_files_sha256"]
    )
    assert not result["formal_admission"] and not result["formal_budget_approved"]
    assert not list(tmp_path.glob("manifest.json"))


@pytest.mark.parametrize("dataset", ["cifar10", "adni"])
def test_basic_actual_os_fit_cost_dual_selection_reclaim_and_replay(tmp_path, dataset, monkeypatch):
    original, calls = PANClassifier.fit, []

    @functools.wraps(original)
    def audit(self, features, labels, **kwargs):
        calls.append((len(features), int(labels.sum()), set(kwargs)))
        return original(self, features, labels, **kwargs)

    monkeypatch.setattr(PANClassifier, "fit", audit)
    planned = plan(product(tmp_path, dataset), dataset)
    result = script.execute(planned, tmp_path / "output")
    assert result["accepted_engineering_run"] and result["selected_models_replayed"]
    assert result["warning_count"] == result["retry_count"] == 0
    assert calls == [(12, 3, {"epoch_callback", "os_or_ts"})]
    assert result["training_view"] == "os" and not result["formal_admission"]
    assert not result["paper_numerical_reproduction"]
    assert set(result["selection"]) == {"PA", "OA"}
    assert 1 <= result["retained_checkpoint_count"] <= 2
    payload = json.loads((tmp_path / "output/manifest.json").read_text())
    assert payload["run_view"] == "os-compatible" and not payload["calibration_applied"]
    row = payload["candidate_runs"][0]
    assert [c["epoch_label"] for c in row["epoch_checkpoints"]] == [0, 1, 2]
    assert [c["training_context"]["optimizer_steps"] for c in row["epoch_checkpoints"]] == [
        6,
        12,
        18,
    ]
    assert row["training_provenance"]["bound_fit_receipt"]["optimizer_steps"] == 18
    assert row["training_provenance"]["checkpoint_scope"].startswith("classifier_C_inference_only")
    with pytest.raises(FileExistsError):
        script.execute(planned, tmp_path / "output")


@pytest.mark.parametrize(
    "fault", ["missing_epoch", "missing_lr", "missing_weight", "prior_override", "custom_D", "view"]
)
def test_param_edge_ambiguous_or_tampered_plan_refused(tmp_path, fault):
    directory = product(tmp_path, "cifar10")
    original = plan(directory)
    parameters = dict(original["parameters_requested"])
    if fault.startswith("missing"):
        parameters.pop(
            {
                "missing_epoch": "max_epochs",
                "missing_lr": "learning_rate",
                "missing_weight": "adversarial_weight",
            }[fault]
        )
    elif fault in {"prior_override", "custom_D"}:
        parameters["class_prior" if fault == "prior_override" else "discriminator"] = None
    else:
        changed = copy.deepcopy(original)
        changed["training_view"] = "ts"
        changed["plan_sha256"] = script.strict_canonical_hash(
            {k: v for k, v in changed.items() if k != "plan_sha256"}
        )
        with pytest.raises(ValueError, match="identity changed"):
            script.execute(changed, tmp_path / "bad")
        assert not (tmp_path / "bad").exists()
        return
    with pytest.raises(ValueError):
        plan(directory, parameters=parameters)


def test_basic_cli_plan_entrypoint_does_not_train_or_create_output(tmp_path, capsys):
    directory = product(tmp_path, "cifar10")
    parameters = tmp_path / "parameters.json"
    parameters.write_text(
        json.dumps(
            dict(
                device="cpu", max_epochs=3, batch_size=4, learning_rate=1e-4, adversarial_weight=1.0
            )
        )
    )
    assert (
        script.main(
            [
                "--method",
                "pan",
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
                str(parameters),
                "--checkpoint-bytes-per-component",
                str(64 * 2**20),
                "--c",
                "0.5",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["training_view"] == "os" and result["checkpoint_epoch_bound"] == 3
    assert not result["formal_admission"]
    assert not list(tmp_path.rglob("manifest.json"))
