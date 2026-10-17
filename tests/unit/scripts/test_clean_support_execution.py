"""Actual three-method CPU training with bound independent support products."""

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment.support_data import (
    load_support_dataset,
    prepare_support_dataset,
    save_support_dataset,
)

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_clean_support_experiment as script  # noqa: E402

pytestmark = pytest.mark.unit


def parameters(method):
    common = {"device": "cpu", "hidden_dim": 4, "batch_size": 32}
    return (
        common
        | {
            "self_pu": {
                "warmup_epochs": 0,
                "self_paced_start": 0,
                "self_paced_end": 1,
                "distill_start": 2,
                "max_epochs": 3,
            },
            "lagam": {"warmup_epochs": 0, "max_epochs": 2, "num_clusters": 2},
            "pulns": {"pretrain_epochs": 1, "episodes": 2, "classifier_epochs": 1},
        }[method]
    )


def product(tmp_path):
    rng = np.random.RandomState(0)
    features = rng.normal(size=(400, 5))
    truth = np.tile([0, 1], 200)
    features[:, 0] += truth
    bundle, support, metadata = prepare_support_dataset(
        features,
        truth,
        dataset="spambase",
        seed=0,
        support_size=40,
        support_seed=7,
        group_policy="not_applicable",
        source="explicit synthetic CPU fixture",
    )
    root = tmp_path / "products"
    save_support_dataset(bundle, support, metadata, root)
    return root


@pytest.mark.parametrize("method", script.METHODS)
def test_basic_actual_method_training_extra_budget_and_selected_replay(method, tmp_path):
    root = product(tmp_path)
    plan = script.build_plan(root, method=method, c_token="0.5", parameters=parameters(method))
    output = tmp_path / "results"
    summary = script.execute(plan, output)
    assert summary["accepted_engineering_run"] and summary["selected_model_replay_verified"]
    assert summary["retry_count"] == summary["warning_count"] == 0
    assert summary["formal_admission"] is False
    assert set(summary["test"]) == {"OA"} and plan["candidate_count"] == 1
    assert summary["extra_supervision"]["support_removed_from_train"]
    assert summary["support_product_audit"]["support_source_verified"] is False
    effects = summary["support_product_audit"]["role_pool_effect"]
    assert effects["train"]["reduced_count"] == 40
    assert all(row["reduced_count"] == 0 for name, row in effects.items() if name != "train")
    provenance = summary["training_provenance"]
    assert provenance["support_samples"] == 40
    assert provenance["total_observed_training_label_budget"] == (
        provenance["observed_labeled_positive_count"] + 40
    )
    if method == "self_pu":
        assert provenance["active_meta_reweight_records"] > 0
    elif method == "lagam":
        assert provenance["virtual_steps"] > 0
    else:
        assert provenance["stage_optimizer_steps"]["policy"] == 2
    with (output / "selected_OA.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    bundle = load_support_dataset(root)[0]
    predictions = script.predict_saved_selection(saved, bundle.test.X)
    assert float(np.mean(predictions == bundle.test.labels)) == summary["test"]["OA"]["accuracy"]
    assert (
        json.loads((output / "summary.json").read_text())["plan_sha256"] == summary["plan_sha256"]
    )
    assert len(list(output.glob("*.pkl"))) == 1
    with pytest.raises(FileExistsError):
        script.execute(plan, output)


@pytest.mark.parametrize("mode", ["gpu", "missing_budget", "prior", "seed", "invalid_budget"])
def test_param_implicit_budget_gpu_or_prior_override_rejected(tmp_path, mode):
    root = product(tmp_path)
    params = parameters("lagam")
    if mode == "gpu":
        params["device"] = "cuda"
    elif mode == "missing_budget":
        del params["max_epochs"]
    elif mode == "prior":
        params["class_prior"] = 0.4
    elif mode == "seed":
        params["random_state"] = 99
    else:
        params["batch_size"] = True
    with pytest.raises(ValueError):
        script.build_plan(root, method="lagam", c_token="0.5", parameters=params)


@pytest.mark.parametrize("kind", ["plan", "product", "overlap"])
def test_edge_changed_inputs_or_output_overlap_refused_before_fit(tmp_path, kind, monkeypatch):
    root = product(tmp_path)
    plan = script.build_plan(root, method="lagam", c_token="0.5", parameters=parameters("lagam"))
    output = tmp_path / "results"
    if kind == "plan":
        plan["parameters_requested"]["max_epochs"] += 1
    elif kind == "product":
        with (root / "support.npz").open("ab") as stream:
            stream.write(b"broken")
    else:
        output = root / "results"

    def never(*args, **kwargs):
        raise AssertionError("invalid input must be rejected before fitting")

    monkeypatch.setattr(script.ExperimentRunner, "fit", never)
    with pytest.raises(ValueError):
        script.execute(plan, output)
    assert not output.exists()


def test_determ_repeated_fixed_plan_has_identical_scores_and_oa_selection(tmp_path):
    root = product(tmp_path)
    plan = script.build_plan(
        root, method="self_pu", c_token="0.5", parameters=parameters("self_pu")
    )
    one = script.execute(plan, tmp_path / "a")
    two = script.execute(plan, tmp_path / "b")
    assert one["selection"] == two["selection"] and one["test"] == two["test"]
    assert one["training_provenance"] == two["training_provenance"]


def test_edge_warning_keeps_evidence_but_cannot_be_accepted(tmp_path, monkeypatch):
    root = product(tmp_path)
    plan = script.build_plan(root, method="lagam", c_token="0.5", parameters=parameters("lagam"))
    actual = script.ExperimentRunner.fit

    def emit(self, *args, **kwargs):
        warnings.warn("synthetic review checkpoint", UserWarning, stacklevel=2)
        return actual(self, *args, **kwargs)

    monkeypatch.setattr(script.ExperimentRunner, "fit", emit)
    output = tmp_path / "warning"
    with pytest.raises(ValueError, match="require review"):
        script.execute(plan, output)
    report = json.loads((output / "summary.json").read_text())
    assert report["warning_count"] == 1 and not report["accepted_engineering_run"]


def test_basic_cli_defaults_to_nontraining_plan(tmp_path, capsys, monkeypatch):
    root = product(tmp_path)
    params = tmp_path / "parameters.json"
    params.write_text(json.dumps(parameters("lagam")))

    def never(*args, **kwargs):
        raise AssertionError("dry-run must not train")

    monkeypatch.setattr(script.ExperimentRunner, "fit", never)
    args = [
        "--artifact-dir",
        str(root),
        "--method",
        "lagam",
        "--c",
        "0.5",
        "--parameters",
        str(params),
    ]
    assert script.main(args) == 0
    assert json.loads(capsys.readouterr().out)["protocols"] == ["OA"]
    assert script.main(args + ["--execute"]) == 1
    assert "error:" in capsys.readouterr().err


def test_basic_new_audit_constructor_contract_receives_all_source_fields(tmp_path, monkeypatch):
    original = script.PULNSExtraSupportTrainer

    class NewAuditTrainer(original):
        def __init__(
            self,
            support,
            *,
            support_origin,
            original_role_counts,
            support_source_verified,
            support_source_receipt,
            **kwargs,
        ):
            self.source_fields = dict(
                support_origin=support_origin,
                original_role_counts=original_role_counts,
                support_source_verified=support_source_verified,
                support_source_receipt=support_source_receipt,
            )
            super().__init__(support, **kwargs)

        def validate_experiment(self, model, bundle, config, protocols):
            super().validate_experiment(model, bundle, config, protocols)
            self._audit.update(self.source_fields)

    root = product(tmp_path)
    monkeypatch.setattr(script, "PULNSExtraSupportTrainer", NewAuditTrainer)
    plan = script.build_plan(root, method="pulns", c_token="0.5", parameters=parameters("pulns"))
    assert plan["support_product_audit"]["native_adapter_source_audit"] is True
    assert plan["extra_supervision"]["support_origin"] == "carved_from_original_pool"
    assert plan["extra_supervision"]["original_role_counts"]["train"] == 288


def test_edge_partial_future_audit_contract_is_rejected(tmp_path, monkeypatch):
    original = script.PULNSExtraSupportTrainer

    class PartialAuditTrainer(original):
        def __init__(self, support, *, support_origin, **kwargs):
            super().__init__(support, **kwargs)

    root = product(tmp_path)
    monkeypatch.setattr(script, "PULNSExtraSupportTrainer", PartialAuditTrainer)
    with pytest.raises(ValueError, match="partial"):
        script.build_plan(root, method="pulns", c_token="0.5", parameters=parameters("pulns"))
