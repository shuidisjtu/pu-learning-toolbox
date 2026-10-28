"""The support CLI preserves canonical c, receipt and protocol identities."""

import json
import sys
from pathlib import Path

import pytest
from test_survey_image_support_execution import tabular_product

from pu_toolbox.experiment.survey_protocol import runner_protocol_context

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_survey_experiment as script  # noqa: E402

pytestmark = pytest.mark.unit


class _ContextCaptured(BaseException):
    pass


def context(monkeypatch, tmp_path, *, copied_receipt=False, prior=0.5, c="0.3"):
    product = tabular_product(tmp_path / "product")
    receipt = product / "support_manifest.json"
    if copied_receipt:
        copied = tmp_path / "unrelated.json"
        copied.write_text(receipt.read_text())
        receipt = copied
    observed = {}

    def capture(model, parts, *, generator, protocols, seed, config, **kwargs):
        from pu_toolbox.experiment.bundle import DatasetBundle

        config["c"] = float(c)
        observed.update(
            runner_protocol_context(
                model, DatasetBundle(*parts), config, seed, generator, protocols
            )
        )
        raise _ContextCaptured

    monkeypatch.setattr(script, "run_one", capture)
    with pytest.raises(_ContextCaptured):
        script.main(
            [
                str(product),
                "--protocol",
                "survey-v3.0",
                "--dataset",
                "spambase",
                "--method",
                "lagam",
                "--training-path",
                "native_2d",
                "--seeds",
                "0",
                "--c",
                c,
                "--class-prior",
                str(prior),
                "--support-dir",
                str(product),
                "--split-ref",
                str(receipt),
                "--device",
                "cpu",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    return observed


def test_basic_shipped_v3_support_receipt_and_c_are_canonical(monkeypatch, tmp_path):
    result = context(monkeypatch, tmp_path)
    assert not {"c", "noncanonical_protocol_matrix", "split_ref_override"}.intersection(
        result["protocol_deviation"]
    )


def test_edge_copied_receipt_remains_an_explicit_override(monkeypatch, tmp_path):
    result = context(monkeypatch, tmp_path, copied_receipt=True)
    assert "split_ref_override" in result["protocol_deviation"]


@pytest.mark.parametrize("c", ["0.1", "0.3", "0.5"])
def test_param_every_frozen_support_c_is_inside_the_grid(c, monkeypatch, tmp_path):
    assert "c" not in context(monkeypatch, tmp_path, c=c)["protocol_deviation"]


def test_determ_support_context_keeps_the_same_protocol_digest(monkeypatch, tmp_path):
    one = context(monkeypatch, tmp_path / "one")
    two = context(monkeypatch, tmp_path / "two")
    assert one["protocol_sha256"] == two["protocol_sha256"]
    assert json.dumps(one["protocol_deviation"]) == json.dumps(two["protocol_deviation"])


@pytest.mark.parametrize("method", ["lagam", "pulns", "self_pu_full_meta"])
@pytest.mark.parametrize("support", [False, True])
def test_basic_runtime_comparison_keys_match_support_selections(method, support):
    from pu_toolbox.experiment.survey_comparison import run_comparison_units
    from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATHS, load_protocol

    protocol = load_protocol(PROTOCOL_PATHS["survey-v3.0"])
    unit = {"dataset": "spambase", "method": method, "training_path": "native_2d"}
    entries = run_comparison_units(
        {"execution_unit": unit},
        {
            "clean_support_provided": support,
            "c_requested_token": "0.1",
            "survey_protocol": {**unit, "mechanism": "scar"},
        },
        survey=protocol,
        profile_id=method,
    )
    assert set(entries) == {"PA+clean-support" if support else "PA", "OA"}


@pytest.mark.parametrize("method", ["lagam", "self_pu_full_meta"])
def test_basic_real_support_cli_completes_with_matching_comparison_units(tmp_path, method):
    product = tabular_product(tmp_path / "product")
    output = tmp_path / "runs"
    script.main(
        [
            str(product),
            "--protocol",
            "survey-v3.0",
            "--dataset",
            "spambase",
            "--method",
            method,
            "--training-path",
            "native_2d",
            "--seeds",
            "0",
            "--c",
            "0.3",
            "--class-prior",
            "0.5",
            "--support-dir",
            str(product),
            "--split-ref",
            str(product / "support_manifest.json"),
            "--device",
            "cpu",
            "--out-dir",
            str(output),
        ]
    )
    manifests = list(output.rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert set(manifest["test_results"]) == {"PA+clean-support", "OA"}
    assert set(manifest["comparison"]) == set(manifest["test_results"])
    assert manifest["run_view"] == "os-compatible"
    assert manifest["calibration_applied"] is False
    support = manifest["extra_supervision"]
    assert support["protocol_policy_basis"] == "survey-v3.0"
    assert support["protocol_pending_review_items"] == []
    assert support["budget_approval"] == "approved_by_protocol_owner_20261021"
    assert manifest["selection_checkpoint_scope"] == "independent_per_epoch"
    assert not {"c", "noncanonical_protocol_matrix", "split_ref_override"}.intersection(
        manifest["protocol_deviation"]
    )
    assert all(run["status"] == "succeeded" for run in manifest["candidate_runs"])


def test_edge_receipt_reassessment_rejects_changed_support_arrays(tmp_path):
    import audit_survey_batches as audit

    product = tabular_product(tmp_path / "products" / "spambase" / "split_0")
    receipt = json.loads((product / "support_manifest.json").read_text())
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"support_root": str(tmp_path / "products")}))
    config = {"batches": [{"name": "support", "plan": str(plan)}]}
    entry = {
        "path": "run.json",
        "batch": "support",
        "payload": {"execution_unit": {"dataset": "spambase"}, "seed": 0, "split_ref": receipt},
    }
    assert audit.verify_support_receipts(config, [entry]) == {"run.json": True}
    (product / "support.npz").write_bytes(b"changed")
    assert audit.verify_support_receipts(config, [entry]) == {"run.json": False}
