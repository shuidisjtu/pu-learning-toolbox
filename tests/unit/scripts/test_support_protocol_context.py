"""The support CLI preserves canonical c, receipt and protocol identities."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
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
                "survey-v3.2",
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


def _support_argv(data_dir, receipt, support_dir, output, method):
    return [
        str(data_dir),
        "--protocol",
        "survey-v3.2",
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
        str(support_dir),
        "--split-ref",
        str(receipt),
        "--device",
        "cpu",
        "--out-dir",
        str(output),
    ]


@pytest.mark.parametrize("method", ["lagam", "pulns", "self_pu_full_meta"])
def test_basic_real_support_cli_keeps_independent_selection_protocols(tmp_path, method):
    product = tabular_product(tmp_path / "product")
    output = tmp_path / "runs"
    assert (
        script.main(
            _support_argv(product, product / "support_manifest.json", product, output, method)
        )
        == 0
    )
    manifests = list(output.rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert set(manifest["test_results"]) == {"PA+clean-support", "OA"}
    assert set(manifest["selection"]) == set(manifest["test_results"])
    assert all(name == artifact["protocol"] for name, artifact in manifest["selection"].items())
    assert manifest["run_view"] == "os-compatible"
    assert manifest["calibration_applied"] is False
    support = manifest["extra_supervision"]
    assert support["protocol_policy_basis"] == "survey-v3.2"
    assert support["protocol_pending_review_items"] == []
    assert support["budget_approval"] == "approved_by_protocol_owner_20261021"
    assert manifest["selection_checkpoint_scope"] == "independent_per_epoch"
    assert not {"c", "noncanonical_protocol_matrix", "split_ref_override"}.intersection(
        manifest["protocol_deviation"]
    )
    assert all(run["status"] == "succeeded" for run in manifest["candidate_runs"])


@pytest.mark.parametrize("method", ["lagam", "pulns"])
@pytest.mark.parametrize("problem", ["missing", "different_features"])
def test_edge_invalid_support_is_refused_before_training_and_evaluation(
    tmp_path, monkeypatch, capsys, method, problem
):
    product = tabular_product(tmp_path / "product")
    data_dir = support_dir = product
    if problem == "missing":
        support_dir = tmp_path / "missing_product"
        expected_error = "missing_product"
    else:
        data_dir = shutil.copytree(product, tmp_path / "other_split")
        train_path = data_dir / "train.npz"
        with np.load(train_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["X"][0, 0] += 1
        np.savez(train_path, **arrays)
        expected_error = "support artifact train does not match the loaded split"

    def never(*args, **kwargs):
        pytest.fail("training and evaluation must not start with invalid support")

    monkeypatch.setattr(script, "run_one", never)
    assert (
        script.main(
            _support_argv(
                data_dir, product / "support_manifest.json", support_dir, tmp_path / "out", method
            )
        )
        == 1
    )
    assert expected_error in capsys.readouterr().err
