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
