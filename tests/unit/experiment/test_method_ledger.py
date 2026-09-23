# tests/unit/experiment/test_method_ledger.py
import json

import pytest

from pu_toolbox.experiment.method_ledger import (
    LEDGER_PATH,
    load_ledger,
    native_sampling_assumption,
)

pytestmark = pytest.mark.unit


def _entry(value):
    return {"native_sampling_assumption": value}


def test_load_ledger_reads_methods_mapping(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"methods": {"nnpu": {}}}), encoding="utf-8")
    assert set(load_ledger(path)["methods"]) == {"nnpu"}


def test_load_ledger_rejects_payload_without_methods(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "1.1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid method ledger"):
        load_ledger(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("os", "os"),
        ("ts", "ts"),
        ("both", "both"),
        ("os（censoring PU，单一 i.i.d. 训练样本）", "os"),
        ("ts（两样本：可靠正例 P + 边缘分布无标签 U）", "ts"),
        ("os (censoring PU)", "os"),
        ("TS（两样本）", "ts"),
    ],
)
def test_normalises_annotated_values(raw, expected):
    """The ledger carries a Chinese note after the enum value; keep the head."""
    assert native_sampling_assumption(_entry(raw)) == expected


@pytest.mark.parametrize("raw", ["", "unknown", "case_control", "tsc"])
def test_rejects_unknown_values(raw):
    with pytest.raises(ValueError, match="native_sampling_assumption"):
        native_sampling_assumption(_entry(raw))


def test_rejects_missing_field():
    with pytest.raises(ValueError, match="native_sampling_assumption"):
        native_sampling_assumption({})


def test_shipped_ledger_normalises_for_every_method():
    """Every shipped entry resolves to a value the training-view gate accepts."""
    methods = load_ledger(LEDGER_PATH)["methods"]
    resolved = {name: native_sampling_assumption(entry) for name, entry in methods.items()}
    assert resolved, "shipped ledger declares no methods"
    assert set(resolved.values()) <= {"os", "ts", "both"}
