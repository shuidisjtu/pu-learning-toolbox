import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]


def test_basic_dist_pu_source_identity_is_not_misread_as_training_fidelity():
    ledger = json.loads(
        (ROOT / "pu_toolbox/experiment/method_ledger.json").read_text(encoding="utf-8")
    )
    entry = ledger["methods"]["dist_pu"]
    assert "cb74be1a87176fd38270873c06374e53905b7354" in entry["code_version"]["source_status"]
    review = ROOT / entry["source_behavior_review_ref"]
    assert review.is_file()
    content = review.read_text(encoding="utf-8")
    for behavior in ("warm-up", "伪标签时序", "histogram", "test loader", "scheduler 计步"):
        assert behavior in content
    assert "warm-up 每 epoch 一步，PU 每 batch 一步" in content
    assert "utils.py::train()` 第 14–30 行" in content
    assert "train.py::train_mixup()` 第 909–958 行" in content
    assert any("batch_size 未生效" in item for item in entry["uncertainty"])
    assert (
        "warm-up scheduler 在每 epoch 末 step，PU scheduler 在每批后 step"
        in entry["uncertainty"][0]
    )
    assert "clean-room" in entry["implementation_status"]
