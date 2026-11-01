"""Durable per-attempt receipts survive failures/retries without changing fit results."""

import json
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.estimators.risk.puet import PUExtraTreesClassifier
from pu_toolbox.experiment.classical_candidate import ClassicalCandidateTrainer

pytestmark = pytest.mark.unit


def data():
    return np.random.RandomState(3).normal(size=(36, 3)), np.tile([0, 1], 18)


def model():
    return PUExtraTreesClassifier(0.5, n_estimators=3, max_depth=3, random_state=1)


def test_basic_determ_opt_in_receipt_equals_successful_manifest_payload(tmp_path):
    features, labels = data()
    direct = model().fit(features, labels)
    recorded = model()
    trainer = ClassicalCandidateTrainer(receipt_dir=tmp_path)
    trainer.fit(recorded, features, labels)
    assert len(trainer.receipt_paths) == 1
    receipt = json.loads(Path(trainer.receipt_paths[0]).read_text())
    assert receipt == trainer.last_receipt == recorded.training_provenance_
    assert receipt["status"] == "completed" and receipt["work_completed"] == 3
    np.testing.assert_array_equal(
        recorded.decision_function(features), direct.decision_function(features)
    )


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_edge_partial_failed_attempt_survives_next_success(tmp_path, monkeypatch, error):
    features, labels = data()
    original = PUExtraTreesClassifier._build_tree
    calls = []

    def fail_second(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise error("injected tree failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PUExtraTreesClassifier, "_build_tree", fail_second)
    trainer = ClassicalCandidateTrainer(receipt_dir=tmp_path)
    fitted = model()
    with pytest.raises(error, match="injected tree failure"):
        trainer.fit(fitted, features, labels)
    assert not fitted._is_fitted and len(trainer.receipt_paths) == 1
    first_path = Path(trainer.receipt_paths[0])
    first_bytes = first_path.read_bytes()
    first = json.loads(first_bytes)
    assert first["error_type"] == error.__name__
    assert first["work_started"] == 2 and first["work_completed"] == 1
    trainer.fit(fitted, features, labels)
    assert fitted._is_fitted and len(trainer.receipt_paths) == 2
    assert trainer.receipt_paths[0] != trainer.receipt_paths[1]
    assert first_path.read_bytes() == first_bytes
    assert json.loads(Path(trainer.receipt_paths[1]).read_text())["status"] == "completed"


def test_param_invalid_receipt_target_rejects_fit_without_success_provenance(tmp_path):
    target = tmp_path / "not-a-directory"
    target.write_text("keep me")
    fitted = model()
    trainer = ClassicalCandidateTrainer(receipt_dir=target)
    with pytest.raises(OSError):
        trainer.fit(fitted, *data())
    assert not fitted._is_fitted and not hasattr(fitted, "training_provenance_")
    assert target.read_text() == "keep me" and trainer.receipt_paths == []
