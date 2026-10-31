"""Four-role PA/OA exports actual non-epoch PUET/RP costs, not admission."""

import json
import pickle

import numpy as np
import pytest

from pu_toolbox.estimators.classic.rank_pruning import RankPruningClassifier
from pu_toolbox.estimators.risk.puet import PUExtraTreesClassifier
from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.classical_candidate import ClassicalCandidateTrainer
from pu_toolbox.experiment.runner import ExperimentRunner

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("method,view", [("puet", "os"), ("puet", "ts"), ("rp", "os")])
def test_param_basic_determ_four_roles_preserve_selection_and_non_epoch_costs(
    tmp_path, method, view
):
    parts = []
    for i in range(4):
        labels = np.tile([0, 1], 18)
        features = np.random.RandomState(4 + i).normal(size=(36, 3)) + labels[:, None]
        parts.append(
            DatasetPart(
                X=features,
                labels=labels,
                indices=np.arange(i * 36, (i + 1) * 36),
                view="clean",
                for_selection=i != 3,
            )
        )
    originals = [(p.X.copy(), p.labels.copy()) for p in parts]
    model = (
        PUExtraTreesClassifier(0.5, n_estimators=3, max_depth=3, random_state=1)
        if method == "puet"
        else RankPruningClassifier(n_cv_folds=3, min_retained=2, random_state=1)
    )
    path = tmp_path / "manifest.json"
    result = ExperimentRunner(
        seed=1,
        class_prior=0.5,
        manifest_path=str(path),
        config={
            "method": method,
            "c": 0.5,
            "os_or_ts": view,
            "trainer": ClassicalCandidateTrainer(),
        },
    ).fit(model, *parts)
    assert set(result.test_metrics) == {"PA", "OA"}
    manifest = json.loads(path.read_text())
    receipt = manifest["candidate_runs"][0]["training_provenance"]
    assert receipt["work_completed"] == (3 if method == "puet" else 4)
    assert receipt["neural_epoch_count"] == 0 and not receipt["formal_admission"]
    assert receipt["calibration_applied"] is (view == "ts")
    assert all(s.epoch is None and s.checkpoint_index is None for s in result.selections.values())
    assert "recipe_registry_sha256" not in manifest
    for p, (x, y) in zip(parts, originals, strict=True):
        np.testing.assert_array_equal(p.X, x)
        np.testing.assert_array_equal(p.labels, y)
    for selected in result.selected_models.values():
        restored = pickle.loads(pickle.dumps(selected))
        np.testing.assert_array_equal(selected.predict(parts[3].X), restored.predict(parts[3].X))


def test_edge_pruning_ts_remains_rejected_and_does_not_claim_calibration(tmp_path):
    trainer = ClassicalCandidateTrainer()
    model = RankPruningClassifier()
    features = np.random.RandomState(1).normal(size=(36, 3))
    with pytest.raises(ValueError, match="native OS"):
        trainer.fit(model, features, np.tile([0, 1], 18), os_or_ts="ts")
    assert trainer.last_receipt["work_started"] == 0
    assert trainer.last_receipt["status"] == "failed" and not model._is_fitted
