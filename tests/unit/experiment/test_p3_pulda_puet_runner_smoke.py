"""Technical runner compatibility, not new frozen-pilot admission or paper replication."""

import json

import numpy as np
import pytest

from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["pulda", "puet"])
def test_pulda_puet_ts_four_role_pa_oa_smoke(tmp_path, method):
    rng = np.random.RandomState(0)
    parts = []
    for index in range(4):
        y = np.tile([0, 1], 12)
        features = rng.normal(size=(24, 3)).astype(np.float32) + y[:, None] * 0.5
        parts.append(
            DatasetPart(
                X=features,
                labels=y,
                indices=np.arange(index * 24, (index + 1) * 24),
                view="clean",
                for_selection=index != 3,
            )
        )
    if method == "pulda":
        pytest.importorskip("torch")
        from pu_toolbox.estimators.risk.pulda import PULDAClassifier

        model = PULDAClassifier(
            class_prior=0.5,
            hidden_dim=4,
            depth=1,
            warmup_epochs=1,
            pu_epochs=1,
            positive_batch_size=4,
            unlabeled_batch_size=8,
            device="cpu",
            random_state=0,
        )
    else:
        from pu_toolbox.estimators.risk.puet import PUExtraTreesClassifier

        model = PUExtraTreesClassifier(class_prior=0.5, n_estimators=3, max_depth=3, random_state=0)
    path = tmp_path / "manifest.json"
    runner = ExperimentRunner(
        seed=0,
        class_prior=0.5,
        manifest_path=str(path),
        config={"method": method, "c": 0.5, "os_or_ts": "ts"},
    )
    result = runner.fit(model, *parts)
    assert set(result.test_metrics) == {"PA", "OA"}
    for metrics in result.test_metrics.values():
        assert np.isfinite(metrics["accuracy"])
        assert np.isfinite(metrics["auc"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["run_view"] == "ts-compatible"
    assert manifest["calibration_applied"] is True
    assert manifest.get("execution_mode") != "versioned_pilot"
    assert "recipe_registry_sha256" not in manifest
