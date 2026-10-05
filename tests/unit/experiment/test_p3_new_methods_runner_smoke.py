# ruff: noqa: N803, N806
"""Independent four-role synthetic runner access, not survey candidate admission."""

import json
from pathlib import Path

import numpy as np
import pytest

from pu_toolbox.experiment import DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner

pytestmark = pytest.mark.unit


def parts():
    rng = np.random.RandomState(6)
    result = []
    for role in range(4):
        y = np.tile([0, 1], 16)
        X = rng.normal(size=(32, 3)).astype("float32") + y[:, None]
        result.append(
            DatasetPart(
                X=X,
                labels=y,
                indices=np.arange(role * 32, (role + 1) * 32),
                view="clean",
                for_selection=role != 3,
            )
        )
    return result


def make_model(name):
    if name == "rp":
        from pu_toolbox import RankPruningClassifier

        return RankPruningClassifier(n_cv_folds=2, random_state=0)
    pytest.importorskip("torch")
    if name == "pan":
        from pu_toolbox import PANClassifier

        return PANClassifier(hidden_dim=4, max_epochs=2, batch_size=8, random_state=0, device="cpu")
    if name == "genpu":
        from pu_toolbox import GenPUClassifier

        return GenPUClassifier(
            class_prior=0.5,
            hidden_dim=4,
            latent_dim=2,
            max_epochs=1,
            classifier_epochs=1,
            batch_size=8,
            random_state=0,
            device="cpu",
        )
    if name == "holistic_pu":
        from pu_toolbox import HolisticPUClassifier

        return HolisticPUClassifier(
            hidden_dim=4, warmup_epochs=3, max_epochs=1, batch_size=8, random_state=0, device="cpu"
        )
    raise AssertionError(name)


@pytest.mark.parametrize(
    "name,view", [("pan", "os"), ("rp", "os"), ("genpu", "ts"), ("holistic_pu", "os")]
)
@pytest.mark.parametrize("reclaim", [False, True])
def test_basic_new_methods_four_role_runner_and_manifest(tmp_path, name, view, reclaim):
    model = make_model(name)
    path = tmp_path / "manifest.json"
    result = ExperimentRunner(
        seed=0,
        class_prior=0.5,
        manifest_path=str(path),
        config={
            "method": name,
            "c": 0.5,
            "os_or_ts": view,
            "checkpoint_bytes_per_component": 4096,
            "reclaim_unselected_checkpoints": reclaim,
        },
    ).fit(model, *parts())
    assert set(result.test_metrics) == {"PA", "OA"}
    for metrics in result.test_metrics.values():
        assert np.isfinite(metrics["accuracy"]) and np.isfinite(metrics["auc"])
    manifest = json.loads(path.read_text())
    assert manifest["run_view"] == f"{view}-compatible"
    assert manifest["calibration_applied"] is (view == "ts")
    assert manifest.get("execution_mode") != "versioned_pilot"
    assert "recipe_registry_sha256" not in manifest
    if name != "rp":
        references = manifest["candidate_runs"][0]["epoch_checkpoints"]
        assert len(references) == {"pan": 2, "genpu": 1, "holistic_pu": 4}[name]
        selected = {record["checkpoint"]["path"] for record in manifest["selection"].values()}
        for reference in references:
            should_reclaim = reclaim and reference["path"] not in selected
            assert reference["reclaimed"] is should_reclaim
            assert Path(reference["path"]).is_file() is not should_reclaim
        assert all(artifact.checkpoint_index is not None for artifact in result.selections.values())


def test_seed_determinism_in_rank_pruning_four_role_runner():
    metrics = []
    for _ in range(2):
        result = ExperimentRunner(
            seed=0,
            class_prior=0.5,
            config={"method": "rp", "c": 0.5, "os_or_ts": "os"},
        ).fit(make_model("rp"), *parts())
        metrics.append(result.test_metrics)
    assert metrics[0] == metrics[1]


def test_param_edge_missing_pulns_support_cannot_steal_selection_labels(tmp_path):
    pytest.importorskip("torch")
    from pu_toolbox import PULNSClassifier

    model = PULNSClassifier(
        hidden_dim=4, pretrain_epochs=1, episodes=1, classifier_epochs=1, batch_size=8, device="cpu"
    )
    runner = ExperimentRunner(
        seed=0,
        class_prior=0.5,
        manifest_path=str(tmp_path / "manifest.json"),
        config={"method": "pulns", "c": 0.5, "os_or_ts": "os"},
    )
    with pytest.raises(RuntimeError, match="all candidate runs failed"):
        runner.fit(model, *parts())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "clean support_data" in json.dumps(manifest["failures"])
    assert manifest["selection"] == {} and manifest["test_results"] == {}
    assert not model._is_fitted
