"""Dist-PU frozen pilot remains feature-adapted."""

# ruff: noqa: E402

import pytest

pytest.importorskip("torch")

from pu_toolbox.estimators.risk.dist_pu import DistPUClassifier
from pu_toolbox.experiment.pilot_plan import planned_runs
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.integration


def test_edge_frozen_pilot_keeps_distpu_fullbatch_features_and_original_candidates():
    protocol = load_protocol()
    profile = protocol["method_profiles"]["dist_pu"]
    assert profile["budget"] == "fullbatch" and profile["model_family"] == "mlp128"
    assert "encoder" not in profile["params"] and "training_recipe" not in profile["params"]
    runs = [run for run in planned_runs(protocol) if run.method == "dist_pu"]
    assert runs and not any(run.training_path == "native_cnn" for run in runs)
    assert {run.training_path for run in runs if run.dataset == "cifar10"} == {
        "cnn_feature_adapter"
    }
    estimator = DistPUClassifier(0.4, **profile["params"])
    assert estimator.encoder is None and estimator.training_recipe == "clean_room"
