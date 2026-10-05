"""Multistage peak disk budgets include warmup, final-stage snapshots and retry."""

import pytest

from pu_toolbox.estimators.risk.pulda import PULDAClassifier
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import DeepFitTrainer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["pulda", "holistic_pu", "genpu", "robust_pu", "split_pu"])
def test_basic_edge_reclaim_budget_counts_both_stages(tmp_path, method):
    if method == "pulda":
        model = PULDAClassifier(class_prior=0.4, warmup_epochs=2, pu_epochs=3)
    elif method == "holistic_pu":
        from pu_toolbox import HolisticPUClassifier

        model = HolisticPUClassifier(warmup_epochs=2, max_epochs=3)
    elif method == "genpu":
        from pu_toolbox import GenPUClassifier

        # GAN epochs cost training time but do not emit classifier snapshots.
        model = GenPUClassifier(class_prior=0.4, max_epochs=9, classifier_epochs=5)
    elif method == "robust_pu":
        from pu_toolbox import RobustPUClassifier

        model = RobustPUClassifier(pretrain_epochs=2, episodes=3, inner_epochs=4)
    else:
        from pu_toolbox import SplitPUClassifier

        model = SplitPUClassifier(teacher_epochs=1, split_epochs=2, rounds=2, student_epochs=1)
    runner = ExperimentRunner(
        manifest_path=str(tmp_path / "manifest.json"),
        config={
            "checkpoint_bytes_per_component": 1000,
            "reclaim_unselected_checkpoints": True,
            "candidates": [{}, {}],
        },
    )
    result = runner._checkpoint_disk_preflight(model, DeepFitTrainer(), {}, ("model",))
    assert model.checkpoint_epoch_count == 5
    assert result["required_bytes"] == 1000 * 5 * 2 * 2


def test_param_explicit_locked_epoch_budget_remains_authoritative(tmp_path):
    model = PULDAClassifier(class_prior=0.4, warmup_epochs=2, pu_epochs=3)
    runner = ExperimentRunner(
        manifest_path=str(tmp_path / "manifest.json"),
        config={
            "checkpoint_bytes_per_component": 1000,
        },
    )
    result = runner._checkpoint_disk_preflight(
        model, DeepFitTrainer(), {"budget": {"epochs": 7}}, ("model",)
    )
    assert result["required_bytes"] == 14000
