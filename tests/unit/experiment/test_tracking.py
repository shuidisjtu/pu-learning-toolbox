# tests/unit/experiment/test_tracking.py
from pu_toolbox.experiment.tracking import EpochRecord, RunResult, RunTrajectory, SelectionArtifact


def test_trajectory_holds_epochs_and_model():
    traj = RunTrajectory(
        epochs=[EpochRecord(epoch=1, metrics={"val_risk": 0.4})],
        model=object(),
        best_epoch=1,
    )
    assert traj.epochs[0].epoch == 1
    assert traj.best_epoch == 1
    assert traj.model is not None


def test_selection_artifact_fields():
    art = SelectionArtifact(
        protocol="PA", run_index=0, epoch=3, threshold=0.5, metrics={"acc": 0.8}
    )
    assert art.protocol == "PA"
    assert art.threshold == 0.5


def test_run_result_defaults():
    r = RunResult(selections={}, test_metrics={}, manifest={})
    assert r.failures == []
