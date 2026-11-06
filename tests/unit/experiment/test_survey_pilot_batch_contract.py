"""Execution partitions preserve the current protocol's runnable grid."""

import pytest

from pu_toolbox.experiment.pilot_plan import (
    PilotScope,
    batches,
    planned_runs,
    select_execution_units,
)
from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit


def _protocol():
    return load_protocol()


def _keys(protocol):
    return {run.key for run in planned_runs(protocol)}


def test_basic_dataset_partitions_cover_every_run_once():
    protocol = _protocol()
    whole = _keys(protocol)
    covered = set()
    for dataset in sorted({unit["dataset"] for unit in protocol["execution_units"]}):
        partition = select_execution_units(protocol, PilotScope(datasets=(dataset,)))
        keys = _keys(partition)
        assert not covered.intersection(keys), dataset
        covered.update(keys)
    assert covered == whole


def test_basic_batches_cover_the_selected_grid_without_overlap():
    protocol = select_execution_units(
        _protocol(), PilotScope(datasets=("cifar10",), methods=("nnpu", "pn_oracle"))
    )
    runs = planned_runs(protocol)
    covered = [key for batch in batches(runs) for key in batch.covers]
    assert len(covered) == len(set(covered))
    assert set(covered) == {run.key for run in runs}


@pytest.mark.parametrize("training_path", ["native_cnn", "cnn_feature_adapter"])
def test_param_oracle_paths_are_independent_c_independent_units(training_path):
    protocol = select_execution_units(
        _protocol(),
        PilotScope(
            datasets=("cifar10",),
            methods=("pn_oracle",),
            training_paths=(training_path,),
        ),
    )
    runs = planned_runs(protocol)
    assert len(runs) == len(protocol["seeds"])
    assert all(run.is_oracle and run.c_token is None for run in runs)
    assert {run.training_path for run in runs} == {training_path}


def test_edge_blocked_units_are_never_planned():
    protocol = _protocol()
    for unit in protocol["execution_units"]:
        if unit["method"] == "kldce":
            unit["runnable"] = False
            unit["non_runnable_reason"] = "disabled in this caller-selected matrix"
    blocked = {
        (unit["dataset"], unit["method"], unit["training_path"])
        for unit in protocol["execution_units"]
        if not unit["runnable"]
    }
    assert blocked
    assert not blocked.intersection(
        (run.dataset, run.method, run.training_path) for run in planned_runs(protocol)
    )
    with pytest.raises(ValueError, match="no execution unit is runnable"):
        select_execution_units(protocol, PilotScope(methods=("kldce",)))


def test_determ_partition_order_does_not_change_the_grid():
    protocol = _protocol()
    forward = select_execution_units(
        protocol, PilotScope(datasets=("imdb", "spambase"), methods=("nnpu", "upu"))
    )
    reverse = select_execution_units(
        protocol, PilotScope(datasets=("spambase", "imdb"), methods=("upu", "nnpu"))
    )
    assert [run.key for run in planned_runs(forward)] == [run.key for run in planned_runs(reverse)]
