# ruff: noqa: S101

"""The leaderboard group carries each method's training view (protocol §5).

Protocol §5 strata results by native assumption, so a reader must be able to
tell an ``os-compatible`` row from a ``ts-compatible`` one.  The view is a
per-method property, not a shared precondition: native assumptions legitimately
differ between the methods of one dataset, so requiring agreement would reject
every real group.
"""

from __future__ import annotations

import pytest

from pu_toolbox.experiment.feature_adapter import (
    LeaderboardRunSpec,
    partition_fair_leaderboard_runs,
)

pytestmark = pytest.mark.unit


def _run(method, run_view="os-compatible", **overrides):
    values = {
        "method": method,
        "dataset": "spambase",
        "training_path": "native_2d",
        "adaptation_level": "source-faithful",
        "run_view": run_view,
        "split_sha256": "a" * 64,
        "representation_sha256": "b" * 64,
        "max_epochs": 200,
        "batch_size_candidates": (64, 128),
        "tuning_candidate_count": 5,
        "seeds": (0, 1, 2, 3, 4),
    }
    values.update(overrides)
    return LeaderboardRunSpec(**values)


def test_group_payload_records_the_view_of_every_method():
    groups = partition_fair_leaderboard_runs(
        [_run("nnpu", "ts-compatible"), _run("lbe", "os-compatible")]
    )
    payload = next(iter(groups.values()))
    assert payload["run_views"] == {"lbe": "os-compatible", "nnpu": "ts-compatible"}


def test_mixed_views_inside_one_dataset_are_allowed():
    """Native assumptions differ by method; that must not fail the fairness gate."""
    groups = partition_fair_leaderboard_runs(
        [_run("nnpu", "ts-compatible"), _run("lbe", "os-compatible")]
    )
    assert len(groups) == 1


def test_same_method_under_two_views_is_refused_and_names_the_view():
    """One leaderboard cannot hold two rows of one method; the view is why."""
    with pytest.raises(ValueError, match="differ in training view"):
        partition_fair_leaderboard_runs(
            [_run("nnpu", "ts-compatible"), _run("nnpu", "os-compatible")]
        )


def test_unknown_view_is_refused():
    with pytest.raises(ValueError, match="run_view"):
        partition_fair_leaderboard_runs([_run("nnpu", "TS-compatible")])
