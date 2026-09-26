# ruff: noqa: N803, N806, F811
"""The aggregation entry point partitions results by the view each run used.

P2.0e records the per-run training view (``run_view`` / ``calibration_applied``)
and requires it to become a fairness-grouping dimension.  Without that, a method
run under both views collides with itself: ``_method_specs`` collapses one
method's runs into a single spec, two views are two specs, and the entry point
refused the whole root with ``appears twice with different specs``.

The partition lives here rather than in ``partition_fair_leaderboard_runs``.
Once the entry point has separated the views, every unit handed to that gate is
single-view, so its public contract -- and the tests that pin it, in
``test_leaderboard_run_view.py`` -- are untouched.  It stays the lower,
fail-closed defence.

A group's logical identity is ``(comparability_group, run_view)``: the same
``comparability_group`` string may appear once per view in ``groups[]``.  The
protocol's ``comparability_group`` value is never suffixed with the view.

``run_view`` is the view a run *actually used*, not the method's
``native_sampling_assumption``: a method declared native to TS but not yet wired
for calibration falls back to ``os-compatible``.  This module implements the
per-run view partition only; it does not implement per-native-assumption
stratification, and no field here should be read as it.
"""

import pytest
from _aggregate_script_helpers import (  # noqa: F401
    aggregate_script,
    aggregate_tree,
    manifest,
    write_tree,
)

pytestmark = pytest.mark.unit

#: The locked protocol's group for the default budget family (``minibatch``).
_GROUP = "spambase/native_2d/minibatch"


def _views(report):
    """Each group's composite identity, in the report's own order."""
    return [(group["comparability_group"], group["run_view"]) for group in report["groups"]]


def test_basic_a_method_run_under_both_views_partitions_into_two_groups(aggregate_script, tmp_path):
    """One method, one (seed, c), two views: two leaderboards, not a collision."""
    root = write_tree(
        tmp_path,
        {
            "nnpu_os": manifest(method="nnpu", run_view="os-compatible"),
            "nnpu_ts": manifest(method="nnpu", run_view="ts-compatible"),
        },
    )
    report = aggregate_tree(aggregate_script, root)

    assert _views(report) == [(_GROUP, "os-compatible"), (_GROUP, "ts-compatible")]


def test_basic_each_group_holds_only_the_runs_of_its_own_view(aggregate_script, tmp_path):
    root = write_tree(
        tmp_path,
        {
            "os": manifest(method="nnpu", run_view="os-compatible"),
            "ts": manifest(method="nnpu", run_view="ts-compatible"),
        },
    )
    report = aggregate_tree(aggregate_script, root)

    by_view = {group["run_view"]: group for group in report["groups"]}
    assert set(by_view) == {"os-compatible", "ts-compatible"}
    for group in by_view.values():
        assert group["methods"] == ["nnpu"]
        assert len(group["units"]) == 1
        assert group["units"][0]["methods"] == ["nnpu"]


def test_basic_a_single_view_tree_keeps_its_group(aggregate_script, tmp_path):
    """The old, single-view tree aggregates exactly as it did before."""
    root = write_tree(tmp_path, {"a": manifest(method="upu"), "b": manifest(method="nnpu")})
    report = aggregate_tree(aggregate_script, root)

    assert len(report["groups"]) == 1
    assert report["groups"][0]["comparability_group"] == _GROUP
    assert report["groups"][0]["run_view"] == "os-compatible"
    assert report["groups"][0]["methods"] == ["nnpu", "upu"]


def test_basic_methods_that_differ_in_view_go_to_separate_groups(aggregate_script, tmp_path):
    """A view is its own leaderboard, so a group need not hold every method.

    ``lbe`` is native to OS and ``nnpu`` runs calibrated, so one comparability
    group legitimately holds two different method sets once the views are
    separated.  Coverage and completeness accounting is P2.2's to define; this
    test pins the partition, not a completeness rule.
    """
    root = write_tree(
        tmp_path,
        {
            "lbe_os": manifest(method="lbe", run_view="os-compatible"),
            "nnpu_ts": manifest(method="nnpu", run_view="ts-compatible"),
        },
    )
    report = aggregate_tree(aggregate_script, root)

    assert _views(report) == [(_GROUP, "os-compatible"), (_GROUP, "ts-compatible")]
    assert [group["methods"] for group in report["groups"]] == [["lbe"], ["nnpu"]]


def test_param_units_stay_inside_the_view_that_produced_them(aggregate_script, tmp_path):
    root = write_tree(
        tmp_path,
        {
            f"nnpu_{view}_{seed}_{c}": manifest(method="nnpu", run_view=view, seed=seed, c=c)
            for view in ("os-compatible", "ts-compatible")
            for seed in (0, 1)
            for c in (0.1, 0.5)
        },
    )
    report = aggregate_tree(aggregate_script, root)

    assert _views(report) == [(_GROUP, "os-compatible"), (_GROUP, "ts-compatible")]
    for group in report["groups"]:
        assert group["methods"] == ["nnpu"]
        assert [(unit["seed"], unit["c"]) for unit in group["units"]] == [
            (0, 0.1),
            (0, 0.5),
            (1, 0.1),
            (1, 0.5),
        ]


def test_edge_a_manifest_without_a_run_view_is_refused(aggregate_script, tmp_path):
    payload = manifest(method="nnpu")
    del payload["run_view"]
    write_tree(tmp_path, {"a": payload})

    with pytest.raises(ValueError, match="run_view"):
        aggregate_tree(aggregate_script, tmp_path)


def test_edge_an_unknown_view_is_refused(aggregate_script, tmp_path):
    write_tree(tmp_path, {"a": manifest(method="nnpu", run_view="TS-compatible")})

    with pytest.raises(ValueError, match="run_view"):
        aggregate_tree(aggregate_script, tmp_path)


@pytest.mark.parametrize(
    ("run_view", "calibration_applied"),
    [("os-compatible", True), ("ts-compatible", False)],
)
def test_edge_a_view_that_disagrees_with_calibration_applied_is_refused(
    aggregate_script, tmp_path, run_view, calibration_applied
):
    """A run that claims a view its calibration flag contradicts is malformed."""
    write_tree(
        tmp_path,
        {"a": manifest(method="nnpu", run_view=run_view, calibration_applied=calibration_applied)},
    )

    with pytest.raises(ValueError, match="calibration_applied"):
        aggregate_tree(aggregate_script, tmp_path)


def test_edge_the_oracle_stays_in_the_os_partition(aggregate_script, tmp_path):
    """The oracle trains on real labels, so it has no calibrated view to run under.

    It is not broadcast into the TS leaderboard: that would put a row in a
    partition no run of it ever produced.
    """
    root = write_tree(
        tmp_path,
        {
            "oracle": manifest(method="pn_oracle", run_view="os-compatible", c_independent=True),
            "nnpu_ts": manifest(method="nnpu", run_view="ts-compatible"),
        },
    )
    report = aggregate_tree(aggregate_script, root)

    by_view = {group["run_view"]: group for group in report["groups"]}
    assert by_view["os-compatible"]["methods"] == ["pn_oracle"]
    assert by_view["ts-compatible"]["methods"] == ["nnpu"]


def test_edge_a_calibrated_oracle_is_refused(aggregate_script, tmp_path):
    """The oracle trains on real labels, so a calibrated view of it cannot exist.

    ``--oracle --os-or-ts ts`` is already refused at the command line; this is
    the aggregation-side half of the same rule, so a hand-built manifest cannot
    slip one into the calibrated leaderboard.
    """
    write_tree(
        tmp_path,
        {"a": manifest(method="pn_oracle", run_view="ts-compatible", c_independent=True)},
    )

    with pytest.raises(ValueError, match="oracle"):
        aggregate_tree(aggregate_script, tmp_path)


def test_determ_repeated_aggregation_yields_an_identical_report(aggregate_script, tmp_path):
    root = write_tree(
        tmp_path,
        {
            "os": manifest(method="nnpu", run_view="os-compatible"),
            "ts": manifest(method="nnpu", run_view="ts-compatible"),
        },
    )

    assert aggregate_tree(aggregate_script, root) == aggregate_tree(aggregate_script, root)
