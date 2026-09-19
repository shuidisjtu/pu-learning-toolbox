# ruff: noqa: N803, N806, F811
"""The aggregation entry point forces the leaderboard-separation gates.

``validate_comparable_manifests`` and ``partition_fair_leaderboard_runs`` both
existed with tests and no production caller, so path separation was a library
function plus a manifest field rather than something any real path went
through.  These tests pin the entry point that changes that -- and the four
ways a naive version of it would be wrong: grouping by dataset instead of
comparability group, partitioning a whole group instead of one (seed, c) unit
(each seed has its own split), building a fairness spec per manifest instead of
per method, and deriving the seed list from the files found.

Budget-family comparability has its own file,
``test_aggregate_survey_budget_fairness.py``.

Scope is deliberately the wiring only.  The expected-unit completeness grid,
anomaly units and reproduction-field auditing belong to P2.2.
"""

import pytest
from _aggregate_script_helpers import (  # noqa: F401
    aggregate_script,
    aggregate_tree,
    manifest,
    write_tree,
)

pytestmark = pytest.mark.unit


def test_basic_comparable_group_partitions_by_path_and_unit(aggregate_script, tmp_path):
    root = write_tree(
        tmp_path,
        {
            "a": manifest(method="upu"),
            "b": manifest(method="kldce"),
            "c": manifest(method="nnpu", seed=1),
        },
    )
    report = aggregate_tree(aggregate_script, root)

    assert report["formal"] is True
    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["comparability_group"] == "spambase/native_2d/minibatch"
    assert group["dataset"] == "spambase"
    assert group["training_path"] == "native_2d"
    # Every method in the group, across both seeds.
    assert group["methods"] == ["kldce", "nnpu", "upu"]
    # One unit per (seed, c), each comparing only the methods that share it --
    # the gate compares seed for exact equality, so this split is not optional.
    assert [(unit["seed"], unit["c"]) for unit in group["units"]] == [(0, 0.5), (1, 0.5)]
    assert [unit["methods"] for unit in group["units"]] == [["kldce", "upu"], ["nnpu"]]
    assert all(unit["state"] == "comparable" for unit in group["units"])
    # The fairness fingerprint sits on the unit, not the group: it covers the
    # split, and every seed has its own.
    assert all(len(unit["fairness_sha256"]) == 64 for unit in group["units"])
    assert report["formal_ready"] is True


def test_edge_table_only_units_reach_the_gate(aggregate_script, tmp_path):
    """native_2d is most of the matrix; a gate that refuses it checks nothing."""
    root = write_tree(tmp_path, {"a": manifest(method="upu", training_path="native_2d")})
    report = aggregate_tree(aggregate_script, root)
    assert report["groups"][0]["training_path"] == "native_2d"
    assert report["refused"] == []


def test_edge_one_method_across_seeds_and_c_values_aggregates(aggregate_script, tmp_path):
    """A method appears once per (seed, c); the fairness spec is per method.

    ``partition_fair_leaderboard_runs`` keys a leaderboard on method and refuses
    duplicates, so building one spec per *manifest* reads a method that ran two
    seeds at two c values as four copies of itself.  A real pilot runs five
    seeds at three c values, so every group in the tree has this shape -- which
    is why the entry point produced no report against one.
    """
    root = write_tree(
        tmp_path,
        {
            f"{method}_s{seed}_c{c}": manifest(method=method, seed=seed, c=c)
            for method in ("upu", "nnpu")
            for seed in (0, 1)
            for c in (0.1, 0.5)
        },
    )
    report = aggregate_tree(aggregate_script, root)

    group = report["groups"][0]
    assert group["methods"] == ["nnpu", "upu"]
    assert [(unit["seed"], unit["c"]) for unit in group["units"]] == [
        (0, 0.1),
        (0, 0.5),
        (1, 0.1),
        (1, 0.5),
    ]
    assert all(unit["state"] == "comparable" for unit in group["units"])


def test_param_a_method_rerun_under_a_different_budget_is_refused(aggregate_script, tmp_path):
    """Units are gated one at a time, so drift between them needs its own check.

    Re-running the missing seeds after a budget change is routine, and it leaves
    a group whose units agree internally while disagreeing with each other.
    Nothing inside a unit compares them, so without a group-level comparison the
    two training budgets rank side by side and the report still calls itself
    ready.
    """
    root = write_tree(
        tmp_path,
        {
            "seed_0": manifest(method="nnpu", seed=0, epochs=200),
            "seed_1": manifest(method="nnpu", seed=1, epochs=150),
        },
    )
    with pytest.raises(ValueError, match="max_epochs"):
        aggregate_tree(aggregate_script, root)


def test_param_a_group_mixing_protocol_versions_is_refused(aggregate_script, tmp_path):
    """The same hazard one level up: a later seed re-run against a newer protocol."""
    root = write_tree(
        tmp_path,
        {
            "seed_0": manifest(method="nnpu", seed=0),
            "seed_1": manifest(method="nnpu", seed=1, protocol_version="survey-v1.3"),
        },
    )
    with pytest.raises(ValueError, match="protocol"):
        aggregate_tree(aggregate_script, root)


def test_param_a_method_rerun_under_a_different_learning_rate_is_refused(
    aggregate_script, tmp_path
):
    """The epoch cap is not the whole budget.

    A re-run can move anything a family defines, so the group-level comparison
    covers the whole dictionary.  It is compared per method, which is what keeps
    it from refusing the classical group's four families.
    """
    first = manifest(method="nnpu", seed=0)
    first["budget"]["learning_rate"] = 0.001
    second = manifest(method="nnpu", seed=1)
    second["budget"]["learning_rate"] = 0.1
    root = write_tree(tmp_path, {"seed_0": first, "seed_1": second})
    with pytest.raises(ValueError, match="budgets"):
        aggregate_tree(aggregate_script, root)


def test_param_cross_method_mismatch_is_refused(aggregate_script, tmp_path):
    """A shared split is the premise of comparing two methods at all."""
    root = write_tree(
        tmp_path,
        {
            "a": manifest(method="upu"),
            "b": manifest(method="kldce", split_marker="z"),
        },
    )
    with pytest.raises(ValueError, match="split_sha256"):
        aggregate_tree(aggregate_script, root)


def test_edge_diagnostic_mode_relaxes_eligibility_only(aggregate_script, tmp_path):
    """The opt-out must not become a way to skip the fairness checks."""
    root = write_tree(
        tmp_path,
        {
            "a": manifest(method="upu", formal_eligible=False, blockers=["collaborator_review"]),
            "b": manifest(method="kldce", formal_eligible=False, blockers=["collaborator_review"]),
        },
    )
    formal = aggregate_tree(aggregate_script, root)
    assert formal["formal_ready"] is False
    assert formal["groups"][0]["units"][0]["state"] == "blocked"
    assert formal["groups"][0]["units"][0]["blockers"] == ["collaborator_review"]

    diagnostic = aggregate_tree(aggregate_script, root, require_formal=False)
    assert diagnostic["formal"] is False
    assert diagnostic["groups"][0]["units"][0]["state"] == "comparable"

    # A real mismatch is still fatal in diagnostic mode.
    bad = write_tree(
        tmp_path / "bad",
        {"a": manifest(method="upu"), "b": manifest(method="kldce", split_marker="z")},
    )
    with pytest.raises(ValueError, match="split_sha256"):
        aggregate_tree(aggregate_script, bad, require_formal=False)


def test_edge_rejected_and_non_protocol_manifests_are_reported_not_aggregated(
    aggregate_script, tmp_path
):
    rejected = manifest(method="upu")
    rejected["execution_mode"] = "rejected_versioned_pilot"
    stripped = manifest(method="kldce")
    del stripped["execution_unit"]
    root = write_tree(tmp_path, {"a": manifest(method="nnpu"), "r": rejected, "s": stripped})

    report = aggregate_tree(aggregate_script, root)
    assert [group["methods"] for group in report["groups"]] == [["nnpu"]]
    assert sorted(item["reason"] for item in report["refused"]) == [
        "no_execution_unit",
        "rejected_versioned_pilot",
    ]
    assert all(item["path"].endswith("manifest.json") for item in report["refused"])


def test_param_a_non_runnable_execution_unit_is_refused(aggregate_script, tmp_path):
    """A row the protocol marks unrunnable is not a result to rank.

    The runner refuses to write such a manifest, so this guards hand-assembled
    trees -- but the matrix names four unrunnable rows, and one reaching a
    leaderboard is what the separation exists to prevent.
    """
    payload = manifest(method="kldce")
    payload["execution_unit"]["runnable"] = False
    root = write_tree(tmp_path, {"a": payload})
    report = aggregate_tree(aggregate_script, root)
    assert report["groups"] == []
    assert [item["reason"] for item in report["refused"]] == ["execution_unit_not_runnable"]

    # A row carrying no flag at all is not one to assume ran either.
    unflagged = manifest(method="kldce")
    del unflagged["execution_unit"]["runnable"]
    root = write_tree(tmp_path / "unflagged", {"a": unflagged})
    report = aggregate_tree(aggregate_script, root)
    assert report["groups"] == []
    assert [item["reason"] for item in report["refused"]] == ["execution_unit_not_runnable"]


def test_param_a_run_whose_candidates_all_failed_is_refused(aggregate_script, tmp_path):
    """A manifest with an empty selection is a failure record, not a result.

    The runner writes it and then re-raises when every candidate was excluded,
    so the artifact is there to explain the failure.  Ranking it would put a run
    that produced nothing beside runs that produced something, and nothing
    inside the manifest distinguishes the two.
    """
    payload = manifest(method="upu")
    payload["selection"] = {}
    payload["test_results"] = {}
    payload["failures"] = [{"stage": "train", "error": "convergence"}]
    root = write_tree(tmp_path, {"a": payload})
    report = aggregate_tree(aggregate_script, root)
    assert report["groups"] == []
    assert [item["reason"] for item in report["refused"]] == ["no_selected_candidate"]


def test_param_manifest_missing_representation_names_the_field(aggregate_script, tmp_path):
    """The gate indexes representation directly, so a KeyError would escape."""
    payload = manifest(method="upu")
    del payload["representation"]
    root = write_tree(tmp_path, {"a": payload})
    with pytest.raises(ValueError, match="representation"):
        aggregate_tree(aggregate_script, root)


def test_param_a_manifest_missing_a_read_field_names_it(aggregate_script, tmp_path):
    """Every field the entry point reads is named, not left to a bare KeyError."""
    payload = manifest(method="upu")
    del payload["candidate_runs"]
    root = write_tree(tmp_path, {"a": payload})
    with pytest.raises(ValueError, match="candidate_runs"):
        aggregate_tree(aggregate_script, root)

    # The grouping reads the dataset off the execution unit, one call after the
    # names checked above.
    payload = manifest(method="upu")
    del payload["execution_unit"]["dataset"]
    root = write_tree(tmp_path / "no_dataset", {"a": payload})
    with pytest.raises(ValueError, match="dataset"):
        aggregate_tree(aggregate_script, root)


def test_determ_repeated_aggregation_yields_identical_report(aggregate_script, tmp_path):
    root = write_tree(
        tmp_path,
        {
            "b": manifest(method="nnpu", seed=1),
            "a": manifest(method="upu"),
            "c": manifest(method="kldce"),
        },
    )
    first = aggregate_tree(aggregate_script, root)
    second = aggregate_tree(aggregate_script, root)
    assert first == second
    assert first["groups"][0]["units"][0]["methods"] == ["kldce", "upu"]
