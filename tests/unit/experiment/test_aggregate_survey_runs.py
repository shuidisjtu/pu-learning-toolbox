# ruff: noqa: N803, N806, F811
"""The aggregation entry point forces the leaderboard-separation gates.

``validate_comparable_manifests`` and ``partition_fair_leaderboard_runs`` both
existed with tests and no production caller, so path separation was a library
function plus a manifest field rather than something any real path went
through.  These tests pin the entry point that changes that -- and the three
ways a naive version of it would be wrong: grouping by dataset instead of
comparability group, feeding a partition straight into the comparability gate
(it spans every seed), and deriving the seed list from the files found.

Scope is deliberately the wiring only.  The expected-unit completeness grid,
anomaly units and reproduction-field auditing belong to P2.2.
"""

import json

import pytest
from _aggregate_script_helpers import aggregate_script, manifest, write_tree  # noqa: F401

from pu_toolbox.experiment.survey_protocol import load_protocol

pytestmark = pytest.mark.unit


def _aggregate(script, root, *, require_formal=True):
    return script.aggregate(
        script.discover_manifests(root), protocol=load_protocol(), require_formal=require_formal
    )


def test_basic_comparable_group_partitions_by_path_and_unit(aggregate_script, tmp_path):
    root = write_tree(
        tmp_path,
        {
            "a": manifest(method="upu"),
            "b": manifest(method="kldce"),
            "c": manifest(method="nnpu", seed=1),
        },
    )
    report = _aggregate(aggregate_script, root)

    assert report["formal"] is True
    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["comparability_group"] == "spambase/native_2d/minibatch"
    assert group["dataset"] == "spambase"
    assert group["training_path"] == "native_2d"
    assert len(group["fairness_sha256"]) == 64
    # The partition payload spans every seed, so its method list is the union.
    assert group["methods"] == ["kldce", "nnpu", "upu"]
    # One unit per (seed, c), each comparing only the methods that share it --
    # the gate compares seed for exact equality, so this split is not optional.
    assert [(unit["seed"], unit["c"]) for unit in group["units"]] == [(0, 0.5), (1, 0.5)]
    assert [unit["methods"] for unit in group["units"]] == [["kldce", "upu"], ["nnpu"]]
    assert all(unit["state"] == "comparable" for unit in group["units"])
    assert report["formal_ready"] is True


def test_edge_table_only_units_reach_the_gate(aggregate_script, tmp_path):
    """native_2d is most of the matrix; a gate that refuses it checks nothing."""
    root = write_tree(tmp_path, {"a": manifest(method="upu", training_path="native_2d")})
    report = _aggregate(aggregate_script, root)
    assert report["groups"][0]["training_path"] == "native_2d"
    assert report["refused"] == []


def test_edge_budget_families_without_epochs_or_batch_size_still_aggregate(
    aggregate_script, tmp_path
):
    """Four of the seven budget families cap no epochs and define no batch size.

    A closed-form solve has neither, so the gate's positive-int fields have to
    come from somewhere other than the budget -- found by running the entry
    point against real artifacts, where synthetic manifests had always
    supplied both.
    """
    root = write_tree(
        tmp_path,
        {
            "a": manifest(method="upu", budget="closed_form", epochs=None, batch_size=None),
            "b": manifest(method="kldce", budget="closed_form", epochs=None, batch_size=None),
        },
    )
    report = _aggregate(aggregate_script, root)
    assert report["groups"][0]["comparability_group"] == "spambase/native_2d/closed_form"
    assert report["groups"][0]["units"][0]["methods"] == ["kldce", "upu"]


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
        _aggregate(aggregate_script, root)


def test_edge_diagnostic_mode_relaxes_eligibility_only(aggregate_script, tmp_path):
    """The opt-out must not become a way to skip the fairness checks."""
    root = write_tree(
        tmp_path,
        {
            "a": manifest(method="upu", formal_eligible=False, blockers=["collaborator_review"]),
            "b": manifest(method="kldce", formal_eligible=False, blockers=["collaborator_review"]),
        },
    )
    formal = _aggregate(aggregate_script, root)
    assert formal["formal_ready"] is False
    assert formal["groups"][0]["units"][0]["state"] == "blocked"
    assert formal["groups"][0]["units"][0]["blockers"] == ["collaborator_review"]

    diagnostic = _aggregate(aggregate_script, root, require_formal=False)
    assert diagnostic["formal"] is False
    assert diagnostic["groups"][0]["units"][0]["state"] == "comparable"

    # A real mismatch is still fatal in diagnostic mode.
    bad = write_tree(
        tmp_path / "bad",
        {"a": manifest(method="upu"), "b": manifest(method="kldce", split_marker="z")},
    )
    with pytest.raises(ValueError, match="split_sha256"):
        _aggregate(aggregate_script, bad, require_formal=False)


def test_edge_rejected_and_non_protocol_manifests_are_reported_not_aggregated(
    aggregate_script, tmp_path
):
    rejected = manifest(method="upu")
    rejected["execution_mode"] = "rejected_versioned_pilot"
    stripped = manifest(method="kldce")
    del stripped["execution_unit"]
    root = write_tree(tmp_path, {"a": manifest(method="nnpu"), "r": rejected, "s": stripped})

    report = _aggregate(aggregate_script, root)
    assert [group["methods"] for group in report["groups"]] == [["nnpu"]]
    assert sorted(item["reason"] for item in report["refused"]) == [
        "no_execution_unit",
        "rejected_versioned_pilot",
    ]
    assert all(item["path"].endswith("manifest.json") for item in report["refused"])


def test_param_manifest_missing_representation_names_the_field(aggregate_script, tmp_path):
    """The gate indexes representation directly, so a KeyError would escape."""
    payload = manifest(method="upu")
    del payload["representation"]
    root = write_tree(tmp_path, {"a": payload})
    with pytest.raises(ValueError, match="representation"):
        _aggregate(aggregate_script, root)


def test_determ_repeated_aggregation_yields_identical_report(aggregate_script, tmp_path):
    root = write_tree(
        tmp_path,
        {
            "b": manifest(method="nnpu", seed=1),
            "a": manifest(method="upu"),
            "c": manifest(method="kldce"),
        },
    )
    first = _aggregate(aggregate_script, root)
    second = _aggregate(aggregate_script, root)
    assert first == second
    assert first["groups"][0]["units"][0]["methods"] == ["kldce", "upu"]


def test_basic_cli_refuses_a_blocked_pilot_and_explains(aggregate_script, tmp_path, capsys):
    """Blocked is a first-class outcome, not a crash and not a silent pass."""
    root = write_tree(
        tmp_path,
        {"a": manifest(method="upu", formal_eligible=False, blockers=["P2.0c_acceptance"])},
    )
    assert aggregate_script.main([str(root)]) == 1
    printed = capsys.readouterr().out
    assert "P2.0c_acceptance" in printed

    assert aggregate_script.main([str(root), "--diagnostic"]) == 0
    diagnostic = capsys.readouterr().out
    assert "NON-FORMAL" in diagnostic


def test_param_cli_rejects_a_missing_directory(aggregate_script, tmp_path, capsys):
    assert aggregate_script.main([str(tmp_path / "nope")]) == 1
    assert "error:" in capsys.readouterr().err


def test_edge_cli_reports_an_empty_tree_as_an_error(aggregate_script, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert aggregate_script.main([str(empty)]) == 1
    assert "no manifests" in capsys.readouterr().err


def test_param_script_is_reachable_as_json(aggregate_script, tmp_path, capsys):
    root = write_tree(tmp_path, {"a": manifest(method="upu")})
    assert aggregate_script.main([str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["groups"][0]["dataset"] == "spambase"
