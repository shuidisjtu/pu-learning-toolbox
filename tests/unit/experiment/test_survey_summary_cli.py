# ruff: noqa: F811

"""The two entry points end to end: what a run of them produces, and must not.

These exercise the wiring the unit suites cannot: that the gate is called once
per group so a refusal carries its members, that a whitelisted parent holding a
flattened copy is refused rather than double-counted, and that the report a
reviewer reads is written where it was asked for and says the same thing twice.

Nothing here asserts a number the aggregation library does not already own; the
subject is the pipeline.
"""

import json

import pytest
from _survey_summary_helpers import (  # noqa: F401
    audit_cli,
    config_for,
    load_protocol,
    manifest,
    summary_cli,
    write_config,
    write_tree,
)

pytestmark = pytest.mark.unit


def _five_seeds(**overrides):
    return {f"seed_{seed}": manifest(seed=seed, **overrides) for seed in range(5)}


def _protocol_grid(**overrides):
    """One SCAR row over the c grid the protocol designed, five seeds each.

    A five-seed single-c stub is what the audit calls an incomplete row -- 20 of the
    30 designed units are absent -- so the one test that asserts a clean verdict
    needs the grid, not just the seeds.
    """
    # The protocol stores the tokens as strings (they are report keys); the fixture
    # builder takes the number, as the runner's own config does.
    c_tokens = [float(token) for token in load_protocol()["c_tokens"]["scar"]]
    return {
        f"c_{c}_seed_{seed}": manifest(c=c, seed=seed, **overrides)
        for c in c_tokens
        for seed in range(5)
    }


def _run_summary(summary_cli, tmp_path, tree, *, expected=5, extra_config=None):
    config = config_for({"B1": (tree, expected)})
    if extra_config:
        config.update(extra_config)
    config_path = write_config(tmp_path / "roots.json", config)
    out_dir = tmp_path / "out"
    code = summary_cli.main(["--config", str(config_path), "--out-dir", str(out_dir)])
    report = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    return code, report, out_dir


def test_basic_a_complete_tree_yields_one_row_per_selection_protocol(summary_cli, tmp_path):
    grid = _protocol_grid()
    tree = write_tree(tmp_path / "b1", grid)

    code, report, _ = _run_summary(summary_cli, tmp_path, tree, expected=len(grid))

    # The one clean verdict in this file, and it takes the designed c grid to earn
    # it: over a five-seed single-c stub the audit reports 20 of 30 units absent,
    # which is a finding the exit code is supposed to carry.
    assert code == 0
    protocols = sorted({row["row_key"]["selection_protocol"] for row in report["rows"]})
    assert protocols == ["OA", "PA"]
    # Two vocabularies, pinned together so neither drifts: the report row keeps the
    # manifest's own dictionary key, and the selector handed to the pre-registered
    # matrix is spelled the way the matrix reads.
    assert sorted({row["result_identity"]["selection_protocol"] for row in report["rows"]}) == [
        "oa",
        "pa",
    ]
    assert report["coverage"]["manifests"] == len(grid)
    assert report["coverage"]["not_reproducible"] == 0


def test_basic_the_row_mean_is_over_the_five_seeds_not_over_the_runs(summary_cli, tmp_path):
    seeds = {f"seed_{seed}": manifest(seed=seed, accuracy=0.5 + 0.1 * seed) for seed in range(5)}
    tree = write_tree(tmp_path / "b1", seeds)

    _, report, _ = _run_summary(summary_cli, tmp_path, tree)

    row = next(r for r in report["rows"] if r["row_key"]["selection_protocol"] == "OA")
    assert row["metric"]["n_observed"] == 5
    assert row["metric"]["n_expected"] == 5
    assert row["metric"]["mean"] == pytest.approx((0.5 + 0.6 + 0.7 + 0.8 + 0.9) / 5)
    assert row["status"] == "formal"


def test_basic_a_missing_seed_makes_the_row_partial_and_names_it(summary_cli, tmp_path):
    seeds = {f"seed_{seed}": manifest(seed=seed) for seed in (0, 1, 3, 4)}
    tree = write_tree(tmp_path / "b1", seeds)

    _, report, _ = _run_summary(summary_cli, tmp_path, tree, expected=5)

    row = next(r for r in report["rows"] if r["row_key"]["selection_protocol"] == "OA")
    assert row["metric"]["missing_seeds"] == [2]
    assert row["status"] == "partial"
    assert row["reasons"] == ["missing_seed"]
    # The mean is over the four that ran; nothing was invented for the fifth.
    assert row["metric"]["n_observed"] == 4


def test_basic_a_manifest_without_environment_identity_is_not_reproducible(summary_cli, tmp_path):
    seeds = _five_seeds()
    seeds["seed_4"] = manifest(seed=4, environment=False)
    tree = write_tree(tmp_path / "b1", seeds)

    _, report, _ = _run_summary(summary_cli, tmp_path, tree, expected=5)

    assert report["coverage"]["not_reproducible"] == 1
    # It still contributes nothing to the row: the mean is the four reproducible runs.
    row = next(r for r in report["rows"] if r["row_key"]["selection_protocol"] == "OA")
    assert row["metric"]["missing_seeds"] == [4]


def test_basic_a_root_holding_a_working_copy_is_refused_not_double_counted(summary_cli, tmp_path):
    # The shape R8 warns about: a batch tree and a flattened copy of it under one
    # whitelisted root.  Counting both would double every unit.
    root = write_tree(tmp_path / "parent", _five_seeds())
    write_tree(root / "B3ab_merged", {"copy": manifest()})

    code, report, _ = _run_summary(summary_cli, tmp_path, root, expected=5)

    assert code == 1
    assert "working-copy" in report.get("error", "")


def test_basic_a_group_the_gate_refuses_still_reports_numbers_but_not_formal(summary_cli, tmp_path):
    # Two methods in one unit disagreeing about the split: the fairness gate
    # refuses the unit, and the finding belongs to the group, not to either file.
    members = {}
    for seed in range(5):
        members[f"np_{seed}"] = manifest(method="nnpu", seed=seed, split_marker="a")
        members[f"lb_{seed}"] = manifest(method="lbe", seed=seed, split_marker="b")
    tree = write_tree(tmp_path / "b1", members)

    _, report, _ = _run_summary(summary_cli, tmp_path, tree, expected=10)

    refused = [check for check in report["checks"] if check["check_id"] == "A08"]
    assert refused
    assert refused[0]["scope"] == "group"
    # Every manifest of the group is a member: the gate named the rule, not a file.
    assert len(refused[0]["members"]) == 10
    assert "split_mismatch" in refused[0]["reasons"]
    assert all(row["status"] == "partial" for row in report["rows"])


def test_edge_a_missing_batch_root_is_a_coverage_failure_not_a_silent_zero(audit_cli, tmp_path):
    config = config_for({"B1": (tmp_path / "nope", 215)})
    config_path = write_config(tmp_path / "roots.json", config)

    code = audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / "audit")])
    report = json.loads((tmp_path / "audit" / "audit.json").read_text(encoding="utf-8"))

    assert code == 1
    coverage = next(c for c in report["checks"] if c["check_id"] == "A02")
    assert coverage["result"] == "fail"
    assert "missing roots" in coverage["message"]


def test_edge_an_empty_tree_still_writes_a_readable_report(audit_cli, tmp_path):
    (tmp_path / "b1").mkdir()
    config = config_for({"B1": (tmp_path / "b1", 0)})
    config_path = write_config(tmp_path / "roots.json", config)

    code = audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / "audit")])

    assert code == 0
    assert (tmp_path / "audit" / "audit.json").exists()
    assert (tmp_path / "audit" / "audit.md").exists()


def test_param_a_config_with_an_unknown_schema_version_is_refused(audit_cli, tmp_path):
    config_path = write_config(
        tmp_path / "roots.json",
        {"schema_version": "something-else", "batches": []},
    )

    code = audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / "audit")])

    assert code == 1
    assert not (tmp_path / "audit" / "audit.json").exists()


def test_param_a_config_without_an_expected_count_is_refused(audit_cli, tmp_path):
    config_path = write_config(
        tmp_path / "roots.json",
        {
            "schema_version": "survey-batch-roots-1",
            "batches": [{"name": "B1", "root": str(tmp_path)}],
        },
    )

    code = audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / "audit")])

    assert code == 1


def test_param_a_config_repeating_a_batch_name_is_refused(audit_cli, tmp_path):
    config_path = write_config(
        tmp_path / "roots.json",
        {
            "schema_version": "survey-batch-roots-1",
            "batches": [
                {"name": "B1", "root": str(tmp_path), "expected_manifests": 1},
                {"name": "B1", "root": str(tmp_path), "expected_manifests": 1},
            ],
        },
    )

    assert audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / "a")]) == 1


def test_determ_two_summary_runs_agree_except_for_the_timestamp(summary_cli, tmp_path):
    tree = write_tree(tmp_path / "b1", _five_seeds())
    config_path = write_config(tmp_path / "roots.json", config_for({"B1": (tree, 5)}))

    first_dir, second_dir = tmp_path / "out1", tmp_path / "out2"
    for out_dir in (first_dir, second_dir):
        summary_cli.main(["--config", str(config_path), "--out-dir", str(out_dir)])

    first = json.loads((first_dir / "summary.json").read_text(encoding="utf-8"))
    second = json.loads((second_dir / "summary.json").read_text(encoding="utf-8"))
    assert first["generated_at"] and second["generated_at"]
    first.pop("generated_at"), second.pop("generated_at")
    assert first == second
    # The CSV and Markdown carry no stamp, so they must be byte-identical.
    assert (first_dir / "summary.csv").read_bytes() == (second_dir / "summary.csv").read_bytes()
    assert (first_dir / "summary.md").read_bytes() == (second_dir / "summary.md").read_bytes()


def test_determ_the_summary_writes_every_artifact_it_promises(summary_cli, tmp_path):
    tree = write_tree(tmp_path / "b1", _five_seeds())

    _, _, out_dir = _run_summary(summary_cli, tmp_path, tree)

    assert sorted(path.name for path in out_dir.iterdir()) == [
        "coverage.csv",
        "refusals.json",
        "summary.csv",
        "summary.json",
        "summary.md",
    ]
