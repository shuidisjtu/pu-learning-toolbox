"""A04: the runs a batch delivered are the runs its plan said it would.

The plan is the batch's own statement of which ``(dataset, method, path, mechanism,
c, seed, view)`` runs it intended; the manifests are what exists.  The check compares
the two as *sets of parsed runs*, because the same plan reached the repository twice
with different bytes (re-serialized) and identical content -- a file hash would call
that a mismatch, and a count would call a swapped run a match.

Honest outcomes are three, not two: a batch with no plan configured is ``not_run``
(the identity was not checked), and so is a tree where only some batches have one --
a verified half must not read as a verified whole.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest
from _survey_summary_helpers import (
    FROZEN_PROTOCOL_SHA256,
    SCRIPTS_DIR,
    config_for,
    manifest,
)
from _survey_summary_helpers import entry as _entry
from _survey_summary_helpers import write_plan as _write_plan

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
checks = importlib.import_module("audit_survey_batches")

pytestmark = pytest.mark.unit


@pytest.fixture
def audit_cli():
    return checks


def _grid():
    """Three runs that differ in seed, c and mechanism, plus the c-independent oracle."""
    return [
        manifest(seed=0),
        manifest(seed=1, c=0.5),
        manifest(seed=2, mechanism="sar_lbe_a"),
        manifest(seed=0, method="pn_oracle", c_independent=True),
    ]


def _entries(payloads, *, batch="B1"):
    return [
        _entry(p, batch=batch, name=f"{batch}/{i}/manifest.json") for i, p in enumerate(payloads)
    ]


def _config(tmp_path, plans):
    """A config over batches named in *plans*, each pointing at its plan path (or None)."""
    config = config_for({name: (tmp_path / name, 0) for name in plans})
    for batch in config["batches"]:
        if plans[batch["name"]] is not None:
            batch["plan"] = str(plans[batch["name"]])
    return config


def test_basic_a04_passes_when_the_manifests_are_exactly_the_planned_runs(tmp_path):
    payloads = _grid()
    plan = _write_plan(tmp_path / "plan.json", payloads)

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan}),
        _entries(payloads),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["check_id"] == "A04"
    assert check["result"] == "pass"
    assert check["observed"]["verified_batches"] == {"B1": 4}


def test_basic_a04_the_oracle_row_matches_a_plan_row_without_mechanism_or_c(tmp_path):
    # The plan writes the oracle as mechanism=None, c_token=None; the manifest records
    # `pn_oracle` and a nominal c.  Reading either literally would orphan the oracle.
    payloads = [manifest(seed=seed, method="pn_oracle", c_independent=True) for seed in range(3)]
    plan = _write_plan(tmp_path / "plan.json", payloads)

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan}),
        _entries(payloads),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "pass"


def test_edge_a04_a_planned_run_with_no_manifest_is_named(tmp_path):
    payloads = _grid()
    plan = _write_plan(tmp_path / "plan.json", payloads)

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan}),
        _entries(payloads[:-1]),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "fail"
    assert check["severity"] == "error"
    assert any(
        "planned but not delivered" in item and "pn_oracle" in item for item in check["evidence"]
    )


def test_edge_a04_a_manifest_outside_the_plan_is_named_with_its_path(tmp_path):
    payloads = _grid()
    plan = _write_plan(tmp_path / "plan.json", payloads[:-1])

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan}),
        _entries(payloads),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "fail"
    assert any(
        "delivered but not planned" in item and str(Path("B1/3/manifest.json")) in item
        for item in check["evidence"]
    )


def test_edge_a04_a_swapped_run_fails_even_though_the_counts_agree(tmp_path):
    payloads = _grid()
    plan = _write_plan(tmp_path / "plan.json", payloads)
    delivered = payloads[:-1] + [manifest(seed=4, method="pn_oracle", c_independent=True)]

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan}),
        _entries(delivered),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "fail"
    assert len(check["evidence"]) == 2


def test_param_a04_a_plan_written_for_another_protocol_is_an_error(tmp_path):
    payloads = _grid()
    plan = _write_plan(tmp_path / "plan.json", payloads, sha256="0" * 64)

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan}),
        _entries(payloads),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "fail"
    assert any("protocol" in item for item in check["evidence"])
    assert "superseded_protocol" in check["reasons"]


def test_param_a04_a_plan_that_repeats_a_run_is_an_error(tmp_path):
    payloads = _grid()
    plan = _write_plan(tmp_path / "plan.json", payloads)
    document = json.loads(plan.read_text(encoding="utf-8"))
    document["runs"].append(document["runs"][0])
    document["totals"]["planned"] += 1
    plan.write_text(json.dumps(document), encoding="utf-8")

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan}),
        _entries(payloads),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "fail"
    assert any("repeats" in item for item in check["evidence"])


def test_param_a04_a_plan_whose_total_disagrees_with_its_runs_is_an_error(tmp_path):
    payloads = _grid()
    plan = _write_plan(tmp_path / "plan.json", payloads, totals={"planned": 99})

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan}),
        _entries(payloads),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "fail"
    assert any("totals.planned" in item for item in check["evidence"])


def test_determ_a04_the_same_plan_reserialized_still_matches(tmp_path):
    payloads = _grid()
    plan = _write_plan(tmp_path / "plan.json", payloads)
    document = json.loads(plan.read_text(encoding="utf-8"))
    document["runs"].reverse()
    plan.write_text(json.dumps(document, indent=4, sort_keys=True), encoding="utf-8")

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan}),
        _entries(payloads),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "pass"


def test_basic_a04_is_not_run_when_no_batch_names_a_plan(tmp_path):
    payloads = _grid()

    check = checks.check_plan(
        _config(tmp_path, {"B1": None}),
        _entries(payloads),
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "not_run"
    assert check["observed"]["verified_batches"] == {}


def test_edge_a04_a_half_covered_tree_is_not_reported_as_verified(tmp_path):
    first, second = _grid()[:2], _grid()[2:]
    plan = _write_plan(tmp_path / "plan.json", first)
    entries = _entries(first, batch="B1") + _entries(second, batch="B2")

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan, "B2": None}),
        entries,
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "not_run"
    assert check["observed"]["verified_batches"] == {"B1": 2}
    assert check["observed"]["unverified_batches"] == ["B2"]


def test_param_a04_a_failure_in_one_batch_outranks_a_gap_in_another(tmp_path):
    first, second = _grid()[:2], _grid()[2:]
    plan = _write_plan(tmp_path / "plan.json", first[:1])
    entries = _entries(first, batch="B1") + _entries(second, batch="B2")

    check = checks.check_plan(
        _config(tmp_path, {"B1": plan, "B2": None}),
        entries,
        frozen_sha256=FROZEN_PROTOCOL_SHA256,
    )

    assert check["result"] == "fail"


def test_param_a04_an_unreadable_plan_is_an_error_not_a_pass(tmp_path):
    payloads = _grid()
    broken = tmp_path / "plan.json"
    broken.write_text("{ not json", encoding="utf-8")

    for plan in (broken, tmp_path / "absent.json"):
        check = checks.check_plan(
            _config(tmp_path, {"B1": plan}),
            _entries(payloads),
            frozen_sha256=FROZEN_PROTOCOL_SHA256,
        )

        assert check["result"] == "fail"
        assert any("cannot read the plan" in item for item in check["evidence"])
