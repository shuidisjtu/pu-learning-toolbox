"""A05: the run log says every delivered run completed, and names the same runs.

The logs of the P2.1 batches carry one ``[method] seed=N -> <path>/manifest.json`` line
per run and a closing ``completed X of Y run(s) in <dataset>; Z still pending`` line.
They carry **no exit code**, so the check reads what is there and says what is not:
the exit code is reported as unverified and never decides the result either way.

A run is matched to its manifest by the path *under the batch root*, because the
absolute prefix is the execution host's and differs from wherever the tree was copied.
A batch with no log is ``not_run``, and so is a tree where only some batches have one --
a verified half must not read as a verified whole.
"""

import importlib
import sys

import pytest
from _survey_summary_helpers import (
    SCRIPTS_DIR,
    config_for,
    manifest,
    write_tree,
)
from _survey_summary_helpers import entry as _entry

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
checks = importlib.import_module("audit_survey_batches")

pytestmark = pytest.mark.unit

_COMPLETED = "completed {done} of {total} run(s) in spambase; {pending} still pending"


def _batch(tmp_path, *, name="B1", runs=3):
    """A written tree of *runs* manifests, their entries, and the config over them."""
    payloads = {f"spambase/nnpu/seed_{i}": manifest(seed=i) for i in range(runs)}
    root = write_tree(tmp_path / name, payloads)
    entries = [
        _entry(payload, batch=name, name=str(root / rel / "manifest.json"))
        for rel, payload in payloads.items()
    ]
    return root, list(payloads), entries


def _log(tmp_path, relatives, *, done=None, total=None, pending=0, extra=(), prefix="/remote/B1"):
    """A log with one arrow line per relative path and a closing completion line."""
    lines = []
    for relative in relatives:
        lines.append(f"[nnpu] seed=0 -> {prefix}/{relative}/manifest.json")
        lines.append("  {'OA': {'accuracy': 0.9, 'auc': 0.9}}")
    lines.extend(extra)
    count = len(relatives)
    lines.append(
        _COMPLETED.format(
            done=count if done is None else done,
            total=count if total is None else total,
            pending=pending,
        )
    )
    path = tmp_path / "run.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _config(root, log, *, name="B1", runs=3):
    config = config_for({name: (root, runs)})
    if log is not None:
        config["batches"][0]["run_log"] = str(log)
    return config


def test_basic_a05_passes_when_the_log_completes_exactly_the_delivered_runs(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    log = _log(tmp_path, relatives)

    check = checks.check_completion(_config(root, log), entries)

    assert check["check_id"] == "A05"
    assert check["result"] == "pass"
    assert check["observed"]["verified_batches"] == {"B1": 3}


def test_basic_a05_never_claims_the_exit_code_was_verified(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    log = _log(tmp_path, relatives)

    check = checks.check_completion(_config(root, log), entries)

    assert check["observed"]["exit_code"] == "unverified"
    assert "exit code" in check["message"]


def test_edge_a05_a_completion_line_short_of_its_total_is_an_error(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    log = _log(tmp_path, relatives, done=2, total=3, pending=1)

    check = checks.check_completion(_config(root, log), entries)

    assert check["result"] == "fail"
    assert any("2 of 3" in item for item in check["evidence"])


def test_edge_a05_pending_runs_fail_even_when_the_counts_agree(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    log = _log(tmp_path, relatives, pending=1)

    check = checks.check_completion(_config(root, log), entries)

    assert check["result"] == "fail"
    assert any("still pending" in item for item in check["evidence"])


def test_edge_a05_a_total_that_is_not_the_delivered_count_is_an_error(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    log = _log(tmp_path, relatives, done=5, total=5)

    check = checks.check_completion(_config(root, log), entries)

    assert check["result"] == "fail"
    assert any("3 manifest" in item for item in check["evidence"])


def test_edge_a05_a_run_logged_twice_is_an_error(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    log = _log(tmp_path, relatives + relatives[:1], done=3, total=3)

    check = checks.check_completion(_config(root, log), entries)

    assert check["result"] == "fail"
    assert any("more than once" in item for item in check["evidence"])


def test_edge_a05_a_logged_run_with_no_manifest_and_a_manifest_never_logged_are_named(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    log = _log(tmp_path, relatives[:2] + ["spambase/nnpu/seed_9"], done=3, total=3)

    check = checks.check_completion(_config(root, log), entries)

    assert check["result"] == "fail"
    assert any(
        "logged but not delivered" in item and "seed_9" in item for item in check["evidence"]
    )
    assert any(
        "delivered but not logged" in item and "seed_2" in item for item in check["evidence"]
    )


def test_param_a05_a_log_without_a_completion_line_is_an_error(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    log = tmp_path / "run.log"
    log.write_text(
        "".join(f"[nnpu] seed=0 -> /remote/B1/{r}/manifest.json\n" for r in relatives),
        encoding="utf-8",
    )

    check = checks.check_completion(_config(root, log), entries)

    assert check["result"] == "fail"
    assert any("no completion line" in item for item in check["evidence"])


def test_param_a05_an_unreadable_log_is_an_error_not_a_pass(tmp_path):
    root, _, entries = _batch(tmp_path)

    check = checks.check_completion(_config(root, tmp_path / "absent.log"), entries)

    assert check["result"] == "fail"
    assert any("cannot read the log" in item for item in check["evidence"])


def test_basic_a05_is_not_run_when_no_batch_names_a_log(tmp_path):
    root, _, entries = _batch(tmp_path)

    check = checks.check_completion(_config(root, None), entries)

    assert check["result"] == "not_run"
    assert check["observed"]["verified_batches"] == {}


def test_edge_a05_a_half_covered_tree_is_not_reported_as_verified(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    other_root, _, other_entries = _batch(tmp_path / "other", name="B2")
    log = _log(tmp_path, relatives)
    config = config_for({"B1": (root, 3), "B2": (other_root, 3)})
    config["batches"][0]["run_log"] = str(log)

    check = checks.check_completion(config, entries + other_entries)

    assert check["result"] == "not_run"
    assert check["observed"]["verified_batches"] == {"B1": 3}
    assert check["observed"]["unverified_batches"] == ["B2"]


def test_param_a05_a_failure_in_one_batch_outranks_a_gap_in_another(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    other_root, _, other_entries = _batch(tmp_path / "other", name="B2")
    log = _log(tmp_path, relatives, pending=2)
    config = config_for({"B1": (root, 3), "B2": (other_root, 3)})
    config["batches"][0]["run_log"] = str(log)

    check = checks.check_completion(config, entries + other_entries)

    assert check["result"] == "fail"


def test_determ_a05_the_same_inputs_give_the_same_check(tmp_path):
    root, relatives, entries = _batch(tmp_path)
    log = _log(tmp_path, relatives, extra=["WARNING noise"])
    config = _config(root, log)

    assert checks.check_completion(config, entries) == checks.check_completion(config, entries)
