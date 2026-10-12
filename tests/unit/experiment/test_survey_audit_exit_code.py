"""A05: an optional per-batch ``exit_code_file`` turns the exit code from unverified into checked.

The P2.1 hosts wrote ``<batch>_exit_code.txt`` beside the run log: one line holding the
runner's exit status.  Only ``0`` is success.  A configured file that is empty, not a
number or non-zero is a ``fail`` -- a file that was named but cannot vouch must not read
as a pass -- and a batch without such a file stays "unverified" rather than "verified".
"""

import importlib
import sys

import pytest
from _survey_summary_helpers import SCRIPTS_DIR, config_for, manifest, write_tree
from _survey_summary_helpers import entry as _entry

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
checks = importlib.import_module("audit_survey_batches")

pytestmark = pytest.mark.unit


def _batch(tmp_path, *, name="B1", runs=2, exit_text="0\n"):
    """A tree, its entries, a matching run log, and an exit-code file holding *exit_text*."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    payloads = {f"spambase/nnpu/seed_{i}": manifest(seed=i) for i in range(runs)}
    root = write_tree(tmp_path / name, payloads)
    entries = [
        _entry(payload, batch=name, name=str(root / rel / "manifest.json"))
        for rel, payload in payloads.items()
    ]
    lines = [f"[nnpu] seed=0 -> /remote/{name}/{rel}/manifest.json" for rel in payloads]
    lines.append(f"completed {runs} of {runs} run(s) in spambase; 0 still pending")
    log = tmp_path / f"{name}_run.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    code = tmp_path / f"{name}_exit_code.txt"
    if exit_text is not None:
        code.write_text(exit_text, encoding="utf-8")
    return root, entries, log, code


def _config(batches):
    """*batches* maps name -> (root, runs, log, exit-code file or None)."""
    config = config_for({name: (root, runs) for name, (root, runs, _, _) in batches.items()})
    for item in config["batches"]:
        _, _, log, code = batches[item["name"]]
        item["run_log"] = str(log)
        if code is not None:
            item["exit_code_file"] = str(code)
    return config


def _check(tmp_path, **kwargs):
    root, entries, log, code = _batch(tmp_path, **kwargs)
    return checks.check_completion(_config({"B1": (root, 2, log, code)}), entries)


def test_basic_a05_a_zero_exit_code_file_verifies_the_exit_code(tmp_path):
    check = _check(tmp_path)

    assert check["result"] == "pass"
    assert check["observed"]["exit_code"] == "verified"
    assert check["observed"]["exit_code_unverified_batches"] == []


def test_basic_a05_without_an_exit_code_file_the_exit_code_stays_unverified(tmp_path):
    root, entries, log, _ = _batch(tmp_path)

    check = checks.check_completion(_config({"B1": (root, 2, log, None)}), entries)

    assert check["result"] == "pass"
    assert check["observed"]["exit_code"] == "unverified"
    assert check["observed"]["exit_code_unverified_batches"] == ["B1"]


@pytest.mark.parametrize("text", ["1\n", "137\n", "-9\n"])
def test_edge_a05_a_non_zero_exit_code_is_an_error(tmp_path, text):
    check = _check(tmp_path, exit_text=text)

    assert check["result"] == "fail"
    assert any("exit code" in item and text.strip() in item for item in check["evidence"])


@pytest.mark.parametrize("text", ["", "\n", "ok\n", "0 1\n", "0\n0\n"])
def test_edge_a05_an_exit_code_file_that_is_not_one_integer_is_an_error(tmp_path, text):
    check = _check(tmp_path, exit_text=text)

    assert check["result"] == "fail"
    assert any("exit code" in item for item in check["evidence"])


def test_edge_a05_a_configured_exit_code_file_that_cannot_be_read_is_an_error(tmp_path):
    root, entries, log, code = _batch(tmp_path, exit_text=None)

    check = checks.check_completion(_config({"B1": (root, 2, log, code)}), entries)

    assert check["result"] == "fail"
    assert any("cannot read the exit code file" in item for item in check["evidence"])


def test_edge_a05_a_zero_exit_code_does_not_excuse_a_log_disagreement(tmp_path):
    root, entries, log, code = _batch(tmp_path)
    log.write_text("completed 2 of 2 run(s) in spambase; 1 still pending\n", encoding="utf-8")

    check = checks.check_completion(_config({"B1": (root, 2, log, code)}), entries)

    assert check["result"] == "fail"
    assert check["observed"]["exit_code"] == "verified"


def test_param_a05_a_mixed_tree_reports_partial_and_names_the_unverified_batch(tmp_path):
    first = _batch(tmp_path / "one", name="B1")
    second = _batch(tmp_path / "two", name="B2", exit_text=None)
    config = _config(
        {
            "B1": (first[0], 2, first[2], first[3]),
            "B2": (second[0], 2, second[2], None),
        }
    )

    check = checks.check_completion(config, first[1] + second[1])

    assert check["result"] == "pass"
    assert check["observed"]["exit_code"] == "partial"
    assert check["observed"]["exit_code_unverified_batches"] == ["B2"]
    assert "B2" in check["message"]


def test_determ_a05_the_same_exit_code_inputs_give_the_same_check(tmp_path):
    root, entries, log, code = _batch(tmp_path)
    config = _config({"B1": (root, 2, log, code)})

    assert checks.check_completion(config, entries) == checks.check_completion(config, entries)
