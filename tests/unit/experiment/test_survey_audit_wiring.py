"""The audit entry point reads its optional A04 inputs from the batch-root config.
``plan`` paths are resolved beside the config file, so a config and its plans move
together; the CLI then runs A04 end to end.  Check semantics live in
``test_survey_audit_plan.py``.
"""

import importlib
import json
import sys

import pytest
from _survey_summary_helpers import (
    SCRIPTS_DIR,
    config_for,
    manifest,
    write_config,
    write_plan,
    write_tree,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
checks = importlib.import_module("audit_survey_batches")

pytestmark = pytest.mark.unit


@pytest.fixture
def audit_cli():
    return checks


def test_param_load_config_rejects_a_plan_that_is_not_a_path_string(tmp_path):
    config = config_for({"B1": (tmp_path, 1)})
    config["batches"][0]["plan"] = 7
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(checks.ConfigError, match="plan"):
        checks.load_config(path)


def test_edge_load_config_resolves_a_relative_plan_beside_the_config(tmp_path):
    config = config_for({"B1": (tmp_path, 1)})
    config["batches"][0]["plan"] = "plans/B1.json"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = checks.load_config(path)

    assert loaded["batches"][0]["plan"] == str(tmp_path / "plans" / "B1.json")


def test_basic_the_entry_point_wires_a04_from_the_config(audit_cli, tmp_path):
    payloads = [manifest(seed=0), manifest(seed=1)]
    root = write_tree(tmp_path / "B1", {f"run{i}": p for i, p in enumerate(payloads)})
    plan = write_plan(tmp_path / "B1_plan.json", payloads)
    config = config_for({"B1": (root, 2)})
    config["batches"][0]["plan"] = plan.name
    config_path = write_config(tmp_path / "config.json", config)
    audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "audit.json").read_text(encoding="utf-8"))
    by_id = {check["check_id"]: check for check in report["checks"]}
    assert by_id["A04"]["result"] == "pass"


def test_edge_the_entry_point_fails_when_a_delivered_run_is_not_planned(audit_cli, tmp_path):
    payloads = [manifest(seed=0), manifest(seed=1)]
    root = write_tree(tmp_path / "B1", {f"run{i}": p for i, p in enumerate(payloads)})
    plan = write_plan(tmp_path / "B1_plan.json", payloads[:1])
    config = config_for({"B1": (root, 2)})
    config["batches"][0]["plan"] = plan.name
    config_path = write_config(tmp_path / "config.json", config)
    code = audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "audit.json").read_text(encoding="utf-8"))
    by_id = {check["check_id"]: check for check in report["checks"]}
    assert by_id["A04"]["result"] == "fail"
    assert code == 1


def test_basic_without_a_plan_the_entry_point_still_reports_a04_as_not_run(audit_cli, tmp_path):
    root = write_tree(tmp_path / "B1", {"run0": manifest()})
    config_path = write_config(tmp_path / "config.json", config_for({"B1": (root, 1)}))
    audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "audit.json").read_text(encoding="utf-8"))
    by_id = {check["check_id"]: check for check in report["checks"]}
    assert by_id["A04"]["result"] == "not_run"


def test_determ_two_audits_of_one_configured_tree_agree_on_every_check(audit_cli, tmp_path):
    payloads = [manifest(seed=0), manifest(seed=1)]
    root = write_tree(tmp_path / "B1", {f"run{i}": p for i, p in enumerate(payloads)})
    plan = write_plan(tmp_path / "B1_plan.json", payloads)
    config = config_for({"B1": (root, 2)})
    config["batches"][0]["plan"] = plan.name
    config_path = write_config(tmp_path / "config.json", config)
    for name in ("first", "second"):
        audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / name)])

    def checks_of(name):
        report = json.loads((tmp_path / name / "audit.json").read_text(encoding="utf-8"))
        return report["checks"]

    assert checks_of("first") == checks_of("second")


@pytest.mark.parametrize("key", ["run_log", "archive_digests", "exit_code_file"])
def test_param_load_config_rejects_an_evidence_input_that_is_not_a_path_string(tmp_path, key):
    config = config_for({"B1": (tmp_path, 1)})
    config["batches"][0][key] = ["x"]
    path = write_config(tmp_path / "config.json", config)
    with pytest.raises(checks.ConfigError, match=key):
        checks.load_config(path)


@pytest.mark.parametrize("key", ["run_log", "archive_digests", "exit_code_file"])
def test_edge_load_config_resolves_a_relative_evidence_input_beside_the_config(tmp_path, key):
    config = config_for({"B1": (tmp_path, 1)})
    config["batches"][0][key] = f"evidence/{key}.txt"
    path = write_config(tmp_path / "config.json", config)
    loaded = checks.load_config(path)
    assert loaded["batches"][0][key] == str(tmp_path / "evidence" / f"{key}.txt")


def test_basic_the_entry_point_wires_a05_and_a11_from_the_config(audit_cli, tmp_path):
    root = write_tree(tmp_path / "B1", {"spambase/nnpu/seed_0": manifest()})
    (tmp_path / "B1_run.log").write_text(
        "[nnpu] seed=0 -> /remote/B1/spambase/nnpu/seed_0/manifest.json\n"
        "completed 1 of 1 run(s) in spambase; 0 still pending\n",
        encoding="utf-8",
    )
    digest = "a" * 64
    (tmp_path / "B1_spambase.tar.gz.sha256").write_text(
        f"{digest}  /remote/B1_spambase/B1_spambase.tar.gz\n", encoding="utf-8"
    )
    config = config_for({"B1": (root, 1)})
    config["batches"][0]["run_log"] = "B1_run.log"
    config["batches"][0]["archive_digests"] = "B1_spambase.tar.gz.sha256"
    config_path = write_config(tmp_path / "config.json", config)
    audit_cli.main(["--config", str(config_path), "--out-dir", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "audit.json").read_text(encoding="utf-8"))
    by_id = {check["check_id"]: check for check in report["checks"]}
    assert by_id["A05"]["result"] == "pass"
    assert by_id["A11"]["result"] == "pass"
