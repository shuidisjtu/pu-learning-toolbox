"""Externally pinned pool bindings survive audit, summary and CSV transport."""

# ruff: noqa: F811

import copy
import csv
import json

import pytest
from _survey_recipe_registry_helpers import pool_registry
from _survey_summary_helpers import (  # noqa: F401
    audit_cli,
    config_for,
    manifest,
    summary_cli,
    write_config,
    write_tree,
)

from pu_toolbox.experiment.survey_protocol import digest
from pu_toolbox.experiment.survey_recipe_audit import (
    audit_recipe_batches,
    validate_consumed_bindings,
)
from pu_toolbox.experiment.survey_recipe_binding import (
    SCHEMA_VERSION,
    bind_recipe_pool,
    candidate_binding_at,
)
from pu_toolbox.experiment.survey_recipe_registry import canonical_digest

pytestmark = pytest.mark.unit


def _inputs(tmp_path, seed=0):
    protocol, registry = pool_registry()
    binding = bind_recipe_pool(
        registry, protocol, method="upu", dataset="spambase", training_path="native_2d"
    )
    payload = manifest(
        method="upu",
        budget="closed_form",
        epochs=None,
        batch_size=None,
        candidates=3,
        seed=seed,
        formal_eligible=False,
        blockers=("synthetic_pool_fixture",),
    )
    payload.update(binding)
    payload["protocol_sha256"] = digest(protocol)
    payload["candidate_runs"] = [
        {
            **candidate_binding_at(binding, index),
            "params": {},
            "status": "excluded" if index == 0 else "succeeded",
            "trajectory_index": None if index == 0 else index - 1,
        }
        for index in range(3)
    ]
    payload["selection"] = {
        name: {**candidate_binding_at(binding, index), "run_index": index - 1}
        for name, index in (("PA", 1), ("OA", 2))
    }
    root = write_tree(tmp_path / "B1", {f"seed_{seed}": payload})
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    config = config_for({"B1": (root, 1)})
    config["schema_version"] = "survey-batch-roots-2"
    config["batches"][0].update(
        recipe_policy="registry_bound",
        recipe_registry=str(registry_path),
        recipe_registry_sha256=canonical_digest(registry),
        recipe_binding_schema_version=SCHEMA_VERSION,
    )
    entries = [{"batch": "B1", "path": root / f"seed_{seed}/manifest.json", "payload": payload}]
    return protocol, registry, config, entries


def test_basic_pool_audit_and_low_level_gate_share_recomputed_identity(tmp_path):
    protocol, _, config, entries = _inputs(tmp_path)
    result = audit_recipe_batches(config, entries, protocol)
    assert result["overall"] == "pass"
    binding = result["method_bindings"]["upu"]
    assert binding["recipe_binding_schema_version"] == SCHEMA_VERSION
    assert "recipe_candidate_id" not in binding
    validate_consumed_bindings(
        [entries[0]["payload"]], result["method_bindings"], protocol=protocol
    )


def test_basic_summary_preserves_per_seed_pa_oa_choice_in_json_csv_markdown(
    audit_cli, summary_cli, tmp_path, monkeypatch
):
    protocol, _, config, _ = _inputs(tmp_path)
    # Synthetic future protocol in memory only, never the frozen file or matrix.
    original_loader = audit_cli.load_protocol

    def load_test_protocol(path=None):
        return protocol if path is None else original_loader(path)

    monkeypatch.setattr(audit_cli, "load_protocol", load_test_protocol)
    monkeypatch.setattr(summary_cli, "load_protocol", load_test_protocol)
    report = summary_cli.build_summary(config)
    assert report["recipe_audit"]["overall"] == "pass"
    assert len(report["rows"]) == 2
    for row in report["rows"]:
        expected_index = 1 if row["row_key"]["selection_protocol"] == "PA" else 2
        assert row["recipe_selection_bindings"][0]["binding"]["candidate_index"] == expected_index
        assert row["status"] != "formal"
    summary_cli.write_report(report, tmp_path / "out")
    with (tmp_path / "out/summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        selected = json.loads(row["recipe_selection_bindings_json"])
        assert row["recipe_binding_schema_version"] == SCHEMA_VERSION
        assert row["recipe_candidates_sha256"]
        assert row["recipe_candidate_id"] == ""  # Never pretend the pool chose one globally.
        assert selected[0]["binding"]["candidate_index"] == (
            1 if row["selection_protocol"] == "PA" else 2
        )
    markdown = (tmp_path / "out/summary.md").read_text(encoding="utf-8")
    assert "upu_test_1" in markdown and "upu_test_2" in markdown


@pytest.mark.parametrize(
    "defect", ["erased", "status", "params", "excluded", "schema", "pool_digest"]
)
def test_param_external_pool_obligation_rejects_tampering(tmp_path, defect):
    protocol, _, config, entries = _inputs(tmp_path)
    payload = entries[0]["payload"]
    if defect == "erased":
        entries[0]["payload"] = manifest(method="upu")
    elif defect == "status":
        payload["candidate_runs"][1]["status"] = "excluded"
    elif defect == "params":
        payload["candidate_runs"][2]["params"] = {"reg_lambda": 42}
    elif defect == "excluded":
        payload["selection"]["OA"] = {**candidate_binding_at(payload, 0), "run_index": 0}
    elif defect == "schema":
        config["batches"][0].pop("recipe_binding_schema_version")
    else:
        payload["recipe_candidates_sha256"] = "a" * 64
    result = audit_recipe_batches(config, entries, protocol)
    assert result["overall"] == "fail"
    assert result["not_reproducible_paths"] == [str(entries[0]["path"])]


@pytest.mark.parametrize("version", [None, "unknown", "survey-recipe-binding-1"])
def test_param_unknown_or_null_pool_version_is_not_an_implicit_default(
    audit_cli, tmp_path, version
):
    _, _, config, _ = _inputs(tmp_path)
    config["batches"][0]["recipe_binding_schema_version"] = version
    path = write_config(tmp_path / "config.json", config)
    with pytest.raises(audit_cli.ConfigError, match="schema version"):
        audit_cli.load_config(path)


@pytest.mark.parametrize("payload", [None, [], {"execution_unit": None}, {"execution_unit": []}])
def test_edge_malformed_pool_manifest_is_a_finding_not_a_crash(tmp_path, payload):
    protocol, _, config, entries = _inputs(tmp_path)
    entries[0]["payload"] = payload
    report = audit_recipe_batches(config, entries, protocol)
    assert report["overall"] == "fail"
    assert report["not_reproducible_paths"] == [str(entries[0]["path"])]


def test_determ_pool_audit_is_input_preserving(tmp_path):
    protocol, _, config, entries = _inputs(tmp_path)
    before = copy.deepcopy((protocol, config, entries))
    first = audit_recipe_batches(config, entries, protocol)
    assert first == audit_recipe_batches(config, entries, protocol)
    assert (protocol, config, entries) == before
