"""External batch obligations guard both audit and numeric summary consumption."""

# ruff: noqa: F811

import copy
import csv
import json

import pytest
from _survey_recipe_registry_helpers import locked_registry
from _survey_summary_helpers import (  # noqa: F401
    audit_cli,
    config_for,
    manifest,
    summary_cli,
    write_config,
    write_tree,
)

from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.experiment.survey_recipe_audit import audit_recipe_batches
from pu_toolbox.experiment.survey_recipe_registry import bind_recipe_candidate, canonical_digest

pytestmark = pytest.mark.unit


def _bound_payload(registry, **kwargs):
    payload = manifest(**kwargs)
    binding = bind_recipe_candidate(
        registry,
        load_protocol(),
        method=kwargs.get("method", "nnpu"),
        dataset="spambase",
        training_path="native_2d",
    )
    payload.update(binding)
    payload["candidate_runs"][0]["params"] = load_protocol()["candidate_pool"][0]
    for child in [*payload["candidate_runs"], *payload["selection"].values()]:
        child.update(binding)
    return payload


def _config(tmp_path, payloads=None, registry=None):
    registry = locked_registry("nnpu") if registry is None else registry
    root = write_tree(tmp_path / "B1", payloads or {"run": _bound_payload(registry)})
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    config = config_for({"B1": (root, len(payloads or {"run": 1}))})
    config["schema_version"] = "survey-batch-roots-2"
    config["batches"][0].update(
        recipe_policy="registry_bound",
        recipe_registry=str(path),
        recipe_registry_sha256=canonical_digest(registry),
    )
    return config, registry


def test_basic_bound_audit_and_summary_keep_full_identity(summary_cli, audit_cli, tmp_path):
    registry = locked_registry("nnpu")
    payloads = {str(seed): _bound_payload(registry, seed=seed) for seed in range(5)}
    config, _ = _config(tmp_path, payloads, registry)
    report = summary_cli.build_summary(config)
    assert report["recipe_audit"]["overall"] == "pass"
    assert len(report["checks"]) == 17
    assert len(report["rows"]) == 2
    for row in report["rows"]:
        assert row["recipe_binding"]["recipe_registry_sha256"] == canonical_digest(registry)
        assert row["metric"]["mean"] == pytest.approx(0.8)
    assert report["run_costs"][0]["recipe_binding"] == report["rows"][0]["recipe_binding"]
    summary_cli.write_report(report, tmp_path / "out")
    with (tmp_path / "out/summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["recipe_registry_sha256"] == canonical_digest(registry) for row in rows)
    assert "配方绑定审计扩展" in (tmp_path / "out/summary.md").read_text(encoding="utf-8")
    assert audit_cli.build_audit(config)["recipe_audit"] == report["recipe_audit"]


@pytest.mark.parametrize("level", ["root", "candidate", "selection", "all"])
def test_param_erased_binding_never_downgrades_to_legacy(
    summary_cli, audit_cli, tmp_path, monkeypatch, level
):
    registry = locked_registry("nnpu")
    payload = _bound_payload(registry)
    targets = {
        "root": [payload],
        "candidate": payload["candidate_runs"],
        "selection": list(payload["selection"].values()),
        "all": [payload, *payload["candidate_runs"], *payload["selection"].values()],
    }
    for target in targets[level]:
        for key in list(target):
            if key.startswith("recipe_") or key in {"resolved_params_sha256", "inner_search_id"}:
                target.pop(key)
    config, _ = _config(tmp_path, {"bad": payload}, registry)

    def forbidden(*args, **kwargs):
        raise AssertionError("numeric aggregation must not run on bad recipe bindings")

    monkeypatch.setattr(audit_cli.gate, "aggregate", forbidden)
    report = summary_cli.build_summary(config)
    assert report["overall"] == "fail"
    assert report["rows"] == report["run_costs"] == []
    assert report["recipe_audit"]["failures"]
    assert report["coverage"]["not_reproducible"] == 1
    summary_cli.write_report(report, tmp_path / "out")
    audit_cli.write_report(audit_cli.build_audit(config), tmp_path / "audit")


@pytest.mark.parametrize(
    "field,value",
    [
        ("recipe_registry_sha256", "a" * 64),
        ("resolved_params_sha256", "b" * 64),
        ("inner_search_id", "invented"),
        ("recipe_profile_id", "unknown"),
        ("protocol_sha256", "c" * 64),
        ("protocol_version", "wrong"),
    ],
)
def test_param_forged_identity_is_refused(audit_cli, tmp_path, field, value):
    config, registry = _config(tmp_path)
    payload = _bound_payload(registry)
    payload[field] = value
    entries = [{"batch": "B1", "path": tmp_path / "manifest.json", "payload": payload}]
    assert audit_recipe_batches(config, entries, load_protocol())["overall"] == "fail"


@pytest.mark.parametrize("change", ["pin", "draft", "absent", "scope", "nested_index"])
def test_param_bad_registry_or_request_fails(summary_cli, tmp_path, change):
    config, registry = _config(tmp_path)
    batch = config["batches"][0]
    if change == "pin":
        batch["recipe_registry_sha256"] = "a" * 64
    elif change == "absent":
        batch["recipe_registry"] = str(tmp_path / "absent.json")
    elif change == "draft":
        registry["recipe_lifecycle"] = "draft"
        (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
        batch["recipe_registry_sha256"] = canonical_digest(registry)
    else:
        payload = _bound_payload(registry)
        if change == "scope":
            payload["execution_unit"]["dataset"] = "unknown"
        else:
            payload["selection"]["PA"]["candidate_index"] = True
        write_tree(tmp_path / "B1", {"run": payload})
    report = summary_cli.build_summary(config)
    assert report["overall"] == "fail"
    assert report["rows"] == []


def test_edge_v1_claims_in_nested_only_require_external_obligation(audit_cli, tmp_path):
    registry = locked_registry("nnpu")
    payload = manifest()
    payload["candidate_runs"][0].update(_bound_payload(registry)["candidate_runs"][0])
    root = write_tree(tmp_path / "B1", {"nested": payload})
    report = audit_cli.build_audit(config_for({"B1": (root, 1)}))
    assert report["overall"] == "fail"
    assert "v2 external" in report["recipe_audit"]["failures"][0]["message"]


def test_basic_legacy_v1_unchanged_and_v2_explicit_legacy_allowed(summary_cli, tmp_path):
    root = write_tree(tmp_path / "B1", {"legacy": manifest()})
    config = config_for({"B1": (root, 1)})
    old = summary_cli.build_summary(config)
    assert "recipe_audit" not in old
    config["schema_version"] = "survey-batch-roots-2"
    config["batches"][0]["recipe_policy"] = "legacy_unbound"
    new = summary_cli.build_summary(config)
    assert new["recipe_audit"]["overall"] == "pass"
    assert new["rows"] == old["rows"]


def test_param_mixed_namespaces_are_not_pooled(summary_cli, tmp_path):
    config, _ = _config(tmp_path)
    legacy = write_tree(tmp_path / "B2", {"legacy": manifest(seed=1)})
    config["batches"].append(
        {
            "name": "B2",
            "root": str(legacy),
            "expected_manifests": 1,
            "recipe_policy": "legacy_unbound",
        }
    )
    report = summary_cli.build_summary(config)
    assert report["overall"] == "fail"
    assert report["rows"] == []
    assert "mixed recipe namespaces" in report["recipe_audit"]["failures"][-1]["message"]


def test_basic_distinct_methods_in_one_registry_are_allowed(audit_cli, tmp_path):
    registry = locked_registry("nnpu")
    registry["profiles"].update(locked_registry("upu")["profiles"])
    config, _ = _config(
        tmp_path,
        {
            "nnpu": _bound_payload(registry),
            "upu": _bound_payload(registry, method="upu"),
        },
        registry,
    )
    extension = audit_cli.build_audit(config)["recipe_audit"]
    assert extension["overall"] == "pass"
    assert set(extension["method_bindings"]) == {"nnpu", "upu"}


@pytest.mark.parametrize("change", ["missing", "pin", "legacy_registry", "v1"])
def test_param_config_requires_explicit_pinned_obligation(audit_cli, tmp_path, change):
    config, _ = _config(tmp_path)
    batch = config["batches"][0]
    if change == "missing":
        batch.pop("recipe_policy")
    elif change == "pin":
        batch["recipe_registry_sha256"] = "no"
    elif change == "legacy_registry":
        batch["recipe_policy"] = "legacy_unbound"
    else:
        config["schema_version"] = "survey-batch-roots-1"
    path = write_config(tmp_path / "roots.json", config)
    with pytest.raises(audit_cli.ConfigError):
        audit_cli.load_config(path)


def test_edge_config_registry_path_resolves_relative_to_config(audit_cli, tmp_path):
    config, _ = _config(tmp_path)
    config["batches"][0]["recipe_registry"] = "registry.json"
    path = write_config(tmp_path / "roots.json", config)
    loaded = audit_cli.load_config(path)
    assert loaded["batches"][0]["recipe_registry"] == str(tmp_path / "registry.json")


def test_determ_recipe_audit_does_not_mutate_inputs(audit_cli, tmp_path):
    config, _ = _config(tmp_path)
    entries = audit_cli.discover_batches(config)
    before = copy.deepcopy((config, entries))
    first = audit_recipe_batches(config, entries, load_protocol())
    assert first == audit_recipe_batches(config, entries, load_protocol())
    assert (config, entries) == before


@pytest.mark.parametrize("different", [False, True])
def test_edge_cross_batch_registry_identity(summary_cli, tmp_path, different):
    config, registry = _config(tmp_path)
    second = copy.deepcopy(registry)
    if different:
        second["recipe_registry_version"] = "test-only-second-registry"
    second_path = tmp_path / "second.json"
    second_path.write_text(json.dumps(second), encoding="utf-8")
    root = write_tree(tmp_path / "B2", {"run": _bound_payload(second, seed=1)})
    config["batches"].append(
        {
            "name": "B2",
            "root": str(root),
            "expected_manifests": 1,
            "recipe_policy": "registry_bound",
            "recipe_registry": str(second_path),
            "recipe_registry_sha256": canonical_digest(second),
        }
    )
    report = summary_cli.build_summary(config)
    assert report["recipe_audit"]["overall"] == ("fail" if different else "pass")
    assert bool(report["rows"]) is (not different)


@pytest.mark.parametrize(
    "malformation",
    [
        "missing_null",
        "missing_status",
        "wrong_method",
        "empty_selection",
        "params",
        "missing_params",
    ],
)
def test_param_partial_or_misattributed_payload_fails(audit_cli, tmp_path, malformation):
    config, registry = _config(tmp_path)
    payload = _bound_payload(registry)
    if malformation == "missing_null":
        payload["selection"]["OA"].pop("inner_search_id")
    elif malformation == "missing_status":
        payload.pop("recipe_binding_status")
    elif malformation == "wrong_method":
        payload["execution_unit"]["method"] = "upu"
    elif malformation == "empty_selection":
        payload["selection"] = {}
    elif malformation == "params":
        payload["candidate_runs"][0]["params"] = {"reg_lambda": 42}
    else:
        payload["candidate_runs"][0].pop("params")
    entries = [{"batch": "B1", "path": tmp_path / "manifest.json", "payload": payload}]
    assert audit_recipe_batches(config, entries, load_protocol())["overall"] == "fail"


def test_edge_legacy_claim_and_cli_failure_are_visible(summary_cli, audit_cli, tmp_path):
    config, _ = _config(tmp_path)
    batch = config["batches"][0]
    batch["recipe_policy"] = "legacy_unbound"
    batch.pop("recipe_registry")
    batch.pop("recipe_registry_sha256")
    path = write_config(tmp_path / "roots.json", config)
    assert audit_cli.main(["--config", str(path), "--out-dir", str(tmp_path / "audit")]) == 1
    assert summary_cli.main(["--config", str(path), "--out-dir", str(tmp_path / "summary")]) == 1
    report = json.loads((tmp_path / "summary/summary.json").read_text(encoding="utf-8"))
    assert report["recipe_audit"]["failures"]
    assert report["rows"] == []
