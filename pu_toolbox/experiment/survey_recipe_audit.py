"""External recipe obligations for new survey batches, separate from A01--A17.

Historical batch-roots-1 inputs remain unchanged when no binding is claimed.
Batch-roots-2 makes the obligation external to the manifest, so deleting all
binding fields cannot turn a new run into an accepted historical artifact.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .pilot_plan import _matches_recipe_binding
from .survey_protocol import digest
from .survey_recipe_registry import (
    REGISTRY_BOUND_MANIFEST_FIELDS,
    bind_recipe_candidate,
    canonical_digest,
    load_registry,
    manifest_has_binding_fields,
    validate_registry,
)

BATCH_SCHEMA_VERSION = "survey-batch-roots-2"
SCHEMA_VERSION = "survey-recipe-batch-audit-1"


def validate_recipe_config(config: Mapping[str, Any], *, base_dir: Path) -> None:
    """Validate and resolve explicit v2 obligations; never infer them from runs."""
    versioned = config.get("schema_version") == BATCH_SCHEMA_VERSION
    for batch in config["batches"]:
        keys = {"recipe_policy", "recipe_registry", "recipe_registry_sha256"} & batch.keys()
        if not versioned:
            if keys:
                raise ValueError("recipe obligations require survey-batch-roots-2")
            continue
        policy = batch.get("recipe_policy")
        if policy not in {"legacy_unbound", "registry_bound"}:
            raise ValueError(f"batch {batch['name']!r} needs an explicit recipe_policy")
        if policy == "legacy_unbound":
            if keys - {"recipe_policy"}:
                raise ValueError("legacy batches must not declare a recipe registry")
            continue
        registry = batch.get("recipe_registry")
        pin = batch.get("recipe_registry_sha256")
        if not isinstance(registry, str) or not registry:
            raise ValueError("registry_bound batch needs a recipe_registry path")
        if (
            not isinstance(pin, str)
            or len(pin) != 64
            or any(c not in "0123456789abcdef" for c in pin)
        ):
            raise ValueError("registry_bound batch needs a canonical recipe_registry_sha256 pin")
        batch["recipe_registry"] = str((base_dir / registry).resolve())


def _claims_binding(payload: Mapping[str, Any]) -> bool:
    """Nested claims count too: deleting just the root must not evade the gate."""
    if (
        "recipe_binding_status" in payload and payload["recipe_binding_status"] != "legacy_unbound"
    ) or manifest_has_binding_fields(payload):
        return True
    candidates = payload.get("candidate_runs", [])
    selections = payload.get("selection", {})
    children = (candidates if isinstance(candidates, list) else []) + (
        list(selections.values()) if isinstance(selections, Mapping) else []
    )
    return any(isinstance(child, Mapping) and _claims_binding(child) for child in children)


def validate_consumed_bindings(
    payloads: list[dict[str, Any]],
    bindings: Mapping[str, Mapping[str, Any]] | None,
    *,
    protocol: Mapping[str, Any],
) -> None:
    """The low-level gate needs externally recomputed method expectations too.

    Callers obtain these from a successful batch audit. Without expectations,
    only legacy artifacts are accepted; the standalone legacy CLI cannot
    silently consume bound artifacts without their registry and batch contract.
    """
    if bindings is None:
        if any(_claims_binding(payload) for payload in payloads):
            raise ValueError(
                "recipe-bound aggregation requires the v2 batch audit/summary entry point"
            )
        return
    fields = {
        "recipe_binding_status",
        *REGISTRY_BOUND_MANIFEST_FIELDS,
        "protocol_version",
        "protocol_sha256",
    }
    if not bindings or any(
        not isinstance(value, Mapping)
        or not fields.issubset(value)
        or value.get("recipe_binding_status") != "registry_bound"
        for value in bindings.values()
    ):
        raise ValueError("aggregation requires complete externally audited recipe identities")
    if len({value["recipe_registry_sha256"] for value in bindings.values()}) != 1:
        raise ValueError("mixed recipe registries cannot share an aggregate")
    for payload in payloads:
        method = (payload.get("execution_unit") or {}).get("method")
        expected = bindings.get(method)
        if expected is None or not _matches_recipe_binding(payload, expected):
            raise ValueError(
                "manifest binding does not match the externally audited recipe identity"
            )
        if not _candidate_params_match(payload, protocol):
            raise ValueError("recorded candidate params disagree with the bound protocol pool")


def _candidate_params_match(payload: Mapping[str, Any], protocol: Mapping[str, Any]) -> bool:
    """A correct identity hash cannot excuse different recorded training params."""
    candidates = payload.get("candidate_runs")
    pool = protocol.get("candidate_pool")
    return (
        isinstance(candidates, list)
        and len(candidates) == 1
        and isinstance(candidates[0], Mapping)
        and "params" in candidates[0]
        and isinstance(pool, list)
        and len(pool) == 1
        and digest(candidates[0]["params"]) == digest(pool[0])
    )


def audit_recipe_batches(
    config: Mapping[str, Any], entries: list[dict[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Recompute bindings and nested identities before any numeric consumption.

    A report consumes one recipe namespace at a time. Different methods may
    have different profile/candidate IDs within that registry. Different
    registries, or bound and legacy batches, require separate reports rather
    than silently pooling their seeds or costs under the old row key.
    """
    loaded = [entry for entry in entries if "payload" in entry]
    versioned = config.get("schema_version") == BATCH_SCHEMA_VERSION
    if not versioned and not any(_claims_binding(e["payload"]) for e in loaded):
        return None
    failures: list[dict[str, str]] = []
    config = copy.deepcopy(config)
    try:
        validate_recipe_config(config, base_dir=Path.cwd())
    except ValueError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "overall": "fail",
            "batches": [],
            "method_bindings": {},
            "failures": [{"batch": "*", "status": "not_reproducible", "message": str(exc)}],
            "not_reproducible_paths": sorted(str(entry["path"]) for entry in loaded),
        }
    batches: list[dict[str, Any]] = []
    namespaces: set[tuple[str, str | None]] = set()
    identities: dict[str, dict[str, Any]] = {}
    registries: dict[str, Any] = {}
    prepared: dict[tuple, dict[str, Any]] = {}
    for batch in config["batches"]:
        members = [entry for entry in loaded if entry["batch"] == batch["name"]]
        policy = batch.get("recipe_policy") if versioned else None
        record = {"name": batch["name"], "recipe_policy": policy, "manifests": len(members)}
        batches.append(record)
        if policy is None:
            failures.append(
                {
                    "batch": batch["name"],
                    "message": "binding claims require a v2 external recipe obligation",
                }
            )
            continue
        if policy == "legacy_unbound":
            namespaces.add((policy, None))
            for entry in members:
                # An explicit legacy label is allowed, but registry fields are not.
                payload = entry["payload"]
                legacy = dict(payload)
                if legacy.get("recipe_binding_status") == "legacy_unbound":
                    legacy.pop("recipe_binding_status")
                if _claims_binding(legacy):
                    failures.append(
                        {
                            "batch": batch["name"],
                            "path": str(entry["path"]),
                            "message": "legacy batch carries a binding claim",
                        }
                    )
            continue
        pin = batch.get("recipe_registry_sha256")
        namespaces.add((policy, pin))
        record["recipe_registry_sha256"] = pin
        try:
            registry_path = batch["recipe_registry"]
            if registry_path not in registries:
                registries[registry_path] = load_registry(registry_path)
            registry = registries[registry_path]
            if canonical_digest(registry) != pin:
                raise ValueError("registry digest differs from external batch pin")
            findings = validate_registry(registry, protocol)
            if findings or registry.get("recipe_lifecycle") != "locked":
                raise ValueError("batch registry must validate and already be locked")
            for entry in members:
                payload = entry["payload"]
                unit = payload.get("execution_unit", {})
                key = (pin, unit.get("method"), unit.get("dataset"), unit.get("training_path"))
                if key not in prepared:
                    prepared[key] = bind_recipe_candidate(
                        registry,
                        protocol,
                        method=key[1],
                        dataset=key[2],
                        training_path=key[3],
                        expected_sha256=pin,
                    )
                expected = dict(prepared[key])
                expected.update(
                    protocol_version=protocol["protocol_version"], protocol_sha256=digest(protocol)
                )
                if not _matches_recipe_binding(payload, expected) or not _candidate_params_match(
                    payload, protocol
                ):
                    failures.append(
                        {
                            "batch": batch["name"],
                            "path": str(entry["path"]),
                            "message": (
                                "root, candidate or selection binding disagrees "
                                "with external obligation (including candidate params)"
                            ),
                        }
                    )
                else:
                    identities[unit["method"]] = expected
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"batch": batch["name"], "message": str(exc)})
    if len(namespaces) > 1:
        failures.append(
            {
                "batch": "*",
                "status": "refused",
                "message": "mixed recipe namespaces require separate audit/summary reports",
            }
        )
    for failure in failures:
        failure.setdefault("status", "not_reproducible")
    bad_batches = {
        failure["batch"]
        for failure in failures
        if "path" not in failure and failure["status"] == "not_reproducible"
    }
    bad_paths = {
        failure["path"]
        for failure in failures
        if "path" in failure and failure["status"] == "not_reproducible"
    }
    bad_paths.update(str(entry["path"]) for entry in loaded if entry["batch"] in bad_batches)
    return {
        "schema_version": SCHEMA_VERSION,
        "overall": "fail" if failures else "pass",
        "batches": batches,
        "method_bindings": identities,
        "failures": failures,
        "not_reproducible_paths": sorted(bad_paths),
    }


def recipe_audit_lines(report: Mapping[str, Any]) -> list[str]:
    """Render the versioned extension without relabeling historical checks."""
    extension = report.get("recipe_audit")
    if extension is None:
        return []
    lines = [
        "",
        "## 配方绑定审计扩展",
        "",
        f"- `{extension['schema_version']}`：{extension['overall']}",
    ]
    for batch in extension["batches"]:
        lines.append(
            f"- {batch['name']}: `{batch['recipe_policy']}` / "
            f"`{batch.get('recipe_registry_sha256', 'unbound')}`"
        )
    for failure in extension["failures"]:
        lines.append(f"- 拒收 {failure.get('path', failure['batch'])}: {failure['message']}")
    for method, binding in sorted(extension["method_bindings"].items()):
        lines.append(
            f"- {method}: `{binding['recipe_profile_id']}` / `{binding['recipe_candidate_id']}`; "
            f"resolved=`{binding['resolved_params_sha256']}`; inner=`{binding['inner_search_id']}`"
        )
    return lines
