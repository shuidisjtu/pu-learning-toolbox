"""Read completed Survey results without replaying batch or protocol audits.

Only identities and evidence recorded by each run are read. This reader checks
whether observations can be combined; it never grants formal admission.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..utils.serialization import strict_canonical_hash
from .survey_summary import (
    RESULT_SELECTION_PROTOCOLS,
    SummaryError,
    aggregate_additional_metric,
    aggregate_metric,
    matrix_selector,
    row_key,
    summarize_costs,
    unrecorded_measurements,
    uses_clean_support,
)

_VERSIONS = ("survey-v3.0", "survey-v3.1", "survey-v3.2")
_METRICS = ("auc", "f1", "precision", "recall")
_MECHANISMS = ("scar", "sar_lbe_a", "sar_lbe_b", "pn_oracle")
_RECIPE_FIELDS = (
    "recipe_binding_status",
    "recipe_binding_schema_version",
    "recipe_registry_version",
    "recipe_registry_sha256",
    "recipe_profile_id",
    "recipe_candidate_id",
    "resolved_params_sha256",
    "inner_search_id",
)


def _filesystem_path(path: Path) -> Path:
    """Use Windows extended paths for I/O, including traversal below short inputs."""
    if os.name != "nt":
        return path
    value = str(path)
    if value.startswith("\\\\?\\"):
        return path
    value = os.path.abspath(value)
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


def _report_path(path: Path) -> Path:
    """Keep one normal absolute identity for normal and extended Windows inputs."""
    value = str(path)
    if os.name == "nt" and value.startswith("\\\\?\\"):
        value = "\\\\" + value[8:] if value[4:8].lower() == "unc\\" else value[4:]
    return Path(value)


def _paths(values: Iterable[str | Path], *, allow_missing: bool = False) -> list[Path]:
    if isinstance(values, str | Path):
        values = (values,)
    found = set()
    for value in values:
        path = _report_path(_filesystem_path(Path(value)).resolve())
        readable = _filesystem_path(path)
        if readable.is_dir():
            manifests = [_report_path(item) for item in readable.rglob("manifest.json")]
            if not manifests and not allow_missing:
                raise SummaryError(f"result directory has no manifest.json files: {path}")
            found.update(manifests)
            if not manifests and allow_missing:
                found.add(path)
        elif readable.is_file() or allow_missing:
            found.add(path)
        else:
            raise SummaryError(f"missing result input: {path}")
    return sorted(found, key=str)


def _signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _recorded_risk(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key in {"numerical_risk", "numerical_risks", "numerical_failure", "quarantined"}
        and value
    }


def _check_selection(payload: Mapping[str, Any], name: str) -> None:
    selected = payload.get("selection", {}).get(name)
    if not isinstance(selected, Mapping):
        raise SummaryError(f"test result {name} has no selection evidence")
    if "protocol" in selected and selected["protocol"] != name:
        raise SummaryError(f"selection {name} carries another selection protocol")
    index = selected.get("candidate_index")
    candidates = payload.get("candidate_runs")
    if type(index) is not int or not isinstance(candidates, list):
        raise SummaryError(f"selection {name} has no candidate identity")
    if any(
        not isinstance(item, Mapping) or type(item.get("candidate_index")) is not int
        for item in candidates
    ):
        raise SummaryError(f"selection {name} has malformed candidate evidence")
    matches = [item for item in candidates if item.get("candidate_index") == index]
    if len(matches) != 1:
        raise SummaryError(f"selection {name} references an unknown candidate")
    candidate = matches[0]
    binding = payload
    if "recipe_candidates" in payload or "recipe_candidates_sha256" in payload:
        pool = payload.get("recipe_candidates")
        if (
            not isinstance(pool, list)
            or any(not isinstance(item, Mapping) for item in pool)
            or payload.get("recipe_candidates_sha256") != strict_canonical_hash(pool)
        ):
            raise SummaryError(f"selection {name} recipe pool digest does not recompute")
        identities = [item for item in pool if item.get("candidate_index") == index]
        if len(identities) != 1:
            raise SummaryError(f"selection {name} recipe pool has no unique candidate")
        binding = identities[0]
        for field in _RECIPE_FIELDS:
            if field in payload and binding.get(field) != payload[field]:
                raise SummaryError(f"selection {name} recipe pool conflicts with its run")
    fields = set(_RECIPE_FIELDS) & (set(binding) | set(candidate) | set(selected))
    for field in fields:
        if any(field not in record for record in (binding, candidate, selected)) or not (
            binding[field] == candidate[field] == selected[field]
        ):
            raise SummaryError(f"selection {name} has conflicting recipe identity: {field}")
    if "params" in selected and selected["params"] != candidate.get("params"):
        raise SummaryError(f"selection {name} params conflict with its candidate")
    for record in (candidate, selected):
        for value_field, hash_field in (
            ("params", "params_sha256"),
            ("resolved_params", "resolved_params_sha256"),
        ):
            if (
                value_field in record
                and hash_field in record
                and strict_canonical_hash(record[value_field]) != record[hash_field]
            ):
                raise SummaryError(f"selection {name} recorded params digest does not recompute")
    if any(item.get("status") in {"excluded", "failed"} for item in matches):
        raise SummaryError(f"selection {name} references an excluded candidate")
    if "run_index" in selected and not any(
        item.get("trajectory_index") == selected["run_index"] for item in matches
    ):
        raise SummaryError(f"selection {name} references an unknown trajectory")
    checkpoint = selected.get("checkpoint")
    if checkpoint is not None:
        recorded = candidate.get("epoch_checkpoints", [])
        if (
            not isinstance(checkpoint, Mapping)
            or not checkpoint.get("path")
            or not checkpoint.get("sha256")
            or not isinstance(recorded, list)
            or any(not isinstance(point, Mapping) for point in recorded)
        ):
            raise SummaryError(f"selection {name} checkpoint is not in its candidate trajectory")
        points = [
            point
            for point in recorded
            if all(point.get(key) == checkpoint.get(key) for key in ("path", "sha256"))
        ]
        if len(points) != 1:
            raise SummaryError(f"selection {name} checkpoint is not unique in its trajectory")
        point = points[0]
        index = selected.get("checkpoint_index")
        if index is not None and (
            type(index) is not int or not 0 <= index < len(recorded) or recorded[index] != point
        ):
            raise SummaryError(f"selection {name} checkpoint index conflicts with its trajectory")
        if (
            selected.get("epoch") is not None
            and point.get("epoch_position") is not None
            and selected["epoch"] != point["epoch_position"]
        ):
            raise SummaryError(f"selection {name} epoch conflicts with its checkpoint")
        for field in ("epoch_position", "epoch_label", "component", "training_context"):
            if field in checkpoint and field in point and checkpoint[field] != point[field]:
                raise SummaryError(f"selection {name} checkpoint has conflicting {field}")


def _read(path: Path, seeds: Sequence[int]) -> dict[str, Any]:
    try:
        payload = json.loads(_filesystem_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SummaryError(f"cannot read result {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("protocol_version") not in _VERSIONS:
        raise SummaryError(f"unsupported result protocol in {path}")
    if type(payload.get("seed")) is not int or payload["seed"] not in seeds:
        raise SummaryError(f"result seed outside expected seeds: {path}")
    unit = payload.get("execution_unit")
    if not isinstance(unit, dict) or any(
        not isinstance(unit.get(key), str) or not unit[key]
        for key in ("method", "dataset", "training_path", "comparability_group")
    ):
        raise SummaryError(f"result has no execution identity: {path}")
    if payload.get("training_path") != unit["training_path"]:
        raise SummaryError(f"result training path conflicts with its execution unit: {path}")
    for name in ("budget", "representation", "generation", "split_ref"):
        if not isinstance(payload.get(name), dict) or not payload[name]:
            raise SummaryError(f"result has no {name} evidence: {path}")
    if not payload["representation"].get("split_sha256") or not payload["representation"].get(
        "feature_sha256"
    ):
        raise SummaryError(f"result has no role/feature identity: {path}")
    for name in ("selection", "test_results"):
        if not isinstance(payload.get(name, {}), dict):
            raise SummaryError(f"result {name} must be an object: {path}")
    candidates = payload.get("candidate_runs", [])
    if not isinstance(candidates, list) or any(
        not isinstance(item, Mapping) or type(item.get("candidate_index")) is not int
        for item in candidates
    ):
        raise SummaryError(f"result has malformed candidate evidence: {path}")
    if len({item["candidate_index"] for item in candidates}) != len(candidates):
        raise SummaryError(f"result has duplicate candidate identities: {path}")
    staged = payload["budget"].get("staged_budget")
    if staged is not None:
        from .survey_staged_budget import validate_contract

        try:
            validate_contract(staged)
        except (ValueError, TypeError, KeyError) as exc:
            raise SummaryError(f"invalid recorded staged budget in {path}: {exc}") from exc
    generation = payload["generation"]
    if any(not isinstance(generation.get(role), dict) for role in ("train", "pu_val")):
        raise SummaryError(f"result has no training/validation role evidence: {path}")
    mechanism = generation["train"].get("mechanism")
    if mechanism not in _MECHANISMS or generation["pu_val"].get("mechanism") != mechanism:
        raise SummaryError(f"result has conflicting labeling mechanisms: {path}")
    if (mechanism == "pn_oracle") != bool(payload.get("c_independent")):
        raise SummaryError(f"oracle result has conflicting c-independent identity: {path}")
    if uses_clean_support(payload) and "PA" in (
        set(payload.get("selection", {})) | set(payload.get("test_results", {}))
    ):
        raise SummaryError(f"clean support cannot be reported as pure PA: {path}")
    if set(payload.get("test_results", {})) - set(RESULT_SELECTION_PROTOCOLS):
        raise SummaryError(f"result has unknown selection protocols: {path}")
    try:
        row_key(payload, selection_protocol="OA")
    except ValueError as exc:
        raise SummaryError(f"invalid result identity in {path}: {exc}") from exc
    return payload


def _check_consistency(payload: dict[str, Any], references: dict, identities: set) -> None:
    key = row_key(payload, selection_protocol="OA")
    seed = payload["seed"]
    identity = (key, seed)
    if identity in identities:
        raise SummaryError(f"duplicate result identity for seed {seed}: {key}")
    identities.add(identity)
    unit = payload["execution_unit"]
    namespace = (unit["dataset"], payload["training_path"])
    if references.setdefault(("namespace", key[:3]), namespace) != namespace:
        raise SummaryError(f"comparability group mixes datasets or training paths: {key[:3]}")
    method_key = (unit["dataset"], *key[:4])
    params = {
        name: value
        for name, value in payload.get("estimator_parameters", {}).items()
        if name not in {"random_state", "seed", "device"}
    }
    config = {
        "protocol_version": payload["protocol_version"],
        "protocol_sha256": payload.get("protocol_sha256"),
        "budget": payload["budget"],
        "backbone": payload.get("backbone"),
        "representation": payload["representation"].get("name"),
        "algorithm_id": payload.get("algorithm_id"),
        "params": params,
        "recipe_registry_version": payload.get("recipe_registry_version"),
        "recipe_registry_sha256": payload.get("recipe_registry_sha256"),
        "supervision": uses_clean_support(payload),
    }
    signature = _signature(config)
    reference = references.setdefault(("config", method_key), signature)
    if reference != signature:
        raise SummaryError(f"conflicting recorded configurations: {method_key}")
    candidates = payload.get("candidate_runs", [])
    if candidates and all("params" in candidate for candidate in candidates):
        candidate_config = {
            "params": sorted(
                ((item["candidate_index"], item["params"]) for item in candidates),
                key=lambda item: item[0],
            ),
            "recipe": {field: payload[field] for field in _RECIPE_FIELDS if field in payload},
            "recipe_candidates_sha256": payload.get("recipe_candidates_sha256"),
        }
        signature = _signature(candidate_config)
        if references.setdefault(("candidates", method_key), signature) != signature:
            raise SummaryError(f"conflicting recorded candidate configurations: {method_key}")
    group_config = {
        "training_path": payload["training_path"],
        "backbone": payload.get("backbone"),
        "representation": payload["representation"].get("name"),
        "budget": {
            field: payload["budget"].get(field)
            for field in ("epochs", "batch_size", "actual_outer_candidates")
        },
        "staged_budget": {
            key: value
            for key, value in payload["budget"].get("staged_budget", {}).items()
            if key != "checkpoint_bytes_per_component"
        },
    }
    signature = _signature(group_config)
    group = (unit["dataset"], *key[:3])
    if references.setdefault(("group_config", group), signature) != signature:
        raise SummaryError(f"conflicting configurations within comparability group: {group}")
    role_key = (unit["dataset"], *key[:3], seed)
    role_evidence = {
        "split_ref": payload["split_ref"].get("indices_sha256"),
        "split": payload["representation"]["split_sha256"],
        "features": payload["representation"]["feature_sha256"],
    }
    signature = _signature(role_evidence)
    if references.setdefault(("roles", role_key), signature) != signature:
        raise SummaryError(f"conflicting recorded role identities: {role_key}")
    label_key = (*role_key, key[-1])
    labels = {
        role: payload["generation"][role].get("label_view_sha256") for role in ("train", "pu_val")
    }
    signature = _signature(labels)
    if references.setdefault(("labels", label_key), signature) != signature:
        raise SummaryError(f"conflicting recorded label identities: {label_key}")


def _comparison_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the actual comparison inputs, without repeating unrelated backbone specs."""
    representation = payload["representation"]
    backbone = payload.get("backbone")
    if isinstance(backbone, Mapping) and isinstance(backbone.get("specs"), Mapping):
        backbone = {
            "name": backbone.get("name"),
            "spec": backbone["specs"].get(backbone.get("name")),
        }
    return copy.deepcopy(
        {
            "split_sha256": representation.get("split_sha256"),
            "feature_sha256": representation.get("feature_sha256"),
            "representation": representation.get("name"),
            "backbone": backbone,
            "budget": payload["budget"],
            "clean_support": uses_clean_support(payload),
        }
    )


def _row(key: tuple, entries: list, seeds: Sequence[int], cost_id: str) -> dict[str, Any]:
    usable = [entry for entry in entries if entry["value"] is not None]
    metric = aggregate_metric(
        ((entry["payload"]["seed"], entry["value"]) for entry in usable),
        metric_name="accuracy",
        expected_seeds=seeds,
    )
    additional = {}
    for name in _METRICS:
        values = []
        reasons = {}
        for entry in usable:
            seed = entry["payload"]["seed"]
            block = entry["payload"]["test_results"][key[4]]
            value = block.get(name)
            if not isinstance(value, int | float) or isinstance(value, bool):
                value = float("nan")
            values.append((seed, value))
            if not math.isfinite(value):
                reasons[str(seed)] = block.get(f"{name}_unavailable_reason") or "not recorded"
        additional[name] = aggregate_additional_metric(
            values, metric_name=name, ran_seeds=metric["seeds_observed"]
        )
        additional[name]["unavailable_reasons"] = reasons
    payload = entries[0]["payload"]
    blockers = sorted(
        {value for item in entries for value in item["payload"].get("formal_blockers", [])}
    )
    reasons = {"formal_admission_not_assessed"}
    if any(item["reproducibility"]["missing"] for item in entries):
        reasons.add("missing_reproducibility")
    if blockers:
        reasons.add("formal_blockers_present")
    if metric["missing_seeds"]:
        reasons.add("missing_seeds")
    reasons.update(item["state"] for item in entries if item["state"] != "available")
    if any(item["payload"].get("protocol_deviation") for item in entries):
        reasons.add("protocol_deviation")
    if any(item["payload"].get("formal_eligible") is False for item in entries):
        reasons.add("formal_eligible_false")
    missing_costs = unrecorded_measurements(
        [item["payload"].get("resources", {}) for item in usable]
    )
    if missing_costs:
        reasons.add("missing_resource_provenance")
    result = {
        "result_identity": matrix_selector(payload, selection_protocol=key[4]),
        "row_key": dict(
            zip(
                (
                    "comparability_group",
                    "run_view",
                    "mechanism",
                    "method",
                    "selection_protocol",
                    "c_token",
                ),
                key,
                strict=True,
            )
        ),
        "cost_id": cost_id,
        "metric": metric,
        "additional_metrics": additional,
        "status": "incomplete" if metric["missing_seeds"] else "partial",
        "reasons": sorted(reasons),
        "formal_blockers": blockers,
        "formal_eligible_by_seed": {
            str(item["payload"]["seed"]): item["payload"].get("formal_eligible") for item in entries
        },
        "unrecorded_measurements": missing_costs,
        "members": sorted(str(item["path"]) for item in entries),
        "source_protocol_versions": sorted(
            {item["payload"]["protocol_version"] for item in entries}
        ),
        "protocol_revision": payload.get("protocol_revision"),
        "source_protocol_revisions": sorted(
            {
                item["payload"]["protocol_revision"]
                for item in entries
                if item["payload"].get("protocol_revision")
            }
        ),
        "comparison_by_seed": {
            str(item["payload"]["seed"]): _comparison_evidence(item["payload"]) for item in usable
        },
        "research_settings_by_seed": {
            str(item["payload"]["seed"]): copy.deepcopy(item["payload"].get("research_settings"))
            for item in entries
        },
        "research_settings_summaries": sorted(
            {
                (item["payload"].get("research_settings") or {}).get("summary")
                or "历史记录：按原 manifest 设置读取"
                for item in entries
            }
        ),
    }
    if uses_clean_support(payload):
        result["pure_pu_comparison_eligible"] = False
        result["extra_supervision_by_seed"] = {
            str(item["payload"]["seed"]): item["payload"].get("extra_supervision", {})
            for item in entries
        }
    return result


def _reproducibility(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read recorded code/lock identity, including an optional shared evidence file."""
    identity = {
        "code_identity": payload.get("code_identity", {}),
        "dependency_lock": payload.get("dependency_lock", {}),
        "environment": payload.get("resources", {}).get("environment", {}),
    }
    missing = []
    batch = payload.get("batch")
    if batch:
        identity["batch"] = copy.deepcopy(batch)
        filename = batch.get("evidence_file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise SummaryError(f"invalid shared evidence reference in {path}")
        candidates = [parent / "batch_evidence" / filename for parent in path.parents]
        packet_path = next((item for item in candidates if _filesystem_path(item).is_file()), None)
        if packet_path is None:
            missing.append("shared_evidence_file")
        else:
            from pu_toolbox.utils.serialization import file_hash

            from .batch_evidence import load

            try:
                readable = _filesystem_path(packet_path)
                packet = load(readable)
                if (
                    file_hash(readable) != batch.get("evidence_sha256")
                    or packet["batch_id"] != batch.get("batch_id")
                    or packet["protocol_sha256"] != payload.get("protocol_sha256")
                ):
                    raise ValueError("shared evidence conflicts with its manifest reference")
                unit = payload["execution_unit"]
                for field in ("method", "dataset", "training_path"):
                    if packet["unit"][field] != unit[field]:
                        raise ValueError(f"shared evidence conflicts on {field}")
                if payload["seed"] not in packet["seeds"]:
                    raise ValueError("shared evidence does not cover this seed")
                identity["code_identity"] = {
                    key: packet[key]
                    for key in ("code_commit", "code_commit_state", "code_commit_dirty")
                }
                identity["dependency_lock"] = packet["dependency_lock"]
                identity["environment"] = packet["environment"]
                identity["evidence_path"] = str(packet_path)
            except (ValueError, OSError, KeyError) as exc:
                raise SummaryError(f"invalid shared evidence for {path}: {exc}") from exc
    if not identity["code_identity"].get("code_commit"):
        missing.append("code_commit")
    if not identity["dependency_lock"].get("sha256"):
        missing.append("dependency_lock")
    if not identity["environment"].get("python_version"):
        missing.append("environment.python_version")
    return {**identity, "state": "unavailable" if missing else "recorded", "missing": missing}


def build_summary(
    manifest_paths: Iterable[str | Path],
    *,
    expected_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    excluded_paths: Iterable[str | Path] = (),
    pending_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Summarize explicit read-only result inputs; exclusions never become observations.

    Pending/excluded paths may identify directories or unreceived manifest files.
    Completeness concerns only rows present in these inputs, not an inferred
    experiment-wide grid. Existing admission blockers remain recorded verbatim.
    """
    seeds = tuple(expected_seeds)
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise SummaryError("expected seeds must be unique nonnegative integers")
    paths = _paths(manifest_paths)
    excluded = set(_paths(excluded_paths, allow_missing=True))
    pending = set(_paths(pending_paths, allow_missing=True))
    if excluded & pending:
        raise SummaryError("result input cannot be both excluded and pending")
    groups: dict[tuple, list] = defaultdict(list)
    costs: dict[tuple, list] = defaultdict(list)
    sources = []
    diagnostic = []
    references: dict = {}
    identities: set = set()
    counts: Counter = Counter()
    for path in sorted(set(paths) | excluded | pending, key=str):
        if path in excluded or path in pending:
            state = "excluded" if path in excluded else "pending"
            sources.append({"path": str(path), "state": state})
            counts[state] += 1
            continue
        payload = _read(path, seeds)
        _check_consistency(payload, references, identities)
        risk = _recorded_risk(payload)
        failed = (
            not payload.get("selection")
            or not payload.get("test_results")
            or payload.get("execution_mode") == "rejected_versioned_pilot"
            or payload["execution_unit"].get("runnable") is not True
        )
        state = "risk" if risk else "failed" if failed else "available"
        source = {
            "path": str(path),
            "state": state,
            "seed": payload["seed"],
            "protocol_version": payload["protocol_version"],
            "protocol_sha256": payload.get("protocol_sha256"),
            "formal_eligible": payload.get("formal_eligible"),
            "formal_blockers": payload.get("formal_blockers", []),
            "protocol_deviation": payload.get("protocol_deviation", []),
            "failures": payload.get("failures", []),
            "numerical_risk": risk,
            "selection": copy.deepcopy(payload.get("selection", {})),
            "budget": copy.deepcopy(payload["budget"]),
            "research_settings": copy.deepcopy(payload.get("research_settings")),
            "reproducibility": _reproducibility(path, payload),
        }
        mechanism = payload["generation"]["train"]["mechanism"]
        protocols = (
            ["OA"]
            if mechanism != "scar"
            else ["PA+clean-support" if uses_clean_support(payload) else "PA", "OA"]
        )
        missing = False
        for name in protocols:
            key = row_key(payload, selection_protocol=name)
            block = payload.get("test_results", {}).get(name, {})
            value = block.get("accuracy")
            valid = (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
                and 0 <= value <= 1
            )
            if valid and not failed:
                _check_selection(payload, name)
            if not valid:
                missing = True
            groups[key].append(
                {
                    "path": path,
                    "payload": payload,
                    "value": float(value) if valid and state == "available" else None,
                    "state": state if valid else "missing_accuracy",
                    "reproducibility": source["reproducibility"],
                }
            )
        if missing and state == "available":
            source["state"] = "missing_accuracy"
        counts[source["state"]] += 1
        if mechanism != "scar" and "PA" in payload.get("test_results", {}):
            diagnostic.append({"path": str(path), "selection_protocol": "PA", "reason": "OA only"})
        sources.append(source)
        cost_key = row_key(payload, selection_protocol="OA")
        if state == "available":
            costs[cost_key].append((payload["seed"], payload.get("resources", {})))
    rows, run_costs = [], []
    for key, entries in sorted(groups.items()):
        cost_key = (*key[:4], "OA", key[5])
        cost_id = "|".join(cost_key)
        rows.append(_row(key, entries, seeds, cost_id))
    for key, observations in sorted(costs.items()):
        run_costs.append(
            {
                "cost_id": "|".join(key),
                "costs": summarize_costs(observations, expected_seeds=seeds),
                "unrecorded_measurements": unrecorded_measurements(
                    [item for _, item in observations]
                ),
            }
        )
    versions = sorted({item["protocol_version"] for item in sources if "protocol_version" in item})
    revisions = sorted(
        {
            item["protocol_revision"]
            for item in sources
            if isinstance(item.get("protocol_revision"), str) and item["protocol_revision"]
        }
    )
    return {
        "schema_version": "survey-summary-1",
        "generated_at": None,
        "protocol_version": versions[0] if len(versions) == 1 else "mixed-v3-results",
        "protocol_revision": (
            revisions[0] if len(revisions) == 1 else ("mixed-v3-revisions" if revisions else None)
        ),
        "protocol_sha256": None,
        "source_protocol_versions": versions,
        "source_protocol_revisions": revisions,
        "metric_unit": "fraction",
        "report_mode": "read_only_results",
        "admission_status": "partial",
        "formal_readiness_assessed": False,
        "overall": "partial"
        if not rows or any(row["status"] == "incomplete" for row in rows)
        else "pass",
        "rows": rows,
        "run_costs": run_costs,
        "sources": sources,
        "diagnostic_results": diagnostic,
        "checks": [],
        "refusals": [],
        "coverage": {
            "manifests": len(paths),
            "rows": len(rows),
            "run_costs": len(run_costs),
            **{
                name: counts[name]
                for name in (
                    "available",
                    "pending",
                    "excluded",
                    "failed",
                    "missing_accuracy",
                    "risk",
                )
            },
            "diagnostic_results": len(diagnostic),
            "rows_by_tier": {"formal": 0, "partial": len(rows), "diagnostic": 0},
        },
    }
