"""Reconcile explicit matrix cells without inferring expectations from manifests.

This preparatory layer neither generates the final Survey matrix nor grants
formal eligibility. Callers supply normalized, audited identities and explicit
exclusions; the existing identity, provenance and fairness gates remain required.
No draft protocol or historical result tree is loaded implicitly.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .survey_protocol import digest
from .survey_summary import RESULT_IDENTITY_FIELDS, manifest_selection_protocol

IDENTITY_FIELDS = (
    *RESULT_IDENTITY_FIELDS,
    "seed",
    "backbone",
    "comparability_group",
    "run_view",
    "calibration_applied",
    "selection_path",
    "budget_identity",
)
_ADDITIONAL_METRICS = ("auc", "f1", "precision", "recall")


def _identity_key(cell: Mapping[str, Any]) -> str:
    identity = cell.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("coverage cell requires an identity object")
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"coverage identity missing fields: {missing}")
    if type(identity["seed"]) is not int or identity["seed"] < 0:
        raise ValueError("coverage seed must be a nonnegative integer")
    if type(identity["calibration_applied"]) is not bool:
        raise ValueError("coverage calibration_applied must be a boolean")
    for field in set(IDENTITY_FIELDS) - {"seed", "calibration_applied"}:
        if not isinstance(identity[field], str) or not identity[field].strip():
            raise ValueError(f"coverage identity {field} must be a nonempty string")
    manifest_selection_protocol(identity["selection_protocol"])
    return digest(dict(identity))


def _index(cells: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    indexed = {}
    for cell in cells:
        key = _identity_key(cell)
        if key in indexed:
            raise ValueError(f"duplicate {label} coverage cell")
        indexed[key] = cell
    return indexed


def expand_expected_cells(
    rows: Sequence[Mapping[str, Any]], *, seeds: Sequence[int]
) -> list[dict[str, Any]]:
    """Expand already-resolved matrix rows, including exclusions, over seeds.

    Rows contain ``identity`` without seed and optionally ``excluded_reason``.
    Mechanisms, selection paths and c-independent oracle rows must come from
    the caller's matrix, not from method-name heuristics in this function.
    """
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("coverage seeds must be nonempty unique nonnegative integers")
    cells = []
    for row in rows:
        if "seed" in row["identity"]:
            raise ValueError("expected matrix row must not already contain a seed")
        for seed in sorted(seeds):
            cell = copy.deepcopy(dict(row))
            cell["identity"]["seed"] = seed
            cells.append(cell)
    _index(cells, "expected")
    return cells


def _unavailable_reason(block: Mapping[str, Any], metric: str) -> str | None:
    value = block.get(metric)
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        return None
    reason = block.get(f"{metric}_unavailable_reason")
    if reason:
        return str(reason)
    if metric not in block:
        return "Legacy manifest does not record this metric."
    return "Manifest metric is unavailable or not a finite number."


def reconcile_result_coverage(
    expected_cells: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report every declared cell, including wholly absent rows and exclusions.

    Observations pair the same normalized ``identity`` with a runner ``manifest``
    and optional ``source``. They must be normalized by the matrix adapter, not
    relabeled to match expectations. This layer does not replace those adapters
    or assess protocol/registry/comparison provenance and fairness.

    ``complete`` means the selected metric exists, never that a run may be ranked.
    Secondary-metric gaps and concurrent eligibility problems remain separate.
    PN results stay c-independent here; display-time broadcasting is downstream.
    """
    expected = _index(expected_cells, "expected")
    actual = _index(observations, "observation")
    results = []
    for key, cell in expected.items():
        result = {"identity": copy.deepcopy(cell["identity"])}
        observation = actual.get(key)
        exclusion = cell.get("excluded_reason")
        if "excluded_reason" in cell and (not isinstance(exclusion, str) or not exclusion.strip()):
            raise ValueError("excluded coverage cell requires a nonempty excluded_reason")
        if exclusion:
            result.update(
                status="excluded",
                excluded_reason=exclusion,
                unexpected_observation=observation is not None,
            )
        elif observation is None:
            result["status"] = "missing"
        else:
            manifest = observation["manifest"]
            protocol = manifest_selection_protocol(cell["identity"]["selection_protocol"])
            block = manifest.get("test_results", {}).get(protocol)
            result.update(
                source=observation.get("source"),
                formal_eligible=manifest.get("formal_eligible"),
                formal_blockers=copy.deepcopy(manifest.get("formal_blockers", [])),
                protocol_deviation=copy.deepcopy(manifest.get("protocol_deviation", [])),
                failures=copy.deepcopy(manifest.get("failures", [])),
            )
            if protocol not in manifest.get("selection", {}) or not isinstance(block, Mapping):
                result["status"] = "incomplete"
            else:
                metric = cell["identity"]["metric"]
                reason = _unavailable_reason(block, metric)
                result["additional_metric_gaps"] = {
                    name: gap
                    for name in _ADDITIONAL_METRICS
                    if (gap := _unavailable_reason(block, name)) is not None
                }
                if reason is not None:
                    result.update(status="metric_unavailable", metric_unavailable_reason=reason)
                else:
                    result["value"] = float(block[metric])
                    if result["protocol_deviation"]:
                        result["status"] = "protocol_deviation"
                    elif manifest.get("formal_eligible") is False or result["formal_blockers"]:
                        result["status"] = "formal_blocked"
                    else:
                        result["status"] = "complete"
        results.append(result)
    unexpected = [
        {"identity": copy.deepcopy(cell["identity"]), "source": cell.get("source")}
        for key, cell in actual.items()
        if key not in expected
    ]
    return {
        "schema_version": "survey-result-coverage-preparation-1",
        "formal_readiness_assessed": False,
        "cells": results,
        "counts": dict(sorted(Counter(cell["status"] for cell in results).items())),
        "unexpected": unexpected,
        "coverage_complete": bool(results)
        and not unexpected
        and all(
            cell["status"] in {"complete", "excluded", "formal_blocked", "protocol_deviation"}
            and not cell.get("unexpected_observation", False)
            for cell in results
        ),
    }
