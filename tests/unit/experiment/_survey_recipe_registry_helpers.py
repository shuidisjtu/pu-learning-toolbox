# tests/unit/experiment/_survey_recipe_registry_helpers.py
"""Shared fixtures for the P4.1 recipe-registry tests.

The fixtures are derived from the *frozen* protocol rather than written out by
hand, so a test that fails is about the rule it names and not about a stale copy
of ``budgets`` or a training path the matrix has since renamed.  Only the fields
a test is actually about are spelled as literals.

Three builders cover every case in the suite: :func:`base_registry` produces a
payload that validates clean, and the tests perturb exactly one thing from
there.
"""

from __future__ import annotations

import copy
from typing import Any

from pu_toolbox.experiment.survey_protocol import digest, load_protocol
from pu_toolbox.experiment.survey_recipe_registry import (
    SCHEMA_VERSION,
    SELECTION_SPEC_REF,
    canonical_digest,
    profile_id,
    resolve_candidate,
    resolved_params_digest,
)

#: The one evidence entry every default profile and candidate cites.
DEFAULT_REF = "project_method_ledger_nnpu"
EXTERNAL_REF = "pubench_hparams_registry"


def codes(findings: list[Any]) -> list[str]:
    """The finding codes, sorted, so a test asserts on the set it expects."""
    return sorted({finding.code for finding in findings})


def unit_rows(protocol: dict[str, Any], method: str) -> list[dict[str, Any]]:
    return [row for row in protocol["execution_units"] if row["method"] == method]


def training_paths(protocol: dict[str, Any], method: str) -> list[str]:
    """The training paths the matrix declares, in declaration order."""
    seen: list[str] = []
    for row in unit_rows(protocol, method):
        if row["training_path"] not in seen:
            seen.append(row["training_path"])
    return seen


def budget_family(protocol: dict[str, Any], method: str) -> str:
    return protocol["method_profiles"][method]["budget"]


def evidence(**overrides: Any) -> dict[str, Any]:
    """One ``source_evidence`` entry: an in-repo file with line numbers."""
    entry: dict[str, Any] = {
        "type": "in_repo",
        "file": "pu_toolbox/experiment/method_ledger.json",
        "lines": "255-301",
    }
    entry.update(overrides)
    return entry


def evidence_index(**overrides: Any) -> dict[str, Any]:
    index = {DEFAULT_REF: evidence(), EXTERNAL_REF: external_evidence()}
    index.update(overrides)
    return index


def external_evidence(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "external_local",
        "path": "F:/Temp/PUBench",
        "file": "core/hparams_registry.py",
        "lines": "20-129",
        "reproducibility": "external_only",
    }
    entry.update(overrides)
    return entry


def inner_search(method: str = "pusb_kernel") -> dict[str, Any]:
    """The identity-and-pointer form the registry records an inner search in.

    No size and no axes: ``budgets.<family>.internal_hyperparameter_pairs``
    declares the count, and ``survey_protocol`` already checks it against the
    grids, so restating either here would be a second declaration of the same
    fact.
    """
    return {
        "inner_search_id": f"{method}_sigma_reg_cv",
        "grid_ref": f"survey_protocol_v1.method_profiles.{method}.params",
    }


def candidate(candidate_id: str = "nnpu_default_v1", **overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "recipe_candidate_id": candidate_id,
        "candidate_role": "current_protocol_binding",
        "overrides": {},
        "resolved_snapshot": None,
        "source_refs": [DEFAULT_REF],
    }
    value.update(overrides)
    return value


def profile(
    method: str = "nnpu",
    *,
    protocol: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """One profile, with its budget and scope read off the frozen matrix."""
    protocol = protocol or load_protocol()
    value: dict[str, Any] = {
        "method": method,
        "recipe_family": "oracle" if method == "pn_oracle" else "pu",
        "variant_id": f"survey_shared_{method}",
        "recipe_lifecycle": "draft",
        "scope": {
            "datasets": sorted({row["dataset"] for row in unit_rows(protocol, method)}),
            "training_paths": training_paths(protocol, method),
        },
        "base_ref": f"survey_protocol_v1.method_profiles.{method}",
        "selection_spec_ref": SELECTION_SPEC_REF,
        "source_refs": [DEFAULT_REF],
        "fixed_overrides": {},
        "candidate_pool": [candidate(f"{method}_default_v1")],
        "inner_search": None,
        "budget_binding_ref": f"survey_protocol_v1.budgets.{budget_family(protocol, method)}",
        "review": {"reviewers": [], "open_questions": []},
    }
    value.update(overrides)
    return value


def excluded_profile(method: str = "kldce", **overrides: Any) -> dict[str, Any]:
    """A profile that carries no candidate, with the reason that makes it legal.

    The reason goes in ``exclusion_reason``, not into the profile's open
    questions: an exclusion is a decision that was taken, and an open question
    is one that was not.
    """
    value = profile(
        method,
        recipe_lifecycle="excluded",
        candidate_pool=None,
        exclusion_reason="flip_probability has no mechanism-safe binding",
    )
    value.update(overrides)
    return value


def base_registry(
    methods: tuple[str, ...] = ("nnpu",),
    *,
    protocol: dict[str, Any] | None = None,
    profiles: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A registry that validates clean unless a test perturbs one thing."""
    protocol = protocol or load_protocol()
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recipe_registry_version": "survey-recipes-v0.1-draft",
        "recipe_lifecycle": "draft",
        "protocol_basis": {
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": digest(protocol),
        },
        "selection_spec_ref": SELECTION_SPEC_REF,
        "profiles": profiles if profiles is not None else {m: profile(m) for m in methods},
        "source_evidence": evidence_index(),
        "validation": {
            "recipe_candidate_ids_unique": True,
            "source_evidence_required": True,
            "test_truth_forbidden": True,
            "parameter_overrides_forbidden_in_v1": True,
        },
    }
    value.update(overrides)
    return value


def protocol_with_outer_candidates(count: int, family: str = "minibatch") -> dict[str, Any]:
    """The frozen protocol with one budget family widened, for pool-size cases.

    A copy rather than a literal so the widened family still carries every other
    declaration (``unit``, ``learning_rate``) the validator reads.

    Both of the protocol's declarations of the count move together -- the budget
    family's ``outer_candidates`` and the top-level ``candidate_pool``.  Widening
    only one would produce a protocol that contradicts itself, and the registry
    validator would rightly report the registry for failing to match both.
    """
    protocol = copy.deepcopy(load_protocol())
    protocol["budgets"][family]["outer_candidates"] = count
    protocol["candidate_pool"] = [{} for _ in range(count)]
    return protocol


def bound_manifest(
    registry_value: dict[str, Any],
    method: str = "nnpu",
    *,
    protocol: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A complete registry-bound manifest for one profile's first candidate."""
    protocol = protocol or load_protocol()
    prof = registry_value["profiles"][method]
    chosen = prof["candidate_pool"][0]
    inner = prof.get("inner_search") or {}
    value: dict[str, Any] = {
        "recipe_binding_status": "registry_bound",
        "recipe_registry_version": registry_value["recipe_registry_version"],
        "recipe_registry_sha256": canonical_digest(registry_value),
        "recipe_profile_id": profile_id(prof),
        "recipe_candidate_id": chosen["recipe_candidate_id"],
        "resolved_params_sha256": resolved_params_digest(resolve_candidate(prof, chosen, protocol)),
        "inner_search_id": inner.get("inner_search_id"),
    }
    value.update(overrides)
    return value
