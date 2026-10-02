"""P4.1 central experiment recipe registry: schema, validation and digests.

The project already holds four parameter-adjacent sources, and none of them
answers "which recipe did *this* survey adopt, out of which candidate pool, at
which version":

``builtin_methods.py``      estimator registration, capabilities, prior needs
``method_ledger.json``      paper/official-code provenance, adaptation level
``survey_protocol_v1.json`` budgets, execution matrix, frozen pilot values
``benchmarks/*/configs``    historical benchmark and tuning configs

The registry is the missing *binding* layer (plan §1.2), and it holds identity,
not values.  A profile points at ``survey_protocol_v1.method_profiles.<method>``
via ``base_ref``; parameter values stay in the protocol, where they already are
single-sourced.  v1 therefore rejects every parameter-level override outright:
a non-empty ``overrides`` is a protocol change, not a recipe choice
(D-P4.1-1, D-P4.1-8).

Design notes
------------
**Two new closed vocabularies, kept out of the existing ones.**
``recipe_lifecycle`` and ``recipe_binding_status`` are deliberately *not* the
audit/summary vocabularies.  Reusing ``accepted``/``pending_review`` for recipe
review, or ``blocked`` for exclusion, would make one string mean two things
depending on which report a reader opened (D-P4.1-3, and the §7.1 table listing
each collision).  They live here rather than in ``survey_summary`` because the
registry is the only producer of them; the audit layer is a consumer.

**Provenance is evidence, not identity.**  ``source_refs``/``source_evidence``
never enter the digest (D-P4.1-4 rule 3): several of them are ``external_local``
paths that do not exist inside the repository, so folding them in would make the
digest machine-dependent and un-recomputable elsewhere.  ``digest_payload()``
is the auditable projection that decides this, and it is public so a reviewer
can see exactly what the digest covers.

**Declared facts stay where they are declared.**  ``outer_candidates``, the
frozen ``candidate_pool`` and ``internal_hyperparameter_pairs`` are already
declared by the protocol, and ``survey_protocol`` already enforces that they
agree with one another.  The registry quotes identity and points at them; it
never restates a count.  Where a check would need a declared counterpart that
exists nowhere, the answer is not to invent the field here -- that is what the
unresolved-questions record beside the draft is for.

**A malformed registry is a finding, not a crash** -- the same rule
``survey_audit`` states for manifests.  ``load_registry`` raises on a payload
that is not a registry at all, while ``validate_registry`` reports every problem
it can see instead of stopping at the first, because its caller's job is to
*list* what is wrong.  Findings stay independent of ``survey_audit.make_check``
so that a pure registry validator can be tested without the audit vocabulary;
stage 2/3 maps these codes into audit findings.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .survey_protocol import digest

SCHEMA_VERSION = "survey-recipe-registry-1.0"

#: Recipe review lifecycle.  Independent closed set -- see D-P4.1-3, which pairs
#: each value with the existing status it is *not* allowed to be confused with.
RECIPE_LIFECYCLE: tuple[str, ...] = (
    "draft",
    "under_review",
    "signed_off",
    "locked",
    "retired",
    "excluded",
)

#: Whether a run bound the registry *before* training.  Says nothing about the
#: result's formal eligibility or the recipe's review state (D-P4.1-2).
RECIPE_BINDING_STATUS: tuple[str, ...] = ("legacy_unbound", "registry_bound")

#: ``oracle`` profiles share the file but must never share a candidate pool with
#: the PU methods they measure (D-P4.1-7).
RECIPE_FAMILIES: tuple[str, ...] = ("pu", "oracle")

#: What a candidate *is* in a recipe pool.  Closed to one value in v1 because
#: v1 registers the current binding and nothing else (D-P4.1-8): an open string
#: is exactly the thing that drifts, and the whole point of the registry is that
#: a recipe cannot be reworded after the fact.  Adding a role later is meant to
#: cost a plan revision.
CANDIDATE_ROLES: tuple[str, ...] = ("current_protocol_binding",)

#: The one selection-policy entry point.  The registry quotes this reference and
#: copies no PA/OA, checkpoint or threshold literal of its own (D-P4.1-1).
SELECTION_SPEC_REF = "survey_protocol_v1.selection_spec"

BASE_REF_PREFIX = "survey_protocol_v1.method_profiles."
BUDGET_BINDING_PREFIX = "survey_protocol_v1.budgets."

#: Top-level manifest fields a registry-bound run must carry (D-P4.1-9).  They
#: belong on the manifest's top level and never inside ``representation``, whose
#: digest is part of the fairness comparison.
REGISTRY_BOUND_MANIFEST_FIELDS: tuple[str, ...] = (
    "recipe_registry_version",
    "recipe_registry_sha256",
    "recipe_profile_id",
    "recipe_candidate_id",
    "resolved_params_sha256",
    "inner_search_id",
)

#: Key names that would mean a candidate was configured against the test split.
#: Matched anywhere inside a candidate, at any depth.
FORBIDDEN_TRUTH_KEYS: frozenset[str] = frozenset({"test", "test_labels", "test_truth"})

#: Where a source reference points.  ``external_local`` is a path on one
#: machine -- registrable, but not something a second reviewer can rebuild from
#: the repository (D-P4.1-4 rules 1 and 4).
SOURCE_REF_TYPES: tuple[str, ...] = ("in_repo", "external_local")

#: Severities a registry finding may carry.  A subset of the audit vocabulary on
#: purpose: the registry has no "not applicable" verdict to report.
FINDING_SEVERITIES: tuple[str, ...] = ("error", "warning", "info")

#: Every code the validators below can emit, in one place so a typo fails the
#: suite instead of reaching a report.  ``REASONS`` in ``survey_summary`` is the
#: analogous closed set for the audit layer.
REASON_CODES: tuple[str, ...] = (
    "unsupported_schema_version",
    "missing_registry_field",
    "unknown_recipe_lifecycle",
    "unknown_recipe_family",
    "unknown_method",
    "duplicate_profile_method",
    "duplicate_profile_id",
    "duplicate_candidate_id",
    "base_ref_mismatch",
    "selection_spec_ref_mismatch",
    "unknown_budget_binding",
    "budget_unit_mismatch",
    "scope_not_in_protocol",
    "scope_contains_run_view",
    "fixed_overrides_forbidden",
    "candidate_overrides_forbidden",
    "empty_candidate_pool",
    "candidate_pool_null_not_excluded",
    "missing_exclusion_reason",
    "unexpected_exclusion_reason",
    "unknown_candidate_role",
    "candidate_count_mismatch",
    "excluded_method_still_runnable",
    "test_truth_in_candidate",
    "missing_resolved_snapshot",
    "resolved_snapshot_mismatch",
    "protocol_basis_mismatch",
    "inner_search_shape_invalid",
    "inner_search_budget_mismatch",
    "missing_inner_search",
    "unknown_grid_ref",
    "oracle_in_pu_candidate_pool",
    "oracle_c_grid_broadcast",
    "unknown_source_ref",
    "source_refs_missing",
    "source_evidence_shape_invalid",
    "source_evidence_external_only",
    "manifest_registry_field_missing",
    "manifest_registry_digest_mismatch",
    "unknown_recipe_candidate",
    "unknown_recipe_binding_status",
    "partial_recipe_binding",
    "resolved_params_digest_mismatch",
    "inner_search_id_mismatch",
)

#: Identity keys the digest covers.  ``source_refs``/``source_evidence``/
#: ``review`` are excluded: the first two are provenance (D-P4.1-4 rule 3), and
#: ``review`` holds reviewers and open questions, which are neither a binding
#: nor an identity -- ``recipe_lifecycle`` is where review *state* lives.
_DIGEST_EXCLUDED_TOP = frozenset({"source_evidence"})
_DIGEST_EXCLUDED_PROFILE = frozenset({"source_refs", "review"})
_DIGEST_EXCLUDED_CANDIDATE = frozenset({"source_refs"})


@dataclass(frozen=True)
class Finding:
    """One defect, named by the closed code that produced it.

    ``path`` is dotted down to the offending field so a reader does not have to
    pick a culprit out of a whole profile, the same reasoning that makes
    ``survey_audit`` carry ``scope``.
    """

    code: str
    path: str
    message: str
    severity: str = "error"

    def __post_init__(self) -> None:
        if self.code not in REASON_CODES:
            raise ValueError(f"unknown registry reason code: {self.code!r}")
        if self.severity not in FINDING_SEVERITIES:
            raise ValueError(f"unknown finding severity: {self.severity!r}")


def _finding(code: str, path: str, message: str, *, severity: str = "error") -> Finding:
    return Finding(code=code, path=path, message=message, severity=severity)


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load a registry payload; a file that is not one fails loud.

    Shape only -- whether the contents hold together is ``validate_registry``'s
    job, which reports rather than raises.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("profiles"), dict):
        raise ValueError(f"{path} is not a valid survey recipe registry file.")
    return value


def profile_id(profile: Mapping[str, Any]) -> str:
    """The stable ``method/variant_id`` identity a manifest records."""
    return f"{profile.get('method')}/{profile.get('variant_id')}"


def manifest_has_binding_fields(manifest: Mapping[str, Any]) -> bool:
    """Whether *any* recipe binding field is present.

    Used for the fail-closed half of D-P4.1-2: a payload carrying one binding
    field is claiming to be registry-bound, so a missing remainder is a defect
    rather than grounds to fall back to legacy.
    """
    return any(field in manifest for field in REGISTRY_BOUND_MANIFEST_FIELDS)


def digest_payload(registry: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical projection the registry digest is computed over.

    Excludes provenance and reviewer identity, and nothing else: everything the
    digest *must* move for -- candidate identity, lifecycle, inner-search
    identity, the protocol basis and every locked ``resolved_snapshot`` -- is
    kept (D-P4.1-4 rule 3).
    """
    projected: dict[str, Any] = {
        key: value for key, value in registry.items() if key not in _DIGEST_EXCLUDED_TOP
    }
    profiles: dict[str, Any] = {}
    for method, profile in (registry.get("profiles") or {}).items():
        if not isinstance(profile, Mapping):
            profiles[method] = profile
            continue
        entry = {k: v for k, v in profile.items() if k not in _DIGEST_EXCLUDED_PROFILE}
        pool = profile.get("candidate_pool")
        if isinstance(pool, Sequence) and not isinstance(pool, (str, bytes)):
            entry["candidate_pool"] = [
                {k: v for k, v in candidate.items() if k not in _DIGEST_EXCLUDED_CANDIDATE}
                if isinstance(candidate, Mapping)
                else candidate
                for candidate in pool
            ]
        profiles[method] = entry
    projected["profiles"] = profiles
    return projected


def canonical_digest(registry: Mapping[str, Any]) -> str:
    """SHA256 over the canonical projection, recomputable by any consumer.

    Shares ``survey_protocol.digest``'s normalization (sorted keys, compact
    separators, ``allow_nan=False``) so the registry digest and the protocol
    digest cannot drift apart in how they are computed (plan §6.1).
    """
    return digest(digest_payload(registry))


def resolved_params_digest(snapshot: Mapping[str, Any]) -> str:
    """``resolved_params_sha256``: the digest of one candidate's snapshot."""
    return digest(snapshot)


def resolve_candidate(
    profile: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand ``base_ref`` and the budget binding into a canonical snapshot.

    This is the derived artifact a locked registry stores per candidate so that
    a second reviewer can rebuild the binding without depending on a later
    revision of the protocol or the provenance records (plan §13 decision 3).
    It is *generated*, never hand-edited -- ``validate_registry`` recomputes it
    and compares, which is what makes a checked-in snapshot auditable.
    """
    method = profile["method"]
    unit = protocol["method_profiles"][method]
    budget_name = unit["budget"]
    inner_search = profile.get("inner_search")
    return {
        "recipe_profile_id": profile_id(profile),
        "recipe_candidate_id": candidate["recipe_candidate_id"],
        "recipe_family": profile["recipe_family"],
        "method": method,
        "base_ref": profile["base_ref"],
        "budget_binding_ref": f"{BUDGET_BINDING_PREFIX}{budget_name}",
        "model_family": unit["model_family"],
        "params": dict(unit["params"]),
        "budget_declaration": dict(protocol["budgets"][budget_name]),
        "inner_search_id": inner_search.get("inner_search_id") if inner_search else None,
        "overrides": dict(candidate.get("overrides") or {}),
    }


def _truth_keys(value: Any, path: str) -> list[Finding]:
    """Find test-truth key names anywhere inside a candidate subtree."""
    findings: list[Finding] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN_TRUTH_KEYS:
                findings.append(
                    _finding(
                        "test_truth_in_candidate",
                        f"{path}.{key}",
                        "test-split identity cannot appear in a candidate configuration",
                    )
                )
            findings.extend(_truth_keys(child, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            findings.extend(_truth_keys(child, f"{path}[{index}]"))
    return findings


def validate_registry(registry: Mapping[str, Any], protocol: Mapping[str, Any]) -> list[Finding]:
    """Every schema and protocol-consistency defect visible in *registry*.

    *protocol* is a parameter rather than loaded here because the caller already
    holds the frozen matrix, and re-validating it per call would be the same
    mistake ``survey_audit.matrix_conflicts`` documents for the comparison
    matrix.
    """
    findings: list[Finding] = []

    if registry.get("schema_version") != SCHEMA_VERSION:
        findings.append(
            _finding(
                "unsupported_schema_version",
                "schema_version",
                f"expected {SCHEMA_VERSION!r}, got {registry.get('schema_version')!r}",
            )
        )
    if not registry.get("recipe_registry_version"):
        findings.append(
            _finding(
                "missing_registry_field",
                "recipe_registry_version",
                "a registry must name its own human-readable version",
            )
        )
    if registry.get("recipe_lifecycle") not in RECIPE_LIFECYCLE:
        findings.append(
            _finding(
                "unknown_recipe_lifecycle",
                "recipe_lifecycle",
                f"must be one of {list(RECIPE_LIFECYCLE)}; "
                f"got {registry.get('recipe_lifecycle')!r}",
            )
        )
    if registry.get("selection_spec_ref") != SELECTION_SPEC_REF:
        findings.append(
            _finding(
                "selection_spec_ref_mismatch",
                "selection_spec_ref",
                f"the only selection-policy entry point is {SELECTION_SPEC_REF!r}",
            )
        )
    findings.extend(_validate_protocol_basis(registry, protocol))

    profiles = registry.get("profiles") or {}
    if not profiles:
        findings.append(
            _finding("missing_registry_field", "profiles", "a registry needs at least one profile")
        )
    seen_methods: set[str] = set()
    seen_profile_ids: set[str] = set()
    seen_candidate_ids: set[str] = set()
    for method, profile in profiles.items():
        findings.extend(
            _validate_profile(
                method=method,
                profile=profile,
                protocol=protocol,
                seen_methods=seen_methods,
                seen_profile_ids=seen_profile_ids,
                seen_candidate_ids=seen_candidate_ids,
            )
        )
    findings.extend(validate_source_evidence(registry))
    return findings


def _validate_protocol_basis(
    registry: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[Finding]:
    """``protocol_basis`` is a source check only -- it writes nothing back."""
    basis = registry.get("protocol_basis")
    if not isinstance(basis, Mapping):
        return [
            _finding(
                "missing_registry_field",
                "protocol_basis",
                "a registry must record the frozen protocol it was drafted against",
            )
        ]
    findings: list[Finding] = []
    if basis.get("protocol_version") != protocol.get("protocol_version"):
        findings.append(
            _finding(
                "protocol_basis_mismatch",
                "protocol_basis.protocol_version",
                f"registry cites {basis.get('protocol_version')!r}, "
                f"protocol is {protocol.get('protocol_version')!r}",
            )
        )
    if basis.get("protocol_sha256") != digest(protocol):
        findings.append(
            _finding(
                "protocol_basis_mismatch",
                "protocol_basis.protocol_sha256",
                "registry cites a protocol digest that does not match the loaded protocol",
            )
        )
    return findings


def _validate_profile(
    *,
    method: str,
    profile: Any,
    protocol: Mapping[str, Any],
    seen_methods: set[str],
    seen_profile_ids: set[str],
    seen_candidate_ids: set[str],
) -> list[Finding]:
    """One profile's schema, scope, candidate pool and inner search."""
    path = f"profiles.{method}"
    if not isinstance(profile, Mapping):
        return [_finding("missing_registry_field", path, "a profile must be an object")]

    findings: list[Finding] = []
    declared_method = profile.get("method")
    if declared_method != method:
        findings.append(
            _finding(
                "missing_registry_field",
                f"{path}.method",
                f"profile key {method!r} must match its own method {declared_method!r}",
            )
        )
    if method in seen_methods:
        findings.append(
            _finding("duplicate_profile_method", f"{path}.method", "one profile per method")
        )
    seen_methods.add(method)
    identity = profile_id(profile)
    if identity in seen_profile_ids:
        findings.append(
            _finding("duplicate_profile_id", path, f"duplicate profile id {identity!r}")
        )
    seen_profile_ids.add(identity)

    if profile.get("recipe_family") not in RECIPE_FAMILIES:
        findings.append(
            _finding(
                "unknown_recipe_family",
                f"{path}.recipe_family",
                f"must be one of {list(RECIPE_FAMILIES)}; got {profile.get('recipe_family')!r}",
            )
        )
    if profile.get("recipe_lifecycle") not in RECIPE_LIFECYCLE:
        findings.append(
            _finding(
                "unknown_recipe_lifecycle",
                f"{path}.recipe_lifecycle",
                f"must be one of {list(RECIPE_LIFECYCLE)}; got {profile.get('recipe_lifecycle')!r}",
            )
        )

    method_profiles = protocol.get("method_profiles") or {}
    if method not in method_profiles:
        findings.append(
            _finding("unknown_method", path, f"{method!r} is not in the frozen protocol")
        )
        return findings

    expected_base = f"{BASE_REF_PREFIX}{method}"
    if profile.get("base_ref") != expected_base:
        findings.append(
            _finding(
                "base_ref_mismatch",
                f"{path}.base_ref",
                f"must reference the frozen protocol profile {expected_base!r}",
            )
        )
    if profile.get("selection_spec_ref") != SELECTION_SPEC_REF:
        findings.append(
            _finding(
                "selection_spec_ref_mismatch",
                f"{path}.selection_spec_ref",
                f"must be {SELECTION_SPEC_REF!r}",
            )
        )
    findings.extend(_validate_budget_binding(path, profile, method_profiles[method], protocol))
    findings.extend(_validate_scope(path, profile, method, protocol))
    if profile.get("fixed_overrides"):
        findings.append(
            _finding(
                "fixed_overrides_forbidden",
                f"{path}.fixed_overrides",
                "v1 registers identity only; a non-empty override is a protocol change",
            )
        )
    findings.extend(_validate_candidates(path, profile, protocol, seen_candidate_ids))
    findings.extend(_validate_inner_search(path, profile, protocol, method_profiles[method]))
    findings.extend(_validate_oracle(path, profile))
    return findings


def _validate_budget_binding(
    path: str,
    profile: Mapping[str, Any],
    unit: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[Finding]:
    """The budget binding must name the family the execution units actually use."""
    findings: list[Finding] = []
    budget_name = unit.get("budget")
    expected = f"{BUDGET_BINDING_PREFIX}{budget_name}"
    if profile.get("budget_binding_ref") != expected:
        findings.append(
            _finding(
                "unknown_budget_binding",
                f"{path}.budget_binding_ref",
                f"must be {expected!r} to match the profile's own budget family",
            )
        )
    units = [
        row
        for row in protocol.get("execution_units", [])
        if row.get("method") == profile.get("method")
    ]
    drifted = sorted({str(row.get("budget")) for row in units if row.get("budget") != budget_name})
    if drifted:
        findings.append(
            _finding(
                "budget_unit_mismatch",
                f"{path}.budget_binding_ref",
                f"execution units use {drifted} while the method profile declares {budget_name!r}",
            )
        )
    return findings


def _validate_scope(
    path: str,
    profile: Mapping[str, Any],
    method: str,
    protocol: Mapping[str, Any],
) -> list[Finding]:
    """``scope`` may only name dimensions the execution matrix declares.

    ``run_view`` is excluded on purpose: it is resolved at run time from the
    manifest and is not a protocol field, so admitting it here would invent a
    scope dimension the matrix cannot corroborate (plan §4.2).
    """
    findings: list[Finding] = []
    scope = profile.get("scope")
    if not isinstance(scope, Mapping):
        findings.append(
            _finding("missing_registry_field", f"{path}.scope", "a profile must declare its scope")
        )
        return findings
    if "run_view" in scope:
        findings.append(
            _finding(
                "scope_contains_run_view",
                f"{path}.scope.run_view",
                "run_view is resolved at run time and is not a protocol scope dimension",
            )
        )
    units = [row for row in protocol.get("execution_units", []) if row.get("method") == method]
    allowed = {
        "datasets": {str(row.get("dataset")) for row in units},
        "training_paths": {str(row.get("training_path")) for row in units},
    }
    for field, declared in allowed.items():
        claimed = scope.get(field)
        if not isinstance(claimed, Sequence) or isinstance(claimed, (str, bytes)):
            findings.append(
                _finding(
                    "missing_registry_field",
                    f"{path}.scope.{field}",
                    "scope dimensions are lists of values the matrix declares",
                )
            )
            continue
        outside = sorted({str(item) for item in claimed} - declared)
        if outside:
            findings.append(
                _finding(
                    "scope_not_in_protocol",
                    f"{path}.scope.{field}",
                    f"{outside} are not declared for {method!r} by any execution unit",
                )
            )
    return findings


def _validate_candidates(
    path: str,
    profile: Mapping[str, Any],
    protocol: Mapping[str, Any],
    seen_candidate_ids: set[str],
) -> list[Finding]:
    """The candidate pool: identity only, counted against the frozen budget.

    A null pool is legal for exactly one reason -- the method is excluded and
    therefore consumes no candidate -- and then the protocol must agree that
    nothing about it is runnable.  The count is checked against the protocol's
    own declarations -- ``budgets.<family>.outer_candidates`` and the length of
    the top-level ``candidate_pool`` -- rather than against a second
    ``outer_candidate_count`` field in the registry: both already exist, and a
    third copy would be one more place for the same number to drift
    (plan §3.4, D-P4.1-8).

    The protocol's ``candidate_pool`` is not decoration: a run's ``candidates``
    is compared against it and any difference is recorded as a protocol
    deviation, and the frozen value is ``[{}]`` -- a single candidate carrying
    no overrides.  So the registry's own pool is the same identity recorded
    twice, and its entries correspond to the protocol's position by position.
    That correspondence is why D-P4.1-1 and D-P4.1-8 forbid parameter-level
    candidates in v1: an override here would be a deviation there, not merely a
    recipe choice.  The correspondence itself is not policed by a separate rule,
    because in v1 any divergence already *is* an override.

    This is the entry point only; the pool's contract, its declared size and each
    candidate are checked by the helpers below.  They answer separate questions
    and fail for separate reasons, and a finding a reader cannot attribute to one
    of them is the kind of report ``survey_audit``'s ``scope`` field exists to
    prevent.
    """
    pool = profile.get("candidate_pool")
    if pool is None:
        return _validate_null_pool(path, profile, protocol)

    findings = _validate_stale_exclusion_reason(path, profile)
    if not isinstance(pool, Sequence) or isinstance(pool, (str, bytes)):
        findings.append(
            _finding(
                "empty_candidate_pool",
                f"{path}.candidate_pool",
                "a candidate pool is either a non-empty list or an explicit null",
            )
        )
        return findings
    if not pool:
        findings.append(_finding("empty_candidate_pool", f"{path}.candidate_pool", "pool is empty"))
    findings.extend(_validate_pool_counts(path, profile, pool, protocol))
    for index, candidate in enumerate(pool):
        findings.extend(
            _validate_one_candidate(
                f"{path}.candidate_pool[{index}]",
                candidate,
                profile,
                protocol,
                seen_candidate_ids,
            )
        )
    return findings


def _validate_null_pool(
    path: str, profile: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[Finding]:
    """A profile that carries no candidate, and what that obliges it to say."""
    findings: list[Finding] = []
    lifecycle = profile.get("recipe_lifecycle")
    if lifecycle != "excluded":
        findings.append(
            _finding(
                "candidate_pool_null_not_excluded",
                f"{path}.candidate_pool",
                "a null candidate pool is only legal for an excluded profile",
            )
        )
    if not profile.get("exclusion_reason"):
        findings.append(
            _finding(
                "missing_exclusion_reason",
                f"{path}.exclusion_reason",
                "a null candidate pool must record why the method carries none",
            )
        )
    method = profile.get("method")
    runnable = sorted(
        {
            str(row.get("dataset"))
            for row in protocol.get("execution_units", [])
            if row.get("method") == method and row.get("runnable")
        }
    )
    # Only meaningful once the lifecycle actually claims exclusion; on any other
    # lifecycle the null pool is already reported as the defect it is, and naming
    # the runnable rows too would suggest the protocol is at fault.
    if lifecycle == "excluded" and runnable:
        findings.append(
            _finding(
                "excluded_method_still_runnable",
                f"{path}.recipe_lifecycle",
                f"an excluded profile still has runnable execution units: {runnable}",
            )
        )
    return findings


def _validate_stale_exclusion_reason(path: str, profile: Mapping[str, Any]) -> list[Finding]:
    """The adopted-pool half of ``exclusion_reason``'s meaning.

    The field says *why this profile carries no candidate*, so it is present
    exactly when there is no pool.  A reason left behind on an adopted profile is
    a stale justification -- the drift this field exists to make visible, not a
    harmless leftover.
    """
    if not profile.get("exclusion_reason"):
        return []
    return [
        _finding(
            "unexpected_exclusion_reason",
            f"{path}.exclusion_reason",
            "an adopted profile has no exclusion to explain",
        )
    ]


def _validate_pool_counts(
    path: str, profile: Mapping[str, Any], pool: Sequence[Any], protocol: Mapping[str, Any]
) -> list[Finding]:
    """The pool's size against both of the protocol's declarations.

    Mirroring the pool's *contents* is not checked separately: in v1 the frozen
    pool is ``[{}]``, so any divergence is already an override, and a rule that
    cannot fire on its own would be a rule nobody can attribute.
    """
    method = profile.get("method")
    budget_name = (protocol.get("method_profiles") or {}).get(method, {}).get("budget")
    budget = (protocol.get("budgets") or {}).get(budget_name, {})
    frozen_pool = protocol.get("candidate_pool")
    declarations: dict[str, Any] = {
        f"budgets.{budget_name}.outer_candidates": budget.get("outer_candidates")
    }
    if isinstance(frozen_pool, Sequence) and not isinstance(frozen_pool, (str, bytes)):
        declarations["the protocol's candidate_pool length"] = len(frozen_pool)
    findings: list[Finding] = []
    for source, declared in declarations.items():
        if isinstance(declared, int) and len(pool) != declared:
            findings.append(
                _finding(
                    "candidate_count_mismatch",
                    f"{path}.candidate_pool",
                    f"{len(pool)} candidate(s) against {declared} declared by {source}",
                )
            )
    return findings


def _validate_one_candidate(
    candidate_path: str,
    candidate: Any,
    profile: Mapping[str, Any],
    protocol: Mapping[str, Any],
    seen_candidate_ids: set[str],
) -> list[Finding]:
    """One candidate's identity, role, overrides and snapshot."""
    if not isinstance(candidate, Mapping):
        return [_finding("missing_registry_field", candidate_path, "a candidate must be an object")]
    findings: list[Finding] = []
    candidate_id = candidate.get("recipe_candidate_id")
    if not candidate_id:
        findings.append(
            _finding(
                "missing_registry_field",
                f"{candidate_path}.recipe_candidate_id",
                "a candidate needs a stable identity, not a list position",
            )
        )
    else:
        if candidate_id in seen_candidate_ids:
            findings.append(
                _finding(
                    "duplicate_candidate_id",
                    f"{candidate_path}.recipe_candidate_id",
                    f"duplicate candidate id {candidate_id!r}",
                )
            )
        seen_candidate_ids.add(candidate_id)
    if candidate.get("candidate_role") not in CANDIDATE_ROLES:
        findings.append(
            _finding(
                "unknown_candidate_role",
                f"{candidate_path}.candidate_role",
                f"must be one of {list(CANDIDATE_ROLES)}; got {candidate.get('candidate_role')!r}",
            )
        )
    if candidate.get("overrides"):
        findings.append(
            _finding(
                "candidate_overrides_forbidden",
                f"{candidate_path}.overrides",
                "v1 registers the current binding: the frozen protocol pool carries no "
                "overrides, so this entry could not mirror it and would be a recorded "
                "protocol deviation",
            )
        )
    findings.extend(_truth_keys(candidate, candidate_path))
    findings.extend(_validate_candidate_snapshot(candidate_path, candidate, profile, protocol))
    return findings


def _validate_candidate_snapshot(
    candidate_path: str,
    candidate: Mapping[str, Any],
    profile: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[Finding]:
    """A locked registry must carry the expansion, and it must still match.

    The snapshot is a derived artifact, so the check recomputes it rather than
    trusting it: one that drifted from ``base_ref`` plus the protocol binding is
    exactly the hand-edited parameter source the registry exists to prevent
    (plan §13 decision 3).
    """
    snapshot = candidate.get("resolved_snapshot")
    if snapshot is None:
        if profile.get("recipe_lifecycle") != "locked":
            return []
        return [
            _finding(
                "missing_resolved_snapshot",
                f"{candidate_path}.resolved_snapshot",
                "a locked registry must carry the expanded snapshot for every candidate",
            )
        ]
    if not isinstance(snapshot, Mapping):
        return []
    expected = _expected_snapshot(profile, candidate, protocol)
    if expected is None or digest(expected) == digest(snapshot):
        return []
    return [
        _finding(
            "resolved_snapshot_mismatch",
            f"{candidate_path}.resolved_snapshot",
            "stored snapshot does not match the expansion of base_ref and the protocol binding",
        )
    ]


def _expected_snapshot(
    profile: Mapping[str, Any], candidate: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any] | None:
    """``resolve_candidate`` guarded for a profile whose refs are already broken."""
    method = profile.get("method")
    if method not in (protocol.get("method_profiles") or {}):
        return None
    if profile.get("base_ref") != f"{BASE_REF_PREFIX}{method}":
        return None
    if not candidate.get("recipe_candidate_id"):
        return None
    try:
        return resolve_candidate(profile, candidate, protocol)
    except KeyError:
        return None


def _validate_inner_search(
    path: str,
    profile: Mapping[str, Any],
    protocol: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> list[Finding]:
    """Inner search is registered as identity and a pointer, never as a grid.

    ``pusb_kernel``'s 72 kernel/reg pairs already exist in the protocol, and
    ``survey_protocol`` already enforces that their product is what
    ``budgets.kernel_cv.internal_hyperparameter_pairs`` declares.  The registry
    therefore records *which* search the 72 belong to and nothing about its
    size: copying the count here would be the same duplicated declaration this
    module refuses for the outer candidate count, and restating the axes would
    give the two lists somewhere to drift apart (plan §3.4, D-P4.1-8).

    What is checkable, and checked: the pointer must aim at the profile's own
    parameters, and a profile whose budget family declares an internal search
    may not leave it unregistered -- that is how the 72 combinations would
    otherwise drop out of the cost accounting unnoticed.
    """
    budget = (protocol.get("budgets") or {}).get(unit.get("budget"), {})
    declared_pairs = budget.get("internal_hyperparameter_pairs")
    inner = profile.get("inner_search")

    if inner is None:
        if declared_pairs is None:
            return []
        return [
            _finding(
                "missing_inner_search",
                f"{path}.inner_search",
                f"budgets.{unit.get('budget')} declares {declared_pairs} internal "
                "combination(s); the profile must register which search they belong to",
            )
        ]
    if not isinstance(inner, Mapping):
        return [
            _finding(
                "inner_search_shape_invalid",
                f"{path}.inner_search",
                "inner search is either null or an object naming its identity and grid",
            )
        ]

    findings: list[Finding] = []
    if not inner.get("inner_search_id"):
        findings.append(
            _finding(
                "missing_registry_field",
                f"{path}.inner_search.inner_search_id",
                "an inner search needs a stable identity for the manifest to carry",
            )
        )
    expected_grid = f"{BASE_REF_PREFIX}{profile.get('method')}.params"
    if inner.get("grid_ref") != expected_grid:
        findings.append(
            _finding(
                "unknown_grid_ref",
                f"{path}.inner_search.grid_ref",
                f"must be {expected_grid!r} to point at this profile's own grid",
            )
        )
    if declared_pairs is None:
        findings.append(
            _finding(
                "inner_search_budget_mismatch",
                f"{path}.inner_search",
                f"budgets.{unit.get('budget')} declares no internal search to bind against",
            )
        )
    return findings


def _validate_oracle(path: str, profile: Mapping[str, Any]) -> list[Finding]:
    """The oracle stands beside the PU methods, never inside their pool.

    ``pn_oracle`` measures the other profiles; letting it share their candidate
    family, or broadcasting it across the c grid the matrix gives a real method,
    would silently produce a comparable-looking row for something that is not
    comparable (D-P4.1-7).  Candidate families cannot actually be shared, since
    ``recipe_candidate_id`` and ``method`` are both globally unique above --
    which is exactly why the remaining vector is the family declaration itself.
    """
    family = profile.get("recipe_family")
    method = profile.get("method")
    if family != "oracle" and method != "pn_oracle":
        return []
    findings: list[Finding] = []
    if family != "oracle":
        findings.append(
            _finding(
                "oracle_in_pu_candidate_pool",
                f"{path}.recipe_family",
                "an oracle profile must declare recipe_family='oracle'",
            )
        )
    if profile.get("c_tokens") or profile.get("c_grid"):
        findings.append(
            _finding(
                "oracle_c_grid_broadcast",
                f"{path}.c_tokens",
                "the oracle is c-independent; a c grid would broadcast it over the c axis",
            )
        )
    return findings


def validate_source_evidence(registry: Mapping[str, Any]) -> list[Finding]:
    """Every profile's evidence actually resolves, and resolves to something.

    ``source_evidence`` is the registry-wide index; profiles and candidates cite
    it by ``ref_id`` so a reference is defined once and cannot drift between the
    two places that quote it.  ``source_refs`` stays out of the digest
    (D-P4.1-4 rule 3), which makes this check the only thing standing between a
    profile and a citation that points nowhere -- so it is a separate pass
    rather than a footnote of the schema walk.

    A reference must land on a file, not a directory (rule 5), and an
    ``external_local`` one must say so and land on lines as well.  Nothing here
    reads the referenced file: an ``in_repo`` path is checkable against the
    repository, but the registry's job is to record provenance, not to police a
    path that may legitimately be deleted later.
    """
    findings: list[Finding] = []
    evidence = registry.get("source_evidence")
    if not isinstance(evidence, Mapping):
        return [
            _finding(
                "source_evidence_shape_invalid",
                "source_evidence",
                "a registry needs a source-evidence index for its profile references",
            )
        ]
    for ref_id, entry in evidence.items():
        path = f"source_evidence.{ref_id}"
        if not isinstance(entry, Mapping):
            findings.append(
                _finding("source_evidence_shape_invalid", path, "an evidence entry is an object")
            )
            continue
        if entry.get("type") not in SOURCE_REF_TYPES:
            findings.append(
                _finding(
                    "source_evidence_shape_invalid",
                    f"{path}.type",
                    f"must be one of {list(SOURCE_REF_TYPES)}; got {entry.get('type')!r}",
                )
            )
            continue
        if not entry.get("file"):
            findings.append(
                _finding(
                    "source_evidence_shape_invalid",
                    f"{path}.file",
                    "a reference must land on a file, not on a repository directory",
                )
            )
        if entry.get("type") == "external_local":
            if entry.get("reproducibility") != "external_only":
                findings.append(
                    _finding(
                        "source_evidence_shape_invalid",
                        f"{path}.reproducibility",
                        "an external_local reference must be marked external_only",
                    )
                )
            if not entry.get("lines"):
                findings.append(
                    _finding(
                        "source_evidence_shape_invalid",
                        f"{path}.lines",
                        "an external reference must cite line numbers",
                    )
                )

    profiles = registry.get("profiles") or {}
    for method, profile in profiles.items():
        if not isinstance(profile, Mapping):
            continue
        findings.extend(
            _check_refs(
                evidence,
                profile.get("source_refs"),
                f"profiles.{method}.source_refs",
                required=profile.get("recipe_lifecycle") != "excluded",
            )
        )
        for index, candidate in enumerate(profile.get("candidate_pool") or []):
            if not isinstance(candidate, Mapping):
                continue
            findings.extend(
                _check_refs(
                    evidence,
                    candidate.get("source_refs"),
                    f"profiles.{method}.candidate_pool[{index}].source_refs",
                    required=False,
                )
            )
        findings.extend(_external_only_warning(profile, evidence, f"profiles.{method}"))
    return findings


def _is_empty_ref_list(value: Any) -> bool:
    """True for an absent or empty list of refs -- a bare string is neither.

    ``str`` is a ``Sequence``, so the guard has to exclude it explicitly: a
    ``source_refs`` written as ``"some_ref"`` is a shape error, not a one-item
    list, and silently iterating it would report three unknown refs named after
    its characters.
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and not value


def _check_refs(
    evidence: Mapping[str, Any],
    refs: Any,
    path: str,
    *,
    required: bool,
) -> list[Finding]:
    """A ``source_refs`` list must be ids that resolve, when it is present."""
    if refs is None or _is_empty_ref_list(refs):
        if required:
            return [
                _finding(
                    "source_refs_missing",
                    path,
                    "an adopted profile must cite at least one source reference",
                )
            ]
        return []
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        return [_finding("source_evidence_shape_invalid", path, "source_refs is a list of ref_ids")]
    findings: list[Finding] = []
    for ref in refs:
        if not isinstance(ref, str) or ref not in evidence:
            findings.append(
                _finding(
                    "unknown_source_ref",
                    path,
                    f"{ref!r} does not resolve in source_evidence",
                )
            )
    return findings


def _external_only_warning(
    profile: Mapping[str, Any], evidence: Mapping[str, Any], path: str
) -> list[Finding]:
    """External-only evidence cannot carry a profile to ``locked`` by itself.

    Reported as a warning rather than an error because it is admissible while a
    profile is still ``draft`` -- the defect is claiming a lock on evidence a
    second reviewer cannot rebuild (D-P4.1-4 rule 4).
    """
    if profile.get("recipe_lifecycle") != "locked":
        return []
    refs = profile.get("source_refs") or []
    types = {
        evidence[ref].get("type")
        for ref in refs
        if isinstance(ref, str) and isinstance(evidence.get(ref), Mapping)
    }
    if types and types == {"external_local"}:
        return [
            _finding(
                "source_evidence_external_only",
                f"{path}.source_refs",
                "a locked profile needs an in-repo snapshot or a citable committed artifact, "
                "not external_local evidence alone",
                severity="warning",
            )
        ]
    return []


def validate_manifest_binding(
    manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> list[Finding]:
    """Whether *manifest*'s recipe binding is complete, resolvable and consistent.

    Fail closed, per D-P4.1-2.  A manifest with *no* binding field at all is a
    legacy artifact and yields nothing here -- B1-B4 are neither rewritten nor
    retroactively judged ``not_reproducible``.  One binding field, or a
    ``registry_bound`` declaration, means the run claimed the binding, and then
    every field must be present and correct; a partial set is never quietly
    downgraded to legacy.

    One case this function *cannot* decide on its own is a registry-bound
    payload stripped of every binding field including the status: with nothing
    left to read, it is indistinguishable from a legacy manifest.  D-P4.1-2
    requires that run to fail anyway, so the discriminator has to be external --
    the batch/version whitelist and the artifact schema, which are stage 3's
    wiring, not a property of the payload.

    The three questions below are asked in order because each one is only
    meaningful once the previous has been answered: whether the manifest claims a
    binding at all, whether it cites the registry it claims, and whether the
    profile and candidate it names hold together.
    """
    verdict = _binding_form_verdict(manifest)
    if verdict is not None:
        return verdict
    findings = _validate_registry_reference(manifest, registry)
    findings.extend(_validate_bound_candidate(manifest, registry, protocol))
    return findings


def _binding_form_verdict(manifest: Mapping[str, Any]) -> list[Finding] | None:
    """The fail-closed form check, or ``None`` when the binding can be resolved.

    A returned list *ends* the check, including the empty list: that is the
    legacy pass, not a reason to keep looking.  ``None`` is the only value that
    means "claimed and complete enough to resolve against the registry".

    A payload carrying one binding field has claimed the binding, so a missing
    remainder is a defect rather than grounds to fall back to legacy
    (D-P4.1-2).
    """
    status = manifest.get("recipe_binding_status")
    present = manifest_has_binding_fields(manifest)
    if status is not None and status not in RECIPE_BINDING_STATUS:
        return [
            _finding(
                "unknown_recipe_binding_status",
                "recipe_binding_status",
                f"must be one of {list(RECIPE_BINDING_STATUS)}; got {status!r}",
            )
        ]
    if status is None and not present:
        return []
    if status == "legacy_unbound":
        if not present:
            return []
        return [
            _finding(
                "partial_recipe_binding",
                "recipe_binding_status",
                "a legacy_unbound manifest must not carry registry fields",
            )
        ]
    missing = [field for field in REGISTRY_BOUND_MANIFEST_FIELDS if field not in manifest]
    if not missing:
        return None
    return [
        _finding(
            "manifest_registry_field_missing",
            "recipe_registry_version",
            f"registry-bound manifest is missing {missing}",
        )
    ]


def _validate_registry_reference(
    manifest: Mapping[str, Any], registry: Mapping[str, Any]
) -> list[Finding]:
    """The manifest must cite the registry it was bound to.

    Both are checked because they answer to different readers: the version for a
    human reading a report, the digest for anything recomputing the binding.
    """
    findings: list[Finding] = []
    if manifest["recipe_registry_version"] != registry.get("recipe_registry_version"):
        findings.append(
            _finding(
                "manifest_registry_field_missing",
                "recipe_registry_version",
                "manifest cites a registry version the bound registry does not have",
            )
        )
    if manifest["recipe_registry_sha256"] != canonical_digest(registry):
        findings.append(
            _finding(
                "manifest_registry_digest_mismatch",
                "recipe_registry_sha256",
                "recorded registry digest does not recompute over the bound registry",
            )
        )
    return findings


def _validate_bound_candidate(
    manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    protocol: Mapping[str, Any] | None,
) -> list[Finding]:
    """The named profile and candidate must resolve, and their fields must agree.

    Failure to resolve ends the check on purpose: an unresolvable profile makes
    every later comparison meaningless, and reporting them anyway would suggest
    several independent defects where there is one.
    """
    findings: list[Finding] = []
    profile = _find_profile(registry, manifest.get("recipe_profile_id"))
    if profile is None:
        return [
            _finding(
                "unknown_recipe_candidate",
                "recipe_profile_id",
                f"{manifest.get('recipe_profile_id')!r} does not resolve in the registry",
            )
        ]
    candidate = _find_candidate(profile, manifest.get("recipe_candidate_id"))
    if candidate is None:
        return [
            _finding(
                "unknown_recipe_candidate",
                "recipe_candidate_id",
                f"{manifest.get('recipe_candidate_id')!r} does not resolve under "
                f"{manifest.get('recipe_profile_id')!r}",
            )
        ]
    expected_inner = (profile.get("inner_search") or {}).get("inner_search_id")
    if manifest["inner_search_id"] != expected_inner:
        findings.append(
            _finding(
                "inner_search_id_mismatch",
                "inner_search_id",
                f"manifest records {manifest['inner_search_id']!r}, candidate's inner search "
                f"is {expected_inner!r}",
            )
        )
    if protocol is not None:
        expected = _expected_snapshot(profile, candidate, protocol)
        if expected is not None and manifest["resolved_params_sha256"] != resolved_params_digest(
            expected
        ):
            findings.append(
                _finding(
                    "resolved_params_digest_mismatch",
                    "resolved_params_sha256",
                    "recorded resolved-parameters digest does not match the candidate's expansion",
                )
            )
    return findings


def _find_profile(registry: Mapping[str, Any], wanted: Any) -> Mapping[str, Any] | None:
    for profile in (registry.get("profiles") or {}).values():
        if isinstance(profile, Mapping) and profile_id(profile) == wanted:
            return profile
    return None


def _find_candidate(profile: Mapping[str, Any], wanted: Any) -> Mapping[str, Any] | None:
    for candidate in profile.get("candidate_pool") or []:
        if isinstance(candidate, Mapping) and candidate.get("recipe_candidate_id") == wanted:
            return candidate
    return None
