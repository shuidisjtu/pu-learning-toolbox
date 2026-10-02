"""P2.2 batch audit: is this artifact set complete, reproducible and handable?

Two rules shape every function here.

* **A check's scope is not decoration.**  A missing field belongs to one
  manifest; a split mismatch belongs to a group of them.  The aggregation gate
  raises on the first bad group and names the rule that broke, never the file
  that broke it -- so attributing that failure to one manifest would manufacture
  a culprit.  Checks therefore carry ``scope`` and ``members``, and the
  group-scoped ones name every manifest in the group rather than the first.
* **A malformed manifest is a finding, not a crash.**  The aggregation entry
  point raises ``ValueError`` on a manifest missing a field it reads, which is
  the right call for an entry point that only reports an exit code.  An audit
  whose job is to *list* what is wrong cannot stop at the first item, so the same
  condition becomes a ``not_reproducible`` finding here.

Code commit and the dependency lock are **not** per-manifest fields: the survey
runner writes neither, and the protocol's requirement for them is satisfied by
the batch plan and evidence packet instead.  A preflight that demanded them per
run would mark every legitimate manifest defective.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .survey_comparison import resolve_comparison_unit
from .survey_summary import (
    REASONS,
    SELECTION_PROTOCOLS,
    SummaryError,
    group_key,
    matrix_selector,
    normalize_c_token,
    reasons_for,
    status_for_gate_error,
)

SCHEMA_VERSION = "survey-audit-1"

#: Outcomes a single check may report.  ``not_run`` is a first-class value: a
#: check that could not run because its inputs are absent is not a pass, and
#: recording it as one is how a gap disappears from a report.  ``not_applicable``
#: is the other half of that honesty, and is kept apart from ``not_run`` on
#: purpose: a check the artifact set cannot raise (reclaim accounting, on batches
#: that never reclaimed) must not degrade the overall verdict the way a check whose
#: inputs are missing does.
CHECK_RESULTS: tuple[str, ...] = ("pass", "warn", "fail", "not_run", "not_applicable")

CHECK_SEVERITIES: tuple[str, ...] = ("error", "warning", "info")

#: What a check's finding is about.  ``manifest`` findings name one file;
#: ``unit`` / ``group`` findings name the members they were found among.
CHECK_SCOPES: tuple[str, ...] = ("manifest", "unit", "group", "batch", "global")

#: The roles a whitelisted batch may carry.  A probe batch's runs are real but
#: protocol §5 clause 1 bars them from the formal leaderboard, so the role is a
#: fact about the batch that both entry points have to read the same way.
BATCH_ROLES: tuple[str, ...] = ("formal", "technical_probe")

#: The probe role, named once.  A summarizer that spelled this string itself
#: could disagree with the whitelist validator about which batches are probes.
PROBE_ROLE = "technical_probe"

#: Per-manifest identity, whose absence makes the result not reproducible
#: (protocol §5 clause 6).  Dotted so a nested gap names the field actually
#: missing rather than the top-level object that contains it.
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "execution_mode",
    "protocol_version",
    "protocol_sha256",
    "execution_unit.method",
    "execution_unit.dataset",
    "execution_unit.comparability_group",
    "training_path",
    "run_view",
    "representation.name",
    "representation.split_sha256",
    "representation.feature_sha256",
    "seed",
    "split_ref.indices_sha256",
    "generation.train",
    "candidate_runs",
)

#: Environment identity, which protocol §5 clause 6 names as "Python/PyTorch/
#: CUDA/GPU information".  Absent, the result cannot be reproduced anywhere else.
#: Measured costs are a different matter: they are reported, but their absence is
#: a reporting gap rather than a reproduction failure.
REQUIRED_ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "resources.environment.python_version",
    "resources.environment.torch_version",
    "resources.environment.gpu_devices",
)

#: Identity recorded once per batch instead of per manifest.  Kept here as the
#: documented answer to "why does no manifest carry a commit?" -- the audit reads
#: these from the batch plan and evidence packet.
BATCH_IDENTITY_FIELDS: tuple[str, ...] = ("code_commit", "dependency_lock")

_MISSING = object()


class AuditError(ValueError):
    """An audit check, or a request to build one, that does not hold together."""


def _dig(payload: Mapping[str, Any], path: str) -> Any:
    """The value at a dotted *path*, or :data:`_MISSING`."""
    node: Any = payload
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def missing_fields(payload: Mapping[str, Any], paths: Sequence[str]) -> list[str]:
    """Which of *paths* are absent, in the order given."""
    return [path for path in paths if _dig(payload, path) is _MISSING]


def matrix_conflicts(payload: Mapping[str, Any], *, comparison: Mapping[str, Any]) -> list[str]:
    """Reason codes for recorded protocols the frozen matrix does not corroborate.

    Protocol §5.2 keeps two statements that must agree, and neither is derived from
    the other: the manifest records what the run *did* (``c_independent``, the
    mechanism, the recorded ``c`` token) and the matrix records what that execution
    row is *expected* to be.  Every selection protocol the manifest recorded a
    result under is therefore resolved against the matrix's own mappings, one of
    which has to match exactly.  A run the matrix does not cover and a run it
    places on a different axis are both findings -- a later comparison is judged
    against the pre-registration, so the claim is checked here rather than trusted.

    A manifest that recorded no results still makes a c-axis claim of its own, so the
    check falls back to both selection protocols rather than concluding there is
    nothing to compare: a claim is checkable whether or not a number was produced
    under it, and an unaudited claim is how the axis drifts unnoticed.

    ``comparison`` is a parameter because loading it re-validates the whole matrix,
    which is too expensive to repeat per manifest; the caller loads it once.
    """
    codes: set[str] = set()
    recorded = payload.get("test_results")
    protocols = (
        sorted(recorded) if isinstance(recorded, Mapping) and recorded else SELECTION_PROTOCOLS
    )
    for protocol in protocols:
        try:
            selector = matrix_selector(payload, selection_protocol=protocol)
        except (KeyError, SummaryError):
            # Already a structural finding; naming it twice would suggest two gaps.
            continue
        try:
            resolve_comparison_unit(selector, comparison=comparison)
        except ValueError:
            # Both directions of a broken axis claim land here: an oracle standing
            # where the matrix has a c grid, and a c-grid run standing where the
            # matrix has the c-independent row.
            codes.add(
                "oracle_c_independent_conflict"
                if payload.get("c_independent")
                else "mechanism_mismatch"
            )
    return sorted(codes)


def preflight_status(
    payload: Mapping[str, Any], *, comparison: Mapping[str, Any] | None = None
) -> tuple[str, list[str], list[str]]:
    """``(status, reasons, missing_fields)`` for one manifest, without raising.

    Structure, plus -- when *comparison* is supplied -- the identity the frozen
    matrix expects.  Whether the run *succeeded* is the aggregation gate's call: a
    run that selected nothing is a fact about the experiment, while a manifest
    missing its split digest is a defect in the artifact, and only the second
    makes the result unreproducible.
    """
    absent = missing_fields(payload, REQUIRED_MANIFEST_FIELDS)
    unrecorded = missing_fields(payload, REQUIRED_ENVIRONMENT_FIELDS)
    codes: set[str] = set()
    if absent or unrecorded:
        # Environment identity is grouped with the structural gap rather than
        # given a softer code: protocol §5 clause 6 names Python/PyTorch/CUDA/GPU
        # information alongside the split manifest, so losing it is the same kind
        # of finding -- and both are manifest fields, whatever their nesting.
        codes.add("missing_manifest_field")
    try:
        normalize_c_token(payload)
    except SummaryError:
        codes.add("missing_c_token")
    if not absent and not unrecorded:
        # The view/calibration pair and the mechanism are read here so that a
        # manifest the aggregation gate cannot assign to a group becomes a finding
        # rather than an exception thrown out of the middle of a report.  Checked
        # only when nothing structural is missing, which would explain it already.
        try:
            group_key(payload)
        except (KeyError, SummaryError, ValueError):
            codes.add("group_key_mismatch")
    if comparison is not None:
        codes.update(matrix_conflicts(payload, comparison=comparison))
    if not codes:
        return "formal", [], []
    status = "not_reproducible"
    return status, reasons_for(status=status, reasons=sorted(codes)), absent + unrecorded


def make_check(
    *,
    check_id: str,
    result: str,
    severity: str,
    scope: str,
    message: str,
    observed: Mapping[str, Any] | None = None,
    expected: Mapping[str, Any] | None = None,
    evidence: Sequence[str] = (),
    members: Sequence[str] = (),
    reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """One audited finding, with the two vocabularies checked rather than trusted."""
    if result not in CHECK_RESULTS:
        raise AuditError(f"unknown check result: {result!r}")
    if severity not in CHECK_SEVERITIES:
        raise AuditError(f"unknown check severity: {severity!r}")
    if scope not in CHECK_SCOPES:
        raise AuditError(f"unknown check scope: {scope!r}")
    if scope in {"unit", "group"} and not members:
        # A group-scoped finding without members is the misattribution this field
        # exists to prevent: it reads as "somewhere in this group" and a reader
        # will supply the wrong file.
        raise AuditError(f"a {scope}-scoped check must name its members")
    if scope == "manifest" and len(members) > 1:
        raise AuditError("a manifest-scoped check names exactly one manifest")
    codes = sorted(set(reasons))
    unknown = sorted(set(codes) - set(REASONS))
    if unknown:
        raise AuditError(f"unknown reason code(s): {unknown}")
    return {
        "check_id": check_id,
        "result": result,
        "severity": severity,
        "scope": scope,
        "observed": dict(observed or {}),
        "expected": dict(expected or {}),
        # Coerced here rather than at each call site: a caller naturally passes
        # the ``Path`` objects it discovered with, and a check that cannot be
        # serialized is a report that cannot be written.
        "evidence": [str(item) for item in evidence],
        "members": [str(item) for item in members],
        "reasons": codes,
        "message": message,
    }


def rollup_gate_check(
    *, check_id: str, failures: Sequence[tuple[Sequence[str], str]]
) -> dict[str, Any]:
    """One check for the fairness gate's refusals, naming every affected member.

    A single check rather than one per refusing group: two findings sharing an
    id cannot be cited unambiguously, and every one of these is the same category
    of problem.  Nothing is lost by merging them -- each group's rule and members
    still appear, in ``evidence`` and ``members``.

    ``members`` is required per failure because the gate's message identifies the
    rule, not the file: the whole failing group is named instead of one culprit
    picked out of it.
    """
    if not failures:
        raise AuditError("a gate roll-up needs at least one failure")
    members: list[str] = []
    reasons: set[str] = set()
    evidence: list[str] = []
    for group_members, message in failures:
        if not group_members:
            raise AuditError("a gate refusal must name the members it was found among")
        status, codes = status_for_gate_error(message)
        if status == "formal":
            # The gate refused something; a formal verdict would drop the finding.
            raise AuditError("a gate refusal cannot be a formal verdict")
        members.extend(str(item) for item in group_members)
        reasons.update(codes)
        evidence.append(f"{len(group_members)} member(s): {message}")
    return make_check(
        check_id=check_id,
        result="fail",
        severity="error",
        scope="group",
        message=f"{len(failures)} group(s) refused by the fairness gate",
        evidence=evidence,
        members=members,
        reasons=sorted(reasons),
    )


def overall_status(checks: Iterable[Mapping[str, Any]]) -> str:
    """``fail`` if any check failed, ``partial`` if any was not run, else ``pass``.

    ``not_applicable`` leaves the verdict alone: a check the artifact set cannot
    raise is not a gap in coverage, which is the separate finding ``not_run``
    records.
    """
    results = {check["result"] for check in checks}
    if "fail" in results:
        return "fail"
    if "not_run" in results:
        return "partial"
    return "pass"
