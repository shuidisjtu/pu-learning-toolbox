"""Versioned profile -> comparison-matrix mapping for ``survey-v3.0``.

The comparison matrix selects its entries by ``method``, and under the
profile-identity schema the execution unit's ``method`` is a *profile* ID rather
than an algorithm.  With no map between the two, every v3 manifest could only
record that it had no lookup path (``missing_versioned_profile_mapping``), and a
reader could not tell a profile whose entry has not been written yet from a
profile that is deliberately outside the matrix.

The map answers exactly one question -- *which pre-registered entry does this
profile inherit, and where does the profile depart from it* -- never "are the
numbers comparable".  Eligibility stays in the comparison file: a deviation that
should cost a profile its numeric verdict is a comparison-level decision with its
own record (plan D31/D37), not a silent downgrade performed here.

Design notes
------------
**Inheritance plus explicit deviations, never a second entry.**  ``self_pu`` and
``self_pu_full_meta`` are one algorithm with two supervision contracts, and the
matrix describes the no-meta ablation.  Re-keying every mapping by profile
would restate each entry once per profile and let the copies
drift; inheriting one entry and recording the profile's own departure keeps a
single source of truth and makes the difference between "same row" and "same
number" visible.

**``not_in_matrix`` describes the matrix, not runnability.**  Two axes meet here
and are deliberately kept apart: the recipe registry says whether a profile may
run, this file says whether the matrix holds an entry for it.  ``kldce`` is
excluded from running under v3 *and* has three registration entries (the
non-runnable marker rows), so it is ``mapped`` with a rationale that says so;
marking it ``not_in_matrix`` would write the opposite of what the matrix holds.
Conversely ``lagam`` runs on one dataset and has no entry anywhere, which is the
P4.2 gap.

**``against`` names an anchor family.**  One profile can relate to the paper and
to PU-Bench's controlled implementation in opposite directions:
``self_pu_full_meta`` is the paper's own full self-calibrated path (no meta-level
deviation from ``chen_table7``) while it departs from ``pu_bench_table1``, whose
meta target is validation U rather than clean labels.  Collapsing the two into
one "deviation from the benchmark" note would lose that.

**A profile missing from the map is a defect, not ``not_in_matrix``.**  The map
is a total function over the protocol's profiles: validation refuses a protocol
profile it does not describe, so "not mapped yet" can never masquerade as "not
comparable".

See ``docs/research/pu_survey/survey_execution_plan.md`` §1.2 and D37.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .survey_comparison import (
    SELECTOR_FIELDS,
    comparison_for_protocol,
    resolve_comparison_unit,
)
from .survey_protocol import digest, load_protocol, resolve_protocol_path

#: The effective map: the one bound to the protocol and matrix the runs use.
#: v1 and v2 stay on disk as the frozen artifacts they are (bound to v4 and to
#: v5 respectively); loading them is still possible by path, never by default.
MAP_PATH = Path(__file__).with_name("survey_comparison_profile_map_v3.json")

SCHEMA_VERSION = "survey-comparison-profile-map-1"

#: Whether the matrix holds an entry the profile inherits.
REGISTRATIONS = ("mapped", "not_in_matrix")

#: What a profile-level deviation is about.  Closed: a new axis is a vocabulary
#: change, not a free-text note.
DEVIATION_AXES = ("supervision", "budget", "representation", "architecture", "protocol")

#: Which anchor family a deviation is stated against.  The paper and the
#: controlled benchmark implementation are different reference points and a
#: profile can deviate from one without deviating from the other.
ANCHOR_FAMILIES = ("chen_table7", "pu_bench_table1", "generic")

#: Manifest statuses derived from this map (plan §5).
REGISTRATION_STATUSES = (
    "mapped_inherited",
    "mapped_with_profile_deviations",
    "not_in_matrix",
)

STATUS_MAPPED_INHERITED = REGISTRATION_STATUSES[0]
STATUS_MAPPED_WITH_DEVIATIONS = REGISTRATION_STATUSES[1]
STATUS_NOT_IN_MATRIX = REGISTRATION_STATUSES[2]

_DEVIATION_FIELDS = ("axis", "ours", "mapped", "against", "note")
_ENTRY_FIELDS = ("registration", "inherits", "deviations", "rationale")
_REQUIRED_TOP_LEVEL = (
    "schema_version",
    "map_version",
    "bound_survey_protocol",
    "bound_comparison",
    "profiles",
)
#: ``(binding label, version field, digest field)``: one table drives both the
#: shape check and the cross-file check, so the two cannot disagree.
_BINDINGS = (
    ("bound_survey_protocol", "protocol_version", "protocol_sha256"),
    ("bound_comparison", "comparison_version", "comparison_sha256"),
)
_PROFILE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def profile_map_available(path: str | Path = MAP_PATH) -> bool:
    """Whether a versioned profile map ships yet.

    The runner keeps working without one -- it falls back to recording that no
    lookup path exists -- so this check is the one place that decides whether
    "no map" is a degraded state rather than a defect.
    """
    return Path(path).exists()


def load_profile_map(path: str | Path = MAP_PATH) -> dict[str, Any]:
    """Load the map and check its own shape.

    Cross-file checks (binding digests, profile coverage) need the protocol and
    the comparison, so they live in :func:`validate_profile_map`; a payload that
    is not a map at all is rejected here rather than there, because the resolver
    must not read an arbitrary object as "no entries".
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("survey profile map must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported survey profile map schema_version")
    for key in _REQUIRED_TOP_LEVEL:
        if key not in payload:
            raise ValueError(f"survey profile map missing {key}")
    _require_text(payload["map_version"], "survey profile map", "map_version")
    for label, version_field, digest_field in _BINDINGS:
        _validate_binding_shape(payload[label], label, version_field, digest_field)
    profiles = payload["profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("survey profile map profiles must be a non-empty object")
    for profile_id, entry in profiles.items():
        _validate_entry(profile_id, entry)
    return payload


def validate_profile_map(
    profile_map: dict[str, Any],
    *,
    protocol: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Prove the map binds the loaded files and covers every protocol profile.

    Returns the counts the gate prints.  Every failure is fatal here: a binding
    mismatch means the map was written against another matrix, a missing profile
    means the resolution path would report "no entry" for a unit the protocol
    declares, and an uninherited comparison method means a row of the matrix is
    reachable by nothing.
    """
    if protocol is None:
        protocol = bound_protocol(profile_map)
    if comparison is None:
        comparison = bound_comparison(profile_map)
    for label, version_field, digest_field in _BINDINGS:
        target = protocol if label == "bound_survey_protocol" else comparison
        _check_binding(profile_map[label], target, label, version_field, digest_field)

    profiles = profile_map["profiles"]
    expected = set(protocol["method_profiles"])
    declared = set(profiles)
    if declared != expected:
        raise ValueError(
            "survey profile map must cover exactly the protocol's profiles: "
            f"missing={sorted(expected - declared)} unexpected={sorted(declared - expected)}"
        )

    selector_methods = {mapping["result_selector"]["method"] for mapping in comparison["mappings"]}
    counts = {"profiles": len(profiles), "mapped": 0, "not_in_matrix": 0, "deviations": 0}
    inherited: set[str] = set()
    for profile_id, entry in profiles.items():
        if entry["registration"] != "mapped":
            counts["not_in_matrix"] += 1
            continue
        counts["mapped"] += 1
        counts["deviations"] += len(entry["deviations"])
        target = entry["inherits"]
        if target not in selector_methods:
            raise ValueError(
                f"profile {profile_id!r} inherits {target!r}, which is not a comparison method"
            )
        inherited.add(target)

    # Reverse direction: a comparison method the protocol also declares as a
    # profile must be inherited by something, or that whole row is unreachable
    # from a v3 run and the map silently drops it.
    unreachable = sorted((selector_methods & expected) - inherited)
    if unreachable:
        raise ValueError(
            "comparison methods the protocol declares as profiles are inherited by no profile: "
            f"{unreachable}"
        )
    return counts


def bound_protocol(profile_map: dict[str, Any]) -> dict[str, Any]:
    """Load the protocol the map binds, by name, rather than the current default."""
    return load_protocol(
        resolve_protocol_path(profile_map["bound_survey_protocol"]["protocol_version"])
    )


def bound_comparison(profile_map: dict[str, Any]) -> dict[str, Any]:
    """The matrix the bound protocol resolves to, by that protocol's version.

    One protocol version has exactly one matrix, so the map does not get to
    pick a different one: reading the shipped default instead is how a v5 map
    ended up validating against v4.  The map's own ``bound_comparison`` is
    then proven to name this file by :func:`validate_profile_map`, which turns
    a map written against the superseded matrix into a clear mismatch rather
    than a silently re-adjudicated result.
    """
    return comparison_for_protocol(bound_protocol(profile_map))


def resolve_profile_mapping(
    profile_id: str,
    result_identity: dict[str, str],
    *,
    protocol: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    profile_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one profile's result unit to the pre-registered entry it inherits.

    ``result_identity`` carries the six non-method selector fields (dataset,
    mechanism, c token, selection protocol, metric, training path); the method is
    the profile's ``inherits`` value, never the profile ID.  A ``not_in_matrix``
    profile never reaches the selector at all.

    ``rationale`` is the inherited entry's, ``profile_rationale`` the map
    entry's: the first explains the row, the second why this profile departs
    from it.  Eligibility is copied from the comparison file untouched.
    """
    if profile_map is None:
        profile_map = load_profile_map()
    if protocol is None:
        protocol = bound_protocol(profile_map)
    if comparison is None:
        comparison = bound_comparison(profile_map)
    validate_profile_map(profile_map, protocol=protocol, comparison=comparison)

    entry = profile_map["profiles"].get(profile_id)
    if entry is None:
        raise ValueError(f"profile has no entry in the survey profile map: {profile_id!r}")
    resolved_base = {
        "profile_id": profile_id,
        "profile_map_version": profile_map["map_version"],
        "profile_map_sha256": digest(profile_map),
        "profile_registration": entry["registration"],
        "profile_rationale": entry["rationale"],
    }
    if entry["registration"] == "not_in_matrix":
        return {
            **resolved_base,
            "status": "not_in_matrix",
            "deviation_count": 0,
        }

    selector = {field: result_identity.get(field) for field in SELECTOR_FIELDS}
    selector["method"] = entry["inherits"]
    resolved = resolve_comparison_unit(selector, comparison=comparison)
    deviations = [{**deviation, "source": "profile_map"} for deviation in entry["deviations"]]
    return {
        **resolved_base,
        "status": "mapped",
        "inherits": entry["inherits"],
        "mapping_id": resolved["mapping_id"],
        "eligibility": resolved["eligibility"],
        "rationale": resolved["rationale"],
        "protocol_differences": [*resolved["protocol_differences"], *deviations],
        "anchors": resolved["anchors"],
        "deviation_count": len(deviations),
    }


def profile_map_binds(profile_map: dict[str, Any], protocol: dict[str, Any]) -> bool:
    """Whether the map was frozen against exactly this matrix.

    A narrowed or otherwise noncanonical matrix carries the profile-identity
    schema but is not the protocol the map was written against.  The map then
    does not apply, and callers record that this matrix has no versioned profile
    mapping instead of reading the map's entries as if they described it.  Drift
    in the *shipped* matrix is caught by the gate, which recomputes both digests
    against the committed files.
    """
    bound = profile_map["bound_survey_protocol"]
    same_version = bound["protocol_version"] == protocol["protocol_version"]
    same_digest = bound["protocol_sha256"] == digest(protocol)
    return same_version and same_digest


def profile_registration_status(
    profile_id: str,
    *,
    protocol: dict[str, Any],
    profile_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The manifest-side status: profile level, so it needs no result selector.

    A run is not one result unit (a SCAR run selects twice), so the manifest can
    only record the *profile's* registration; which entry each result unit lands
    on is resolved from the result identity at aggregation time.
    """
    if profile_map is None:
        profile_map = load_profile_map()
    _check_binding(
        profile_map["bound_survey_protocol"],
        protocol,
        "bound_survey_protocol",
        "protocol_version",
        "protocol_sha256",
    )
    entry = profile_map["profiles"].get(profile_id)
    if entry is None:
        raise ValueError(f"profile has no entry in the survey profile map: {profile_id!r}")
    deviation_count = len(entry["deviations"])
    if entry["registration"] == "not_in_matrix":
        status = STATUS_NOT_IN_MATRIX
    elif deviation_count:
        status = STATUS_MAPPED_WITH_DEVIATIONS
    else:
        status = STATUS_MAPPED_INHERITED
    return {
        "comparison_registration_status": status,
        "profile_map_version": profile_map["map_version"],
        "profile_map_sha256": digest(profile_map),
        "profile_map_deviation_count": deviation_count,
    }


def _require_text(value: Any, owner: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} requires a non-empty {field}")


def _validate_binding_shape(bound: Any, label: str, version_field: str, digest_field: str) -> None:
    if not isinstance(bound, dict):
        raise ValueError(f"{label} must be an object")
    for field in (version_field, digest_field):
        _require_text(bound.get(field), label, field)


def _check_binding(
    bound: dict[str, Any],
    target: dict[str, Any],
    label: str,
    version_field: str,
    digest_field: str,
) -> None:
    """Reject a map written against another matrix, in both directions."""
    if bound[version_field] != target[version_field]:
        raise ValueError(
            f"{label}.{version_field} disagrees with the loaded file: "
            f"{bound[version_field]!r} != {target[version_field]!r}"
        )
    if bound[digest_field] != digest(target):
        raise ValueError(f"{label}.{digest_field} disagrees with the loaded file's digest")


def _validate_entry(profile_id: Any, entry: Any) -> None:
    if not isinstance(profile_id, str) or not _PROFILE_NAME.fullmatch(profile_id):
        raise ValueError("survey profile map keys must be safe lowercase profile IDs")
    if not isinstance(entry, dict):
        raise ValueError(f"profile map entry must be an object: {profile_id!r}")
    missing = [field for field in _ENTRY_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"profile map entry {profile_id!r} missing {missing}")
    registration = entry["registration"]
    if registration not in REGISTRATIONS:
        raise ValueError(f"unknown registration for profile {profile_id!r}: {registration!r}")
    _require_text(entry["rationale"], f"profile map entry {profile_id!r}", "rationale")
    deviations = entry["deviations"]
    if not isinstance(deviations, list):
        raise ValueError(f"profile map entry {profile_id!r} deviations must be a list")
    inherits = entry["inherits"]
    if registration == "mapped":
        _require_text(inherits, f"mapped profile {profile_id!r}", "inherits")
    else:
        if inherits is not None:
            raise ValueError(
                f"not_in_matrix profile {profile_id!r} must not name an inherited entry"
            )
        if deviations:
            raise ValueError(f"not_in_matrix profile {profile_id!r} cannot carry deviations")
    for deviation in deviations:
        _validate_deviation(profile_id, deviation)


def _validate_deviation(profile_id: Any, deviation: Any) -> None:
    if not isinstance(deviation, dict):
        raise ValueError(f"profile map entry {profile_id!r} deviations must be objects")
    if tuple(sorted(deviation)) != tuple(sorted(_DEVIATION_FIELDS)):
        raise ValueError(
            f"profile map entry {profile_id!r} deviation must carry exactly "
            f"{list(_DEVIATION_FIELDS)}"
        )
    for field in _DEVIATION_FIELDS:
        _require_text(deviation[field], f"profile {profile_id!r} deviation", field)
    if deviation["axis"] not in DEVIATION_AXES:
        raise ValueError(f"profile {profile_id!r} deviation axis is not recognized")
    if deviation["against"] not in ANCHOR_FAMILIES:
        raise ValueError(f"profile {profile_id!r} deviation against is not a known anchor family")
