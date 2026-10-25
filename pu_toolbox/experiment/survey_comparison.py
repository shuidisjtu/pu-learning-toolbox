"""Pre-registered external comparison for the survey pilot.

This is a second truth source, deliberately separate from
``survey_protocol_v2.json``: the execution matrix says what will be run, this
says what the numbers may later be compared against, and how.  Nothing here
runs training or reads results; the module only freezes the audit rules.

Design notes:

* Anchor values come from published papers -- the benchmark repositories do
  not ship result products, so the code can never be the value source.
  Protocol metadata (class mapping, preprocessing, validation ratio, seeds,
  selection criterion, backbone) is taken from the locked repositories
  instead, because they keep moving after publication.  Where the two
  disagree the code wins and the discrepancy is recorded in ``contradictions``
  (see P2.0c execution plan 6.1 / 5.7).
* ``uncertainty_kind`` decides the pooled standard error: some benchmarks
  report a standard deviation, others a standard error of the mean.  Treating
  one as the other double-counts the sample size, so the field is mandatory
  and the two branches are implemented separately.
* ``value_producer`` answers "who computed this number", measured against the
  method's own authors rather than against the paper it was read from.  A
  source reporting somebody else's method is a third-party reproduction even
  though the number carries that source's name; a benchmark's maintainers
  running it are ``benchmark_maintainers``.  The field is descriptive only:
  it never enters a verdict, and exists so that a reader cannot mistake a
  reproduction for an original result.
* Eligibility is a closed enum; nothing may be compared numerically without
  an explicit, reviewable classification.

See docs/research/pu_survey/delivery/p2_0c_delivery.md, P2.0c.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pu_toolbox.utils.serialization import strict_canonical_hash

COMPARISON_PATH = Path(__file__).with_name("survey_comparison_v4.json")
#: The shipped matrices, keyed by the protocol version each one was frozen
#: against.  A version resolves to the file carrying *its* binding: a
#: survey-v3.0 run reads the v3-bound matrix rather than whichever file
#: happens to be newest, so a historical version keeps re-reading the rules it
#: was pre-registered under.  ``survey_protocol.PROTOCOL_PATHS`` makes the same
#: promise for the protocols, and the two tables are what keep them aligned.
COMPARISON_PATHS: dict[str, Path] = {
    "survey-v1": Path(__file__).with_name("survey_comparison_v3.json"),
    "survey-v1.1": Path(__file__).with_name("survey_comparison_v3.json"),
    "survey-v1.2": Path(__file__).with_name("survey_comparison_v3.json"),
    "survey-v2.0": COMPARISON_PATH,
    "survey-v3.0": Path(__file__).with_name("survey_comparison_v6.json"),
}

#: The six eligibility classes, in canonical order.  A comparison file that
#: declares anything else (an extra class, a missing one) is rejected rather
#: than treated as a superset.
ELIGIBILITY_CLASSES = (
    "numeric",
    "magnitude_and_trend",
    "background_only",
    "no_direct_anchor",
    "blocked_pending_pa_criterion",
    "non_runnable",
)

_ELIGIBILITY = frozenset(ELIGIBILITY_CLASSES)
_UNCERTAINTY_KINDS = frozenset({"std", "sem"})
_VALUE_PRODUCERS = frozenset(
    {"original_authors", "third_party_reproduction", "benchmark_maintainers"}
)
_PROVENANCE = frozenset({"certain", "uncertain"})
_REVIEW_STATES = frozenset({"pending_review", "accepted"})
_REVIEW_STATUSES = frozenset({"pending_collaborator_review", "changes_requested", "accepted"})
_METRICS = frozenset({"accuracy", "error_rate"})
_COMPARISON_METRIC = "accuracy"
_MAPPING_SCOPES = frozenset({"unit", "row"})
#: Sources without a label-frequency axis (PUBench, the original method papers)
#: cannot be resolved to one c token.  Rather than invent a c for them, a
#: row-level mapping marks the mechanism, axis and selection semantics as
#: "not aligned for this whole execution row".
_ROW_LEVEL = "row_level"
_ROW_LEVEL_FIELDS = ("labeling_mechanism", "c_token", "selection_protocol")
#: Marker for the one coverage entry a non-runnable row carries, so the matrix
#: records the gap without fabricating c / mechanism expansions for it.
_NON_RUNNABLE = "non_runnable"
_PN_ORACLE = "pn_oracle"
#: Selection protocols our own SCAR units carry; SAR rows are OA-only.
_UNIT_SELECTION_PROTOCOLS = ("pa", "oa")
_SAR_MECHANISMS = ("sar_lbe_a", "sar_lbe_b")
_RESULT_UNITS = frozenset({"fraction", "percentage_points"})
SELECTOR_FIELDS = (
    "method",
    "dataset",
    "labeling_mechanism",
    "c_token",
    "selection_protocol",
    "metric",
    "training_path",
)
#: The profile map resolves the same selector with ``method`` substituted, so it
#: imports this tuple instead of restating it.
_SELECTOR_FIELDS = SELECTOR_FIELDS
_REQUIRED_TOP_LEVEL = (
    "schema_version",
    "comparison_version",
    "review_status",
    "bound_survey_protocol",
    "metric_unit",
    "result_input_unit",
    "eligibility_classes",
    "decision_rules",
    "sources",
    "anchors",
    "mappings",
    "contradictions",
    "formal_blockers",
)
_REQUIRED_ANCHOR = (
    "anchor_id",
    "source_id",
    "paper_location",
    "method_identity",
    "dataset",
    "labeling_mechanism",
    "c_token",
    "selection_protocol",
    "metric",
    "mean_percent",
    "spread_percent",
    "uncertainty_kind",
    "n_repeats",
    "value_producer",
    "protocol_provenance",
    "review_state",
    "verified_by",
)


def comparison_digest(value: Any) -> str:
    """Hash canonical JSON, independent of whitespace and key order.

    Kept as a named entry point because the comparison scripts and tests
    import it; the recipe itself is
    :func:`~pu_toolbox.utils.serialization.strict_canonical_hash`.
    """
    return strict_canonical_hash(value)


def load_comparison_protocol(
    path: str | Path = COMPARISON_PATH, *, survey: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load and validate the comparison rules before any result is produced.

    ``survey`` is the execution protocol this file must be bound to; when it is
    omitted the protocol the file itself declares is loaded, so a matrix cannot
    be read under another version's protocol.  Only the version is enforced: the
    recorded digest is provenance, and the shipped protocol is edited in place,
    so requiring the digest to match would rewrite every artifact on each edit
    without catching anything the version check does not.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("survey comparison protocol must be an object")
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported survey comparison schema_version")
    for key in _REQUIRED_TOP_LEVEL:
        if key not in payload:
            raise ValueError(f"survey comparison missing {key}")

    if not isinstance(payload["comparison_version"], str) or not payload["comparison_version"]:
        raise ValueError("comparison_version must be a non-empty string")
    if payload["review_status"] not in _REVIEW_STATUSES:
        raise ValueError("review_status is not recognized")
    if payload["metric_unit"] != "percentage_points":
        raise ValueError("metric_unit must be percentage_points for anchors")
    if payload["result_input_unit"] not in _RESULT_UNITS:
        raise ValueError("result_input_unit is not recognized")
    if list(payload["eligibility_classes"]) != list(ELIGIBILITY_CLASSES):
        raise ValueError("eligibility_classes must match the six-class specification exactly")

    blockers = payload["formal_blockers"]
    if (
        not isinstance(blockers, list)
        or any(not isinstance(item, str) or not item for item in blockers)
        or len(blockers) != len(set(blockers))
    ):
        raise ValueError("formal_blockers must be unique non-empty strings")

    _bound = _validate_binding(payload["bound_survey_protocol"], survey)
    sources = _validate_sources(payload["sources"])
    anchors = _validate_anchors(payload["anchors"], sources)
    mappings = _validate_mappings(payload["mappings"], anchors)
    contradictions = _validate_contradictions(payload["contradictions"], sources, anchors)

    if payload["review_status"] == "accepted":
        if blockers:
            raise ValueError("an accepted comparison must not have formal_blockers")
        for anchor in anchors.values():
            if anchor["review_state"] != "accepted":
                raise ValueError(
                    "an accepted comparison requires every anchor review_state accepted: "
                    f"{anchor['anchor_id']}"
                )
        for mapping in mappings.values():
            if mapping["review_state"] != "accepted":
                raise ValueError(
                    "an accepted comparison requires every mapping review_state accepted: "
                    f"{mapping['mapping_id']}"
                )
        for contradiction in contradictions.values():
            if contradiction["review_state"] != "accepted":
                raise ValueError(
                    "an accepted comparison requires every contradiction review_state accepted: "
                    f"{contradiction['contradiction_id']}"
                )

    comparison_digest(payload)  # also reject non-finite values
    return payload


def resolve_comparison_path(value: str | Path) -> Path:
    """The matrix file *value* denotes: a shipped version name, or the path it gives."""
    name = str(value)
    if name in COMPARISON_PATHS:
        return COMPARISON_PATHS[name]
    return Path(value).resolve()


def comparison_for_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """The shipped matrix bound to *protocol*, selected by the protocol version.

    The version -- not the path the caller happens to hold, and not the default
    file -- picks the matrix, so a v3 run cannot pick up the v2-bound rules the
    way it could while the default was a single hardcoded path.  The binding is
    then verified against the protocol itself: a registry entry pointing at the
    wrong matrix raises here, instead of producing a plausible-looking verdict.
    """
    return load_comparison_protocol(
        resolve_comparison_path(protocol["protocol_version"]), survey=protocol
    )


def comparison_for_version(version: str) -> dict[str, Any]:
    """The shipped matrix of *version*, for callers holding a derived protocol.

    A narrowed or in-memory protocol carries the same version but a different
    digest, and no matrix was frozen against it; resolving through the shipped
    file of that version is what lets such a run still record the
    pre-registered entry it lands on.  :func:`comparison_for_protocol` stays
    strict for callers that must prove they hold the frozen matrix.
    """
    from .survey_protocol import load_protocol, resolve_protocol_path

    return comparison_for_protocol(load_protocol(resolve_protocol_path(version)))


def _validate_binding(bound: Any, survey: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(bound, dict):
        raise ValueError("bound_survey_protocol must be an object")
    for key in ("protocol_version", "protocol_sha256"):
        if not isinstance(bound.get(key), str) or not bound[key]:
            raise ValueError(f"bound_survey_protocol missing {key}")
    if survey is None:
        from .survey_protocol import load_protocol, resolve_protocol_path

        # Read the protocol this matrix declares rather than the shipped
        # default: the default moves on with every protocol bump, and loading
        # it here is what let a v3 run quietly record a v2-bound block.
        survey = load_protocol(resolve_protocol_path(bound["protocol_version"]))
    if bound["protocol_version"] != survey["protocol_version"]:
        raise ValueError(
            "bound survey protocol_version disagrees with the execution protocol: "
            f"{bound['protocol_version']!r} != {survey['protocol_version']!r}"
        )
    return bound


def _validate_sources(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("sources must be a non-empty object list")
    sources: dict[str, dict[str, Any]] = {}
    for source in raw:
        if not isinstance(source, dict):
            raise ValueError("sources must be a non-empty object list")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("every source needs a non-empty source_id")
        if source_id in sources:
            raise ValueError(f"duplicate source_id: {source_id}")
        sources[source_id] = source
    return sources


def _validate_anchors(raw: Any, sources: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("anchors must be a non-empty object list")
    anchors: dict[str, dict[str, Any]] = {}
    for anchor in raw:
        if not isinstance(anchor, dict):
            raise ValueError("anchors must be a non-empty object list")
        for field in _REQUIRED_ANCHOR:
            if field not in anchor:
                raise ValueError(f"anchor missing {field}")
        anchor_id = anchor["anchor_id"]
        if not isinstance(anchor_id, str) or not anchor_id:
            raise ValueError("anchor_id must be a non-empty string")
        if anchor_id in anchors:
            raise ValueError(f"duplicate anchor_id: {anchor_id}")
        if anchor["source_id"] not in sources:
            raise ValueError(f"anchor references an unknown source_id: {anchor['source_id']}")
        if anchor["metric"] not in _METRICS:
            raise ValueError(f"anchor metric is not recognized: {anchor['metric']!r}")
        if anchor["uncertainty_kind"] not in _UNCERTAINTY_KINDS:
            raise ValueError("anchor uncertainty_kind must be std or sem")
        if anchor["value_producer"] not in _VALUE_PRODUCERS:
            raise ValueError(
                f"anchor value_producer is not recognized: {anchor['value_producer']!r}"
            )
        if anchor["protocol_provenance"] not in _PROVENANCE:
            raise ValueError("anchor protocol_provenance must be certain or uncertain")
        if anchor["review_state"] not in _REVIEW_STATES:
            raise ValueError("anchor review_state must be pending_review or accepted")
        if anchor["dataset"] != "cifar10" and anchor["dataset"] not in {
            "spambase",
            "imdb",
        }:
            raise ValueError(f"anchor dataset is not a pilot dataset: {anchor['dataset']}")
        _validate_anchor_numbers(anchor)
        anchors[anchor_id] = anchor
    return anchors


def _validate_anchor_numbers(anchor: dict[str, Any]) -> None:
    mean = anchor["mean_percent"]
    spread = anchor["spread_percent"]
    repeats = anchor["n_repeats"]
    if type(mean) is not int and type(mean) is not float:
        raise ValueError("anchor mean_percent must be numeric")
    if type(spread) is not int and type(spread) is not float:
        raise ValueError("anchor spread_percent must be numeric")
    if not 0 <= float(mean) <= 100:
        raise ValueError("anchor mean_percent must be within [0, 100] percentage points")
    if float(spread) < 0:
        raise ValueError("anchor spread_percent must not be negative")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("anchor n_repeats must be a positive integer")
    if anchor["metric"] == "error_rate":
        converted = anchor.get("metric_conversion")
        if not isinstance(converted, dict):
            raise ValueError("an error_rate anchor needs an explicit metric_conversion")
        if converted.get("to") != _COMPARISON_METRIC:
            raise ValueError("metric_conversion must convert to accuracy")
        for key in ("rule", "mean_percent", "spread_percent"):
            if key not in converted:
                raise ValueError(f"metric_conversion missing {key}")
        if not 0 <= float(converted["mean_percent"]) <= 100:
            raise ValueError("metric_conversion mean_percent must be within [0, 100]")
        if float(converted["spread_percent"]) < 0:
            raise ValueError("metric_conversion spread_percent must not be negative")


def _validate_mappings(raw: Any, anchors: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("mappings must be a non-empty object list")
    mappings: dict[str, dict[str, Any]] = {}
    for mapping in raw:
        if not isinstance(mapping, dict):
            raise ValueError("mappings must be a non-empty object list")
        mapping_id = mapping.get("mapping_id")
        if not isinstance(mapping_id, str) or not mapping_id:
            raise ValueError("every mapping needs a non-empty mapping_id")
        if mapping_id in mappings:
            raise ValueError(f"duplicate mapping_id: {mapping_id}")
        if mapping.get("scope") not in _MAPPING_SCOPES:
            raise ValueError("mapping scope must be unit or row")
        selector = mapping.get("result_selector")
        if not isinstance(selector, dict):
            raise ValueError("mapping result_selector must be an object")
        for field in _SELECTOR_FIELDS:
            if not isinstance(selector.get(field), str) or not selector[field]:
                raise ValueError(f"result_selector missing {field}")
        row_markers = [field for field in _ROW_LEVEL_FIELDS if selector[field] == _ROW_LEVEL]
        if mapping["scope"] == "row":
            if len(row_markers) != len(_ROW_LEVEL_FIELDS):
                raise ValueError(
                    "a row-level mapping must mark labeling_mechanism, c_token and "
                    "selection_protocol as row_level"
                )
        elif row_markers:
            raise ValueError(
                f"a unit-level mapping must not use the row_level marker: {', '.join(row_markers)}"
            )
        if mapping.get("eligibility") not in _ELIGIBILITY:
            raise ValueError(
                f"mapping eligibility is not recognized: {mapping.get('eligibility')!r}"
            )
        if mapping.get("review_state") not in _REVIEW_STATES:
            raise ValueError("mapping review_state must be pending_review or accepted")
        anchor_ids = mapping.get("anchor_ids")
        if not isinstance(anchor_ids, list) or any(
            not isinstance(item, str) for item in anchor_ids
        ):
            raise ValueError("mapping anchor_ids must be a string list")
        for anchor_id in anchor_ids:
            if anchor_id not in anchors:
                raise ValueError(f"mapping references an unknown anchor: {anchor_id}")
        if mapping["eligibility"] == "numeric":
            if not anchor_ids:
                raise ValueError("a numeric mapping needs at least one anchor")
            for anchor_id in anchor_ids:
                anchor = anchors[anchor_id]
                if anchor["protocol_provenance"] != "certain":
                    raise ValueError(
                        "an anchor with protocol_provenance=uncertain cannot be numeric: "
                        f"{anchor_id}"
                    )
        if not isinstance(mapping.get("rationale"), str) or not mapping["rationale"]:
            raise ValueError("mapping needs a non-empty rationale")
        mappings[mapping_id] = mapping
    return mappings


def _validate_contradictions(
    raw: Any,
    sources: dict[str, dict[str, Any]],
    anchors: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("contradictions must be an object list")
    contradictions: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("contradictions must be an object list")
        contradiction_id = item.get("contradiction_id")
        if not isinstance(contradiction_id, str) or not contradiction_id:
            raise ValueError("every contradiction needs a non-empty contradiction_id")
        if contradiction_id in contradictions:
            raise ValueError(f"duplicate contradiction_id: {contradiction_id}")
        if item.get("source_id") not in sources:
            raise ValueError("a contradiction references an unknown source_id")
        if not isinstance(item.get("code_commit"), str) or not item["code_commit"]:
            raise ValueError("a contradiction must bind the code_commit it was taken from")
        if item.get("resolution") != "code":
            raise ValueError(
                "contradiction resolution must be 'code': protocol metadata follows the "
                "locked repository (see the P2.0c plan 6.1)"
            )
        if item.get("review_state") not in _REVIEW_STATES:
            raise ValueError("contradiction review_state is not recognized")
        affected = item.get("affected_anchors")
        if not isinstance(affected, list) or any(not isinstance(a, str) for a in affected):
            raise ValueError("contradiction affected_anchors must be a string list")
        for anchor_id in affected:
            if anchor_id not in anchors:
                raise ValueError(
                    f"contradiction affected_anchors references an unknown anchor: {anchor_id}"
                )
            if anchors[anchor_id]["protocol_provenance"] != "uncertain":
                raise ValueError(
                    "an anchor affected by an unresolved contradiction must be "
                    f"protocol_provenance=uncertain: {anchor_id}"
                )
        contradictions[contradiction_id] = item
    return contradictions


def resolve_comparison_unit(
    result_identity: dict[str, str], *, comparison: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve one project result unit to its pre-registered comparison entry.

    Exactly one mapping must match; a missing or ambiguous match is a coverage
    defect, not something to paper over at evaluation time.
    """
    if comparison is None:
        comparison = load_comparison_protocol()
    selector = {field: result_identity.get(field) for field in _SELECTOR_FIELDS}
    matches = [
        mapping
        for mapping in comparison["mappings"]
        if all(mapping["result_selector"][field] == selector[field] for field in _SELECTOR_FIELDS)
    ]
    if not matches:
        raise ValueError(f"no comparison mapping covers result unit: {selector}")
    if len(matches) > 1:
        raise ValueError(
            "result unit resolves to more than one mapping: "
            f"{[mapping['mapping_id'] for mapping in matches]}"
        )
    mapping = matches[0]
    anchors = [
        anchor for anchor in comparison["anchors"] if anchor["anchor_id"] in mapping["anchor_ids"]
    ]
    return {
        "mapping_id": mapping["mapping_id"],
        "eligibility": mapping["eligibility"],
        "protocol_differences": list(mapping.get("protocol_differences", [])),
        "rationale": mapping["rationale"],
        "anchors": anchors,
    }


def _selector_key(selector: dict[str, str]) -> tuple[str, ...]:
    return tuple(selector[field] for field in _SELECTOR_FIELDS)


def expected_result_units(
    survey: dict[str, Any], *, profile_map: dict[str, Any] | None = None
) -> tuple[dict[str, str], ...]:
    """Enumerate every pilot result unit the comparison matrix must cover.

    Derived from the execution protocol, never hand-listed, so a protocol bump
    cannot silently leave units uncovered.  Runnable PU rows expand over the
    SCAR c-grid times PA/OA plus the SAR c-grids (OA only); each PN oracle row
    yields one c-independent unit; each non-runnable row yields one marker unit
    instead of fabricated c/mechanism expansions.

    ``profile_map`` translates each row's profile to the algorithm whose entry
    the matrix registers (plan D40): a ``mapped`` profile inherits that entry, so
    two profiles of one algorithm must *not* produce two entries, and repeating
    them would let the same anchor be counted twice.  Rows whose profile is
    ``not_in_matrix`` are outside the matrix and are dropped here; they are
    counted by :func:`validate_comparison_coverage` instead of silently vanishing.
    """
    c_tokens = survey["c_tokens"]
    units: list[dict[str, str]] = []
    for row in survey["execution_units"]:
        profile = row["method"]
        if profile_map is not None:
            registration = profile_map["profiles"].get(profile)
            if registration is None:
                raise ValueError(f"profile has no entry in the survey profile map: {profile!r}")
            if registration["registration"] != "mapped":
                continue
            profile = registration["inherits"]
        identity = {
            "method": profile,
            "dataset": row["dataset"],
            "training_path": row["training_path"],
            "metric": _COMPARISON_METRIC,
        }
        if not row["runnable"]:
            marker = _NON_RUNNABLE
            units.append(
                {
                    **identity,
                    "labeling_mechanism": marker,
                    "c_token": marker,
                    "selection_protocol": marker,
                }
            )
            continue
        if row["method"] == _PN_ORACLE:
            units.append(
                {
                    **identity,
                    "labeling_mechanism": "c_independent",
                    "c_token": "c_independent",
                    "selection_protocol": "oa",
                }
            )
            continue
        for c_token in c_tokens["scar"]:
            for protocol in _UNIT_SELECTION_PROTOCOLS:
                units.append(
                    {
                        **identity,
                        "labeling_mechanism": "scar",
                        "c_token": c_token,
                        "selection_protocol": protocol,
                    }
                )
        for mechanism in _SAR_MECHANISMS:
            for c_token in c_tokens[mechanism]:
                units.append(
                    {
                        **identity,
                        "labeling_mechanism": mechanism,
                        "c_token": c_token,
                        "selection_protocol": "oa",
                    }
                )
    if profile_map is None:
        return tuple(units)
    # Two profiles of one algorithm inherit the same entry (D40), so they expand
    # to the same units; dedupe on the selector while keeping first-seen order so
    # the enumeration stays a set rather than a multiset.
    seen: set[tuple[str, ...]] = set()
    unique: list[dict[str, str]] = []
    for unit in units:
        key = _selector_key(unit)
        if key not in seen:
            seen.add(key)
            unique.append(unit)
    return tuple(unique)


def validate_comparison_coverage(
    survey: dict[str, Any],
    comparison: dict[str, Any],
    *,
    profile_map: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Prove the matrix covers every result unit exactly once.

    Three independent failures are possible and each one is fatal: a unit with
    no mapping, a unit covered twice, and a mapping that covers nothing (which
    usually means a typo'd token that would otherwise silently drop a whole
    row).  Anchors must land on exactly one mapping so the same published
    number is never counted as two independent observations.
    """
    expected = expected_result_units(survey, profile_map=profile_map)
    unit_index: dict[tuple[str, ...], str] = {}
    for mapping in comparison["mappings"]:
        if mapping["scope"] != "unit":
            continue
        key = _selector_key(mapping["result_selector"])
        if key in unit_index:
            raise ValueError(
                "two unit-level mappings cover the same result unit: "
                f"{unit_index[key]} and {mapping['mapping_id']}"
            )
        unit_index[key] = mapping["mapping_id"]

    expected_keys = {_selector_key(unit) for unit in expected}
    missing = [unit for unit in expected if _selector_key(unit) not in unit_index]
    if missing:
        sample = missing[0]
        raise ValueError(
            f"{len(missing)} result unit(s) have no comparison mapping, first: {sample}"
        )
    orphan = [name for key, name in unit_index.items() if key not in expected_keys]
    if orphan:
        raise ValueError(
            f"{len(orphan)} unit-level mapping(s) cover no expected result unit: {orphan[:3]}"
        )

    if profile_map is None:
        rows = {
            (row["method"], row["dataset"], row["training_path"])
            for row in survey["execution_units"]
        }
        outside_matrix = 0
    else:
        # A row named by a profile outside the matrix has no algorithm identity
        # to check a row-level mapping against, so it is counted, not matched.
        rows = set()
        outside_matrix = 0
        for row in survey["execution_units"]:
            registration = profile_map["profiles"].get(row["method"])
            if registration is None:
                raise ValueError(
                    f"profile has no entry in the survey profile map: {row['method']!r}"
                )
            if registration["registration"] != "mapped":
                outside_matrix += 1
                continue
            rows.add((registration["inherits"], row["dataset"], row["training_path"]))
    row_level = [mapping for mapping in comparison["mappings"] if mapping["scope"] == "row"]
    for mapping in row_level:
        selector = mapping["result_selector"]
        identity = (selector["method"], selector["dataset"], selector["training_path"])
        if identity not in rows:
            raise ValueError(
                f"row-level mapping covers no execution row: {mapping['mapping_id']} {identity}"
            )

    anchor_use: dict[str, list[str]] = {}
    for mapping in comparison["mappings"]:
        for anchor_id in mapping["anchor_ids"]:
            anchor_use.setdefault(anchor_id, []).append(mapping["mapping_id"])
    orphan_anchors = [
        anchor["anchor_id"]
        for anchor in comparison["anchors"]
        if anchor["anchor_id"] not in anchor_use
    ]
    if orphan_anchors:
        raise ValueError(f"anchor(s) referenced by no mapping: {orphan_anchors[:3]}")
    reused = {name: users for name, users in anchor_use.items() if len(users) > 1}
    if reused:
        raise ValueError(
            "an anchor may be referenced by exactly one mapping, but these are reused: "
            f"{sorted(reused)[:3]}"
        )

    return {
        "units": len(expected),
        "unit_mappings": len(unit_index),
        "row_mappings": len(row_level),
        "anchors": len(anchor_use),
        "outside_matrix": outside_matrix,
    }


def _to_percentage_points(result: dict[str, Any]) -> tuple[float, float]:
    unit = result["metric_unit"]
    if unit not in _RESULT_UNITS:
        raise ValueError("result metric_unit must be fraction or percentage_points")
    mean = float(result["mean"])
    spread = float(result["std"])
    if spread < 0:
        raise ValueError("result std must not be negative")
    if unit == "fraction":
        if not 0 <= mean <= 1:
            raise ValueError("a fraction result mean must be within [0, 1]")
        return mean * 100.0, spread * 100.0
    if not 0 <= mean <= 100:
        raise ValueError("a percentage_points result mean must be within [0, 100]")
    return mean, spread


def _anchor_percentage_points(anchor: dict[str, Any]) -> tuple[float, float]:
    if anchor["metric"] == "error_rate":
        converted = anchor["metric_conversion"]
        return float(converted["mean_percent"]), float(converted["spread_percent"])
    return float(anchor["mean_percent"]), float(anchor["spread_percent"])


def evaluate_anchor(result_summary: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered numeric rule; never conclude more than it can.

    Only ``numeric`` mappings may reach this function.  Crossing the threshold
    returns an investigation trigger -- this module is not allowed to declare
    an implementation error, because that judgement needs the root-cause pass
    described in the P2.0c plan 7.2.
    """
    eligibility = resolved["eligibility"]
    if eligibility != "numeric":
        raise ValueError(f"evaluate_anchor only accepts numeric mappings, got {eligibility!r}")
    anchors = resolved["anchors"]
    if not anchors:
        raise ValueError("a numeric mapping needs at least one anchor")
    for anchor in anchors:
        if anchor["protocol_provenance"] != "certain":
            raise ValueError(f"anchor protocol_provenance is uncertain: {anchor['anchor_id']}")

    mean_ours_pp, spread_ours_pp = _to_percentage_points(result_summary)
    n_ours = result_summary.get("n_repeats")
    if type(n_ours) is not int or n_ours < 2:
        raise ValueError("result n_repeats must be an integer of at least 2")

    per_anchor = []
    for anchor in anchors:
        mean_anchor_pp, spread_anchor_pp = _anchor_percentage_points(anchor)
        n_anchor = anchor["n_repeats"]
        delta_pp = abs(mean_ours_pp - mean_anchor_pp)
        if anchor["uncertainty_kind"] == "std":
            sampling = spread_anchor_pp**2 / n_anchor
        else:
            # The published spread already is the standard error of the mean.
            sampling = spread_anchor_pp**2
        se_pooled_pp = (sampling + spread_ours_pp**2 / n_ours) ** 0.5
        limit_pp = max(3.0, 2.0 * se_pooled_pp)
        per_anchor.append(
            {
                "anchor_id": anchor["anchor_id"],
                "mean_anchor_pp": mean_anchor_pp,
                "spread_anchor_pp": spread_anchor_pp,
                "uncertainty_kind": anchor["uncertainty_kind"],
                "n_anchor": n_anchor,
                "delta_pp": delta_pp,
                "se_pooled_pp": se_pooled_pp,
                "limit_pp": limit_pp,
                "status": "consistent" if delta_pp <= limit_pp else "investigate",
            }
        )

    worst = max(per_anchor, key=lambda item: item["delta_pp"] / max(item["limit_pp"], 1e-12))
    status = (
        "investigate"
        if any(item["status"] == "investigate" for item in per_anchor)
        else ("consistent")
    )
    return {
        "mapping_id": resolved["mapping_id"],
        "status": status,
        "mean_ours_pp": mean_ours_pp,
        "spread_ours_pp": spread_ours_pp,
        "n_ours": n_ours,
        "mean_anchor_pp": worst["mean_anchor_pp"],
        "spread_anchor_pp": worst["spread_anchor_pp"],
        "uncertainty_kind": worst["uncertainty_kind"],
        "n_anchor": worst["n_anchor"],
        "delta_pp": worst["delta_pp"],
        "se_pooled_pp": worst["se_pooled_pp"],
        "limit_pp": worst["limit_pp"],
        "per_anchor": per_anchor,
        "protocol_differences": list(resolved.get("protocol_differences", [])),
    }


COMPARISON_CONCLUSIONS = (
    "implementation_error",
    "protocol_difference_explains",
    "unexplained_warning",
)
_ESCALATING_CONCLUSION = "unexplained_warning"


def _resolve_entry(
    result_identity: dict[str, str],
    *,
    comparison: dict[str, Any],
    profile_id: str | None,
    survey: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve one result unit through the matrix, or through the profile map.

    A profile outside the matrix raises rather than returning a half-filled
    entry: there is no mapping to record, and borrowing the base algorithm's
    entry is exactly the substitution D40 forbids.
    """
    if profile_id is None:
        return resolve_comparison_unit(result_identity, comparison=comparison)
    from .survey_profile_map import resolve_profile_mapping

    resolved = resolve_profile_mapping(
        profile_id, result_identity, protocol=survey, comparison=comparison
    )
    if resolved.get("status") == "not_in_matrix":
        raise ValueError(
            f"profile {profile_id!r} is outside the comparison matrix: "
            f"{resolved['profile_rationale']}"
        )
    return resolved


def comparison_context(
    result_identity: dict[str, str],
    *,
    comparison: dict[str, Any] | None = None,
    profile_id: str | None = None,
    survey: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fields a run manifest records so every result row states its comparison.
    Only the version, digest and resolved mapping id belong in the run
    manifest: the actual mean/spread across seeds and any literature verdict
    are aggregation work (P2.2), not run work (P2.1).

    ``profile_id`` switches the lookup to the versioned profile map: under the
    profile-identity schema the unit's ``method`` is a profile, and the entry it
    inherits is the one its base algorithm registered.
    """
    if comparison is None:
        # A caller that knows the protocol gets the matrix bound to it; only a
        # caller that names neither keeps the historical default.
        comparison = (
            comparison_for_protocol(survey) if survey is not None else load_comparison_protocol()
        )
    profile_fields: dict[str, Any] = {}
    resolved = _resolve_entry(
        result_identity, comparison=comparison, profile_id=profile_id, survey=survey
    )
    if profile_id is not None:
        profile_fields = {
            "profile_id": profile_id,
            "profile_map_version": resolved["profile_map_version"],
            "profile_map_sha256": resolved["profile_map_sha256"],
            "profile_registration": resolved["profile_registration"],
            "profile_deviation_count": resolved["deviation_count"],
        }
    return {
        "comparison_version": comparison["comparison_version"],
        "comparison_sha256": comparison_digest(comparison),
        "mapping_id": resolved["mapping_id"],
        "eligibility": resolved["eligibility"],
        "review_status": comparison["review_status"],
        "comparison_blockers": list(comparison["formal_blockers"]),
        **profile_fields,
    }


def run_comparison_units(
    protocol_context: dict[str, Any],
    config: dict[str, Any],
    *,
    survey: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    profile_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """The comparison entries one run's manifest records, keyed by selection protocol.

    A run is not one result unit.  A SCAR run selects twice -- once on the PU
    view, once on the clean view -- and the matrix registers those as two units
    under two different mappings, so a manifest carrying a single ``mapping_id``
    would leave half of its own result unadjudicable.  The keys are the protocol
    names the manifest's own ``selection`` block uses, so P2.2 joins the two by
    name rather than by position.

    The units are taken from :func:`expected_result_units` rather than restated
    here: which units a run produces is exactly what the coverage gate proves,
    and a run that expanded its own grid differently would resolve to a mapping
    the gate never counted.

    Empty when the matrix covers none of them, which is not an error.  The
    execution protocol lets ``--c`` be overridden and asks only that the
    deviation be recorded, so a run at a c the matrix never registered is off
    the grid by construction and its ``protocol_deviation`` already says so;
    refusing it would delete a documented capability.  A *pre-registered* unit
    the matrix fails to cover is a different failure -- it is caught once
    before any run, by :func:`validate_comparison_coverage`, and it reaches
    here as :func:`comparison_context` failing to resolve.
    """
    if survey is None:
        from .survey_protocol import load_protocol

        survey = load_protocol(config["survey_protocol"]["path"])
    if profile_id is not None:
        from .survey_profile_map import (
            STATUS_NOT_IN_MATRIX,
            load_profile_map,
            profile_map_binds,
            profile_registration_status,
        )

        profile_map = load_profile_map()
        if not profile_map_binds(profile_map, survey):
            # A narrowed or otherwise noncanonical matrix has no versioned
            # profile mapping; the manifest's registration status already says so.
            return {}
        registration = profile_registration_status(
            profile_id, protocol=survey, profile_map=profile_map
        )
        if registration["comparison_registration_status"] == STATUS_NOT_IN_MATRIX:
            # A profile outside the matrix has no entry to record.  Its manifest
            # already carries that status, so an empty block says the same thing
            # without borrowing the base algorithm's pre-registered row.
            return {}
    if comparison is None:
        # Resolve through the *shipped* matrix of this protocol version: a
        # narrowed copy still records the pre-registered entry its unit lands on,
        # and the noncanonical matrix is already on the manifest as a deviation.
        comparison = comparison_for_version(survey["protocol_version"])
    row = protocol_context["execution_unit"]
    units = [unit for unit in expected_result_units(survey) if _is_run_unit(unit, row, config)]
    entries: dict[str, dict[str, Any]] = {}
    for unit in units:
        # One unit per selection protocol is what makes the entry keys name the
        # manifest's own selection keys; two would mean one of them silently
        # overwrote the other.
        key = unit["selection_protocol"].upper()
        if key in entries:
            raise ValueError(f"a run produces at most one result unit per protocol, got two {key}")
        entries[key] = comparison_context(
            unit, comparison=comparison, profile_id=profile_id, survey=survey
        )
    return entries


def _is_run_unit(unit: dict[str, str], row: dict[str, Any], config: dict[str, Any]) -> bool:
    """Whether one expected result unit is the unit this run produces.

    A PN oracle row trains on real labels and generates no PU label view, so it
    contributes its one c-independent unit whatever mechanism the CLI carries to
    satisfy the runner's c-token binding.
    """
    if (unit["method"], unit["dataset"], unit["training_path"]) != (
        row["method"],
        row["dataset"],
        row["training_path"],
    ):
        return False
    if row["method"] == _PN_ORACLE:
        return True
    request = config["survey_protocol"]
    return unit["labeling_mechanism"] == request["mechanism"] and unit["c_token"] == config.get(
        "c_requested_token"
    )


def build_comparison_report(
    result_summary: dict[str, Any],
    *,
    comparison: dict[str, Any] | None = None,
    profile_id: str | None = None,
    survey: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the aggregation-side comparison artifact for one result unit.
    Never touches the result summary: this layer annotates, it does not
    re-measure, so the original metrics and any ranking stay exactly as
    produced.  A unit that cannot be adjudicated still gets a report -- the
    point is that its reason is recorded rather than left implicit.

    ``profile_id`` resolves the unit through the versioned profile map, the
    same way a run does; an aggregation over v3 rows therefore reads the
    entry the profile inherits instead of looking the profile up as a method.
    """
    if comparison is None:
        comparison = (
            comparison_for_protocol(survey) if survey is not None else load_comparison_protocol()
        )
    resolved = _resolve_entry(
        result_summary["result_identity"],
        comparison=comparison,
        profile_id=profile_id,
        survey=survey,
    )
    report: dict[str, Any] = {
        "mapping_id": resolved["mapping_id"],
        "eligibility": resolved["eligibility"],
        "rationale": resolved["rationale"],
        "protocol_differences": list(resolved["protocol_differences"]),
        "anchor_ids": [anchor["anchor_id"] for anchor in resolved["anchors"]],
        "comparison_version": comparison["comparison_version"],
        "comparison_sha256": comparison_digest(comparison),
    }
    if resolved["eligibility"] != "numeric":
        report["status"] = "not_comparable"
        return report
    verdict = evaluate_anchor(result_summary, resolved)
    report["status"] = verdict["status"]
    # Carry our own converted figures so the artifact is auditable without the
    # run output it came from.
    report["mean_ours_pp"] = verdict["mean_ours_pp"]
    report["spread_ours_pp"] = verdict["spread_ours_pp"]
    report["n_ours"] = verdict["n_ours"]
    report["anchor_comparisons"] = verdict["per_anchor"]
    if verdict["status"] == "investigate":
        report["investigation"] = {"state": "open", "conclusion": None, "evidence": ""}
    return report


def record_investigation(
    report: dict[str, Any], *, conclusion: str, evidence: str
) -> dict[str, Any]:
    """Attach a root-cause conclusion to a triggered comparison.
    Returns a copy: the report handed in is never mutated in place, and the
    result summary is not reachable from here at all.  Only
    ``unexplained_warning`` sets ``escalates`` -- the threshold is never revised
    in response to a result, and neither is the implementation status.
    """
    if report.get("status") != "investigate":
        raise ValueError("only a triggered investigation can receive a conclusion")
    if conclusion not in COMPARISON_CONCLUSIONS:
        raise ValueError(
            f"unknown investigation conclusion {conclusion!r}; "
            f"choose from {list(COMPARISON_CONCLUSIONS)}"
        )
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("an investigation conclusion needs non-empty evidence")
    recorded = copy.deepcopy(report)
    recorded["investigation"] = {
        "state": "recorded",
        "conclusion": conclusion,
        "evidence": evidence,
        "escalates": conclusion == _ESCALATING_CONCLUSION,
    }
    return recorded
