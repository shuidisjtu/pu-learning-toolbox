"""P2.2 numeric summary: turn pilot manifests into aggregate report rows.

Three things this module is careful about, each pinned by tests:

* **``blocked`` is not a status.**  ``scripts/aggregate_survey_runs.py`` reports a
  ``(seed, c)`` unit's ``state`` as ``comparable``/``blocked`` -- the gate's own
  vocabulary.  The delivery status is a different, closed set
  (:data:`STATUSES`), and a gate-blocked unit becomes ``partial`` carrying the
  reason ``fairness_gate_blocked``.  Copying the gate's word into ``status``
  would put a value in the field that no report is allowed to consume.
* **A row is keyed over methods, not over seeds.**  Seeds are the aggregation
  axis: the five repeats collapse into one mean plus a sample standard deviation
  (protocol §5 clause 2).  A row key that kept ``seed`` would split each method
  into five rows and no mean would ever exist.
* **The comparison adapter's field names are not ours to choose.**  The
  pre-registered matrix reads ``result_identity`` / ``mean`` / ``std`` /
  ``n_repeats`` / ``metric_unit``; :func:`to_result_summary` emits exactly that
  and nothing else, because a locally invented name would only fail later, at
  the matrix boundary.

Pure functions over manifest dicts -- no file access, no training, no test truth.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "survey-summary-1"

#: Delivery status vocabulary.  A report row's ``status`` is one of these and
#: nothing else.  ``not_reproducible`` is protocol §5 clause 6; the other six are
#: presentation layers over the same runs.
STATUSES: tuple[str, ...] = (
    "formal",
    "partial",
    "technical_probe",
    "historical",
    "refused",
    "incomplete",
    "not_reproducible",
)

#: Highest priority first, so the first member of a row's candidate set wins.
#: ``refused`` leads because a refused run produced no result to report at all.
#: ``not_reproducible`` outranks ``partial``: the former is a protocol hard rule,
#: the latter only says "there is a number but it may not be ranked".
STATUS_PRIORITY: tuple[str, ...] = (
    "refused",
    "not_reproducible",
    "incomplete",
    "technical_probe",
    "historical",
    "partial",
    "formal",
)

#: Reason codes.  Free-form prose is not allowed here: the taxonomy is
#: machine-checked, one ``status`` may be explained by several reasons, and
#: ``status == "formal"`` must be exactly ``reasons == []``.
REASONS: tuple[str, ...] = (
    # gate-side outcomes, translated from the aggregation entry point
    "rejected_versioned_pilot",
    "execution_unit_not_runnable",
    "no_selected_candidate",
    "fairness_gate_blocked",
    "split_mismatch",
    "budget_mismatch",
    "mechanism_mismatch",
    "representation_mismatch",
    "group_key_mismatch",
    # provenance / reproduction gaps (protocol §5 clause 6)
    "missing_manifest_field",
    "missing_resource_provenance",
    "oracle_c_independent_conflict",
    "superseded_protocol",
    # completeness gaps
    "missing_seed",
    "missing_c_token",
    "run_in_progress",
    # formal-eligibility blockers, for a run that still produced metrics
    "formal_blockers_present",
    "protocol_deviation",
    "pending_collaborator_review",
    # scope notes
    "probe_only",
)

#: The seven ``result_identity`` fields the matrix selects on, in its own order.
RESULT_IDENTITY_FIELDS: tuple[str, ...] = (
    "method",
    "dataset",
    "labeling_mechanism",
    "c_token",
    "selection_protocol",
    "metric",
    "training_path",
)

#: Unit of the ``mean``/``std`` pair this module emits and the matrix reads back.
METRIC_UNIT = "fraction"


class SummaryError(ValueError):
    """A manifest, row or metric block that cannot be summarized as asked."""


def normalize_c_token(manifest: Mapping[str, Any]) -> str:
    """The normalized ``c_token`` of *manifest*.

    Three sources have to be reconciled, and the order below is what reconciles
    them:

    * A c-independent row (the PN oracle) is its own token, and the flag is read
      **before** any ``c`` value.  An oracle manifest carries a nominal
      ``c_requested`` of ``0.1`` with ``c_independent`` set, so a reader that
      looked at ``c`` first would file the oracle as an ordinary 0.1 run.
      ``""`` would be equally wrong in the other direction: it reads as "this
      row's ``c`` is unknown", the opposite of what the oracle asserts.
    * Otherwise the recorded ``c_requested_token`` is the token.  It is already a
      string, so no float formatting can turn ``0.05`` into ``0.05000000000000001``.
    * ``generation.train.c_requested`` is the fallback, and is also the value the
      aggregation entry point's ``unit_key`` keys on.  When both are present they
      therefore have to agree: a disagreement would file one run under two keys
      depending on which reader asked.
    """
    if manifest.get("c_independent"):
        return "c_independent"
    train = manifest.get("generation", {}).get("train")
    if not isinstance(train, Mapping):
        raise SummaryError("manifest generation is missing the train label view")
    token = manifest.get("c_requested_token")
    requested = train.get("c_requested")
    if token is None and requested is None:
        raise SummaryError("manifest has neither c_independent nor a c token")
    if token is not None and requested is not None and str(token) != f"{float(requested):g}":
        raise SummaryError(
            f"manifest c_requested_token {token!r} disagrees with "
            f"generation.train.c_requested {requested!r}"
        )
    return str(token) if token is not None else f"{float(requested):g}"


def mechanism_of(manifest: Mapping[str, Any]) -> str:
    """The labeling mechanism this manifest ran under."""
    if manifest.get("c_independent"):
        return "c_independent"
    train = manifest.get("generation", {}).get("train")
    if not isinstance(train, Mapping) or "mechanism" not in train:
        raise SummaryError("manifest generation is missing the train mechanism")
    return str(train["mechanism"])


def row_key(manifest: Mapping[str, Any], *, selection_protocol: str) -> tuple[str, ...]:
    """``numeric_report_row_key``: the key one mean/std row is reported under.

    Contains no ``seed`` -- that is the aggregation axis, and leaving it in would
    make each method report five rows of one observation instead of one row of
    five (see the module docstring).  ``selection_protocol`` is a *derived*
    dimension: the manifest stores results under ``PA``/``OA`` dictionary keys
    and records no such field of its own.
    """
    unit = manifest["execution_unit"]
    return (
        unit["comparability_group"],
        manifest["run_view"],
        mechanism_of(manifest),
        unit["method"],
        selection_protocol,
        normalize_c_token(manifest),
    )


def choose_status(candidates: Iterable[str]) -> str:
    """Resolve several candidate statuses to the single highest-priority one."""
    wanted = set(candidates)
    unknown = sorted(wanted - set(STATUSES))
    if unknown:
        raise SummaryError(f"unknown status value(s): {unknown}; blocked is not a status")
    for status in STATUS_PRIORITY:
        if status in wanted:
            return status
    raise SummaryError("no status candidates given")


def reasons_for(*, status: str, reasons: Sequence[str]) -> list[str]:
    """Validate ``reasons`` against *status*: formal iff empty, else non-empty."""
    cleaned = sorted(set(reasons))
    unknown = sorted(set(cleaned) - set(REASONS))
    if unknown:
        raise SummaryError(f"unknown reason code(s): {unknown}")
    if status == "formal" and cleaned:
        raise SummaryError(f"formal rows carry no reasons, got {cleaned}")
    if status != "formal" and not cleaned:
        raise SummaryError(f"status {status!r} needs at least one reason")
    return cleaned


def _finite(value: float, *, what: str) -> float:
    if not math.isfinite(value):
        raise SummaryError(f"{what} must be finite, got {value!r}")
    return float(value)


def sample_std(values: Sequence[float]) -> float | None:
    """Sample standard deviation (``n - 1``), or ``None`` below two observations.

    Sample rather than population: the five repeats are a sample of the runs the
    seed could have produced, and the pilot reports the spread of that sample.
    A single observation has no spread to report, and returning ``0.0`` there
    would claim certainty the run does not have.
    """
    if len(values) < 2:
        return None
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def aggregate_metric(
    observations: Iterable[tuple[int, float]],
    *,
    metric_name: str,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Collapse ``(seed, value)`` pairs into one row's metric block.

    ``mean``/``std`` stay in fraction: that is what the pre-registered matrix
    reads.  The percentage-point pair is a display convenience derived from them,
    never the other way round -- scaling first and averaging after would change
    the last digit.
    """
    pairs = sorted(
        (int(seed), _finite(float(value), what=f"{metric_name} value"))
        for seed, value in observations
    )
    seeds = [seed for seed, _ in pairs]
    if len(set(seeds)) != len(seeds):
        raise SummaryError(f"metric {metric_name!r} repeats a seed: {sorted(seeds)}")
    expected = sorted(int(seed) for seed in expected_seeds)
    unexpected = sorted(set(seeds) - set(expected))
    if unexpected:
        # The seed list is the protocol's, not the tree's.  Counting a seed the
        # protocol never asked for would let one anomalous run outvote the five
        # that were designed, and the extra seed would not even show as missing.
        raise SummaryError(
            f"metric {metric_name!r} observed seed(s) {unexpected} "
            f"outside the protocol's {expected}"
        )
    values = [value for _, value in pairs]
    mean = math.fsum(values) / len(values) if values else None
    std = sample_std(values)
    return {
        "metric_name": metric_name,
        "mean": mean,
        "std": std,
        "mean_percent": None if mean is None else mean * 100.0,
        "std_percent": None if std is None else std * 100.0,
        "metric_unit": METRIC_UNIT,
        "n_expected": len(expected),
        "n_observed": len(values),
        "seeds_observed": seeds,
        "missing_seeds": sorted(set(expected) - set(seeds)),
    }


def summarize_costs(
    observations: Iterable[tuple[int, Mapping[str, Any]]],
    *,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Collapse per-run ``resources`` into the cost block protocol §5 asks for.

    Cost belongs to the *run*, not to a ``PA``/``OA`` result row: one manifest
    pays for one training pass and then scores both selection protocols from it.
    Copying the same seconds onto both rows and summing would double-count, so
    the row keys on the run and the metric rows only reference it.

    Peak GPU memory is a maximum, not a sum -- protocol §5 clause 4 defines it as
    the largest allocation over the whole process, and adding peaks across runs
    would describe a machine nobody ran on.
    """
    pairs = sorted((int(seed), dict(resources)) for seed, resources in observations)
    seeds = [seed for seed, _ in pairs]
    expected = sorted(int(seed) for seed in expected_seeds)
    peaks = [item.get("peak_gpu_memory_bytes") for _, item in pairs]
    usable_peaks = [int(value) for value in peaks if value is not None]
    single = [
        float(cost["elapsed_seconds"])
        for _, item in pairs
        for cost in item.get("single_configuration_costs", [])
        if cost.get("elapsed_seconds") is not None
    ]
    tuning = [
        float(item["tuning"]["elapsed_seconds"])
        for _, item in pairs
        if isinstance(item.get("tuning"), Mapping)
        and item["tuning"].get("elapsed_seconds") is not None
    ]
    return {
        "peak_gpu_memory_bytes": max(usable_peaks) if usable_peaks else None,
        "single_configuration_cost_seconds": (
            {
                "mean": math.fsum(single) / len(single),
                "sum": math.fsum(single),
                "n_observed": len(single),
            }
            if single
            else None
        ),
        "tuning_cost_seconds": (
            {"sum": math.fsum(tuning), "n_observed": len(tuning)} if tuning else None
        ),
        "seeds_observed": seeds,
        "missing_seeds": sorted(set(expected) - set(seeds)),
    }


def to_result_summary(row: Mapping[str, Any], *, metric: str = "accuracy") -> dict[str, Any]:
    """Adapt a report row to the pre-registered matrix's input contract.

    The matrix reads ``result_identity`` / ``mean`` / ``std`` / ``n_repeats`` /
    ``metric_unit`` and validates ``metric_unit`` against a two-member set.  This
    function is the only place that translation happens, so a rename here fails
    one test instead of every comparison silently.
    """
    metric_block = row["metric"]
    if metric_block.get("metric_name") != metric:
        raise SummaryError(
            f"row carries metric {metric_block.get('metric_name')!r}, not {metric!r}"
        )
    mean, std, n_repeats = metric_block["mean"], metric_block["std"], metric_block["n_observed"]
    if mean is None or std is None:
        raise SummaryError(
            f"cannot build a result summary from n_observed={n_repeats} "
            "(the matrix needs a mean and a spread)"
        )
    if n_repeats < 2:
        raise SummaryError(f"result n_repeats must be at least 2, got {n_repeats}")
    identity = row["result_identity"]
    missing = [field for field in RESULT_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise SummaryError(f"result_identity is missing {missing}")
    ordered_identity = {field: identity[field] for field in RESULT_IDENTITY_FIELDS}
    return {
        "result_identity": ordered_identity,
        "mean": mean,
        "std": std,
        "n_repeats": n_repeats,
        "metric_unit": metric_block["metric_unit"],
    }


#: Refusal reasons the aggregation entry point returns for a run it will not
#: aggregate at all.  Each becomes a ``refused`` row carrying its own code: the
#: run produced no result, so there is nothing to present even partially.
GATE_REFUSAL_REASONS: frozenset[str] = frozenset(
    {"rejected_versioned_pilot", "execution_unit_not_runnable", "no_selected_candidate"}
)

#: Substrings of the aggregation gate's refusals, mapped to a finer reason code.
#: The gate's messages are part of its contract -- its own tests match on them --
#: so reading them reads a stable interface rather than scraping free text.  The
#: fallback is the coarse ``fairness_gate_blocked``, so a message this table does
#: not recognize is reported as a fairness refusal rather than dropped.
GATE_MESSAGE_REASONS: tuple[tuple[str, str], ...] = (
    ("representation_sha256 differs", "representation_mismatch"),
    ("split_sha256", "split_mismatch"),
    ("different budgets across units", "budget_mismatch"),
    ("mixes protocol versions", "superseded_protocol"),
    ("spans training paths", "group_key_mismatch"),
    ("duplicate method", "mechanism_mismatch"),
)

#: ``formal_blockers`` values this pilot records, and the code each becomes.  An
#: unmapped blocker still yields ``formal_blockers_present``: a blocker added
#: later must not vanish from a report because this table was not extended.
BLOCKER_REASONS: Mapping[str, str] = {
    "protocol_deviation": "protocol_deviation",
    "collaborator_review": "pending_collaborator_review",
}


def status_for_refusal(reason: str) -> tuple[str, list[str]]:
    """The status of a run the aggregation entry point refused outright."""
    if reason not in GATE_REFUSAL_REASONS:
        raise SummaryError(f"not a gate refusal reason: {reason!r}")
    return "refused", reasons_for(status="refused", reasons=[reason])


def status_for_gate_block(blockers: Iterable[str]) -> tuple[str, list[str]]:
    """The status of a unit the gate held back whose metrics nevertheless exist.

    ``partial``, never ``blocked``: that word is the gate's own vocabulary for a
    unit's ``state``, and a report that copied it would put a value in ``status``
    the closed set does not admit.
    """
    present = sorted(set(blockers))
    if not present:
        raise SummaryError("a gate-blocked unit must name at least one blocker")
    codes = {"formal_blockers_present"}
    codes.update(BLOCKER_REASONS.get(blocker, "formal_blockers_present") for blocker in present)
    return "partial", reasons_for(status="partial", reasons=sorted(codes))


def status_for_gate_error(message: str) -> tuple[str, list[str]]:
    """The status of a group or unit the gate refused to compare at all.

    Group-level refusals cannot be attributed to one manifest -- the gate raises
    on the first bad group and says which rule broke, not which file did it.  The
    caller therefore attaches the affected members separately.
    """
    for marker, reason in GATE_MESSAGE_REASONS:
        if marker in message:
            return "partial", reasons_for(
                status="partial", reasons=[reason, "fairness_gate_blocked"]
            )
    return "partial", reasons_for(status="partial", reasons=["fairness_gate_blocked"])


def status_for_completeness(
    *, missing_seeds: Sequence[int], total_seeds: int, run_in_progress: bool = False
) -> tuple[str, list[str]]:
    """``formal`` when every protocol seed ran, otherwise as incomplete as it looks.

    A subset of seeds is still a result and belongs in the partial table.  Only
    when every seed is absent does the row become ``incomplete`` -- and a run
    still writing is ``incomplete`` too, because its absent seeds are absent
    because it has not finished rather than because anything failed.
    """
    if run_in_progress:
        return "incomplete", reasons_for(status="incomplete", reasons=["run_in_progress"])
    if not missing_seeds:
        return "formal", []
    status = "incomplete" if len(missing_seeds) >= total_seeds else "partial"
    return status, reasons_for(status=status, reasons=["missing_seed"])


def resolve_status(candidates: Iterable[str], reasons: Iterable[str]) -> tuple[str, list[str]]:
    """Combine several checks' verdicts into one status and the merged reasons.

    Raises when the winner is ``formal`` but reasons were collected: a row that
    is formally reportable cannot also carry a defect.  That is the invariant
    ``status == "formal"`` iff ``reasons == []``, enforced where the two meet
    rather than left for a reader to notice.
    """
    status = choose_status(candidates)
    return status, reasons_for(status=status, reasons=list(reasons))
