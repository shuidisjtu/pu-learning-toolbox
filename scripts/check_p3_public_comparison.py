#!/usr/bin/env python3
"""Check P1-2 source inventory, units and boundaries without issuing verdicts.

This read-only draft checker never consumes experiment results, trains models,
downloads papers, signs reviews or changes the frozen comparison matrix.
Optional paper/source directories validate recorded bytes, not human readings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from scripts.check_p3_admission_evidence import (
    EVIDENCE_REF,
    METHODS,
    ROOT,
    repo_file,
    require,
    validate_source_checkout,
)

DRAFT_REF = "docs/research/pu_survey/data/p3_public_comparison_draft_20261005.json"
BOUND_FILES = (
    "uv.lock",
    "pu_toolbox/experiment/survey_protocol_v1.json",
    "pu_toolbox/experiment/survey_comparison_v1.json",
    "pu_toolbox/experiment/survey_comparison_v2.json",
    "pu_toolbox/experiment/survey_comparison_v3.json",
    EVIDENCE_REF,
)
BOUND_FILES_DIGEST_POLICY = "parsed_json_toml_strict_canonical_json_sha256"
CONFIG_PARSERS = {".json": json.loads, ".lock": tomllib.loads}
DATASETS = ("spambase", "imdb", "cifar10")
SOURCE_TYPES = {
    "puet": ("accuracy", "percentage_points", "std", 5),
    "gradpu": ("accuracy", "percentage_points", "unspecified_deviation", 3),
    "robust_pu": ("error_rate", "fraction", "std", 10),
    "split_pu": ("accuracy", "percentage_points", "std", 5),
}


def finite_number(value, name):
    """Refuse booleans, strings and non-finite numeric facts."""
    require(type(value) in (int, float) and math.isfinite(value), f"bad {name}")
    return value


def normalize_reading(row):
    """Convert only metric direction/units, never infer SD/SEM or significance."""
    require(row["metric"] in {"accuracy", "error_rate"}, "bad metric")
    require(row["mean_unit"] == "percentage_points", "bad mean unit")
    mean = finite_number(row["mean"], "mean")
    require(0 <= mean <= 100, "mean out of range")
    require(isinstance(row["spread_token"], str), "spread token must preserve source text")
    spread = float(row["spread_token"])
    require(math.isfinite(spread) and spread >= 0, "bad spread")
    require(row["spread_unit"] in {"fraction", "percentage_points"}, "bad spread unit")
    if row["spread_unit"] == "fraction":
        spread *= 100
    require(spread <= 100, "spread out of range")
    return {
        "reading_id": row["reading_id"],
        "accuracy_mean_pp": round(100 - mean if row["metric"] == "error_rate" else mean, 12),
        "spread_pp": round(spread, 12),
        "uncertainty_kind": row["uncertainty_kind"],
        "numeric_eligible": False,
    }


def config_digest(path):
    """Digest parsed configuration: types and array order, never formatting.

    Newlines, indentation, object key order and comments are representation, not
    configuration, so hashing recorded bytes would report them as drift.  Only
    formats named in ``CONFIG_PARSERS`` are parsed: an unknown suffix is refused
    rather than guessed as TOML, and there is no raw-byte fallback that would
    hide a binding still written under the old byte policy.
    """
    from pu_toolbox.utils.serialization import strict_canonical_hash

    parse = CONFIG_PARSERS.get(path.suffix)
    require(parse is not None, f"unsupported configuration format: {path.name}")
    return strict_canonical_hash(parse(path.read_text(encoding="utf-8")))


def validate_draft(draft, ledger, protocol, *, root=ROOT):
    """Validate coverage/reference consistency; this cannot verify paper truths."""
    json.dumps(draft, allow_nan=False)
    require(draft["schema_version"] == "survey-p3-public-comparison-draft-1", "bad schema")
    require(draft["review_status"] == "pending_method_owner_review", "review must remain pending")
    require(draft["owner_signatures"] == [], "signatures must remain empty")
    require(
        draft["reference_scope"] == "historical_pilot_context_not_new_method_recipe",
        "recipe overclaim",
    )
    policy = draft["numeric_policy"]
    require(
        set(policy) == {"enabled", "tolerance_pp", "policy_ref"}
        and policy["enabled"] is False
        and policy["tolerance_pp"] is None
        and policy["policy_ref"] is None,
        "numeric policy overclaim",
    )
    require(
        draft["new_method_results"] == [] and draft["numeric_verdicts"] == [], "invented results"
    )
    require(draft["admission_evidence_ref"] == EVIDENCE_REF, "evidence ref drift")
    digest_policy = draft.get("bound_files_digest_policy")
    require(digest_policy is not None, "missing bound file digest policy")
    require(
        digest_policy == BOUND_FILES_DIGEST_POLICY,
        f"unknown bound file digest policy: {digest_policy!r}",
    )
    require(set(draft["bound_files_sha256"]) == set(BOUND_FILES), "missing frozen bindings")
    for reference, expected in draft["bound_files_sha256"].items():
        require(
            config_digest(repo_file(reference, root)) == expected,
            "bound file drift",
        )
    profiles = draft["source_profiles"]
    require(set(profiles) == set(METHODS), "missing source profiles")
    for method, profile in profiles.items():
        require(method not in protocol["method_profiles"], "old pilot admission")
        require(ledger["methods"][method]["public_comparison_ref"] == DRAFT_REF, "ledger ref drift")
        require(profile["paper_url"].startswith("https://"), "bad paper URL")
        require(
            profile["protocol_facts"] and profile["evidence_locations"] and profile["unresolved"],
            "missing source boundaries",
        )
        if method == "pulda":
            require(profile["paper_status"] == "full_text_unavailable", "unverified PULDA paper")
            require(
                profile["paper_file"] is None and profile["paper_sha256"] is None,
                "invented PULDA bytes",
            )
        else:
            require(profile["paper_status"] == "table_visually_checked", "missing visual review")
            require(profile["paper_file"] == method + ".pdf", "bad paper file")
            require(
                isinstance(profile["paper_sha256"], str)
                and len(profile["paper_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in profile["paper_sha256"]),
                "bad paper digest",
            )
            require(
                tuple(
                    profile[k] for k in ("metric", "spread_unit", "uncertainty_kind", "n_repeats")
                )
                == SOURCE_TYPES[method],
                "source unit/type drift",
            )
    ids, identities, normalized = set(), set(), []
    for row in draft["readings"]:
        require(row["reading_id"] not in ids, "duplicate reading id")
        ids.add(row["reading_id"])
        require(
            row["method"] in SOURCE_TYPES and row["dataset"] in DATASETS,
            "bad reading method/dataset",
        )
        profile = profiles[row["method"]]
        require(
            row["method_variant"] == profile["method_variant"]
            and row["paper_location"] == profile["paper_location"],
            "wrong table column/variant",
        )
        for key in ("metric", "spread_unit", "uncertainty_kind", "n_repeats"):
            require(
                type(row[key]) is type(profile[key]) and row[key] == profile[key],
                "reading source metadata drift",
            )
        require(row["value_producer"] == "original_authors", "wrong value producer")
        require(
            row["review_state"] == "pending_review" and row["numeric_eligible"] is False,
            "reading acceptance overclaim",
        )
        axis = row["axis"]
        expected_axis = (
            "unlabeled_class_prior" if row["method"] == "robust_pu" else "n_labeled_positive"
        )
        require(axis["name"] == expected_axis, "pi/n_P is not c")
        value = finite_number(axis["value"], "axis value")
        require(
            0 < value < 1
            if expected_axis == "unlabeled_class_prior"
            else type(value) is int and value > 0,
            "bad axis value",
        )
        identity = (row["method"], row["dataset"], axis["name"], value)
        require(identity not in identities, "duplicate reading cell")
        identities.add(identity)
        normalized.append(normalize_reading(row))
    expected_identities = {
        (method, "cifar10", "n_labeled_positive", count)
        for method, counts in {
            "puet": (1000,),
            "gradpu": (1000, 3000),
            "split_pu": (500, 1000, 3000),
        }.items()
        for count in counts
    } | {
        ("robust_pu", dataset, "unlabeled_class_prior", prior)
        for dataset in ("cifar10", "spambase")
        for prior in (0.2, 0.4, 0.6)
    }
    require(identities == expected_identities, "reading inventory drift")
    cells = set()
    counts = Counter()
    for cell in draft["coverage"]:
        key = (cell["method"], cell["dataset"])
        require(key not in cells, "duplicate coverage")
        cells.add(key)
        require(key[0] in METHODS and key[1] in DATASETS, "ghost coverage")
        refs = [r["reading_id"] for r in draft["readings"] if (r["method"], r["dataset"]) == key]
        require(cell["reading_ids"] == refs, "missing/ghost reading refs")
        disposition = (
            "source_unavailable"
            if profiles[key[0]]["paper_status"] == "full_text_unavailable"
            else "protocol_mismatch"
            if refs
            else "no_observed_reading"
        )
        require(
            cell["disposition"] == disposition and cell["numeric_eligible"] is False,
            "coverage overclaim",
        )
        require(
            "new_method_not_in_frozen_pilot" in cell["blockers"]
            and "new_result_artifacts_and_owner_review_missing" in cell["blockers"],
            "missing blockers",
        )
        counts[disposition] += 1
    require(cells == {(m, d) for m in METHODS for d in DATASETS}, "incomplete coverage")
    require(
        {c["id"] for c in draft["contradictions"]}
        == {"split_teacher_epochs_paper_vs_code", "robust_mean_spread_mixed_units"},
        "lost discrepancies",
    )
    return {
        "ok": True,
        "readings_checked": len(ids),
        "coverage_cells": len(cells),
        "dispositions": dict(counts),
        "normalized_readings": normalized,
        "numeric_verdicts": [],
        "formal_admission": False,
    }


def check_papers(profiles, directory):
    """Read local PDF bytes only; do not parse PDFs or execute embedded content."""
    count = 0
    for profile in profiles.values():
        if profile["paper_file"] is not None:
            data = repo_file(profile["paper_file"], directory).read_bytes()
            require(
                data.startswith(b"%PDF-")
                and hashlib.sha256(data).hexdigest() == profile["paper_sha256"],
                "paper byte mismatch",
            )
            count += 1
    return count


def main(argv=None):
    """Emit a draft-only receipt, optionally checking local paper and source bytes."""
    from pu_toolbox.experiment.method_ledger import load_ledger
    from pu_toolbox.experiment.survey_protocol import load_protocol

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=ROOT / DRAFT_REF)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--source-dir", action="append", default=[], metavar="METHOD=PATH")
    args = parser.parse_args(argv)
    try:
        draft = json.loads(args.draft.read_text(encoding="utf-8"))
        receipt = validate_draft(draft, load_ledger(), load_protocol())
        receipt["paper_files_checked"] = (
            check_papers(draft["source_profiles"], args.paper_dir) if args.paper_dir else 0
        )
        evidence = json.loads((ROOT / EVIDENCE_REF).read_text(encoding="utf-8"))
        checked = {}
        for item in args.source_dir:
            method, separator, directory = item.partition("=")
            require(
                separator and directory and method in METHODS and method not in checked,
                "bad source-dir",
            )
            source = dict(evidence["methods"][method]["source"])
            source["files"] = [
                *source["files"],
                *draft["source_profiles"][method]["additional_source_files"],
            ]
            checked[method] = validate_source_checkout(source, Path(directory))
        receipt["source_files_checked"] = checked
        print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
