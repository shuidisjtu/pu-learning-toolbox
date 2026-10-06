"""Canonical references to repository-local Survey evidence artifacts.

Generated Survey drafts and provenance payloads must use this module instead of
repeating document paths across scripts. Moving an evidence document therefore
requires one mapping update followed by regeneration of committed drafts.
"""

from __future__ import annotations

from pathlib import Path

SURVEY_ARTIFACT_REFS: dict[str, str] = {
    "candidate_cnn_storage_review_20261005": (
        "docs/research/pu_survey/admission/candidate_cnn_storage_review_20261005.md"
    ),
    "candidate_storage_probe_20261005": (
        "docs/research/pu_survey/data/candidate_storage_probe_20261005.json"
    ),
    "independent_progress_20261005": (
        "docs/research/pu_survey/reviews/independent_progress_20261005.md"
    ),
    "pn_oracle_integration": ("docs/research/pu_survey/admission/pn_oracle_integration.md"),
}


def survey_artifact_ref(name: str) -> str:
    """Return the canonical repository-relative path for a Survey artifact."""
    try:
        return SURVEY_ARTIFACT_REFS[name]
    except KeyError as error:
        raise KeyError(f"unknown Survey artifact reference: {name}") from error


def survey_artifact_path(name: str, *, repo_root: Path | None = None) -> Path:
    """Resolve a canonical Survey artifact path for local validation."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / survey_artifact_ref(name)
