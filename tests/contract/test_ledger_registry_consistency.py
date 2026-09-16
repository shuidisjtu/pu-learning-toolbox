"""Ledger <-> registry consistency contract tests (issue #42).

The experiment-layer method ledger
(``pu_toolbox/experiment/method_ledger.json``) declares itself the
"programmatic truth source" for survey routing, while ``pu_toolbox/registry/``
is the code-side truth source for algorithm metadata.  These tests pin the two
together so ledger edits cannot drift from the registry -- in particular across
the ``pusb`` / ``pusb_kernel`` split, where one method card covers a linear
baseline (no class prior) and an official-aligned kernel entry (class prior
required), and the registry is the gate's truth source.

One test per invariant:

1. every ledger key is a registered algorithm and the key set is exactly the
   expected eight entries (``pusb`` and ``pusb_kernel`` stay separate);
2. ``modality_backbone.code_capability`` mirrors the registry capability
   fields (``native_architectures`` / ``input_ndims`` / ``encoder_parameter`` /
   ``trains_encoder``);
3. the head of ``code_version.source_status`` (everything before its
   parenthetical note) equals the registry ``source_status`` enum value;
4. ``code_version.upstream_url`` is a bare HTTP URL -- no parenthetical note
   mixed into it;
5. ``calibration_applied`` is a JSON boolean, not a string carrying prose;
6. ``prior_semantics`` mentions "population" exactly when the registry entry
   requires a class prior (``requires_class_prior``).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from pu_toolbox.registry import get_metadata, register_all_builtin_methods

_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "pu_toolbox" / "experiment" / "method_ledger.json"
)

_EXPECTED_METHOD_KEYS = {
    "upu",
    "kldce",
    "dist_pu",
    "pusb",
    "pusb_kernel",
    "lbe",
    "nnpu",
    "self_pu",
}

# Ledger notes open their annotation with either an ASCII or a full-width
# parenthesis; the head is everything before it.
_NOTE_OPEN = re.compile(r"[（(]")


@pytest.fixture(scope="module")
def ledger_methods() -> dict[str, dict]:
    """The ledger ``methods`` mapping, with the builtin registry populated."""
    register_all_builtin_methods()  # idempotent; safe alongside other modules
    payload = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
    return payload["methods"]


def _as_set(value: object) -> set:
    """Normalize a ledger JSON list (or ``null``) and a registry frozenset."""
    return set(value) if value else set()


@pytest.mark.contract
def test_method_keys_are_the_eight_registered_algorithms(ledger_methods: dict[str, dict]) -> None:
    """Keys resolve through the registry, as the exact post-split key set."""
    for name in ledger_methods:
        assert get_metadata(name).name == name, f"{name!r} is not a registered algorithm"
    assert set(ledger_methods) == _EXPECTED_METHOD_KEYS


@pytest.mark.contract
def test_code_capability_mirrors_registry_metadata(ledger_methods: dict[str, dict]) -> None:
    """The declared 4-field capability block equals the registry values."""
    for name, entry in ledger_methods.items():
        declared = entry["modality_backbone"]["code_capability"]
        meta = get_metadata(name)
        assert _as_set(declared["native_architectures"]) == _as_set(meta.native_architectures), name
        assert _as_set(declared["input_ndims"]) == _as_set(meta.input_ndims), name
        assert declared["encoder_parameter"] == meta.encoder_parameter, name
        assert declared["trains_encoder"] == meta.trains_encoder, name


@pytest.mark.contract
def test_source_status_note_head_matches_registry_enum(ledger_methods: dict[str, dict]) -> None:
    """The ledger's source_status note head equals the registry enum value."""
    for name, entry in ledger_methods.items():
        note = entry["code_version"]["source_status"]
        head = _NOTE_OPEN.split(note, maxsplit=1)[0].strip().lower()
        expected = get_metadata(name).source_status.value.lower()
        assert head == expected, f"{name}: ledger note {note!r} vs registry {expected!r}"


@pytest.mark.contract
def test_upstream_url_is_a_bare_http_url(ledger_methods: dict[str, dict]) -> None:
    """``upstream_url`` carries a real URL only -- no parenthetical prose."""
    for name, entry in ledger_methods.items():
        url = entry["code_version"]["upstream_url"]
        assert url.startswith("http"), f"{name}: {url!r}"
        assert not _NOTE_OPEN.search(url), f"{name}: URL carries a note: {url!r}"


@pytest.mark.contract
def test_calibration_applied_is_a_json_boolean(ledger_methods: dict[str, dict]) -> None:
    """``calibration_applied`` is a JSON boolean, not a string with prose."""
    for name, entry in ledger_methods.items():
        value = entry["calibration_applied"]
        assert isinstance(value, bool), f"{name}: {type(value).__name__} {value!r}"


@pytest.mark.contract
def test_prior_semantics_consistent_with_registry_class_prior(
    ledger_methods: dict[str, dict],
) -> None:
    """``population`` in ``prior_semantics`` iff the registry requires a prior."""
    for name, entry in ledger_methods.items():
        requires = get_metadata(name).requires_class_prior
        mentions = "population" in entry["prior_semantics"]
        assert mentions == requires, (
            f"{name}: prior_semantics={entry['prior_semantics']!r} "
            f"but registry requires_class_prior={requires}"
        )


@pytest.mark.contract
def test_heng958_owner_reviews_are_complete_and_traceable(
    ledger_methods: dict[str, dict],
) -> None:
    """The two HENG958-owned rows carry review status and local evidence."""
    repository_root = _LEDGER_PATH.parents[2]
    for name in ("nnpu", "self_pu"):
        review = ledger_methods[name]["owner_review"]
        assert review["status"] == "completed", name
        date.fromisoformat(review["reviewed_on"])
        evidence_path, separator, anchor = review["evidence"].partition("#")
        assert separator and anchor, name
        assert (repository_root / evidence_path).is_file(), name
        assert review["conclusion"].strip(), name
