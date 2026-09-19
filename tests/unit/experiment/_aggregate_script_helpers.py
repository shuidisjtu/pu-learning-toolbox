# tests/unit/experiment/_aggregate_script_helpers.py

# ruff: noqa: N803, S101

"""Shared builders for the ``scripts/aggregate_survey_runs.py`` tests.

Not collected by pytest (``python_files = ["test_*.py"]``).  The aggregation
entry point reads manifests as plain dicts, so these build them directly --
no runner, no training.  That keeps the tests about grouping, gating and
reporting rather than about producing artifacts.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import load_protocol

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/aggregate_survey_runs.py"

_HEX = "0123456789abcdef"

#: Budget family -> the ``comparability_group`` suffix the locked protocol uses.
#:
#: Four classical families share one group named after the method class, not the
#: family: a closed-form solve and an EM fit are different budgets the protocol
#: lists side by side.  The remaining families are named for themselves.  A
#: fixture that derives the group suffix from the family name would keep those
#: four apart, which is the one shape the grouping is there to handle.
_GROUP_OF_FAMILY = {
    "closed_form": "classical",
    "alternating": "classical",
    "kernel_cv": "classical",
    "em": "classical",
    "minibatch": "minibatch",
    "fullbatch": "fullbatch",
    "two_student_sampled": "two_student_sampled",
}


def _digest(marker: str) -> str:
    return (marker * 64)[:64]


def _load_script_module():
    spec = importlib.util.spec_from_file_location("aggregate_survey_runs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolves field annotations through
    # sys.modules[cls.__module__], which the spec loader alone does not populate.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aggregate_script():
    return _load_script_module()


def manifest(
    *,
    method="upu",
    dataset="spambase",
    training_path="native_2d",
    budget="minibatch",
    group=None,
    seed=0,
    c=0.5,
    epochs=200,
    batch_size=64,
    split_marker="a",
    representation_marker="b",
    candidates=1,
    formal_eligible=True,
    blockers=(),
    c_independent=False,
    protocol_version="survey-v1.2",
    runnable=True,
):
    """A versioned-pilot manifest with only the fields aggregation reads.

    ``epochs`` and ``batch_size`` are omitted when passed as ``None``, which is
    what the four budget families without either actually look like in a real
    manifest.  ``batch_size`` may also be a description rather than a number:
    the fullbatch family records that the estimator's batch size is unused.

    ``group`` defaults to the group the locked protocol gives this budget
    family, which is not always named after the family -- see
    ``_GROUP_OF_FAMILY``.
    """
    representation = {
        "split_sha256": _digest(split_marker),
        "feature_sha256": {"train": _digest(representation_marker)},
    }
    budget_payload = {"unit": budget, "actual_outer_candidates": candidates}
    if epochs is not None:
        budget_payload["epochs"] = epochs
    if batch_size is not None:
        budget_payload["batch_size"] = batch_size
    unit = {
        "method": method,
        "dataset": dataset,
        "training_path": training_path,
        "budget": budget,
        "comparability_group": group
        or f"{dataset}/{training_path}/{_GROUP_OF_FAMILY.get(budget, budget)}",
        "runnable": runnable,
    }
    generation = {
        role: {
            "mechanism": "pn_oracle" if c_independent else "scar",
            "c_requested": c,
            "label_view_sha256": _digest("c"),
        }
        for role in ("train", "pu_val")
    }
    return {
        "execution_mode": "versioned_pilot",
        "protocol_version": protocol_version,
        "protocol_sha256": _digest("d"),
        "execution_unit": unit,
        "training_path": training_path,
        "adaptation_level": "benchmark-adapted",
        "representation": representation,
        "budget": budget_payload,
        "candidate_runs": [{"candidate_index": index} for index in range(candidates)],
        # A run that got as far as selecting something records it here; the one
        # failure mode that still writes a manifest leaves this empty.
        "selection": {"OA": {"candidate_index": 0}},
        "generation": generation,
        "seed": seed,
        "formal_eligible": formal_eligible,
        "formal_blockers": list(blockers),
        **({"c_independent": True, "broadcast_c_values": [0.1, 0.5]} if c_independent else {}),
    }


def aggregate_tree(script, root, *, require_formal=True):
    """Run the entry point over a written tree against the locked protocol."""
    return script.aggregate(
        script.discover_manifests(root),
        protocol=load_protocol(),
        require_formal=require_formal,
    )


def write_tree(root, manifests):
    """Write each manifest to ``<root>/<name>/manifest.json`` and return the root."""
    for name, payload in manifests.items():
        path = root / name / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")
    return root
