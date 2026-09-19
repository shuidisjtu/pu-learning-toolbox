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

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts/aggregate_survey_runs.py"

_HEX = "0123456789abcdef"


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
):
    """A versioned-pilot manifest with only the fields aggregation reads."""
    representation = {
        "split_sha256": _digest(split_marker),
        "feature_sha256": {"train": _digest(representation_marker)},
    }
    unit = {
        "method": method,
        "dataset": dataset,
        "training_path": training_path,
        "budget": "minibatch",
        "comparability_group": f"{dataset}/{training_path}/minibatch",
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
        "protocol_version": "survey-v1.2",
        "protocol_sha256": _digest("d"),
        "execution_unit": unit,
        "training_path": training_path,
        "adaptation_level": "benchmark-adapted",
        "representation": representation,
        "budget": {"epochs": epochs, "batch_size": batch_size},
        "candidate_runs": [{"candidate_index": index} for index in range(candidates)],
        "generation": generation,
        "seed": seed,
        "formal_eligible": formal_eligible,
        "formal_blockers": list(blockers),
        **({"c_independent": True, "broadcast_c_values": [0.1, 0.5]} if c_independent else {}),
    }


def write_tree(root, manifests):
    """Write each manifest to ``<root>/<name>/manifest.json`` and return the root."""
    for name, payload in manifests.items():
        path = root / name / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")
    return root
