# tests/unit/experiment/_survey_summary_helpers.py

# ruff: noqa: N803, S101

"""Builders for the P2.2 audit / summary entry-point tests.

Not collected by pytest (``python_files = ["test_*.py"]``).  Both entry points
read manifests as plain dicts and call the aggregation gate over them, so these
build a manifest the gate *accepts* and the preflight *passes* -- which is more
than the older ``_aggregate_script_helpers`` needs.  The extra fields are exactly
the ones P2.2 added a reader for: the environment identity protocol §5 clause 6
requires, the split reference, the c token, and the test results a row is built
from.

``budget`` is assembled in the shape ``budget_fairness_fields`` reads, so a
fixture here differs from a real manifest only in its values.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

from pu_toolbox.experiment.survey_protocol import digest, load_protocol

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"

_HEX = "0123456789abcdef"

#: The digest a manifest must report for the protocol check to pass.
FROZEN_PROTOCOL_SHA256 = digest(load_protocol())


def _digest(marker: str) -> str:
    return (marker * 64)[:64]


def manifest(
    *,
    method="nnpu",
    dataset="spambase",
    training_path="native_2d",
    budget="minibatch",
    group=None,
    seed=0,
    c=0.1,
    epochs=200,
    batch_size=64,
    split_marker="a",
    feature_marker="b",
    candidates=1,
    run_view=None,
    mechanism=None,
    c_independent=False,
    formal_eligible=True,
    blockers=(),
    accuracy=0.80,
    auc=0.85,
    environment=True,
):
    """One versioned-pilot manifest the gate accepts and the preflight passes.

    ``run_view`` and ``calibration_applied`` are derived together: the production
    view validator refuses a manifest where they disagree, and the oracle is
    refused a calibrated view outright.  ``environment=False`` produces the shape
    protocol §5 clause 6 calls unreproducible, for the tests that need it.
    """
    if run_view is None:
        run_view = "os-compatible" if c_independent else "ts-compatible"
    if mechanism is None:
        mechanism = "pn_oracle" if c_independent else "scar"

    budget_payload = {"unit": budget, "actual_outer_candidates": candidates}
    if epochs is not None:
        budget_payload["epochs"] = epochs
    if batch_size is not None:
        budget_payload["batch_size"] = batch_size

    label_view = {"mechanism": mechanism, "c_requested": c, "label_view_sha256": _digest("c")}
    payload = {
        "execution_mode": "versioned_pilot",
        "protocol_version": "survey-v1.2",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "execution_unit": {
            "method": method,
            "dataset": dataset,
            "training_path": training_path,
            "budget": budget,
            "comparability_group": group or f"{dataset}/{training_path}/{budget}",
            "runnable": True,
        },
        "training_path": training_path,
        "adaptation_level": "benchmark-adapted",
        "run_view": run_view,
        "calibration_applied": run_view == "ts-compatible",
        "representation": {
            "name": "tabular_mlp",
            "split_sha256": _digest(split_marker),
            "feature_sha256": {"train": _digest(feature_marker)},
        },
        "budget": budget_payload,
        "candidate_runs": [{"candidate_index": index} for index in range(candidates)],
        "selection": {"OA": {"candidate_index": 0}},
        "generation": {"train": label_view, "pu_val": dict(label_view)},
        "seed": seed,
        "split_ref": {"indices_sha256": _digest("i")},
        "test_results": {
            protocol: {"accuracy": accuracy, "auc": auc, "auc_unavailable_reason": None}
            for protocol in ("PA", "OA")
        },
        "formal_eligible": formal_eligible,
        "formal_blockers": list(blockers),
        "resources": {
            "peak_gpu_memory_bytes": 8_687_000_000,
            "single_configuration_costs": [{"elapsed_seconds": 100.0}],
            "tuning": {"elapsed_seconds": 200.0},
        },
    }
    if environment:
        payload["resources"]["environment"] = {
            "python_version": "3.10.8",
            "torch_version": "2.13.0+cu130",
            "gpu_devices": [{"index": 0, "name": "NVIDIA vGPU-32GB"}],
        }
    if c_independent:
        payload["c_independent"] = True
        payload["broadcast_c_values"] = [c]
    else:
        payload["c_requested_token"] = f"{c:g}"
    return payload


def write_tree(root: Path, manifests: dict[str, dict]) -> Path:
    """Write each manifest to ``<root>/<name>/manifest.json``."""
    for name, payload in manifests.items():
        path = root / name / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return root


def config_for(batches: dict[str, tuple[Path, int]], **overrides) -> dict:
    """A batch-root config over ``{name: (root, expected_count)}``."""
    payload = {
        "schema_version": "survey-batch-roots-1",
        "batches": [
            {"name": name, "root": str(root), "expected_manifests": expected}
            for name, (root, expected) in batches.items()
        ],
    }
    payload.update(overrides)
    return payload


def write_config(path: Path, config: dict) -> Path:
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _load(name: str):
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def audit_cli():
    return _load("audit_survey_batches")


@pytest.fixture(scope="module")
def summary_cli():
    return _load("summarize_survey_results")
