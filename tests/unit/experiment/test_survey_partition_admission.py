"""Admission of complete grids assembled from independently evidenced batches."""

import copy
import json

import pytest
from test_survey_audit_checks import _admission_entries, checks

pytestmark = pytest.mark.unit


def _partitioned_admission_entries(tmp_path):
    from pu_toolbox.experiment import batch_evidence

    protocol, entries, passed = _admission_entries(tmp_path)
    unit = {
        "dataset": "spambase",
        "method": "lagam",
        "training_path": "native_2d",
        "mechanism": "scar",
    }
    environment = {
        "state": "recorded",
        "python_version": "3.11",
        "python_implementation": "CPython",
        "platform": "test",
        "torch_version": "test",
        "cuda_available": False,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "gpu_devices": [],
        "gpu_driver_version": None,
        "numpy_version": "1",
        "scikit_learn_version": "1",
        "hardware": {"machine": "x86_64", "processor": "test", "cpu_count": 4},
        "thread_environment": {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        },
    }
    for seed in protocol["seeds"]:
        token_groups = (
            [[token] for token in protocol["c_tokens"]["scar"]]
            if seed == 0
            else [protocol["c_tokens"]["scar"]]
        )
        for tokens in token_groups:
            packet = batch_evidence.build_evidence(
                unit=unit,
                seeds=[seed],
                c_tokens=tokens,
                protocol_version=protocol["protocol_version"],
                protocol_sha256=checks.digest(protocol),
                code={
                    "code_commit": "a" * 40,
                    "code_commit_state": "recorded",
                    "code_commit_dirty": False,
                },
                lock={"path": "uv.lock", "sha256": "b" * 64, "state": "recorded"},
                environment=environment,
            )
            path = batch_evidence.write(tmp_path / "batch_evidence", packet)
            for entry in entries:
                payload = entry["payload"]
                if payload["seed"] == seed and checks.normalize_c_token(payload) in tokens:
                    payload["batch"] = batch_evidence.manifest_block(packet, path)
                    payload["protocol_deviation"] = ["seed_subset"]
                    if seed == 0:
                        payload["protocol_deviation"].append("c_grid")
                    payload["formal_blockers"].append("protocol_deviation")
                    entry["path"].write_text(json.dumps(payload), encoding="utf-8")
    return protocol, entries, passed, {"batches": [{"name": "B1", "root": str(tmp_path)}]}


def test_basic_partitioned_grid_can_be_readjudicated_without_mutating_manifests(tmp_path):
    protocol, entries, passed, config = _partitioned_admission_entries(tmp_path)
    before = {str(e["path"]): e["path"].read_bytes() for e in entries}
    report = checks.reassess_admission(entries, protocol=protocol, checks=passed, config=config)
    assert all(item["eligible"] for item in report["runs"].values())
    assert all(
        "protocol_deviation" in item["discharged_blockers"] for item in report["runs"].values()
    )
    assert before == {str(e["path"]): e["path"].read_bytes() for e in entries}


@pytest.mark.parametrize(
    "problem",
    [
        "incomplete",
        "duplicate",
        "wrong_env",
        "missing_packet",
        "other_deviation",
        "missing_hardware",
        "tampered_packet",
        "wrong_marker",
        "failed_audit",
    ],
)
def test_param_edge_partitioned_grid_does_not_discharge_without_evidence(tmp_path, problem):
    protocol, entries, passed, config = _partitioned_admission_entries(tmp_path)
    if problem == "incomplete":
        entries.pop()
    elif problem == "duplicate":
        entries.append(copy.deepcopy(entries[0]))
    elif problem in {"wrong_env", "missing_hardware"}:
        from pu_toolbox.experiment import batch_evidence

        block = entries[0]["payload"]["batch"]
        path = tmp_path / "batch_evidence" / block["evidence_file"]
        packet = batch_evidence.load(path)
        if problem == "wrong_env":
            packet["environment"]["hardware"]["processor"] = "different"
        else:
            del packet["environment"]["hardware"]
        packet["batch_id"] = batch_evidence.batch_identity(packet)
        replacement = batch_evidence.write(tmp_path / "batch_evidence", packet)
        for entry in entries:
            if entry["payload"]["batch"] == block:
                entry["payload"]["batch"] = batch_evidence.manifest_block(packet, replacement)
                entry["path"].write_text(json.dumps(entry["payload"]), encoding="utf-8")
    elif problem == "missing_packet":
        entries[0]["payload"].pop("batch")
    elif problem == "tampered_packet":
        block = entries[0]["payload"]["batch"]
        path = tmp_path / "batch_evidence" / block["evidence_file"]
        path.write_text("{}", encoding="utf-8")
    elif problem == "wrong_marker":
        entries[0]["payload"]["protocol_deviation"].remove("c_grid")
    elif problem == "failed_audit":
        passed[0]["result"] = "fail"
    else:
        entries[0]["payload"]["protocol_deviation"].append("constructor_override")
    report = checks.reassess_admission(entries, protocol=protocol, checks=passed, config=config)
    assert all(not item["eligible"] for item in report["runs"].values())


def test_determ_partitioned_reassessment_is_stable(tmp_path):
    protocol, entries, passed, config = _partitioned_admission_entries(tmp_path)
    first = checks.reassess_admission(entries, protocol=protocol, checks=passed, config=config)
    assert (
        checks.reassess_admission(entries, protocol=protocol, checks=passed, config=config) == first
    )


def test_basic_partition_rejection_is_visible_in_summary():
    import importlib

    from _survey_summary_helpers import manifest

    summary = importlib.import_module("summarize_survey_results")
    entries = []
    for seed in range(5):
        payload = manifest(seed=seed)
        entries.append(
            {
                "path": f"seed_{seed}",
                "payload": payload,
                "role": "formal",
                "admission_reassessment": {
                    "eligible": False,
                    "partition_grid_checked": True,
                    "remaining_blockers": [],
                },
            }
        )
    observations, _, _ = summary.collect_observations(entries, metrics=summary.CARRIED_METRICS)
    rows = summary.build_rows(
        observations, expected_seeds=list(range(5)), gate_reasons={}, unrecorded_by_run={}
    )
    assert rows
    assert all(row["status"] == "partial" for row in rows)
    assert all("formal_blockers_present" in row["reasons"] for row in rows)
