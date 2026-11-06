"""Windows long-path I/O does not change the public identity of result inputs."""

import json
import os
from pathlib import Path

import pytest
from _survey_summary_helpers import manifest

from pu_toolbox.experiment import batch_evidence
from pu_toolbox.experiment.results import _filesystem_path, _report_path, build_summary

pytestmark = [pytest.mark.unit, pytest.mark.skipif(os.name != "nt", reason="Windows path API")]


def _extended(path):
    return Path("\\\\?\\" + str(path))


def _long_tree(root, location):
    run = root / "run"
    deep = run / "checkpoints" / ("a" * 80) / ("b" * 80) / ("c" * 80)
    _extended(deep).mkdir(parents=True)
    path = deep / "manifest.json" if location == "manifest" else run / "manifest.json"
    _extended(path).write_text(json.dumps(manifest()), encoding="utf-8")
    assert len(str(deep)) > 260
    return path


@pytest.mark.parametrize("location", ["checkpoint", "manifest"])
def test_param_basic_determ_long_discovery_and_reads_keep_normal_public_paths(tmp_path, location):
    path = _long_tree(tmp_path, location)
    before = _extended(path).read_bytes()

    report = build_summary([tmp_path, _extended(tmp_path), _extended(path)], expected_seeds=[0])

    assert report["coverage"]["available"] == 1
    assert report["sources"][0]["path"] == str(path)
    assert all(row["members"] == [str(path)] for row in report["rows"])
    assert _extended(path).read_bytes() == before
    assert build_summary([path], expected_seeds=[0]) == report


def test_edge_long_pending_and_excluded_inputs_use_the_same_identity(tmp_path):
    path = _long_tree(tmp_path, "manifest")

    report = build_summary([tmp_path], expected_seeds=[0], pending_paths=[_extended(path)])
    excluded = build_summary([path], expected_seeds=[0], excluded_paths=[_extended(path)])

    assert report["sources"] == [{"path": str(path), "state": "pending"}]
    assert excluded["sources"] == [{"path": str(path), "state": "excluded"}]
    assert report["rows"] == excluded["rows"] == []


@pytest.mark.parametrize(
    ("normal", "extended"),
    [
        (r"E:\results\run", r"\\?\E:\results\run"),
        (r"\\server\share\results", r"\\?\UNC\server\share\results"),
    ],
)
def test_param_extended_drive_and_unc_paths_have_one_determ_public_identity(normal, extended):
    assert _filesystem_path(Path(normal)) == Path(extended)
    assert _filesystem_path(Path(extended)) == Path(extended)
    assert _report_path(Path(extended)) == Path(normal)
    assert _report_path(Path(normal)) == Path(normal)


def test_edge_shared_evidence_remains_readable_under_a_long_result_root(tmp_path):
    path = _long_tree(tmp_path, "manifest")
    payload = json.loads(_extended(path).read_text(encoding="utf-8"))
    packet = batch_evidence.build_evidence(
        unit={
            "dataset": "spambase",
            "method": "nnpu",
            "training_path": "native_2d",
            "mechanism": "scar",
        },
        seeds=(0,),
        c_tokens=("0.1",),
        protocol_version=payload["protocol_version"],
        protocol_sha256=payload["protocol_sha256"],
        code={"code_commit": "a" * 40, "code_commit_dirty": False, "code_commit_state": "recorded"},
        lock={"path": "uv.lock", "sha256": "b" * 64, "state": "recorded"},
        environment={"python_version": "3.11"},
    )
    packet_path = path.parent / "batch_evidence" / batch_evidence.evidence_name(packet)
    _extended(packet_path.parent).mkdir()
    _extended(packet_path).write_text(
        batch_evidence.serialise(packet), encoding="utf-8", newline="\n"
    )
    payload["batch"] = batch_evidence.manifest_block(packet, _extended(packet_path))
    _extended(path).write_text(json.dumps(payload), encoding="utf-8")
    before = _extended(packet_path).read_bytes()

    report = build_summary([tmp_path], expected_seeds=[0])

    identity = report["sources"][0]["reproducibility"]
    assert identity["state"] == "recorded"
    assert identity["evidence_path"] == str(packet_path)
    assert _extended(packet_path).read_bytes() == before
