# tests/unit/experiment/test_batch_evidence.py

"""The batch evidence packet: what a run cites so that its result is reproducible.

Protocol §5 clause 6 wants a code commit and a dependency lock behind every
result; D44 moved them from each manifest to one packet per batch.  Three claims
make that substitution safe, and these tests pin each of them: the packet
carries the two fields ``survey_audit.BATCH_IDENTITY_FIELDS`` names, its identity
follows its content rather than its location, and a run cannot check a packet
that describes a different batch.
"""

import json
from pathlib import Path

import pytest

from pu_toolbox.experiment import batch_evidence as be
from pu_toolbox.experiment.survey_audit import BATCH_IDENTITY_FIELDS
from pu_toolbox.utils.serialization import file_hash

pytestmark = pytest.mark.unit

PROTOCOL_SHA = "a" * 64
OTHER_PROTOCOL_SHA = "d" * 64
COMMIT = "b" * 40

CODE = {"code_commit": COMMIT, "code_commit_dirty": False, "code_commit_state": "recorded"}
LOCK = {"path": "uv.lock", "sha256": "c" * 64, "state": "recorded"}
ENVIRONMENT = {"python_version": "3.10.8", "torch_version": "2.13.0+cu130"}

UNIT = {
    "dataset": "spambase",
    "method": "lagam",
    "training_path": "native_2d",
    "mechanism": "scar",
}


def packet(**overrides) -> dict:
    """A packet whose every probe is supplied, so it does not read this machine."""
    arguments = {
        "unit": UNIT,
        "seeds": (0, 1),
        "c_tokens": ("0.1",),
        "protocol_version": "survey-v3.0",
        "protocol_sha256": PROTOCOL_SHA,
        "code": CODE,
        "lock": LOCK,
        "environment": ENVIRONMENT,
    }
    arguments.update(overrides)
    return be.build_evidence(**arguments)


def request(**overrides) -> dict:
    """The four facts a unit script holds when it verifies a packet."""
    arguments = {
        "unit": dict(UNIT),
        "seeds": (0, 1),
        "c_tokens": ("0.1",),
        "protocol_sha256": PROTOCOL_SHA,
    }
    arguments.update(overrides)
    return arguments


def test_basic_a_packet_carries_the_two_audited_identity_fields():
    payload = packet()

    for field in BATCH_IDENTITY_FIELDS:
        assert field in payload
    assert payload["code_commit"] == COMMIT
    assert payload["dependency_lock"] == LOCK
    assert payload["unit"] == UNIT
    assert payload["run_count"] == 2
    assert payload["protocol_sha256"] == PROTOCOL_SHA


def test_basic_a_matching_request_is_accepted_whatever_order_it_spells():
    be.require_matches(packet(), **request(seeds=(1, 0)))


def test_basic_a_written_packet_loads_back_and_locates_itself(tmp_path):
    payload = packet()
    path = be.write(tmp_path / be.DIRECTORY, payload)

    assert path.parent.name == be.DIRECTORY
    assert path.name.endswith(f"{payload['batch_id'][:12]}.json")
    assert be.load(path) == payload
    block = be.manifest_block(payload, path)
    assert block["batch_id"] == payload["batch_id"]
    assert block["evidence_file"] == path.name
    assert block["evidence_sha256"] == file_hash(path)
    assert Path(path).parent == tmp_path / be.DIRECTORY


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"unit": {**UNIT, "dataset": "cifar10"}}, "describes dataset='spambase'"),
        ({"unit": {**UNIT, "method": "pulns"}}, "describes method='lagam'"),
        ({"unit": {**UNIT, "training_path": "native_cnn"}}, "training_path='native_2d'"),
        ({"unit": {**UNIT, "mechanism": None}}, "describes mechanism='scar'"),
        ({"seeds": (0,)}, "covers seed"),
        ({"c_tokens": ("0.3",)}, "covers c"),
        ({"protocol_sha256": OTHER_PROTOCOL_SHA}, "another protocol matrix"),
    ],
    ids=["dataset", "method", "training-path", "mechanism", "seeds", "c-tokens", "protocol"],
)
def test_param_a_request_the_packet_does_not_describe_is_refused(override, message):
    with pytest.raises(ValueError, match=message):
        be.require_matches(packet(), **request(**override))


def test_param_a_packet_of_another_schema_is_refused(tmp_path):
    payload = packet()
    payload["schema_version"] = "batch-evidence-v0"
    path = tmp_path / be.DIRECTORY / "packet.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="batch-evidence-v0"):
        be.load(path)


def test_edge_a_packet_whose_content_was_edited_no_longer_matches(tmp_path):
    path = be.write(tmp_path / be.DIRECTORY, packet())
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["seeds"] = [0, 1, 2]
    path.write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its own batch_id"):
        be.load(path)


def test_edge_an_unreadable_commit_and_a_missing_lock_are_recorded_honestly(tmp_path, monkeypatch):
    monkeypatch.setattr(be, "resolve_code_commit", lambda cwd=None: (None, None))

    code = be.code_identity()
    payload = packet(code=code)
    lock = be.dependency_lock(tmp_path / "absent" / "uv.lock")

    assert code["code_commit"] is None
    assert payload["code_commit"] is None
    assert code["code_commit_state"] == "unavailable"
    assert payload["code_commit_state"] == "unavailable"
    assert lock == {"path": "uv.lock", "sha256": None, "state": "unavailable"}


def test_edge_a_unit_part_that_would_escape_the_file_name_is_refused():
    with pytest.raises(ValueError, match="cannot name a file"):
        be.evidence_name(packet(unit={**UNIT, "dataset": "../spambase"}))


def test_edge_a_packet_that_cannot_be_trusted_is_refused(tmp_path):
    packet_path = be.write(tmp_path / be.DIRECTORY, packet())
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not a readable batch evidence packet"):
        be.load(broken)

    incomplete = json.loads(packet_path.read_text(encoding="utf-8"))
    del incomplete["dependency_lock"]
    broken.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="is missing"):
        be.load(broken)

    shapeless = dict(packet())
    shapeless["unit"] = "spambase"
    shapeless["batch_id"] = be.batch_identity(shapeless)
    with pytest.raises(ValueError, match="must be a mapping"):
        be.require_matches(shapeless, **request())

    elsewhere = tmp_path / "not_the_evidence_dir"
    elsewhere.mkdir()
    copied = elsewhere / packet_path.name
    copied.write_bytes(packet_path.read_bytes())
    with pytest.raises(ValueError, match="not inside a batch_evidence directory"):
        be.manifest_block(packet(), copied)

    with pytest.raises(ValueError, match="different bytes"):
        packet_path.write_text(packet_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        be.write(tmp_path / be.DIRECTORY, packet())


def test_determ_the_same_batch_builds_the_same_bytes(tmp_path):
    first = packet()
    second = packet()

    assert be.serialise(first) == be.serialise(second)
    assert be.write(tmp_path / be.DIRECTORY, first) == be.write(tmp_path / be.DIRECTORY, second)
    assert len(list((tmp_path / be.DIRECTORY).iterdir())) == 1


def test_determ_another_commit_keeps_both_packets(tmp_path):
    first = be.write(tmp_path / be.DIRECTORY, packet())
    second = be.write(
        tmp_path / be.DIRECTORY,
        packet(code={**CODE, "code_commit": "e" * 40}),
    )

    assert first != second
    assert {be.load(first)["code_commit"], be.load(second)["code_commit"]} == {COMMIT, "e" * 40}
    assert len(list((tmp_path / be.DIRECTORY).iterdir())) == 2


def test_basic_environment_identity_records_partition_runtime_context(monkeypatch):
    monkeypatch.setattr(
        be,
        "runtime_environment",
        lambda: {
            "machine": "x86_64",
            "processor": "test-cpu",
            "python_version": "3.11",
            "python_implementation": "CPython",
            "platform": "test",
            "torch_version": None,
            "cuda_available": False,
            "cuda_runtime_version": None,
            "cudnn_version": None,
            "gpu_devices": [],
            "gpu_driver_version": None,
            "numpy_version": "1",
            "scikit_learn_version": "1",
        },
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    result = be.environment_identity()
    assert result["hardware"]["processor"] == "test-cpu"
    assert result["hardware"]["cpu_count"] > 0
    assert result["thread_environment"]["OMP_NUM_THREADS"] == "4"
    assert result["thread_environment"]["MKL_NUM_THREADS"] is None
    assert result["gpu_driver_version"] is None
