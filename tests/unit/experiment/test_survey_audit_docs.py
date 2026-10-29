"""A14 / A16 / A17: the documentation and test evidence the delivery rests on is in place.

None of these reads a manifest.  They look at files in the repository, so a pass says
"the evidence is there with its stable marker", never "its content is right": that stays
a reviewer's judgement, and the check messages say so.

* A14: each batch has its stage snapshot, and the snapshot has its "本快照不含" section.
* A16: when ``self_pu`` is delivered, its snapshot names the ablation variant *and* says
  the methodological review was not obtained.
* A17: the tests that keep the test set out of selection exist; whether they pass is
  the test suite's verdict, not this audit's.
"""

import importlib
import sys

import pytest
from _survey_summary_helpers import SCRIPTS_DIR, config_for, manifest
from _survey_summary_helpers import entry as _entry

from pu_toolbox.experiment import batch_evidence

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
checks = importlib.import_module("audit_survey_batches")

pytestmark = pytest.mark.unit

_BATCHES = ("B1", "B2", "B3a", "B3b", "B4")
_ABLATION = "`self_pu` 用的变体是无 clean-validation 元重加权的已裁决消融变体。\n"
_NOT_REVIEWED = "该变体的方法学合作者复核未获。\n"


def _repo(tmp_path, *, boundary=True, ablation=True, not_reviewed=True, guards=True):
    """A repository skeleton holding every file the three checks look for."""
    delivery = tmp_path / checks.DELIVERY_DIR
    delivery.mkdir(parents=True)
    for name in set(checks.SNAPSHOT_FILES.values()):
        body = "# 快照\n\n## 5. 本快照不含\n\n无排名。\n" if boundary else "# 快照\n"
        if name == checks.ABLATION_SNAPSHOT:
            body += (_ABLATION if ablation else "") + (_NOT_REVIEWED if not_reviewed else "")
        (delivery / name).write_text(body, encoding="utf-8")
    for relative, test_name in checks.ISOLATION_TESTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def {test_name}():\n    pass\n" if guards else "", encoding="utf-8")
    return tmp_path


def _config(names=_BATCHES):
    return config_for({name: (".", 1) for name in names})


def _loaded(method="self_pu"):
    return [_entry(manifest(method=method), batch="B3b")]


def test_basic_a14_passes_when_every_batch_has_a_snapshot_with_its_boundary_section(tmp_path):
    check = checks.check_snapshots(_config(), _repo(tmp_path))

    assert check["check_id"] == "A14"
    assert check["result"] == "pass"
    assert "not judged" in check["message"]


def test_edge_a14_a_missing_snapshot_is_an_error_naming_the_batch(tmp_path):
    repo = _repo(tmp_path)
    (repo / checks.DELIVERY_DIR / checks.SNAPSHOT_FILES["B2"]).unlink()

    check = checks.check_snapshots(_config(), repo)

    assert check["result"] == "fail"
    assert any(item.startswith("B2:") for item in check["evidence"])


def test_edge_a14_a_snapshot_without_its_boundary_section_is_an_error(tmp_path):
    check = checks.check_snapshots(_config(), _repo(tmp_path, boundary=False))

    assert check["result"] == "fail"
    assert any("本快照不含" in item for item in check["evidence"])


def test_param_a14_a_batch_with_no_known_snapshot_is_an_error_not_skipped(tmp_path):
    check = checks.check_snapshots(_config(("B1", "B9")), _repo(tmp_path))

    assert check["result"] == "fail"
    assert any(item.startswith("B9:") for item in check["evidence"])


def test_basic_a16_passes_when_the_variant_and_the_missing_review_are_both_disclosed(tmp_path):
    check = checks.check_ablation_disclosure(_loaded(), _repo(tmp_path))

    assert check["check_id"] == "A16"
    assert check["result"] == "pass"


@pytest.mark.parametrize("missing", ["ablation", "not_reviewed"])
def test_edge_a16_either_half_of_the_disclosure_missing_is_an_error(tmp_path, missing):
    repo = _repo(tmp_path, **{missing: False})

    check = checks.check_ablation_disclosure(_loaded(), repo)

    assert check["result"] == "fail"


def test_edge_a16_is_not_applicable_when_no_self_pu_run_is_delivered(tmp_path):
    check = checks.check_ablation_disclosure(_loaded("nnpu"), _repo(tmp_path, ablation=False))

    assert check["result"] == "not_applicable"


def test_edge_a16_a_missing_snapshot_file_is_an_error(tmp_path):
    repo = _repo(tmp_path)
    (repo / checks.DELIVERY_DIR / checks.ABLATION_SNAPSHOT).unlink()

    check = checks.check_ablation_disclosure(_loaded(), repo)

    assert check["result"] == "fail"


def test_basic_a17_passes_when_every_isolation_guard_test_is_present(tmp_path):
    check = checks.check_isolation_tests(_repo(tmp_path))

    assert check["check_id"] == "A17"
    assert check["result"] == "pass"
    assert "test suite" in check["message"]


def test_edge_a17_a_missing_guard_test_is_an_error_naming_it(tmp_path):
    check = checks.check_isolation_tests(_repo(tmp_path, guards=False))

    assert check["result"] == "fail"
    assert len(check["evidence"]) == len(checks.ISOLATION_TESTS)


def test_determ_the_three_checks_give_the_same_answer_twice(tmp_path):
    repo = _repo(tmp_path)

    def run():
        return (
            checks.check_snapshots(_config(), repo),
            checks.check_ablation_disclosure(_loaded(), repo),
            checks.check_isolation_tests(repo),
        )

    assert run() == run()


def _packet_batch(tmp_path):
    root = tmp_path / "results"
    payload = manifest(method="lagam")
    packet = batch_evidence.build_evidence(
        unit={
            "dataset": "spambase",
            "method": "lagam",
            "training_path": "native_2d",
            "mechanism": "scar",
        },
        seeds=(0,),
        c_tokens=("0.1",),
        protocol_version=payload["protocol_version"],
        protocol_sha256=payload["protocol_sha256"],
        code={"code_commit": "a" * 40, "code_commit_dirty": False, "code_commit_state": "recorded"},
        lock={"path": "uv.lock", "sha256": "b" * 64, "state": "recorded"},
        environment={"python_version": "3.10.8"},
    )
    path = batch_evidence.write(root / batch_evidence.DIRECTORY, packet)
    payload["batch"] = batch_evidence.manifest_block(packet, path)
    return config_for({"support": (root, 1)}), [_entry(payload, batch="support")], path


def test_basic_a14_accepts_verified_d52_packets_for_new_batch_names(tmp_path):
    config, entries, _ = _packet_batch(tmp_path)

    check = checks.check_snapshots(config, tmp_path, entries=entries)

    assert check["result"] == "pass"
    assert check["observed"]["packet_batches"] == ["support"]


@pytest.mark.parametrize("damage", ["missing", "changed", "unit", "seed", "c", "protocol"])
def test_param_a14_refuses_new_batches_with_missing_or_changed_packets(tmp_path, damage):
    config, entries, path = _packet_batch(tmp_path)
    if damage == "missing":
        path.unlink()
    elif damage == "changed":
        path.write_text("{}", encoding="utf-8")
    else:
        packet = batch_evidence.load(path)
        if damage == "unit":
            packet["unit"]["dataset"] = "connect_4"
        elif damage == "seed":
            packet["seeds"] = [1]
        elif damage == "c":
            packet["c_tokens"] = ["0.3"]
        else:
            packet["protocol_sha256"] = "f" * 64
        packet["batch_id"] = batch_evidence.batch_identity(packet)
        replacement = batch_evidence.write(path.parent, packet)
        entries[0]["payload"]["batch"] = batch_evidence.manifest_block(packet, replacement)

    check = checks.check_snapshots(config, tmp_path, entries=entries)

    assert check["result"] == "fail"
    assert any("support:" in item for item in check["evidence"])


def test_edge_a14_requires_every_new_batch_manifest_to_cite_its_evidence(tmp_path):
    config, entries, _ = _packet_batch(tmp_path)
    entries.append(_entry(manifest(seed=1), batch="support"))

    assert checks.check_snapshots(config, tmp_path, entries=entries)["result"] == "fail"


def test_determ_a14_packet_validation_does_not_depend_on_prior_calls(tmp_path):
    config, entries, _ = _packet_batch(tmp_path)

    first = checks.check_snapshots(config, tmp_path, entries=entries)
    assert checks.check_snapshots(config, tmp_path, entries=entries) == first
