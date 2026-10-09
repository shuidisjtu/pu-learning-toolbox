"""A11: a batch's archive digest file is well formed and names its own batch's archive.

The evidence is a coreutils ``sha256sum`` file: ``<64 lowercase hex>`` and two spaces
and the archive's path on the execution host.  Most archives are not on the machine
that audits, so the digest is recomputed only where the archive sits beside its digest
file; everywhere else the check says it checked the *form* and the *name*, and that the
digest was not recomputed -- it never says the archive was recovered, which is a
reviewer's decision.  A batch without a digest file is ``not_run``, and so is a tree
where only some batches have one.
"""

import hashlib
import importlib
import sys

import pytest
from _survey_summary_helpers import SCRIPTS_DIR, config_for, manifest
from _survey_summary_helpers import entry as _entry

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
checks = importlib.import_module("audit_survey_batches")

pytestmark = pytest.mark.unit

_HEX = "f3458eec1f0316ae7d212861438afe84f78533ccb8b29962a8d2859a517fe559"
_REMOTE = "/root/autodl-tmp/pu-survey-backup-stage"


def _digest_file(directory, *lines, name="B1_spambase.tar.gz.sha256"):
    path = directory / name
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _config(tmp_path, digests, *, batch="B1"):
    config = config_for({batch: (tmp_path / batch, 1)})
    if digests is not None:
        config["batches"][0]["archive_digests"] = str(digests)
    return config


def _entries(batch="B1"):
    return [_entry(manifest(), batch=batch, name=f"{batch}/manifest.json")]


def test_basic_a11_passes_a_well_formed_digest_file_for_the_batchs_archive(tmp_path):
    digests = _digest_file(tmp_path, f"{_HEX}  {_REMOTE}/B1_spambase/B1_spambase.tar.gz")

    check = checks.check_archives(_config(tmp_path, digests), _entries())

    assert check["check_id"] == "A11"
    assert check["result"] == "pass"
    assert check["observed"]["verified_batches"] == {"B1": {"archives": 1, "recomputed": 0}}


def test_basic_a11_says_the_digest_was_not_recomputed_when_the_archive_is_absent(tmp_path):
    digests = _digest_file(tmp_path, f"{_HEX}  {_REMOTE}/B1_spambase/B1_spambase.tar.gz")

    check = checks.check_archives(_config(tmp_path, digests), _entries())

    assert "not recomputed" in check["message"]
    assert "recovery is not tested" in check["message"]


def test_basic_a11_recomputes_the_digest_when_the_archive_sits_beside_its_file(tmp_path):
    archive = tmp_path / "B1_spambase.tar.gz"
    archive.write_bytes(b"archive bytes")
    real = hashlib.sha256(b"archive bytes").hexdigest()
    digests = _digest_file(tmp_path, f"{real}  {_REMOTE}/B1_spambase/B1_spambase.tar.gz")

    check = checks.check_archives(_config(tmp_path, digests), _entries())

    assert check["result"] == "pass"
    assert check["observed"]["verified_batches"] == {"B1": {"archives": 1, "recomputed": 1}}


def test_edge_a11_a_local_archive_with_another_digest_is_an_error(tmp_path):
    (tmp_path / "B1_spambase.tar.gz").write_bytes(b"archive bytes")
    digests = _digest_file(tmp_path, f"{_HEX}  {_REMOTE}/B1_spambase/B1_spambase.tar.gz")

    check = checks.check_archives(_config(tmp_path, digests), _entries())

    assert check["result"] == "fail"
    assert any("recomputed" in item and _HEX in item for item in check["evidence"])


@pytest.mark.parametrize(
    "line",
    [
        f"{_HEX[:-1]}  {_REMOTE}/B1_spambase.tar.gz",
        f"{_HEX.upper()}  {_REMOTE}/B1_spambase.tar.gz",
        f"{_HEX} {_REMOTE}/B1_spambase.tar.gz",
        f"{_HEX}  ",
        "not a digest line",
    ],
)
def test_param_a11_a_malformed_digest_line_is_an_error(tmp_path, line):
    digests = _digest_file(tmp_path, line)

    check = checks.check_archives(_config(tmp_path, digests), _entries())

    assert check["result"] == "fail"
    assert any("line 1" in item for item in check["evidence"])


def test_edge_a11_the_same_archive_listed_twice_is_an_error(tmp_path):
    line = f"{_HEX}  {_REMOTE}/B1_spambase/B1_spambase.tar.gz"
    digests = _digest_file(tmp_path, line, line)

    check = checks.check_archives(_config(tmp_path, digests), _entries())

    assert check["result"] == "fail"
    assert any("more than once" in item for item in check["evidence"])


def test_edge_a11_an_archive_named_for_another_batch_is_an_error(tmp_path):
    digests = _digest_file(tmp_path, f"{_HEX}  {_REMOTE}/B4/B4_cifar10.tar.gz")

    check = checks.check_archives(_config(tmp_path, digests), _entries())

    assert check["result"] == "fail"
    assert any("does not belong to batch B1" in item for item in check["evidence"])


def test_param_a11_an_empty_digest_file_is_an_error(tmp_path):
    digests = _digest_file(tmp_path)

    check = checks.check_archives(_config(tmp_path, digests), _entries())

    assert check["result"] == "fail"
    assert any("lists no archive" in item for item in check["evidence"])


def test_param_a11_an_unreadable_digest_file_is_an_error_not_a_pass(tmp_path):
    check = checks.check_archives(_config(tmp_path, tmp_path / "absent.sha256"), _entries())

    assert check["result"] == "fail"
    assert any("cannot read" in item for item in check["evidence"])


def test_basic_a11_is_not_run_when_no_batch_names_a_digest_file(tmp_path):
    check = checks.check_archives(_config(tmp_path, None), _entries())

    assert check["result"] == "not_run"
    assert check["observed"]["verified_batches"] == {}


def test_edge_a11_a_half_covered_tree_is_not_reported_as_verified(tmp_path):
    digests = _digest_file(tmp_path, f"{_HEX}  {_REMOTE}/B1_spambase/B1_spambase.tar.gz")
    config = config_for({"B1": (tmp_path / "B1", 1), "B2": (tmp_path / "B2", 1)})
    config["batches"][0]["archive_digests"] = str(digests)

    check = checks.check_archives(config, _entries("B1") + _entries("B2"))

    assert check["result"] == "not_run"
    assert check["observed"]["unverified_batches"] == ["B2"]
    assert list(check["observed"]["verified_batches"]) == ["B1"]


def test_param_a11_a_failure_in_one_batch_outranks_a_gap_in_another(tmp_path):
    digests = _digest_file(tmp_path, "garbage")
    config = config_for({"B1": (tmp_path / "B1", 1), "B2": (tmp_path / "B2", 1)})
    config["batches"][0]["archive_digests"] = str(digests)

    check = checks.check_archives(config, _entries("B1") + _entries("B2"))

    assert check["result"] == "fail"


def test_determ_a11_the_same_inputs_give_the_same_check(tmp_path):
    digests = _digest_file(tmp_path, f"{_HEX}  {_REMOTE}/B1_spambase/B1_spambase.tar.gz")
    config = _config(tmp_path, digests)

    assert checks.check_archives(config, _entries()) == checks.check_archives(config, _entries())
