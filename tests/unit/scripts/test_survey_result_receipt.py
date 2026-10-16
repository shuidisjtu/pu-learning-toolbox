"""Exact byte receipts precede, and never substitute for, P2.2 replay."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import verify_survey_result_receipt as receipt  # noqa: E402

pytestmark = pytest.mark.unit


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path):
    """Independent tree-digest construction, with deliberately nonalphabetic batch order."""
    roots, batches, total_lines = {}, {}, []
    for name in ("B2", "B1"):
        root = tmp_path / name
        roots[name] = root
        (root / "run").mkdir(parents=True)
        payload = json.dumps({"synthetic": name}).encode()
        (root / "run/manifest.json").write_bytes(payload)
        line = f"{_sha(payload)}  run/manifest.json\n"
        batches[name] = {"manifests": 1, "bytes": len(payload), "tree_sha256": _sha(line.encode())}
        total_lines.append(f"{_sha(payload)}  {name}/run/manifest.json\n")
    index = {
        "schema_version": "survey-p22-artifacts-index-1",
        "produced_by_commit": "synthetic-not-a-project-release",
        "inputs": {
            "batches": batches,
            "total": {
                "manifests": 2,
                "bytes": sum(batch["bytes"] for batch in batches.values()),
                "tree_sha256": _sha("".join(total_lines).encode()),
            },
        },
        "artifacts": {"01_audit/audit.json": {"bytes": 2, "sha256": _sha(b"{}")}},
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return path, roots, index


def test_basic_exact_identity_passes_without_formal_acceptance(tmp_path):
    path, roots, _ = _fixture(tmp_path)
    report = receipt.review_receipt(path, roots)
    assert report["status"] == "verified"
    assert report["formal_acceptance"] is False
    assert report["reports"]["status"] == "not_requested"
    assert report["total"]["observed"] == report["total"]["expected"]


def test_determ_root_argument_order_does_not_change_baseline_tree_order(tmp_path):
    path, roots, _ = _fixture(tmp_path)
    assert receipt.review_receipt(path, roots) == receipt.review_receipt(
        path, dict(reversed(list(roots.items())))
    )


@pytest.mark.parametrize("mode", ["omitted", "absent", "empty"])
def test_param_partial_deliveries_never_claim_complete_identity(tmp_path, mode):
    path, roots, _ = _fixture(tmp_path)
    if mode == "omitted":
        roots.pop("B1")
    elif mode == "absent":
        roots["B1"] = tmp_path / "absent"
    else:
        (roots["B1"] / "run/manifest.json").unlink()
    report = receipt.review_receipt(path, roots)
    assert report["status"] == "incomplete"
    assert report["missing"]
    assert "total" not in report


@pytest.mark.parametrize("mode", ["same_size", "extra", "renamed", "newline"])
def test_param_changed_bytes_counts_or_relative_paths_are_rejected(tmp_path, mode):
    path, roots, _ = _fixture(tmp_path)
    manifest = roots["B1"] / "run/manifest.json"
    if mode == "same_size":
        manifest.write_bytes(manifest.read_bytes().replace(b"B1", b"XX"))
    elif mode == "extra":
        (roots["B1"] / "extra").mkdir()
        (roots["B1"] / "extra/manifest.json").write_bytes(b"{}")
    elif mode == "renamed":
        (roots["B1"] / "run").rename(roots["B1"] / "renamed")
    else:
        manifest.write_bytes(manifest.read_bytes() + b"\n")
    report = receipt.review_receipt(path, roots)
    assert report["status"] == "failed"
    assert report["problems"]


@pytest.mark.parametrize("mode", ["unknown", "same", "nested", "merged"])
def test_param_ambiguous_roots_fail_before_reading_manifests(tmp_path, mode):
    path, roots, _ = _fixture(tmp_path)
    if mode == "unknown":
        roots["unknown"] = tmp_path
    elif mode == "same":
        roots["B1"] = roots["B2"]
    elif mode == "nested":
        roots["B1"] = roots["B2"] / "run"
    else:
        (roots["B1"] / "B3ab_merged").mkdir()
    with pytest.raises(ValueError):
        receipt.review_receipt(path, roots)


@pytest.mark.parametrize("mode", ["root", "directory", "manifest", "report"])
def test_param_symlinks_are_not_hashed_as_original_artifacts(tmp_path, mode):
    path, roots, _ = _fixture(tmp_path)
    reports = None
    link = tmp_path / "link"
    try:
        if mode == "root":
            link.symlink_to(roots["B1"], target_is_directory=True)
            roots["B1"] = link
        elif mode == "directory":
            (roots["B1"] / "linked").symlink_to(roots["B2"], target_is_directory=True)
        elif mode == "manifest":
            manifest = roots["B1"] / "run/manifest.json"
            manifest.unlink()
            manifest.symlink_to(roots["B2"] / "run/manifest.json")
        else:
            reports = tmp_path / "reports"
            (reports / "01_audit").mkdir(parents=True)
            (reports / "01_audit/audit.json").symlink_to(path)
    except OSError as error:
        pytest.skip(f"symlinks unavailable on this platform: {error}")
    with pytest.raises(ValueError):
        receipt.review_receipt(path, roots, reports=reports)


@pytest.mark.parametrize("mode", ["matching", "missing", "changed"])
def test_param_original_report_receipts_are_explicit_and_byte_exact(tmp_path, mode):
    path, roots, _ = _fixture(tmp_path)
    reports = tmp_path / "reports"
    (reports / "01_audit").mkdir(parents=True)
    if mode != "missing":
        (reports / "01_audit/audit.json").write_bytes(b"{}" if mode == "matching" else b"[]")
    report = receipt.review_receipt(path, roots, reports=reports)
    assert (
        report["status"]
        == {"matching": "verified", "missing": "incomplete", "changed": "failed"}[mode]
    )


@pytest.mark.parametrize("field", ["schema", "hash", "bytes", "total", "path"])
def test_param_bad_index_fails_without_using_an_invented_baseline(tmp_path, field):
    path, roots, index = _fixture(tmp_path)
    if field == "schema":
        index["schema_version"] = "unknown"
    elif field == "hash":
        index["inputs"]["batches"]["B1"]["tree_sha256"] = "bad"
    elif field == "bytes":
        index["inputs"]["batches"]["B1"]["bytes"] = True
    elif field == "total":
        index["inputs"]["total"]["manifests"] = 99
    else:
        index["artifacts"]["../escape"] = index["artifacts"]["01_audit/audit.json"]
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError):
        receipt.review_receipt(path, roots)


def test_edge_overall_digest_is_checked_independently_of_batch_digests(tmp_path):
    path, roots, index = _fixture(tmp_path)
    index["inputs"]["total"]["tree_sha256"] = "0" * 64
    path.write_text(json.dumps(index), encoding="utf-8")
    report = receipt.review_receipt(path, roots)
    assert report["status"] == "failed"
    assert report["problems"] == ["overall manifest tree identity differs"]


def test_edge_upload_mutation_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "in-flight.json"
    path.write_bytes(b"{}")
    original = Path.lstat
    calls = 0

    def changed(item):
        nonlocal calls
        if item == path:
            calls += 1
            if calls == 2:
                path.write_bytes(b"changed")
        return original(item)

    monkeypatch.setattr(Path, "lstat", changed)
    with pytest.raises(ValueError, match="changed during"):
        receipt._file_identity(path)


def test_determ_verification_never_rewrites_source_bytes(tmp_path):
    path, roots, _ = _fixture(tmp_path)
    before = {item: item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    receipt.review_receipt(path, roots)
    assert all(item.read_bytes() == data for item, data in before.items())


@pytest.mark.parametrize("payload", ['{"schema_version":NaN}', '{"inputs":{},"inputs":{}}'])
def test_param_nonfinite_or_duplicate_index_fields_fail_closed(tmp_path, capsys, payload):
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    assert receipt.main(["--index", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


@pytest.mark.parametrize("mode,expected", [("complete", 0), ("partial", 2), ("duplicate", 1)])
def test_param_cli_exit_codes_do_not_conflate_missing_and_mismatched(
    tmp_path, capsys, mode, expected
):
    path, roots, _ = _fixture(tmp_path)
    args = ["--index", str(path)]
    for name, root in roots.items():
        if mode != "partial" or name == "B1":
            args.extend(["--batch-root", f"{name}={root}"])
    if mode == "duplicate":
        args.extend(["--batch-root", f"B1={roots['B1']}"])
    assert receipt.main(args) == expected
    report = json.loads(capsys.readouterr().out)
    assert report["formal_acceptance"] is False
