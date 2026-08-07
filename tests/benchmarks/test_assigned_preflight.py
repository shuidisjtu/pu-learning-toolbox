"""Tests for assigned-method paper-run readiness auditing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.assigned_methods.preflight_paper import (
    _parse_method_paths,
    audit_locked_configs,
)

pytestmark = pytest.mark.unit


def _capabilities():
    return {
        "python_version": "3.7.0",
        "platform": "test",
        "packages": {
            "chainer": "7.8.1",
            "densratio": "0.3.0",
            "numpy": "1.19.2",
            "scikit-learn": "0.24.1",
            "torch": "1.8.1",
            "torchvision": "0.9.1",
        },
        "cuda_available": True,
        "cuda_device_count": 1,
        "commands": {"matlab": "/usr/bin/matlab"},
    }


def _write_demo_config(tmp_path: Path, *, gaps=None):
    config_dir = tmp_path / "official"
    config_dir.mkdir()
    (tmp_path / "sources.json").write_text(
        json.dumps({"sources": {"demo": {"status": "locked"}}}), encoding="utf-8"
    )
    config = {
        "schema_version": 1,
        "protocol": "paper_like",
        "method": "demo",
        "status": "locked_not_executed",
        "source_lock": "../sources.json#sources/demo",
        "runner": {"entrypoint": "run.py"},
        "execution_requirements": {
            "source_checkout_required": True,
            "data_root_required": True,
            "cuda_required": False,
            "required_commands": [],
            "toolbox_implementation_gaps": [] if gaps is None else gaps,
        },
    }
    (config_dir / "demo.json").write_text(json.dumps(config), encoding="utf-8")
    return config_dir


@pytest.mark.unit
def test_basic_repository_configs_report_separate_readiness_axes():
    config_dir = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "assigned_methods"
        / "configs"
        / "official"
    )
    report = audit_locked_configs(config_dir, capabilities=_capabilities())
    assert len(report["methods"]) == 5
    assert report["all_ready_for_official_execution"] is False
    assert report["all_ready_for_toolbox_replication"] is False
    cpe = next(item for item in report["methods"] if item["method"] == "class_prior_estimation")
    assert cpe["source_lock_status"] == "partial"
    assert any("Immutable official source" in item for item in cpe["official_execution_blockers"])


@pytest.mark.unit
def test_param_supplied_resources_can_clear_official_but_not_toolbox_blockers(tmp_path):
    config_dir = _write_demo_config(tmp_path, gaps=["clean-room implementation differs"])
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    (source / "run.py").write_text("# entrypoint\n", encoding="utf-8")
    report = audit_locked_configs(
        config_dir,
        source_roots={"demo": source},
        data_roots={"demo": data},
        capabilities=_capabilities(),
    )
    method = report["methods"][0]
    assert method["ready_for_official_execution"] is True
    assert method["ready_for_toolbox_replication"] is False
    assert method["official_execution_blockers"] == []
    assert method["toolbox_replication_blockers"] == ["clean-room implementation differs"]


@pytest.mark.unit
def test_edge_invalid_mapping_and_missing_source_file_are_reported(tmp_path):
    with pytest.raises(ValueError, match="METHOD=PATH"):
        _parse_method_paths(["demo"], "--source-root")
    with pytest.raises(ValueError, match="duplicate"):
        _parse_method_paths(["demo=/a", "demo=/b"], "--source-root")

    config_dir = _write_demo_config(tmp_path)
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    data.mkdir()
    report = audit_locked_configs(
        config_dir,
        source_roots={"demo": source},
        data_roots={"demo": data},
        capabilities=_capabilities(),
    )
    assert report["methods"][0]["official_execution_blockers"] == [
        "Official source file is missing: run.py"
    ]
    with pytest.raises(ValueError, match="unknown methods"):
        audit_locked_configs(
            config_dir,
            source_roots={"typo": source},
            capabilities=_capabilities(),
        )


@pytest.mark.unit
def test_determ_method_reports_are_stable_for_identical_inputs(tmp_path):
    config_dir = _write_demo_config(tmp_path)
    first = audit_locked_configs(config_dir, capabilities=_capabilities())
    second = audit_locked_configs(config_dir, capabilities=_capabilities())
    assert first["methods"] == second["methods"]
    assert first["capabilities"] == second["capabilities"]
