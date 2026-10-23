"""Read-only v3 planning CLI; no estimator run or implicit approvals."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pu_toolbox.experiment import survey_v3_proposal as planning

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import prepare_survey_v3_proposal as script  # noqa: E402

pytestmark = pytest.mark.unit


def test_basic_summary_reports_real_scope_without_runnable_units(capsys):
    assert script.main(["--summary"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["target_algorithm_count"] == 22 and result["P3_1_algorithm_count"] == 13
    assert result["blocked_dataset_profile_requests"] == 184
    assert result["runnable_request_count"] == 0 and not result["formal_admission"]


def test_determ_recorded_check_and_materialized_units_remain_pure(capsys):
    artifact = (
        planning.ROOT / "docs/research/pu_survey/data/survey_v3_profile_unit_proposal_20261023.json"
    )
    before = artifact.read_bytes()
    assert script.main(["--check", str(artifact), "--units"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 184 and all(not row["runnable"] for row in rows)
    assert artifact.read_bytes() == before


@pytest.mark.parametrize("payload", [{}, {"schema_version": "unknown"}, {"formal_admission": True}])
def test_param_bad_artifact_fails_instead_of_replacing_it(payload, tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        script.main(["--check", str(path)])
    assert error.value.code == 1 and "error:" in capsys.readouterr().err
    assert path.read_bytes() == before


def test_edge_subprocess_can_plan_without_torch_installed(tmp_path):
    source = """
import sys
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('torch deliberately unavailable', name='torch')
sys.meta_path.insert(0, NoTorch())
from pu_toolbox.experiment.survey_v3_proposal import build_proposal, expand_unit_requests
assert len(expand_unit_requests(build_proposal())) == 184
assert 'torch' not in sys.modules
"""
    process = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(planning.ROOT)},
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert list(tmp_path.iterdir()) == []


def test_param_conflicting_output_modes_are_not_silently_ignored(capsys):
    with pytest.raises(SystemExit) as error:
        script.main(["--summary", "--units"])
    assert error.value.code == 2 and "not allowed" in capsys.readouterr().err
