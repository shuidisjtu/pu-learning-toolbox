# tests/unit/experiment/test_survey_report_entry_protocol.py

# ruff: noqa: F811, S101

"""Which protocol the audit and summary entry points read a tree under.

Both used to default to ``survey-v2.0``, so a v3 tree was judged by the v2 seed
list and the v2 matrix unless the caller remembered ``--protocol``.  The tree a
caller hands over today was run under ``survey-v3.0``; these pin that the report
says so without being asked, and that an explicit choice is still honoured.
"""

import json

import pytest
from _survey_summary_helpers import (  # noqa: F401
    audit_cli,
    config_for,
    manifest,
    summary_cli,
    write_config,
    write_tree,
)

pytestmark = pytest.mark.unit

_ENTRIES = {"audit": ("audit_cli", "audit.json"), "summary": ("summary_cli", "summary.json")}


def _run(request, entry, tmp_path, *extra):
    cli = request.getfixturevalue(_ENTRIES[entry][0])
    tree = write_tree(tmp_path / "tree", {"a": manifest()})
    config = write_config(tmp_path / "roots.json", config_for({"B1": (tree, 1)}))
    out = tmp_path / "out"
    code = cli.main(["--config", str(config), "--out-dir", str(out), *extra])
    report = json.loads((out / _ENTRIES[entry][1]).read_text(encoding="utf-8"))
    return code, report


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_basic_the_report_is_read_under_the_current_protocol_by_default(request, entry, tmp_path):
    _, report = _run(request, entry, tmp_path)

    assert report["protocol_version"] == "survey-v3.0"


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_param_an_explicit_protocol_is_honoured(request, entry, tmp_path):
    _, report = _run(request, entry, tmp_path, "--protocol", "survey-v2.0")

    assert report["protocol_version"] == "survey-v2.0"


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_edge_an_unknown_protocol_file_is_an_error_not_a_silent_default(
    request, entry, tmp_path, capsys
):
    code, _ = None, None
    cli = request.getfixturevalue(_ENTRIES[entry][0])
    tree = write_tree(tmp_path / "tree", {"a": manifest()})
    config = write_config(tmp_path / "roots.json", config_for({"B1": (tree, 1)}))

    code = cli.main(
        [
            "--config",
            str(config),
            "--out-dir",
            str(tmp_path / "out"),
            "--protocol",
            str(tmp_path / "nope.json"),
        ]
    )

    assert code == 1
    assert "error" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_determ_the_default_is_the_same_as_naming_v3_explicitly(request, entry, tmp_path):
    _, default = _run(request, entry, tmp_path / "d")
    _, named = _run(request, entry, tmp_path / "n", "--protocol", "survey-v3.0")

    assert default["protocol_sha256"] == named["protocol_sha256"]
