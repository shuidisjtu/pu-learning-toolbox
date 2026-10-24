"""Source-bound scope and blocked unit requests do not approve an experiment."""

import ast
import json

import pytest

from pu_toolbox.experiment import survey_v3_proposal as planning
from pu_toolbox.experiment.survey_protocol import load_protocol
from pu_toolbox.utils.serialization import strict_canonical_hash

pytestmark = pytest.mark.unit
ARTIFACT = (
    planning.ROOT / "docs/research/pu_survey/data/survey_v3_profile_unit_proposal_20261023.json"
)


def test_basic_exact_algorithm_scope_keeps_p3mix_vpu_and_tedn_distinct():
    data = planning.build_proposal()
    assert len(data["scope"]["target_algorithms"]) == 22
    assert len(data["scope"]["p3_1_algorithms"]) == 13
    assert "p3mix" in data["scope"]["p3_1_algorithms"]
    assert "vpu" not in data["scope"]["p3_1_algorithms"]
    assert data["scope"]["additional_target_algorithms"] == ["vpu"]
    assert data["scope"]["registered_outside_target_scope"] == ["lbe_neural", "pusb", "tedn"]
    assert len(data["method_profile_requests"]) == 23
    assert data["method_profile_requests"]["self_pu_full_meta"]["algorithm"] == "self_pu"


def test_basic_profile_requests_copy_evidence_not_approve_defaults_or_erase_ablation():
    data = planning.build_proposal()
    original = load_protocol(planning.ROOT / planning.PROTOCOL)
    for method, profile in data["method_profile_requests"].items():
        assert all(
            profile[key] is None
            for key in (
                "proposed_params",
                "proposed_budget",
                "proposed_candidates",
                "proposed_stage_eligibility",
            )
        )
        assert not profile["formal_admission"]
        if method in original["method_profiles"]:
            assert profile["inherited_profile"] == original["method_profiles"][method]
    assert not data["method_profile_requests"]["self_pu"]["inherited_profile"]["params"][
        "require_validation"
    ]
    full = data["method_profile_requests"]["self_pu_full_meta"]
    assert full["required_semantics"]["require_validation"] and full["requires_clean_support"]
    assert full["formal_training_view"] is None and full[
        "available_engineering_training_views"
    ] == ["os", "ts"]
    assert data["basis"]["inherited_execution_units"] == original["execution_units"]
    assert data["basis"]["inherited_review_items"] == original["review_items"]
    assert data["owner_signatures"] == []


def test_edge_all_cells_are_unique_and_blocked_including_unsupported_support_datasets():
    data = planning.build_proposal()
    rows = planning.validate_proposal(data)
    assert len(rows) == len({(r["dataset"], r["profile_id"]) for r in rows}) == 23 * 8
    assert all(
        not row["runnable"] and not row["formal_admission"] and row["blockers"] for row in rows
    )
    assert sum("dataset_outside_existing_support_policy" in row["blockers"] for row in rows) == 9
    assert (
        sum(
            "subject_isolation_and_placeholder_source_scope_required" in row["blockers"]
            for row in rows
        )
        == 23
    )
    oracle = next(
        row for row in rows if row["profile_id"] == "pn_oracle" and row["dataset"] == "cifar10"
    )
    assert "native_cnn" in oracle["path_options"]
    assert data["deferred_experiments"]["pn_oracle/native_cnn/five_seed"].startswith("deferred")


def test_determ_committed_proposal_matches_builder_and_its_own_hash():
    """Pin the proposal content; source digests refresh on demand, not on every edit.

    Hashing the generating sources used to make this case fail on any unrelated
    edit to a shared module, forcing a regeneration for every commit. Content
    tampering is still caught (the builder must reproduce the committed body and
    the body must match its own proposal_sha256); stale source digests are
    reported by ``prepare_survey_v3_proposal.py --check`` when someone asks.
    """
    expected = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    volatile = ("source_files_sha256", "proposal_sha256")
    built = planning.build_proposal()
    assert {k: v for k, v in built.items() if k not in volatile} == {
        k: v for k, v in expected.items() if k not in volatile
    }
    assert expected["proposal_sha256"] == strict_canonical_hash(
        {k: v for k, v in expected.items() if k != "proposal_sha256"}
    )

    expected["method_profile_requests"]["self_pu_full_meta"]["implementation_facts"][
        "ledger_snapshot"
    ]["run_view"] = "changed"
    assert (
        expected["method_profile_requests"]["self_pu"]["implementation_facts"]["ledger_snapshot"][
            "run_view"
        ]
        != "changed"
    )


@pytest.mark.parametrize("fault", ["budget", "params", "formal", "signature", "ghost", "coverage"])
def test_param_rehashed_unapproved_values_or_scope_tampering_are_refused(fault):
    data = planning.build_proposal()
    profile = data["method_profile_requests"]["pulda"]
    if fault == "budget":
        profile["proposed_budget"] = {"epochs": 120}
    elif fault == "params":
        profile["proposed_params"] = {"hidden_dim": 64}
    elif fault == "formal":
        data["formal_admission"] = True
    elif fault == "signature":
        data["owner_signatures"] = [{"name": "fabricated"}]
    elif fault == "ghost":
        data["scope"]["target_algorithms"].append("ghost")
    else:
        data["execution_unit_requests"][0]["datasets"].pop()
    data["proposal_sha256"] = strict_canonical_hash(
        {k: v for k, v in data.items() if k != "proposal_sha256"}
    )
    with pytest.raises(ValueError, match="current sources/scope"):
        planning.validate_proposal(data)


def test_edge_planning_artifact_cannot_be_consumed_as_an_execution_protocol(tmp_path):
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(planning.build_proposal()), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_protocol(path)


def test_edge_nonliteral_constructor_default_is_not_evaluated():
    klass = ast.parse("class C:\n def __init__(self, a=2, *, b=forbidden()): pass\n").body[0]
    observed = planning._constructor(klass)
    assert observed["literal_defaults"] == {"a": 2}
    assert observed["unevaluated_expressions"] == {"b": "forbidden()"}
    assert not observed["defaults_are_approved_budget"]


def test_param_missing_or_external_files_are_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside").write_text("outside", encoding="utf-8")
    for reference in ("missing", "../outside"):
        with pytest.raises(ValueError, match="missing or external"):
            planning._file(root, reference)


def test_determ_planning_source_hash_is_newline_portable(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 1\n")
    lf = planning._text_hash(path)
    path.write_bytes(b"value = 1\r\n")
    assert planning._text_hash(path) == lf


def test_edge_missing_class_and_scope_drift_are_not_silently_repaired(monkeypatch):
    monkeypatch.setattr(planning, "_classes", lambda root: {})
    with pytest.raises(ValueError, match="implementation class"):
        planning.build_proposal()
