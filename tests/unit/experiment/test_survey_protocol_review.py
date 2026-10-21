"""Review-ledger and archive guarantees for the versioned protocol matrices.

Kept separate from ``test_survey_protocol.py`` so the matrix contract tests stay
within the suite's per-file budget; these three tests pin the v1 archive and the
v2 review record rather than the execution matrix itself.
"""

import json

import pytest

from pu_toolbox.experiment.survey_protocol import PROTOCOL_PATH, digest, load_protocol

pytestmark = pytest.mark.unit


def test_basic_v1_archive_stays_byte_pinned_for_dated_evidence():
    """The archived v1.2 matrix must not move: dated P3 evidence binds its bytes."""
    v1 = load_protocol(PROTOCOL_PATH.with_name("survey_protocol_v1.json"))
    assert v1["protocol_version"] == "survey-v1.2"
    assert digest(v1) == "287c2f45387f02714e5925b35dcb04f3f64e1e3faf8f740cab861fc5747dd84a"
    assert "review_items" not in v1


def test_basic_v2_review_ledger_records_accepted_and_pending_items():
    protocol = load_protocol()
    items = {item["item_id"]: item["status"] for item in protocol["review_items"]}
    assert items["v2_p4_clean_support_policy"] == "pending_collaborator_review"
    assert items["v2_p10_self_pu_scope"] == "pending_collaborator_review"
    assert sum(status == "accepted" for status in items.values()) == 8
    assert set(items.values()) == {"accepted", "pending_collaborator_review"}


def test_edge_duplicate_review_item_id_is_rejected(tmp_path):
    protocol = load_protocol()
    protocol["review_items"].append(dict(protocol["review_items"][0]))
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="duplicate survey protocol review item"):
        load_protocol(path)


@pytest.mark.parametrize("status", ["approved", "pending", ""])
def test_param_review_item_status_must_be_a_closed_value(status, tmp_path):
    protocol = load_protocol()
    protocol["review_items"][0]["status"] = status
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="review item status"):
        load_protocol(path)


def test_determ_review_ledger_digest_moves_with_a_status_change():
    protocol = load_protocol()
    before = digest(protocol)
    changed = json.loads(json.dumps(protocol))
    changed["review_items"][3]["status"] = "accepted"
    assert digest(changed) != before
    assert digest(json.loads(json.dumps(protocol))) == before
