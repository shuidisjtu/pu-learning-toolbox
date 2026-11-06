"""Review-state validation never promotes an execution protocol implicitly."""

import json

import pytest

from pu_toolbox.experiment.survey_protocol import digest, load_protocol, resolve_protocol_path

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", ["survey-v3.2"])
def test_basic_supported_protocols_preserve_accepted_policy_items(version):
    protocol = load_protocol(resolve_protocol_path(version))
    items = {item["item_id"]: item["status"] for item in protocol["review_items"]}
    assert items["v2_p4_clean_support_policy"] == "accepted"
    assert items["v2_p10_self_pu_scope"] == "accepted"
    assert protocol["formal_blockers"]


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
    changed["review_items"][3]["status"] = "pending_collaborator_review"
    assert digest(changed) != before
    assert digest(json.loads(json.dumps(protocol))) == before
