# tests/unit/utils/test_serialization_hashes.py

"""``canonical_hash`` and ``strict_canonical_hash`` are the two JSON digests.

They differ in exactly one parameter: ``allow_nan``.  On finite payloads they
are the same function, which is what lets the strict one stand in for the
copies that were scattered across the experiment layer.  They are nonetheless
two contracts -- lenient, because a report payload has to be writable either
way; strict, because these digests are artifact identities and a value that
cannot round-trip must not be silently hashed into one -- so the divergence is
pinned here rather than left to whoever next thinks to "unify" them.
"""

import pytest

from pu_toolbox.utils.serialization import canonical_hash, strict_canonical_hash

pytestmark = pytest.mark.unit


def test_basic_strict_hash_agrees_with_canonical_hash_on_finite_payloads():
    """The one-parameter difference is invisible until a payload is non-finite."""
    payloads = [
        {"b": 1, "a": [1, 2, 3]},
        {"nested": {"deep": {"list": [True, None, "x"]}}},
        [1, 2, 3],
        [],
        {"text": "中文语料", "emoji": "🚀"},
        {"float": 1.5, "int": 7, "bool": True, "none": None},
    ]
    for payload in payloads:
        assert strict_canonical_hash(payload) == canonical_hash(payload)


def test_param_strict_hash_refuses_every_non_finite_number():
    """NaN and both infinities raise rather than being written into a digest."""
    for not_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            strict_canonical_hash({"value": not_finite})
        with pytest.raises(ValueError):
            strict_canonical_hash([not_finite])


def test_edge_key_order_does_not_move_either_digest():
    first = {"b": [1, 2], "a": {"y": 1, "x": 2}}
    reordered = {"a": {"x": 2, "y": 1}, "b": [1, 2]}
    assert strict_canonical_hash(first) == strict_canonical_hash(reordered)
    assert canonical_hash(first) == canonical_hash(reordered)


def test_edge_strict_hash_accepts_the_non_mapping_payloads_callers_pass():
    """``feature_adapter`` hashes a list; the parameter is ``Any`` for that."""
    assert strict_canonical_hash([1, 2, 3]) == canonical_hash([1, 2, 3])
    assert strict_canonical_hash([1, 2, 3]) != strict_canonical_hash([3, 2, 1])
    assert len(strict_canonical_hash([])) == 64


def test_determ_canonical_hash_still_writes_non_finite_numbers():
    """The lenient contract: a report payload has to be writable either way.

    This is deliberately the opposite of
    ``test_param_strict_hash_refuses_every_non_finite_number``.  If a future
    change makes these two agree on non-finite input, one of the two contracts
    was destroyed and this test says so.
    """
    digest = canonical_hash({"value": float("nan")})
    assert len(digest) == 64
    assert digest != canonical_hash({"value": float("inf")})


def test_param_the_delegating_public_names_carry_the_strict_contract():
    """The two public entry points must reach the strict helper, not the lenient one.

    ``survey_protocol.digest`` and ``survey_comparison.comparison_digest`` are
    the names the experiment layer and the survey scripts call, and both are
    supposed to be artifact-identity digests.  Every payload reachable from the
    frozen artifacts is finite, and on finite input the two helpers agree
    byte-for-byte -- so a value freeze cannot tell them apart.  This asserts
    the contract itself, which is the only way a future rerouting to the
    lenient helper gets caught.
    """
    from pu_toolbox.experiment.survey_comparison import comparison_digest
    from pu_toolbox.experiment.survey_protocol import digest as protocol_digest

    for not_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            protocol_digest({"value": not_finite})
        with pytest.raises(ValueError):
            comparison_digest({"value": not_finite})

    # The lenient helper keeps writing them -- that is its contract, not a bug.
    assert len(canonical_hash({"value": float("nan")})) == 64
