# tests/unit/utils/test_content_hashes.py

"""``array_hash`` and ``file_hash`` are the two binary-content digests.

The experiment layer hashed arrays and files in six places; these are the two
that survive.  The lists they feed are artifact identities -- ``feature_sha256``,
``train_data_sha256``, ``cache_sha256``, ``manifest_sha256`` -- so the encoding
is pinned here rather than left to whichever copy a future edit happens to find.
"""

import hashlib
import json
import pathlib

import numpy as np
import pytest

from pu_toolbox.utils.serialization import array_hash, file_hash

pytestmark = pytest.mark.unit


def test_basic_array_hash_covers_dtype_shape_and_bytes():
    """A shape-preserving corruption must still change the digest."""
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = hashlib.sha256()
    expected.update(str(values.dtype).encode())
    expected.update(b"[3, 4]")
    expected.update(values.tobytes())
    assert array_hash(values) == expected.hexdigest()

    reshaped = values.reshape(4, 3)
    assert array_hash(values) != array_hash(reshaped), "shape must be part of the digest"
    assert array_hash(values) != array_hash(values.astype(np.float64)), "dtype too"


def test_param_array_hash_matches_every_encoding_it_replaced():
    """The nine input classes the three predecessors were measured on."""
    rng = np.random.default_rng(0)
    cases = {
        "int64 1-D": np.arange(10, dtype=np.int64),
        "float32 2-D": np.arange(12, dtype=np.float32).reshape(3, 4),
        "float64 3-D": rng.normal(size=(2, 3, 4)),
        "bool": np.array([True, False, True]),
        "non-contiguous transpose": np.arange(12, dtype=np.float64).reshape(3, 4).T,
        "non-contiguous slice": np.arange(20, dtype=np.int32)[::2],
        "object": np.array([1, "a", None], dtype=object),
        "empty": np.array([], dtype=np.float32),
        "non-native byte order": np.arange(5, dtype=">i4"),
    }
    for label, values in cases.items():
        reference = hashlib.sha256()
        reference.update(str(values.dtype).encode())
        # ``json.dumps`` writes ``[3, 4]`` with a space; ``str(list(shape))`` does
        # not match it, which is why the shape goes through the same encoder.
        reference.update(json.dumps(values.shape).encode())
        reference.update(np.ascontiguousarray(values).tobytes())
        assert array_hash(values) == reference.hexdigest(), label


def test_edge_array_hash_reads_the_shape_of_the_callers_array():
    """A 0-d array hashes as ``[]``, not as ``[1]``.

    ``np.ascontiguousarray`` promotes a 0-d input to shape ``(1,)``, so reading
    the shape off the contiguous copy would silently rename the array.  The
    project ruling is that the digest describes the array the caller passed.
    The two ``_array_sha256`` copies this helper replaces behaved this way; the
    ``array_digest`` copy did not, and that is the behaviour this converges
    away.  No frozen artifact covers 0-d input, so this literal -- captured from
    the two ``_array_sha256`` implementations before the convergence -- is the
    only guard.
    """
    zero_d = np.array(3.5)
    assert zero_d.shape == ()
    assert np.ascontiguousarray(zero_d).shape == (1,), "the promotion this test guards against"
    assert array_hash(zero_d) == "cb4f6a2cea26f9fc45898e0924e00f299aa23f980b8ade1a62256326d9a6fdc8"


def test_edge_file_hash_is_chunk_boundary_independent():
    """Chunking is the only real risk in a streaming digest, so pin its edges.

    The payload must outgrow the chunk size, or every slice below is the same
    buffer and the multi-chunk path -- the one that can go wrong -- never runs.
    """
    boundary = 1024 * 1024
    values = bytes(range(256)) * 8192  # exactly 2 * boundary
    assert len(values) == 2 * boundary
    payloads = {
        "empty": b"",
        "one byte": b"x",
        "boundary-1": values[: boundary - 1],
        "boundary": values[:boundary],
        "boundary+1": values[: boundary + 1],
        "two boundaries": values[: 2 * boundary],
    }
    assert len({len(payload) for payload in payloads.values()}) == len(payloads), (
        "each case must be a distinct length, or the slices are not distinct"
    )

    tmp_path = pathlib.Path(__import__("tempfile").mkdtemp())
    try:
        for label, payload in payloads.items():
            path = tmp_path / f"{label.replace(' ', '_')}.bin"
            path.write_bytes(payload)
            assert file_hash(path) == hashlib.sha256(payload).hexdigest(), label
            assert file_hash(str(path)) == file_hash(path), f"{label}: str and Path agree"
    finally:
        for leftover in tmp_path.iterdir():
            leftover.unlink()
        tmp_path.rmdir()


def test_determ_hashes_are_pure_and_do_not_touch_their_inputs():
    values = np.arange(6, dtype=np.int64)
    original = values.copy()
    assert array_hash(values) == array_hash(values)
    assert np.array_equal(values, original), "the caller's array must not be written to"

    tmp_path = pathlib.Path(__import__("tempfile").mkdtemp())
    try:
        path = tmp_path / "payload.bin"
        path.write_bytes(b"content")
        before = path.read_bytes()
        assert file_hash(path) == file_hash(path)
        assert path.read_bytes() == before
    finally:
        (tmp_path / "payload.bin").unlink()
        tmp_path.rmdir()
