import json

import numpy as np
import pytest

from pu_toolbox.experiment import (
    SBERT_EMBEDDING_DIMENSION,
    SBERT_MODEL_NAME,
    encode_survey_texts,
)

pytestmark = pytest.mark.unit


class FakeEncoder:
    def __init__(self, *, fill=1.0, shape=None):
        self.fill = fill
        self.shape = shape
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((tuple(texts), kwargs))
        shape = self.shape or (len(texts), SBERT_EMBEDDING_DIMENSION)
        return np.full(shape, self.fill, dtype=np.float64)


def test_basic_encodes_and_records_content_addressed_cache(tmp_path):
    encoder = FakeEncoder(fill=2.5)
    embeddings, manifest = encode_survey_texts(
        ["positive review", "negative review"],
        cache_dir=tmp_path,
        revision="abc123",
        batch_size=8,
        normalize_embeddings=True,
        encoder=encoder,
    )

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32
    assert manifest["model_name"] == SBERT_MODEL_NAME
    assert manifest["revision"] == "abc123"
    assert manifest["embedding_dimension"] == 384
    assert manifest["cache_hit"] is False
    assert len(manifest["cache_sha256"]) == 64
    assert (tmp_path / manifest["cache_file"]).is_file()
    assert (tmp_path / manifest["metadata_file"]).is_file()
    assert encoder.calls[0][1]["normalize_embeddings"] is True


def test_determ_second_call_uses_verified_cache_without_encoder(tmp_path):
    texts = ["same", "inputs"]
    first, first_manifest = encode_survey_texts(
        texts,
        cache_dir=tmp_path,
        revision="locked-revision",
        encoder=FakeEncoder(fill=3.0),
    )

    second, second_manifest = encode_survey_texts(
        texts,
        cache_dir=tmp_path,
        revision="locked-revision",
        encoder=object(),
    )

    assert np.array_equal(first, second)
    assert first_manifest["cache_key"] == second_manifest["cache_key"]
    assert second_manifest["cache_hit"] is True


def test_param_content_revision_and_normalization_change_cache_key(tmp_path):
    keys = []
    for texts, revision, normalize in [
        (["a"], "r1", False),
        (["b"], "r1", False),
        (["a"], "r2", False),
        (["a"], "r1", True),
    ]:
        _, manifest = encode_survey_texts(
            texts,
            cache_dir=tmp_path,
            revision=revision,
            normalize_embeddings=normalize,
            encoder=FakeEncoder(),
        )
        keys.append(manifest["cache_key"])

    assert len(set(keys)) == 4


def test_edge_rejects_invalid_inputs_and_invalid_embeddings(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        encode_survey_texts(["x"], cache_dir=tmp_path, revision="", encoder=FakeEncoder())
    with pytest.raises(TypeError, match="sequence"):
        encode_survey_texts("x", cache_dir=tmp_path, revision="r", encoder=FakeEncoder())
    with pytest.raises(ValueError, match="shape"):
        encode_survey_texts(
            ["x"],
            cache_dir=tmp_path,
            revision="wrong-shape",
            encoder=FakeEncoder(shape=(1, 10)),
        )
    with pytest.raises(ValueError, match="finite"):
        encode_survey_texts(
            ["x"],
            cache_dir=tmp_path,
            revision="non-finite",
            encoder=FakeEncoder(fill=np.nan),
        )


def test_edge_detects_cache_tampering(tmp_path):
    _, manifest = encode_survey_texts(
        ["x"], cache_dir=tmp_path, revision="r", encoder=FakeEncoder()
    )
    cache_path = tmp_path / manifest["cache_file"]
    cache_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        encode_survey_texts(["x"], cache_dir=tmp_path, revision="r", encoder=FakeEncoder())


def test_edge_detects_metadata_tampering(tmp_path):
    _, manifest = encode_survey_texts(
        ["x"], cache_dir=tmp_path, revision="r", encoder=FakeEncoder()
    )
    metadata_path = tmp_path / manifest["metadata_file"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["embedding_dimension"] = 12
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata mismatch"):
        encode_survey_texts(["x"], cache_dir=tmp_path, revision="r", encoder=FakeEncoder())
