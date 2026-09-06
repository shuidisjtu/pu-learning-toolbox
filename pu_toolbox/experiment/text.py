"""Fixed SBERT preprocessing and content-addressed embedding cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_EMBEDDING_DIMENSION = 384
_CACHE_SCHEMA_VERSION = "1.0"


class _TextEncoder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


def encode_survey_texts(
    texts: Sequence[str],
    *,
    cache_dir: str | Path,
    revision: str,
    batch_size: int = 32,
    normalize_embeddings: bool = False,
    encoder: _TextEncoder | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode survey text with the protocol-locked SBERT model.

    ``revision`` is mandatory so every artifact identifies the exact model
    revision used. The cache key binds that revision, every input text and
    the normalization setting. The returned manifest includes the cached
    ``.npy`` SHA-256 and whether this invocation was a cache hit.

    ``encoder`` accepts a preloaded compatible object. It primarily supports
    offline tests and managed environments; outputs still must be finite
    float arrays with the locked 384-dimensional shape.
    """
    text_values = _validate_inputs(texts, revision=revision, batch_size=batch_size)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    texts_sha256 = _json_sha256(text_values)
    cache_inputs = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "model_name": SBERT_MODEL_NAME,
        "revision": revision,
        "normalize_embeddings": bool(normalize_embeddings),
        "texts_sha256": texts_sha256,
    }
    cache_key = _json_sha256(cache_inputs)
    embedding_path = cache_root / f"{cache_key}.npy"
    metadata_path = cache_root / f"{cache_key}.json"

    if embedding_path.exists() or metadata_path.exists():
        if not embedding_path.is_file() or not metadata_path.is_file():
            raise ValueError(
                f"incomplete SBERT cache entry {cache_key}; remove both cache files and retry."
            )
        metadata = _load_metadata(metadata_path)
        expected = {
            **cache_inputs,
            "cache_key": cache_key,
            "text_count": len(text_values),
            "embedding_dimension": SBERT_EMBEDDING_DIMENSION,
            "dtype": "float32",
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(
                f"SBERT cache metadata mismatch for {cache_key}: {', '.join(mismatches)}."
            )
        actual_sha256 = _file_sha256(embedding_path)
        if metadata.get("cache_sha256") != actual_sha256:
            raise ValueError(f"SBERT cache checksum mismatch for {cache_key}.")
        try:
            embeddings = np.load(embedding_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load SBERT cache entry {cache_key}.") from exc
        embeddings = _validated_embeddings(embeddings, len(text_values))
        result_manifest = dict(metadata)
        result_manifest["cache_hit"] = True
        return embeddings, result_manifest

    resolved_encoder, backend = _resolve_encoder(encoder, revision)
    try:
        raw_embeddings = resolved_encoder.encode(
            text_values,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
    except Exception as exc:
        raise RuntimeError(f"SBERT encoding failed for revision {revision!r}: {exc}") from exc
    embeddings = _validated_embeddings(raw_embeddings, len(text_values))
    _atomic_save_array(embedding_path, embeddings)

    metadata = {
        **cache_inputs,
        "cache_key": cache_key,
        "cache_file": embedding_path.name,
        "cache_sha256": _file_sha256(embedding_path),
        "metadata_file": metadata_path.name,
        "text_count": len(text_values),
        "embedding_dimension": SBERT_EMBEDDING_DIMENSION,
        "dtype": "float32",
        "encoder_backend": backend,
        "cache_hit": False,
    }
    _atomic_write_json(metadata_path, metadata)
    return embeddings, dict(metadata)


def _validate_inputs(texts: Sequence[str], *, revision: str, batch_size: int) -> tuple[str, ...]:
    if isinstance(texts, str | bytes):
        raise TypeError("texts must be a sequence of strings, not one string.")
    text_values = tuple(texts)
    if not text_values:
        raise ValueError("texts must not be empty.")
    if any(not isinstance(value, str) for value in text_values):
        raise TypeError("every text value must be a string.")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty model revision or commit hash.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    return text_values


def _resolve_encoder(encoder: _TextEncoder | None, revision: str) -> tuple[_TextEncoder, str]:
    if encoder is not None:
        if not callable(getattr(encoder, "encode", None)):
            raise TypeError("encoder must provide a callable encode method.")
        encoder_type = type(encoder)
        return encoder, f"injected:{encoder_type.__module__}.{encoder_type.__qualname__}"
    try:
        from sentence_transformers import SentenceTransformer
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "SBERT preprocessing requires compatible optional dependencies; install "
            "'pu-toolbox[text]' in a clean environment."
        ) from exc
    try:
        return SentenceTransformer(SBERT_MODEL_NAME, revision=revision), "sentence-transformers"
    except Exception as exc:
        raise RuntimeError(
            f"cannot load {SBERT_MODEL_NAME!r} at revision {revision!r}: {exc}"
        ) from exc


def _validated_embeddings(values: Any, text_count: int) -> np.ndarray:
    embeddings = np.asarray(values)
    expected_shape = (text_count, SBERT_EMBEDDING_DIMENSION)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"SBERT embeddings must have shape {expected_shape}; got {embeddings.shape}."
        )
    if not np.issubdtype(embeddings.dtype, np.number):
        raise ValueError("SBERT embeddings must be numeric.")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("SBERT embeddings must contain only finite values.")
    return embeddings


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load SBERT cache metadata {path.name}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"SBERT cache metadata {path.name} must be a JSON object.")
    return value


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as output:
            np.save(output, values, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
