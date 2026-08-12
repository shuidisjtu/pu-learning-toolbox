"""Shared serialization helpers for report-like objects.

Every report type (``PipelineReport``, ``PUDiagnosticReport``,
``PUSensitivityAnalysis``, ``PUDataProfile``) renders strict JSON and
Markdown with the same conventions (NaN/Inf -> ``None``, ``unavailable``
for missing table cells, ``|`` escaping).  These helpers used to be
copied per module; they live here so the conventions stay in sync.

See ``docs/architecture.md`` §6 for the report serialization contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

__all__ = [
    "canonical_hash",
    "escape_markdown",
    "format_from_suffix",
    "format_value",
    "json_safe",
]

ReportFormat = Literal["json", "markdown", "csv"]


def canonical_hash(document: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable mapping."""
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: Any) -> Any:
    """Recursively convert values to strict JSON-safe types (NaN/Inf -> None)."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def format_value(value: Any) -> str:
    """Format a value for Markdown tables, ``unavailable`` for missing/non-finite."""
    if value is None:
        return "unavailable"
    try:
        if not np.isfinite(value):
            return "unavailable"
    except TypeError:
        return escape_markdown(str(value))
    return f"{float(value):.6f}"


def escape_markdown(value: str) -> str:
    """Escape Markdown-table-breaking characters in a cell value."""
    return value.replace("|", "\\|").replace("\n", " ")


def format_from_suffix(path: Path) -> ReportFormat:
    """Infer the output format from the file suffix (strict: unknown raises)."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".csv":
        return "csv"
    raise ValueError("Cannot infer report format. Use a .json/.md suffix or pass format=.")
