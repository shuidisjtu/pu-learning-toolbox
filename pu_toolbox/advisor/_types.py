"""Data types for algorithm recommendations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..registry.metadata import AlgorithmMetadata

__all__ = [
    "MethodCandidate",
    "RecommendationResult",
]


@dataclass(frozen=True)
class MethodCandidate:
    """A single recommended method with score and rationale."""

    name: str
    score: float
    rank: int
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: AlgorithmMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "rank": self.rank,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "family": str(self.metadata.family.value),
            "backend": str(self.metadata.backend.value),
            "maturity": str(self.metadata.maturity.value),
        }


@dataclass(frozen=True)
class RecommendationResult:
    """Ranked algorithm recommendations with filters and warnings."""

    candidates: tuple[MethodCandidate, ...]
    filters_applied: dict[str, Any]
    global_warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "candidates": [c.to_dict() for c in self.candidates],
            "filters_applied": self.filters_applied,
            "global_warnings": list(self.global_warnings),
            "provenance": self.provenance,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def to_markdown(self) -> str:
        lines = [
            "# PU Method Recommendations",
            "",
        ]
        if self.global_warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in self.global_warnings:
                lines.append(f"- {w}")
            lines.append("")
        lines.extend(
            [
                "## Recommended Methods",
                "",
                "| Rank | Method | Score | Family | Backend | Maturity |",
                "|---:|---|---:|---|---|---|",
            ]
        )
        for c in self.candidates:
            lines.append(
                f"| {c.rank} | {c.name} | {c.score:.1f} "
                f"| {c.metadata.family.value} | {c.metadata.backend.value} "
                f"| {c.metadata.maturity.value} |"
            )
        lines.append("")
        for c in self.candidates:
            lines.append(f"### {c.rank}. {c.name}")
            lines.append("")
            if c.reasons:
                for r in c.reasons:
                    lines.append(f"- {r}")
            if c.warnings:
                lines.append("")
                for w in c.warnings:
                    lines.append(f"- **Warning**: {w}")
            lines.append("")
        return "\n".join(lines)

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["json", "markdown"] | None = None,
    ) -> Path:
        destination = Path(path)
        fmt = format
        if fmt is None:
            suffix = destination.suffix.lower()
            if suffix == ".json":
                fmt = "json"
            elif suffix in {".md", ".markdown"}:
                fmt = "markdown"
            else:
                raise ValueError(
                    "Cannot infer report format. Use a .json/.md suffix or pass format=."
                )
        if fmt not in {"json", "markdown"}:
            raise ValueError(f"Unknown format {fmt!r}; expected 'json' or 'markdown'.")
        content = self.to_markdown() if fmt == "markdown" else self.to_json() + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination
