"""Algorithm metadata dataclass and validation.

Defines the canonical :class:`AlgorithmMetadata` used by the registry
and documentation generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

from ..core.tags import (
    AlgorithmFamily,
    Assumption,
    Backend,
    ImplementationStatus,
    Maturity,
    Scenario,
    SourceStatus,
)


@dataclass
class AlgorithmMetadata:
    """Canonical metadata for a single PU algorithm registered in the toolbox.

    Every field maps directly to the registry schema defined in
    ``docs/architecture.md`` §8 and ``docs/method_selection.md`` §9.
    """

    name: str
    """Unique algorithm name (snake_case, e.g. ``"elkan_noto"``)."""

    aliases: list[str] = field(default_factory=list)
    """Alternative names users might type (e.g. ``["elkan-noto", "EN"]``)."""

    family: AlgorithmFamily = AlgorithmFamily.CLASSIC_CALIBRATION
    """Algorithm family for grouping and recommendation."""

    paper: str = ""
    """Paper title or citation string."""

    scenario: list[Scenario] = field(default_factory=lambda: [Scenario.UNKNOWN])
    """Applicable data collection scenarios."""

    assumption: list[Assumption] = field(default_factory=lambda: [Assumption.UNKNOWN])
    """Labeling-mechanism assumptions the algorithm requires or expects."""

    requires_class_prior: bool = False
    """Whether the algorithm needs π = P(y=1) at training time."""

    supports_sparse: bool = False
    """Whether the algorithm accepts scipy sparse matrices."""

    supports_gpu: bool = False
    """Whether the algorithm can leverage a GPU."""

    backend: Backend = Backend.NUMPY
    """Primary computational backend."""

    maturity: Maturity = Maturity.STABLE
    """Implementation stability level."""

    implementation_status: ImplementationStatus = ImplementationStatus.API_ONLY
    """How the algorithm is implemented in the toolbox."""

    source_status: SourceStatus = SourceStatus.UNKNOWN
    """Availability of author/official source code."""

    upstream_url: str | None = None
    """URL to the author's repository or code package."""

    license: str | None = None
    """SPDX license identifier (if known) or ``"needs_review"``."""

    @property
    def trainable(self) -> bool:
        """``True`` when the algorithm has a working implementation.

        Derived from :attr:`implementation_status` — safe to query after
        mutating ``implementation_status`` in-place.
        """
        return self.implementation_status not in (ImplementationStatus.API_ONLY,)

    def __post_init__(self) -> None:
        _validate_metadata(self)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON / YAML export."""
        result: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (list, tuple)):
                result[f.name] = [v.value if isinstance(v, Enum) else v for v in value]
            elif isinstance(value, Enum):
                result[f.name] = value.value
            else:
                result[f.name] = value
        # Include derived property
        result["trainable"] = self.trainable
        return result


def _validate_metadata(meta: AlgorithmMetadata) -> None:
    """Validate required fields on AlgorithmMetadata (called by __post_init__)."""
    if not meta.name or not meta.name.strip():
        raise ValueError("AlgorithmMetadata.name must be a non-empty string")
    if not meta.paper:
        raise ValueError(f"AlgorithmMetadata.paper must be a non-empty string (got {meta.name!r})")
