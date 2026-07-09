"""Source policy metadata for external code integration.

Defines :class:`SourcePolicy` — a value object that describes the
provenance, license, and integration strategy for a paper's author code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core.tags import SourceStatus


@dataclass
class SourcePolicy:
    """Describes the provenance and integration strategy for external code.

    Each algorithm that has author / official source code registers a
    ``SourcePolicy`` alongside its :class:`~.metadata.AlgorithmMetadata`.
    The registry and advisor use this information to decide whether to
    prefer an adapter over a native implementation, and to surface
    license / dependency caveats to the user.
    """

    source_status: SourceStatus = SourceStatus.UNKNOWN
    """Classification of the external code's relationship to the paper."""

    upstream_url: str | None = None
    """URL to the author's repository, code package, or homepage."""

    license: str | None = None
    """SPDX identifier (``"MIT"``, ``"BSD"``, ``"GPL"``, etc.),
    ``"needs_review"``, or ``None`` if not yet investigated."""

    integration_mode: Literal[
        "adapter",
        "native_reimplementation",
        "adapter_then_native_reimplementation",
        "reference_only",
    ] = "adapter"
    """Recommended integration strategy for the toolbox."""

    validation_target: str | None = None
    """Description of the reproduction target (e.g. default experiment
    from the paper) used for alignment testing."""

    def is_available(self) -> bool:
        """Return True if the upstream source is known and reachable."""
        return (
            self.source_status not in (SourceStatus.NOT_FOUND, SourceStatus.UNKNOWN)
            and self.upstream_url is not None
        )

    def can_redistribute(self) -> bool:
        """Heuristic for whether the source can be redistributed.

        Returns True for permissive licenses (MIT, BSD, Apache).
        GPL, proprietary, and unknown licenses return False.
        """
        if self.license is None:
            return False
        permissive = {"MIT", "BSD", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0"}
        return self.license in permissive
