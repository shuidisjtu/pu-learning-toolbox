"""Source metadata — provenance facts for external code (status, url, license).

This module provides :class:`SourceMetadata` as the canonical name for
source-provenance metadata.  The underlying implementation lives in
:class:`~.source_policy.SourcePolicy`; ``SourceMetadata`` is a
user-facing alias that matches the documented registry schema.
"""

from .source_policy import SourcePolicy as SourceMetadata

__all__ = ["SourceMetadata"]
