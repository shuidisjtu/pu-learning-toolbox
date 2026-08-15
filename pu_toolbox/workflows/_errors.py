"""Workflow exception types shared by internal orchestration modules."""

from ..core.exceptions import PULearningError


class PipelineError(PULearningError):
    """Workflow orchestration failed (unresolvable classifier/prior, etc.)."""
