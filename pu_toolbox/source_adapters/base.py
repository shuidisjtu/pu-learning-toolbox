"""Base class for source adapters that wrap third-party / official author code.

Adapters sit behind the unified estimator API and handle dependency
checks, input conversion, seed propagation, log capture, and output
normalisation so that the toolbox user never interacts with raw
third-party code directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.base import BasePUClassifier
from ..core.tags import Backend, ImplementationStatus, SourceStatus


class BaseSourceAdapter(ABC):
    """Abstract base class for wrapping author/official source code.

    Adapters sit behind the unified estimator API and handle dependency
    checks, input conversion, seed propagation, log capture, and output
    normalisation so that the toolbox user never interacts with raw
    third-party code directly.
    """

    source_status: SourceStatus = SourceStatus.UNKNOWN
    upstream_url: str | None = None
    license: str = "unknown"
    backend: Backend = Backend.UNKNOWN
    implementation_status: ImplementationStatus = ImplementationStatus.OFFICIAL_ADAPTER

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the external source / dependency is usable.

        Returns ``True`` only if all required packages, files, and
        licences are satisfied.
        """
        ...

    @abstractmethod
    def build_estimator(self, **kwargs) -> BasePUClassifier:
        """Build a PU classifier wrapping the external source.

        Parameters
        ----------
        **kwargs
            Forwarded to the underlying implementation constructor.

        Returns
        -------
        BasePUClassifier
        """
        ...

    def run_reproduction_test(self, config: dict | None = None) -> dict:
        """Run a paper-like reproduction test (if supported).

        Parameters
        ----------
        config : dict or None
            Benchmark configuration (dataset, seed, metrics, etc.).

        Returns
        -------
        dict
            Dictionary of metric-name → value.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement run_reproduction_test()."
        )
