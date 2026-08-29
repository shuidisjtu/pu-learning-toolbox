# ruff: noqa: N802, N803, N806
"""Unit tests for check_architecture_capability (signature vs capability)."""

from __future__ import annotations

import numpy as np
import pytest

from pu_toolbox.core.base import BasePUClassifier
from pu_toolbox.workflows._errors import PipelineError
from pu_toolbox.workflows._models import check_architecture_capability


class _Capable(BasePUClassifier):
    native_architectures = frozenset({"cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True

    def __init__(self, *, encoder=None):
        self.encoder = encoder

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


class _SigYesCapNo(BasePUClassifier):
    """Signature declares encoder but capability metadata does not (drift)."""

    def __init__(self, *, encoder=None):
        self.encoder = encoder

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


class _SigNoCapYes(BasePUClassifier):
    """Capability claims cnn but signature has no encoder param (drift)."""

    native_architectures = frozenset({"cnn"})
    input_ndims = frozenset({2, 4})
    encoder_parameter = "encoder"
    trains_encoder = True

    def fit(self, X, y_pu, *, class_prior=None, sample_weight=None):
        return self

    def _predict(self, X):
        return np.zeros(len(X))

    def _decision_function(self, X):
        return np.zeros(len(X))


@pytest.mark.unit
def test_capable_class_passes_cnn_check():
    check_architecture_capability(_Capable, "cnn", "fake")


@pytest.mark.unit
def test_signature_yes_capability_no_raises():
    with pytest.raises(PipelineError, match="mismatch"):
        check_architecture_capability(_SigYesCapNo, "cnn", "fake")


@pytest.mark.unit
def test_signature_no_capability_yes_raises():
    with pytest.raises(PipelineError, match="mismatch"):
        check_architecture_capability(_SigNoCapYes, "cnn", "fake")


@pytest.mark.unit
def test_mlp_architecture_never_checked():
    check_architecture_capability(_SigYesCapNo, "mlp", "fake")
    check_architecture_capability(_SigNoCapYes, "mlp", "fake")
