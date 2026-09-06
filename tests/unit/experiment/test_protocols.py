# tests/unit/experiment/test_protocols.py

# ruff: noqa: N803

import numpy as np

from pu_toolbox.experiment.protocols import Generator, SelectionProtocol, Trainer


def test_abc_classes_exist():
    assert Generator.__abstractmethods__
    assert Trainer.__abstractmethods__
    assert SelectionProtocol.__abstractmethods__


def test_generator_return_shape():
    class G(Generator):
        def generate(self, X, y_true, c, seed):
            return np.zeros(len(y_true), dtype=int), {"c_realized": c}

    y_pu, meta = G().generate(np.zeros((3, 2)), np.array([1, 0, 1]), 0.5, 1)
    assert y_pu.shape == (3,)
    assert meta["c_realized"] == 0.5
