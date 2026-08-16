# ruff: noqa: N803, S101

"""CLI --save-model regression for CNN deep classifiers (E2 WConPU / E3 InfoMax PU)."""

import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.cli import main  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# (classifier, extra CLI args for short training)
_DEEP_CASES = [
    ("wconpu", ["--max-epochs", "1"]),
    (
        "infomax_pu",
        [
            "--classifier-param",
            "representation_epochs=2",
            "--classifier-param",
            "classifier_epochs=2",
        ],
    ),
]


@pytest.mark.parametrize("classifier, extra_args", _DEEP_CASES)
def test_cli_save_model_cnn_roundtrip(tmp_path: Path, classifier: str, extra_args: list[str]):
    # 4-D NCHW image array (header-less .npy, the CLI CNN path).
    rng = np.random.RandomState(42)
    x = rng.randn(32, 3, 16, 16).astype("float32")
    # Fixed label construction: 9 positives + 23 unlabeled. 32 samples, 2 CV
    # folds -> every fold keeps labeled positives (PUStratifiedKFold).
    y_pu = np.zeros(32, dtype=int)
    y_pu[:9] = 1
    data = tmp_path / "images.npy"
    labels = tmp_path / "y_pu.csv"
    out = tmp_path / "results"
    np.save(data, x)
    np.savetxt(labels, y_pu, fmt="%d", header="label", comments="")

    # main() returns None on success; user/runtime errors raise SystemExit
    # (sys.exit(1)) or argparse SystemExit(2), which fail the test itself.
    main(
        [
            "run",
            "--data",
            str(data),
            "--labels",
            str(labels),
            "--out-dir",
            str(out),
            "--classifier",
            classifier,
            "--architecture",
            "cnn",
            "--backbone",
            "cnn13",
            "--device",
            "cpu",
            "--cv",
            "2",
            "--class-prior",
            "0.3",
            "--save-model",
            *extra_args,
        ]
    )
    model_file = out / "model.pkl"
    assert model_file.exists()
    model = pickle.loads(model_file.read_bytes())
    pred = model.predict(x[:2])
    assert pred.shape == (2,)
