"""P3 technical adapters can consume the shared frozen-CNN feature path.

This is a synthetic interface smoke, not a CIFAR benchmark or native-CNN
claim.  The image encoder is deliberately tiny; the shared adapter owns the
four-way role provenance and every estimator receives the same 2-D features.
"""

# ruff: noqa: N803, N806

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pu_toolbox.estimators.deep.grad_pu import GradPUClassifier  # noqa: E402
from pu_toolbox.estimators.deep.lagam import LaGAMClassifier  # noqa: E402
from pu_toolbox.estimators.deep.robust_pu import RobustPUClassifier  # noqa: E402
from pu_toolbox.estimators.deep.split_pu import SplitPUClassifier  # noqa: E402
from pu_toolbox.estimators.risk.cvir import CVIRClassifier  # noqa: E402
from pu_toolbox.estimators.risk.puet import PUExtraTreesClassifier  # noqa: E402
from pu_toolbox.estimators.risk.pulda import PULDAClassifier  # noqa: E402
from pu_toolbox.estimators.risk.vpu import VPUClassifier  # noqa: E402
from pu_toolbox.experiment import (  # noqa: E402
    DatasetBundle,
    DatasetPart,
    adapt_image_bundle_to_features,
)

pytestmark = pytest.mark.unit


def _features():
    rng = np.random.default_rng(22)
    parts = {}
    start = 0
    for role, size in (("train", 16), ("pu_val", 8), ("clean_val", 8), ("test", 8)):
        labels = np.tile([1, 0], size // 2)
        images = rng.normal(size=(size, 1, 4, 4)).astype(np.float32)
        images[labels == 1] += 0.25
        parts[role] = DatasetPart(
            X=images,
            labels=labels,
            view="clean",
            indices=np.arange(start, start + size),
            for_selection=role != "test",
        )
        start += size
    torch.manual_seed(5)
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 4),
    )
    return adapt_image_bundle_to_features(
        DatasetBundle(**parts),
        encoder,
        feature_version="p3-synthetic-smoke-v1",
        backbone_manifest={"name": "tiny-test-encoder", "claim": "interface-only"},
        batch_size=5,
    )


@pytest.mark.parametrize(
    "name,construct,view",
    [
        ("vpu", lambda: VPUClassifier(max_epochs=1, batch_size=8, random_state=3), "ts"),
        (
            "pulda",
            lambda: PULDAClassifier(
                class_prior=0.4,
                warmup_epochs=1,
                pu_epochs=1,
                positive_batch_size=4,
                unlabeled_batch_size=8,
                random_state=3,
            ),
            "ts",
        ),
        (
            "puet",
            lambda: PUExtraTreesClassifier(class_prior=0.4, n_estimators=2, random_state=3),
            "ts",
        ),
        ("gradpu", lambda: GradPUClassifier(max_epochs=1, batch_size=8, random_state=3), "ts"),
        (
            "robust_pu",
            lambda: RobustPUClassifier(
                class_prior=0.4, pretrain_epochs=1, episodes=1, batch_size=8, random_state=3
            ),
            "ts",
        ),
        (
            "split_pu",
            lambda: SplitPUClassifier(
                class_prior=0.4,
                teacher_epochs=1,
                split_epochs=1,
                student_epochs=1,
                rounds=1,
                batch_size=8,
                random_state=3,
            ),
            "ts",
        ),
        (
            "cvir",
            lambda: CVIRClassifier(
                unlabeled_positive_prior=0.25,
                warm_start_epochs=1,
                max_epochs=1,
                batch_size=8,
                random_state=3,
            ),
            "os",
        ),
        (
            "lagam",
            lambda: LaGAMClassifier(
                warmup_epochs=1,
                max_epochs=2,
                batch_size=8,
                num_clusters=2,
                random_state=3,
            ),
            "os",
        ),
    ],
)
def test_shared_image_feature_path_is_2d_and_fits(name, construct, view):
    adapted, manifest = _features()
    assert manifest["training_path"] == "cnn_feature_adapter"
    assert manifest["encoder_mode"] == "eval_no_grad"
    assert adapted.train.X.shape == (16, 4)
    np.testing.assert_array_equal(adapted.train.indices, np.arange(16))
    # The training label view is explicitly PU and never taken from test.
    y_pu = np.zeros(len(adapted.train.X), dtype=int)
    y_pu[np.flatnonzero(adapted.train.labels == 1)[:4]] = 1
    model = construct()
    kwargs = {}
    if name == "lagam":
        kwargs["support_data"] = (adapted.clean_val.X, adapted.clean_val.labels)
    elif name != "cvir":
        kwargs["os_or_ts"] = view
    model.fit(adapted.train.X, y_pu, **kwargs)
    scores = model.decision_function(adapted.test.X)
    assert scores.shape == (len(adapted.test.X),)
    assert np.isfinite(scores).all()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "construct",
    [
        lambda: VPUClassifier(max_epochs=1, batch_size=8, random_state=3, device="cuda"),
        lambda: PULDAClassifier(
            class_prior=0.4,
            warmup_epochs=1,
            pu_epochs=1,
            positive_batch_size=4,
            unlabeled_batch_size=8,
            random_state=3,
            device="cuda",
        ),
        lambda: GradPUClassifier(max_epochs=1, batch_size=8, random_state=3, device="cuda"),
        lambda: RobustPUClassifier(
            class_prior=0.4,
            pretrain_epochs=1,
            episodes=1,
            batch_size=8,
            random_state=3,
            device="cuda",
        ),
        lambda: SplitPUClassifier(
            class_prior=0.4,
            teacher_epochs=1,
            split_epochs=1,
            student_epochs=1,
            rounds=1,
            batch_size=8,
            random_state=3,
            device="cuda",
        ),
    ],
)
def test_calibrated_feature_path_uses_cuda(construct):
    if not torch.cuda.is_available():
        if os.environ.get("PU_REQUIRE_CUDA") == "1":
            pytest.fail("PU_REQUIRE_CUDA=1 requires CUDA")
        pytest.skip("CUDA unavailable")
    adapted, _ = _features()
    y_pu = np.zeros(len(adapted.train.X), dtype=int)
    y_pu[np.flatnonzero(adapted.train.labels == 1)[:4]] = 1
    model = construct().fit(adapted.train.X, y_pu, os_or_ts="ts")
    assert model.calibration_applied_
    assert next(model.model_.parameters()).device.type == "cuda"
    assert np.isfinite(model.decision_function(adapted.test.X)).all()
