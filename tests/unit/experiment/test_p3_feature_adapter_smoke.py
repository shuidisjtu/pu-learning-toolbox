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

from pu_toolbox.estimators.classic.rank_pruning import RankPruningClassifier  # noqa: E402
from pu_toolbox.estimators.deep.gen_pu import GenPUClassifier  # noqa: E402
from pu_toolbox.estimators.deep.grad_pu import GradPUClassifier  # noqa: E402
from pu_toolbox.estimators.deep.holistic_pu import HolisticPUClassifier  # noqa: E402
from pu_toolbox.estimators.deep.lagam import LaGAMClassifier  # noqa: E402
from pu_toolbox.estimators.deep.pan import PANClassifier  # noqa: E402
from pu_toolbox.estimators.deep.pulns import PULNSClassifier  # noqa: E402
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


def _features(*, independent_support=False):
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
    adapted = adapt_image_bundle_to_features(
        DatasetBundle(**parts),
        encoder,
        feature_version="p3-synthetic-smoke-v1",
        backbone_manifest={"name": "tiny-test-encoder", "claim": "interface-only"},
        batch_size=5,
    )
    if not independent_support:
        return adapted
    # Fifth role: no selection/test rows or labels are reused for RL rewards.
    support_images = rng.normal(size=(8, 1, 4, 4)).astype(np.float32)
    support_labels = np.tile([1, 0], 4)
    support_images[support_labels == 1] += 0.25
    encoder.eval()
    with torch.no_grad():
        support_features = encoder(torch.from_numpy(support_images)).numpy()
    return (*adapted, (support_features, support_labels), np.arange(start, start + 8))


@pytest.mark.parametrize(
    "name,construct,view",
    [
        ("vpu", lambda: VPUClassifier(max_epochs=1, batch_size=8, random_state=3), "ts"),
        ("pan", lambda: PANClassifier(max_epochs=1, batch_size=8, random_state=3), "os"),
        ("rp", lambda: RankPruningClassifier(n_cv_folds=2, random_state=3), "os"),
        (
            "genpu",
            lambda: GenPUClassifier(
                class_prior=0.4,
                hidden_dim=4,
                latent_dim=3,
                max_epochs=1,
                classifier_epochs=1,
                batch_size=8,
                random_state=3,
            ),
            "ts",
        ),
        (
            "holistic_pu",
            lambda: HolisticPUClassifier(
                hidden_dim=4,
                warmup_epochs=3,
                max_epochs=1,
                batch_size=8,
                random_state=3,
            ),
            "os",
        ),
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
def test_determ_shared_image_feature_path_is_2d_and_fits(name, construct, view):
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
    if name == "pan":
        repeated_bundle, repeated_manifest = _features()
        assert repeated_manifest == manifest
        for role in ("train", "pu_val", "clean_val", "test"):
            np.testing.assert_array_equal(
                getattr(adapted, role).X, getattr(repeated_bundle, role).X
            )
        repeated = construct().fit(repeated_bundle.train.X, y_pu, os_or_ts=view)
        np.testing.assert_array_equal(repeated.decision_function(repeated_bundle.test.X), scores)


def test_pulns_shared_features_require_a_separate_fifth_support_role():
    adapted, manifest, support, support_ids = _features(independent_support=True)
    for role in (adapted.train, adapted.pu_val, adapted.clean_val, adapted.test):
        assert not np.intersect1d(role.indices, support_ids).size
    y_pu = np.zeros(len(adapted.train.X), dtype=int)
    y_pu[np.flatnonzero(adapted.train.labels == 1)[:4]] = 1
    model = PULNSClassifier(
        hidden_dim=4,
        pretrain_epochs=1,
        episodes=1,
        classifier_epochs=1,
        batch_size=8,
        random_state=3,
    ).fit(
        adapted.train.X,
        y_pu,
        support_data=support,
        train_indices=adapted.train.indices,
        support_indices=support_ids,
    )
    assert manifest["encoder_mode"] == "eval_no_grad"
    assert np.isfinite(model.decision_function(adapted.test.X)).all()


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
