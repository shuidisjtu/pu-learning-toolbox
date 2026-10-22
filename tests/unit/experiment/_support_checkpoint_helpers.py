"""Small, stable CPU fixtures for the explicit support checkpoint variants."""

import numpy as np

from pu_toolbox.estimators.deep.lagam import LaGAMClassifier
from pu_toolbox.estimators.deep.self_pu import SelfPUClassifier
from pu_toolbox.experiment.bundle import DatasetPart
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPACleanSupport
from pu_toolbox.experiment.support_checkpoints import (
    LaGAMCleanSupportCheckpointTrainer,
    SelfPUCleanSupportCheckpointTrainer,
)


def inputs():
    rng = np.random.RandomState(7)
    features = rng.normal(size=(60, 3)).astype("float32")
    truth = np.tile([0, 1], 30)
    parts = tuple(
        DatasetPart(features[a:b].copy(), truth[a:b].copy(), "clean", np.arange(a, b), i != 3)
        for i, (a, b) in enumerate(((0, 30), (30, 40), (40, 50), (50, 60)))
    )
    support = DatasetPart(
        rng.normal(size=(8, 3)).astype("float32"),
        np.tile([0, 1], 4),
        "clean",
        np.arange(60, 68),
        False,
    )
    return parts, support


def model(method):
    common = dict(hidden_dim=4, batch_size=8, max_epochs=3, random_state=0, device="cpu")
    if method == "self_pu":
        return SelfPUClassifier(
            class_prior=0.5,
            warmup_epochs=0,
            self_paced_start=0,
            self_paced_end=1,
            distill_start=2,
            **common,
        )
    return LaGAMClassifier(warmup_epochs=1, num_clusters=2, **common)


def adapter(method, support):
    cls = (
        SelfPUCleanSupportCheckpointTrainer
        if method == "self_pu"
        else LaGAMCleanSupportCheckpointTrainer
    )
    return cls(
        support,
        dataset="spambase",
        source="synthetic independent clean support",
        prior_estimation_indices=[],
        original_train_count=30,
        support_removed_from_train=False,
        preparation_seed=7,
        sampling_rule="fixed external synthetic cohort",
        group_policy="not_applicable",
    )


def run(
    method, directory, *, parts=None, support=None, candidates=None, config=None, protocols=None
):
    if parts is None:
        parts, support = inputs()
    options = dict(
        c=0.5,
        trainer=adapter(method, support),
        candidates=[{}] if candidates is None else candidates,
        checkpoint_bytes_per_component=65536,
        reclaim_unselected_checkpoints=True,
    )
    options.update(config or {})
    return ExperimentRunner(
        seed=0,
        class_prior=0.5,
        protocols=[ProtocolPACleanSupport(), ProtocolOA()] if protocols is None else protocols,
        manifest_path=str(directory / "manifest.json"),
        config=options,
    ).fit(model(method), *parts)
