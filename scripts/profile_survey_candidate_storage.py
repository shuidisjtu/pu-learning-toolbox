"""Synthetic candidate storage probes; not formal budgets or GPU resource audits.

Print strict JSON only; checkpoints live in a temporary directory and are
cleaned after verification. No user experiment artifacts are removed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np

METHODS = ("pulda", "puet", "pan", "rp", "pulns", "genpu", "holistic_pu")


def _epoch_storage(model, features, labels, **fit_kwargs):
    from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

    with tempfile.TemporaryDirectory(prefix="pu-storage-probe-") as root:
        trajectory = EpochCheckpointTrainer(checkpoint_dir=Path(root)).fit(
            model, features, labels, **fit_kwargs
        )
        sizes = [Path(snapshot.path).stat().st_size for snapshot in trajectory.checkpoints]
        np.testing.assert_allclose(
            trajectory.checkpoints[-1].restore(device="cpu").decision_function(features),
            model.decision_function(features),
            atol=1e-6,
        )
        return {
            "epoch_count": len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "total": sum(sizes),
            "roundtrip_verified": True,
        }


def profile_storage(method, *, seed=0, input_dim=3):
    if method not in METHODS:
        raise ValueError(f"storage probe supports {', '.join(METHODS)} only")
    if type(input_dim) is not int or input_dim < 1:
        raise ValueError("input_dim must be a positive integer")
    rng = np.random.RandomState(seed)
    labels = np.r_[np.ones(16, dtype=int), np.zeros(32, dtype=int)]
    features = rng.normal(size=(48, input_dim)).astype(np.float32)
    started = time.perf_counter()
    training_view = "os"
    support_roles = None
    if method == "pulda":
        from pu_toolbox.estimators.risk.pulda import PULDAClassifier
        from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer

        model = PULDAClassifier(
            class_prior=0.4,
            hidden_dim=4,
            depth=1,
            warmup_epochs=1,
            pu_epochs=1,
            positive_batch_size=8,
            unlabeled_batch_size=16,
            random_state=seed,
            device="cpu",
        )
        with tempfile.TemporaryDirectory(prefix="pu-storage-probe-") as root:
            trajectory = EpochCheckpointTrainer(checkpoint_dir=Path(root)).fit(
                model, features, labels
            )
            sizes = [Path(snapshot.path).stat().st_size for snapshot in trajectory.checkpoints]
            last = trajectory.checkpoints[-1].restore(device="cpu")
            np.testing.assert_allclose(
                last.decision_function(features), model.decision_function(features), atol=1e-6
            )
            snapshot_bytes = {
                "epoch_count": len(sizes),
                "min": min(sizes),
                "max": max(sizes),
                "total": sum(sizes),
                "roundtrip_verified": True,
            }
    elif method == "puet":
        from pu_toolbox.estimators.risk.puet import PUExtraTreesClassifier

        model = PUExtraTreesClassifier(
            class_prior=0.4, n_estimators=3, max_depth=3, random_state=seed
        ).fit(features, labels)
        snapshot_bytes = None  # Tree fitting is not an epoch trajectory.
    else:
        kwargs = {"random_state": seed}
        fit_kwargs = {}
        if method == "rp":
            from pu_toolbox.estimators.classic.rank_pruning import RankPruningClassifier

            model = RankPruningClassifier(n_cv_folds=2, **kwargs)
        else:
            kwargs.update(hidden_dim=4, batch_size=8, device="cpu")
            if method == "pan":
                from pu_toolbox.estimators.deep.pan import PANClassifier

                model = PANClassifier(max_epochs=2, **kwargs)
            elif method == "pulns":
                from pu_toolbox.estimators.deep.pulns import PULNSClassifier

                model = PULNSClassifier(
                    pretrain_epochs=1, episodes=2, classifier_epochs=1, **kwargs
                )
                fit_kwargs.update(
                    support_data=(
                        rng.normal(size=(8, input_dim)).astype(np.float32),
                        np.tile([0, 1], 4),
                    ),
                    train_indices=np.arange(48),
                    support_indices=np.arange(48, 56),
                )
                support_roles = {"rows": 8, "labels": "clean_pn", "disjoint_ids": True}
            elif method == "genpu":
                from pu_toolbox.estimators.deep.gen_pu import GenPUClassifier

                model = GenPUClassifier(
                    class_prior=0.4, latent_dim=2, max_epochs=1, classifier_epochs=1, **kwargs
                )
                training_view = "ts"
                fit_kwargs["os_or_ts"] = training_view
            else:
                from pu_toolbox.estimators.deep.holistic_pu import HolisticPUClassifier

                model = HolisticPUClassifier(warmup_epochs=3, max_epochs=1, **kwargs)
        if method in {"pan", "holistic_pu", "genpu"}:
            snapshot_bytes = _epoch_storage(model, features, labels, **fit_kwargs)
        else:
            model.fit(features, labels, **fit_kwargs)
            # These methods still have no per-epoch model selection hook.
            snapshot_bytes = None
    elapsed_seconds = time.perf_counter() - started
    serialized = pickle.dumps(model, protocol=5)
    restored = pickle.loads(serialized)  # Only freshly created, trusted in-memory bytes.
    np.testing.assert_allclose(
        restored.decision_function(features), model.decision_function(features), atol=1e-6
    )
    return {
        "method": method,
        "seed": seed,
        "input_dim": input_dim,
        "status": "synthetic_storage_probe_not_formal_admission",
        "device": "cpu",
        "training_view": training_view,
        "clean_support": support_roles,
        "elapsed_seconds_including_imports": elapsed_seconds,
        "optimizer_steps": getattr(model, "optimizer_steps_", None),
        "parameters": model.get_params(deep=False),
        "input_rows": len(labels),
        "epoch_weights_bytes": snapshot_bytes,
        "estimator_pickle_bytes": len(serialized),
        "pickle_roundtrip_verified": True,
        "peak_disk_budget_note": (
            "all epochs * components * candidates * retry attempts; not final survivors"
        ),
        "gpu_resource_audit": "not_applicable" if method in {"puet", "rp"} else "not_performed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=[*METHODS, "both", "all"], default="both")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=0)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--input-dim", type=int, default=3)
    args = parser.parse_args()
    methods = (
        list(METHODS)
        if args.method == "all"
        else ["pulda", "puet"]
        if args.method == "both"
        else [args.method]
    )
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": [
                    profile_storage(method, seed=seed, input_dim=args.input_dim)
                    for method in methods
                    for seed in seeds
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
