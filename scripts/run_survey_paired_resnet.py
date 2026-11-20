"""Run one audited, standalone two-arm ResNet study fit without changing Survey v3.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pu_toolbox.estimators.deep._encoder_policy import (
    is_policy_encoder,
    set_encoder_training_policy,
)
from pu_toolbox.experiment import DatasetBundle, ExperimentRunner
from pu_toolbox.experiment.checkpoints import EpochCheckpointTrainer, load_selected_checkpoint
from pu_toolbox.experiment.feature_adapter import encoder_state_sha256
from pu_toolbox.experiment.method_ledger import load_ledger
from pu_toolbox.experiment.resources import process_lifetime_peak_rss_bytes, runtime_environment
from pu_toolbox.experiment.runner import _binary_test_metrics
from pu_toolbox.experiment.strategies import (
    ProtocolOA,
    ProtocolPA,
    ProtocolPACleanSupport,
    SCARGenerator,
)
from pu_toolbox.experiment.support_checkpoints import SelfPUCleanSupportCheckpointTrainer
from pu_toolbox.experiment.support_data import load_support_dataset
from pu_toolbox.experiment.survey_execution import (
    prepare_image_bundle,
    prepare_image_support,
    score_model,
)
from pu_toolbox.experiment.survey_profile_identity import profile_training_view_request
from pu_toolbox.experiment.survey_staged_budget import verify_trajectory
from pu_toolbox.experiment.training_views import resolve_training_view
from pu_toolbox.prior import population_to_native_unlabeled
from pu_toolbox.prior._evidence import risk_pool_target
from pu_toolbox.registry import get_algorithm, get_metadata
from pu_toolbox.utils.serialization import file_hash, strict_canonical_hash
from scripts.run_survey_experiment import load_split_parts, resolve_split_ref

ROOT = Path(__file__).resolve().parents[1]
BASIS = ROOT / "pu_toolbox/experiment/survey_protocol_v3_2.json"
METHODS = ("robust_pu", "split_pu", "holistic_pu", "self_pu", "self_pu_full_meta", "dist_pu")
DATASETS = {
    "cifar10": "splits/cifar10",
    "fashion_mnist": "review-p1_4r-received-20261015-rndXxi/splits_four/fashion_mnist",
}
SUPPORT_ROOT = "clean-support-20261022-yg0avvvx"
STUDY = "paired-resnet18-scar-c03-v1"
C = 0.3
COMPONENT_LIMIT = 64 * 1024**2
DIST_PARAMS = {
    "training_recipe": "author_staged",
    "distribution_loss": "author_histogram",
    "warmup_epochs": 60,
    "pu_epochs": 60,
    "warmup_learning_rate": 5e-4,
    "warmup_weight_decay": 5e-3,
    "pu_learning_rate": 5e-5,
    "pu_weight_decay": 1e-3,
    "warmup_entropy_weight": 2e-3,
    "pu_entropy_weight": 4e-3,
    "mixup_alpha": 6.0,
    "mixup_entropy_weight": 4e-2,
    "mixup_bce_weight": 5.0,
}


def _json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _source_commit(full):
    command = ["git", "-C", str(ROOT)]
    commit = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        [*command, "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if full and dirty:
        raise ValueError("full-budget study requires committed clean source")
    return commit, bool(dirty)


def _load(args):
    root = args.data_root
    folder = root / DATASETS[args.dataset] / f"split_{args.seed}"
    ref = resolve_split_ref(folder, None)
    if ref.get("dataset") != args.dataset or ref.get("seed") != args.seed:
        raise ValueError("wrong source split seed or dataset")
    base = DatasetBundle(*load_split_parts(folder))
    for role in ("train", "pu_val", "clean_val", "test"):
        part = getattr(base, role)
        if len(part.labels) != len(part.X) or len(part.indices) != len(part.X):
            raise ValueError(f"invalid source {role} geometry")
        if ref["role_sizes"][role] != len(part.X):
            raise ValueError(f"invalid source {role} size receipt")
    source_paths = [folder / f"{role}.npz" for role in ("train", "pu_val", "clean_val", "test")]
    source_files = {p.name: file_hash(p) for p in source_paths}
    support = product = None
    if args.method == "self_pu_full_meta":
        sdir = root / SUPPORT_ROOT / args.dataset / f"split_{args.seed}"
        bundle, support, product = load_support_dataset(sdir)
        if len(support.X) != 500 or product.get("dataset") != args.dataset:
            raise ValueError("unexpected full-meta support data or label budget")
        if product["support_adapter_options"]["preparation_seed"] != args.seed:
            raise ValueError("clean-support seed differs from source split seed")
        for role in ("pu_val", "clean_val", "test"):
            left, right = getattr(base, role), getattr(bundle, role)
            for name in ("indices", "labels", "X"):
                if not np.array_equal(getattr(left, name), getattr(right, name)):
                    raise ValueError(f"clean-support product used a different {role}")
        order = np.argsort(base.train.indices)
        for part in (bundle.train, support):
            positions = order[np.searchsorted(base.train.indices[order], part.indices)]
            if not np.array_equal(base.train.indices[positions], part.indices):
                raise ValueError("support product does not use original train indices")
            if not (
                np.array_equal(base.train.labels[positions], part.labels)
                and np.array_equal(base.train.X[positions], part.X)
            ):
                raise ValueError("support product does not match original train data")
        if len(bundle.train.X) + len(support.X) != len(base.train.X):
            raise ValueError("support product does not partition the original train split")
        del base
    else:
        bundle = base
    if args.probe and support is None:
        # Deterministic tiny validation run from these inputs, never a grid result.
        def subset(part, limit):
            positions = np.sort(
                np.concatenate([np.flatnonzero(part.labels == value)[:limit] for value in (0, 1)])
            )
            return replace(
                part,
                X=part.X[positions],
                labels=part.labels[positions],
                indices=part.indices[positions],
            )

        bundle = DatasetBundle(
            subset(bundle.train, 40),
            subset(bundle.pu_val, 20),
            subset(bundle.clean_val, 20),
            subset(bundle.test, 20),
        )
    return bundle, ref, support, product, source_files


def _model(args, protocol, encoder, prior):
    import torch
    from torch import nn

    profile = protocol["method_profiles"][args.method]
    params = dict(profile["params"])
    name = profile["algorithm"]
    if name == "dist_pu":
        params.update(DIST_PARAMS)
    if args.probe:
        reductions = {
            "robust_pu": dict(pretrain_epochs=1, episodes=2),
            "split_pu": dict(teacher_epochs=1, split_epochs=1, student_epochs=1),
            "holistic_pu": dict(warmup_epochs=2, max_epochs=1),
            "self_pu": dict(
                warmup_epochs=0, self_paced_start=0, self_paced_end=2, max_epochs=2, distill_start=2
            ),
            "dist_pu": dict(warmup_epochs=1, pu_epochs=1),
        }
        params.update(reductions[name])
    cls = get_algorithm(name)
    params["random_state"] = args.seed
    params["device"] = args.device
    params.update(dict.fromkeys(get_metadata(name).required_prior_parameters, prior))
    arm = args.arm == "trainable"
    set_encoder_training_policy(encoder, trainable=arm)
    head_hash = None
    if name == "self_pu":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            head = score_model(512, "mlp128")
        head_hash = encoder_state_sha256(head)
        estimator = cls(backbone=nn.Sequential(encoder, head), **params)
    else:
        estimator = cls(encoder=encoder, **params)
    locked_params = {
        k: v
        for k, v in estimator.get_params(deep=False).items()
        if k not in ("encoder", "backbone", "model", "device")
    }
    return estimator, locked_params, head_hash


class EncoderAudit:
    """Check actual trained networks at every paid snapshot, not config flags."""

    def __init__(self, initial_encoder, arm):
        import torch

        self.baseline = encoder_state_sha256(initial_encoder)
        self.conv = next(m for m in initial_encoder.modules() if isinstance(m, torch.nn.Conv2d))
        self.initial_kernel = self.conv.weight.detach().cpu().clone()
        self.arm = arm
        self.records = []
        self.changed = set()
        self.latest_kernels = {}
        self.changed_by_stage = {}
        self.observed_stages = set()
        self.head_states = {}
        self.head_updates = set()

    def record(self, epoch, fitted):
        import torch

        names = (
            ("student_1_", "student_2_", "teacher_1_", "teacher_2_")
            if hasattr(fitted, "teacher_1_")
            else ("model_",)
        )
        stage = getattr(fitted, "checkpoint_stage_", "fit")
        self.observed_stages.add(stage)
        for name in names:
            model = getattr(fitted, name)
            encoders = [m for m in model.modules() if is_policy_encoder(m)]
            if len(encoders) != 1:
                raise ValueError(f"{name} did not preserve exactly one paired encoder")
            encoder = encoders[0]
            kernel = (
                next(m.weight for m in encoder.modules() if isinstance(m, torch.nn.Conv2d))
                .detach()
                .cpu()
            )
            previous = self.latest_kernels.get(name, self.initial_kernel)
            if not torch.equal(kernel, previous):
                self.changed_by_stage.setdefault(stage, set()).add(name)
            self.latest_kernels[name] = kernel.clone()
            moved = not torch.equal(kernel, self.initial_kernel)
            if self.arm == "frozen":
                if encoder_state_sha256(encoder) != self.baseline or any(
                    p.requires_grad for p in encoder.parameters()
                ):
                    raise ValueError("frozen encoder changed parameter or buffer state")
                if any(m.training for m in encoder.modules()):
                    raise ValueError("frozen encoder was left in train mode")
                encoder_parameters = {id(p) for p in encoder.parameters()}
                head_hash = hashlib.sha256()
                for parameter in model.parameters():
                    if id(parameter) not in encoder_parameters:
                        head_hash.update(parameter.detach().cpu().numpy().tobytes())
                current = head_hash.hexdigest()
                if self.head_states.setdefault(name, current) != current:
                    self.head_updates.add(name)
            if moved:
                self.changed.add(name)
        self.records.append(int(epoch))
        print(f"PAIRED_EPOCH {epoch}", flush=True)

    def result(self):
        if not self.records or (self.arm == "trainable" and not self.changed):
            raise ValueError("no convolution weight update on the trainable arm")
        if (
            self.arm == "trainable"
            and "pretrain" in self.observed_stages
            and {"pretrain", "self_paced"} - self.changed_by_stage.keys()
        ):
            raise ValueError("Robust-PU encoder did not move in both training stages")
        if self.arm == "frozen" and not self.head_updates:
            raise ValueError("frozen arm did not update its classifier head between snapshots")
        return {
            "arm": self.arm,
            "epochs_checked": self.records,
            "changed_networks": sorted(self.changed),
            "convolution_updates_by_stage": {
                stage: sorted(networks) for stage, networks in self.changed_by_stage.items()
            },
            "initial_encoder": self.baseline,
            "frozen_arm_head_updates": sorted(self.head_updates),
        }


def _training_prior(algorithm, population, observed_labels, view):
    if not get_metadata(algorithm).required_prior_parameters:
        return None
    if view == "ts":
        return population
    labeled_fraction = float(np.mean(observed_labels == 1))
    return population_to_native_unlabeled(population, labeled_fraction, input_sampling="os")


def _assert_consumed_prior(estimator, view):
    parameter = getattr(estimator, "_runner_prior_parameter", None)
    records = [
        row for row in getattr(estimator, "_prior_consumptions_", ()) if row["partition"] == "train"
    ]
    if parameter is None:
        if records:
            raise ValueError("estimator consumed an undeclared prior in the paired run")
        return
    expected = getattr(estimator, parameter)
    target = risk_pool_target(view == "ts")
    primary_uses = [row for row in records if row["source"] == f"estimator.{parameter}"]
    if (
        not primary_uses
        or any(row["target"] != target for row in primary_uses)
        or any(row["value"] != expected for row in records)
    ):
        raise ValueError(f"estimator did not consume {parameter} as requested for {target}")


def _checkpoint_peak(trajectory):
    # All epoch files coexist before offline selection and pruning.
    sizes = [Path(cp.path).stat().st_size for cp in trajectory.checkpoints]
    if max(sizes) > COMPONENT_LIMIT:
        raise ValueError("checkpoint exceeds paired per-component byte cap")
    return sum(sizes)


class AuditedCheckpointTrainer(EpochCheckpointTrainer):
    records_epoch_checkpoints = True

    def __init__(self, audit, contract, checkpoint_epochs):
        super().__init__()
        self.audit = audit
        self.contract = contract
        self._epochs = checkpoint_epochs

    def bind_checkpoint_directory(self, root):
        self.checkpoint_dir = root

    def checkpoint_epoch_bound(self):
        return self._epochs

    def _checkpoint_context(self, fitted, epoch):
        self.audit.record(epoch, fitted)
        return super()._checkpoint_context(fitted, epoch)

    def fit(self, estimator, X, y, *, val_pu=None, os_or_ts=None):  # noqa: N803
        # External PA and OA select among the recorded epochs.
        trajectory = super().fit(estimator, X, y, os_or_ts=os_or_ts)
        _assert_consumed_prior(estimator, os_or_ts)
        self.checkpoint_peak_bytes = _checkpoint_peak(trajectory)
        if self.contract is not None:
            verify_trajectory(
                estimator, trajectory, self.contract, X, y, os_or_ts, architecture="cnn"
            )
        return trajectory


class AuditedSupportTrainer(SelfPUCleanSupportCheckpointTrainer):
    def validate_experiment(self, model, bundle, config, protocols):
        super().validate_experiment(model, bundle, config, protocols)
        # This pre-registered standalone recipe evaluates all completed epochs,
        # like the Survey support grid (not the exploratory meta-only path).
        self._capture_all_epochs = True
        self._audit.update(selection_scope="completed_epochs", warmup_snapshots_eligible=True)

    def _fit_checkpoint_trajectory(self, estimator, X, y, risk_prior, os_or_ts):  # noqa: N803
        trajectory = super()._fit_checkpoint_trajectory(estimator, X, y, risk_prior, os_or_ts)
        self.checkpoint_peak_bytes = _checkpoint_peak(trajectory)
        return trajectory

    def _fit_support_estimator(self, estimator, X, y, *, risk_prior, epoch_callback=None):  # noqa: N803
        def record(epoch, fitted):
            self.audit.record(epoch, fitted)
            epoch_callback(epoch, fitted)

        super()._fit_support_estimator(
            estimator, X, y, risk_prior=risk_prior, epoch_callback=record
        )
        _assert_consumed_prior(estimator, "os")

    def _finish_support_fit(self, estimator, y):
        super()._finish_support_fit(estimator, y)
        estimator.training_provenance_["training_native_u_prior"] = estimator.risk_prior_


def run(args):
    import torch

    started = time.monotonic()
    source_commit, dirty = _source_commit(not args.probe)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(BASIS.read_text(encoding="utf-8"))
    profile = protocol["method_profiles"][args.method]
    bundle, split_ref, support, product, source_files = _load(args)
    split_hash = strict_canonical_hash(
        {
            role: getattr(bundle, role).indices.tolist()
            for role in ("train", "pu_val", "clean_val", "test")
        }
    )
    bundle, encoder, image_info, prep = prepare_image_bundle(
        bundle, protocol, args.seed, dataset=args.dataset, with_preprocessing=True
    )
    generator = SCARGenerator()
    train_pu, train_meta = generator.generate(bundle.train.X, bundle.train.labels, C, args.seed)
    _, val_meta = generator.generate(bundle.pu_val.X, bundle.pu_val.labels, C, args.seed)
    population = float(split_ref["class_prior"]["population"])
    algorithm = profile["algorithm"]
    cls = get_algorithm(algorithm)
    view = resolve_training_view(
        load_ledger(),
        algorithm,
        profile_training_view_request(protocol, args.method, None),
        is_oracle=False,
        estimator_class=cls,
    )
    prior = _training_prior(algorithm, population, train_pu, view)
    estimator, params, head_hash = _model(args, protocol, encoder, prior)
    audit = EncoderAudit(encoder, args.arm)
    capacity_epochs = int(getattr(estimator, "checkpoint_epoch_count", params.get("max_epochs", 0)))
    if capacity_epochs <= 0:
        raise ValueError("paired recipe did not declare an epoch bound")
    components = len(getattr(estimator, "epoch_components", ("model",)))
    required = capacity_epochs * components * COMPONENT_LIMIT + 2 * 1024**3
    if shutil.disk_usage(args.output).free < required:
        raise OSError(f"insufficient checkpoint storage: need {required} bytes")
    protocols = [ProtocolPA(), ProtocolOA()]
    budget = protocol["budgets"][profile["budget"]].get("staged_budget")
    if support is None:
        trainer = AuditedCheckpointTrainer(audit, None if args.probe else budget, capacity_epochs)
    else:
        prepared = prepare_image_support(support, training_path="native_cnn", preprocessing=prep)
        trainer = AuditedSupportTrainer(prepared, **product["support_adapter_options"])
        trainer.audit = audit
        protocols = [ProtocolPACleanSupport(), ProtocolOA()]
    # The study is separate from and not admitted into any frozen Survey version.
    identity = {
        "study": STUDY,
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "mode": "probe" if args.probe else "full",
        "c": C,
        "source_commit": source_commit,
        "basis_sha256": file_hash(BASIS),
        "source_files": source_files,
        "split_hash": split_hash,
        "support_receipt": None if product is None else product["receipt_sha256"],
        "label_hashes": [m["label_view_sha256"] for m in (train_meta, val_meta)],
        "sampling": "os",
        "risk_view": view,
        "class_prior_population": population,
        "class_prior_training": prior,
        "parameters": params,
        "encoder": image_info,
        "head_initial_state": head_hash,
        "selection_names": [p.name for p in protocols],
        "recipe_registry_basis_sha256": file_hash(
            BASIS.with_name("survey_recipe_registry_v1_3.json")
        ),
    }
    _json(args.output / "request.json", identity | {"arm": args.arm, "dirty": dirty})
    runner = ExperimentRunner(
        seed=args.seed,
        generator=generator,
        protocols=protocols,
        manifest_path=str(args.output / "manifest.json"),
        # PA reads the population prior from split_ref; the estimator is bound to its risk pool.
        config={
            "c": C,
            "os_or_ts": view,
            "trainer": trainer,
            "training_path": "native_cnn",
            "image_manifest": image_info,
            "split_ref": split_ref,
            "candidate_max_attempts": 1,
            "checkpoint_bytes_per_component": COMPONENT_LIMIT,
            "reclaim_unselected_checkpoints": True,
            **({"architecture": "cnn"} if algorithm != "self_pu" else {}),
        },
    )
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = runner.fit(estimator, bundle.train, bundle.pu_val, bundle.clean_val, bundle.test)
    payload = result.manifest
    candidates = payload["candidate_runs"]
    expected = {p.name for p in protocols}
    if payload["failures"] or set(result.test_metrics) != expected or len(candidates) != 1:
        raise ValueError("one successful fit and both declared PA/OA tracks are required")
    actual_audit = audit.result()
    checkpoints = candidates[0]["epoch_checkpoints"]
    observed_components = Counter(cp["epoch_position"] for cp in checkpoints)
    if (
        not observed_components
        or set(observed_components.values()) != {components}
        or len(observed_components) > capacity_epochs
        or (args.method != "split_pu" and len(observed_components) != capacity_epochs)
    ):
        raise ValueError("checkpoint coverage differs from the locked run budget")
    if [payload["generation"][role]["label_view_sha256"] for role in ("train", "pu_val")] != (
        identity["label_hashes"]
    ):
        raise ValueError("runner PU label view differs from the paired request")
    selection_priors = [
        (row["protocol"], row["target"], row["value"])
        for row in payload["research_settings"]["prior"]
        if row["consumer"] == "pa_risk" and row["partition"] == "pu_val"
    ]
    if selection_priors != [(protocols[0].name, "population", population)]:
        raise ValueError("PA selection used the wrong population prior")
    selected = {}
    for name, selection in payload["selection"].items():
        ck = selection["checkpoint"]
        template = result.selected_models[name].model_
        replay = load_selected_checkpoint(selection, template)
        pred = replay.predict(bundle.test.X)
        scores = replay.decision_function(bundle.test.X)
        replay_metrics = {
            "accuracy": float(np.mean(pred == bundle.test.labels)),
            "auc": float(roc_auc_score(bundle.test.labels, scores)),
            **_binary_test_metrics(bundle.test.labels, pred),
        }
        for metric in ("accuracy", "auc", "f1", "precision", "recall"):
            expected_value = result.test_metrics[name][metric]
            actual = replay_metrics[metric]
            if not (
                (actual is None and expected_value is None)
                or (
                    actual is not None
                    and expected_value is not None
                    and np.isclose(actual, expected_value, rtol=0, atol=1e-10)
                )
            ):
                raise ValueError(f"{name} checkpoint replay differs at {metric}")
        selected[ck["path"]] = ck["sha256"]
    env = runtime_environment()
    payload["resources"]["peak_checkpoint_payload_bytes"] = trainer.checkpoint_peak_bytes
    payload.update(
        paired_study=identity,
        encoder_audit=actual_audit,
        paired_arm=args.arm,
        environment=env,
        formal_eligible=False,
        checkpoint_replay_verified=True,
        training_path="native_cnn",
    )
    _json(args.output / "manifest.json", payload)
    _json(
        args.output / "accepted.json",
        {
            "schema_version": 1,
            "study": STUDY,
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "arm": args.arm,
            "mode": identity["mode"],
            "source_commit": source_commit,
            "pair_identity": strict_canonical_hash(identity),
            "metrics": result.test_metrics,
            "environment": env,
            "environment_identity": strict_canonical_hash(env),
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())
            if args.device == "cuda"
            else 0,
            "peak_rss_bytes": int(process_lifetime_peak_rss_bytes() or 0),
            "peak_checkpoint_payload_bytes": trainer.checkpoint_peak_bytes,
            "manifest_sha256": file_hash(args.output / "manifest.json"),
            "checkpoints": [{"path": p, "sha256": sha} for p, sha in selected.items()],
        },
    )
    print(f"PAIRED_ACCEPTED {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--arm", choices=("frozen", "trainable"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
