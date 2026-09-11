"""ExperimentRunner — Template Method orchestration of the four-way protocol.

Design notes: fixed skeleton; every variation point is an injected
strategy (Generator/Trainer/SelectionProtocol) per implementation_plan.md
§1.4. Does NOT subclass — users inject their own strategy objects.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone

from . import resources as resource_tools
from .bundle import DatasetBundle, DatasetPart, validate_bundle
from .manifest import write_manifest
from .strategies import DeepFitTrainer, ProtocolOA, ProtocolPA, SCARGenerator
from .tracking import RunResult, RunTrajectory


class ExperimentRunner:
    """Configured runner for a four-way PU experiment (§2.4).

    Best-epoch semantics: every ``SelectionArtifact.epoch`` mirrors
    ``RunTrajectory.best_epoch`` — a 1-based *position* inside the
    selected run's ``epochs`` list (``epochs[epoch - 1]`` holds the
    selected record). It is unrelated to the estimator's own epoch
    labels (``EpochRecord.epoch``); the runner never assumes equality.
    Callers who restore the checkpoint for a re-run must index
    ``trajectory.epochs[artifact.epoch - 1]``.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        generator=None,
        protocols=None,
        class_prior: float | None = None,
        threshold_candidates=None,
        manifest_path: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.seed = seed
        self.generator = generator or SCARGenerator()
        self.protocols = protocols or [ProtocolPA(), ProtocolOA()]
        self.class_prior = class_prior
        self.threshold_candidates = threshold_candidates
        self.manifest_path = manifest_path
        self.config = config or {}

    def fit(
        self,
        model,
        train: DatasetPart,
        pu_val: DatasetPart,
        clean_val: DatasetPart,
        test: DatasetPart,
    ) -> RunResult:
        """Run the full pipeline; see bundle contract (views must be clean).

        Steps: validate → generate label views → verify the view contract →
        train the candidate pool → offline selection per protocol →
        independent test evaluation → manifest + ``RunResult``.

        Raises
        ------
        ValueError
            If the generator's declared view is not ``"pu"``/``"clean"``, if its
            reported mechanism contradicts that view, if a ``"clean"``
            declaration does not actually carry the real labels, if a clean-view
            run is handed a ``class_prior`` or a trainer that does not declare
            ``trains_on_real_labels``, if a real-label (PN-oracle) trainer is
            paired with a ``"pu"`` view, if ``config["trainer"]`` is a class
            rather than an instance, or if the generated pu-val PU view ends up
            with no labeled positive (the pu-val split has no real positive or
            the labeling rate ``c`` is too low).  Fail loudly instead of
            silently producing a wrong result.
        """
        t0 = time.perf_counter()
        bundle = DatasetBundle(train=train, pu_val=pu_val, clean_val=clean_val, test=test)
        validate_bundle(bundle)
        _validate_model_capability(model, bundle, self.config.get("architecture"))

        # 2. generate label views (SCAR / SAR / clean via the injected generator)
        generation_t0 = time.perf_counter()
        c = self.config.get("c", 0.1)
        y_view_train, meta_train = self.generator.generate(train.X, train.labels, c, self.seed)
        y_view_val, meta_val = self.generator.generate(pu_val.X, pu_val.labels, c, self.seed)
        # The generator declares the view it produces: "pu" for SCAR/SAR, and
        # "clean" for the PN oracle (real labels passed through).  Keeping the
        # declaration on the strategy keeps PA structurally excluded — a clean
        # view makes ProtocolPA raise instead of emitting a fake PA row.
        view = getattr(self.generator, "output_view", "pu")
        if view not in ("pu", "clean"):
            # Fail closed: a typo such as "Clean" would otherwise skip both the
            # clean-view verification below and the supervised-trainer guard.
            raise ValueError(f"generator declared output_view={view!r}; expected 'pu' or 'clean'.")
        for role, generated_meta in (("train", meta_train), ("pu_val", meta_val)):
            # The generator's two self-descriptions must agree: the pn_oracle
            # mechanism means real labels, i.e. the clean view.  Keeping the
            # oracle's mechanism while declaring a PU view would skip every
            # clean-view guard below and still be recorded as an oracle row.
            if view == "pu" and _mechanism(generated_meta) == "pn_oracle":
                raise ValueError(
                    f"generator reports mechanism='pn_oracle' for {role} but declares "
                    "output_view='pu'; the PN oracle mechanism implies real labels, "
                    "which is the 'clean' view. Declare output_view='clean' (as "
                    "CleanLabelGenerator does) or report a PU mechanism."
                )
        train_view = DatasetPart(X=train.X, labels=y_view_train, view=view, indices=train.indices)
        pu_val_view = DatasetPart(X=pu_val.X, labels=y_view_val, view=view, indices=pu_val.indices)

        # The trainer's declared label semantics and the view must agree: a PU
        # trainer reads label 0 as "unlabeled", a real-label (PN-oracle) trainer
        # reads it as a real negative. Mismatching them still trains something
        # and still writes a manifest, so both directions fail loudly here.
        # Resolved once and threaded into _train: the trainer this guard approved
        # is the one that trains, even if the config is mutated meanwhile.
        trainer = self._trainer()
        trains_on_real_labels = getattr(trainer, "trains_on_real_labels", False)

        if view == "clean":
            # A "clean" declaration is verified, never trusted: a generator that
            # declares real labels but emits marked ones would otherwise produce
            # a silent fake oracle carrying a pn_oracle manifest.
            for role, produced, original in (
                ("train", train_view.labels, train.labels),
                ("pu_val", pu_val_view.labels, pu_val.labels),
            ):
                if not np.array_equal(np.asarray(produced), np.asarray(original)):
                    raise ValueError(
                        f"generator declared output_view='clean' but produced different "
                        f"{role} labels; a clean view must carry the real labels."
                    )
            if self.class_prior is not None:
                # A prior on a real-label run would stop the oracle being an
                # upper bound at all, so refuse rather than silently ignore it.
                raise ValueError(
                    "class_prior applies to PU methods; a clean-view (PN oracle) run "
                    "trains on real labels and must not receive a prior. Drop "
                    "class_prior for clean-view runs."
                )
            if not trains_on_real_labels:
                # Defaulting to "PU trainer" is the fail-closed reading: it also
                # covers the default DeepFitTrainer and any trainer that never
                # declares the flag.
                raise ValueError(
                    f"{type(trainer).__name__} does not declare "
                    "trains_on_real_labels=True, so it cannot train on a clean "
                    "(PN oracle) view — a PU objective would read every real "
                    "negative as unlabeled. Use SupervisedTrainer, or declare the "
                    "flag on your own real-label trainer."
                )
        elif trains_on_real_labels:
            # A real-label (PN-oracle) trainer needs a clean view. Pairing it with
            # a PU-view generator trains on marked labels and silently produces a
            # fake upper bound — the mis-wiring this path shipped with.
            raise ValueError(
                f"{type(trainer).__name__} declares trains_on_real_labels=True "
                f"(PN oracle) but the generator produced a {view!r} view; it needs "
                "a clean label view. Use CleanLabelGenerator as the generator, or "
                "drop the real-label trainer. See docs/research/pu_survey/"
                "pn_oracle_integration.md §1."
            )

        # PA needs a labeled positive in its val view. The oracle trains on real
        # labels and runs OA only, so this generated-view check does not apply.
        if view == "pu" and int(np.sum(pu_val_view.labels == 1)) == 0:
            raise ValueError(
                "generated pu_val PU view has no labeled positive; "
                "ensure pu_val contains real positives (and/or raise c)."
            )
        generation_elapsed = time.perf_counter() - generation_t0

        # 3. train the candidate pool. A failed attempt is retried once from
        #    a fresh clone; a twice-failed candidate is excluded from ranking.
        candidates = self.config.get("candidates", [{}])
        trajectories = []
        failures = []
        candidate_runs = []
        tuning_t0 = time.perf_counter()
        for candidate_index, params in enumerate(candidates):
            errors = []
            attempt_resources = []
            for attempt in (1, 2):
                attempt_t0 = time.perf_counter()
                gpu_device = resource_tools.begin_peak_gpu_memory_measurement(model, params)
                try:
                    est = _clone_candidate(model, params, self.seed)
                    _validate_model_capability(
                        est,
                        bundle,
                        self.config.get("architecture"),
                    )
                    trajectory = self._train(est, train_view, pu_val_view, trainer)
                    _validate_trajectory(trajectory, pu_val_view)
                except Exception as exc:  # noqa: BLE001 - recorded retry boundary
                    errors.append(_exception_record(exc, attempt))
                    attempt_resources.append(
                        _attempt_resources(
                            attempt,
                            "failed",
                            attempt_t0,
                            gpu_device,
                        )
                    )
                    continue

                attempt_resources.append(
                    _attempt_resources(
                        attempt,
                        "succeeded",
                        attempt_t0,
                        gpu_device,
                    )
                )
                trajectory_index = len(trajectories)
                trajectories.append(trajectory)
                status = "recovered" if errors else "succeeded"
                candidate_runs.append(
                    {
                        "candidate_index": candidate_index,
                        "trajectory_index": trajectory_index,
                        "params": _artifact_value(params),
                        "seed": self.seed,
                        "attempts": attempt,
                        "status": status,
                        "resources": _candidate_resources(attempt_resources),
                    }
                )
                if errors:
                    failures.append(
                        {
                            "candidate_index": candidate_index,
                            "params": _artifact_value(params),
                            "seed": self.seed,
                            "attempts": attempt,
                            "status": status,
                            "errors": errors,
                        }
                    )
                break
            else:
                candidate_runs.append(
                    {
                        "candidate_index": candidate_index,
                        "trajectory_index": None,
                        "params": _artifact_value(params),
                        "seed": self.seed,
                        "attempts": 2,
                        "status": "excluded",
                        "resources": _candidate_resources(attempt_resources),
                    }
                )
                failures.append(
                    {
                        "candidate_index": candidate_index,
                        "params": _artifact_value(params),
                        "seed": self.seed,
                        "attempts": 2,
                        "status": "excluded",
                        "errors": errors,
                    }
                )

        tuning_elapsed = time.perf_counter() - tuning_t0
        resources = _resource_summary(
            candidate_runs,
            generation_elapsed=generation_elapsed,
            tuning_elapsed=tuning_elapsed,
        )

        if not trajectories:
            elapsed = time.perf_counter() - t0
            resources["runner_elapsed_seconds"] = elapsed
            manifest = {
                "seed": self.seed,
                "split_ref": self.config.get("split_ref", {}),
                "generation": {"train": meta_train, "pu_val": meta_val},
                "candidate_runs": candidate_runs,
                "selection": {},
                "test_results": {},
                "elapsed": elapsed,
                "failures": failures,
                "resources": resources,
            }
            self._write_manifest(manifest)
            raise RuntimeError(
                "all candidate runs failed after one same-seed retry; "
                "inspect manifest['failures'] for details."
            )

        # 4. offline selection: OA on the clean view; PA and any other protocol
        #    get the generated val view. That view holds PU labels on a PU run
        #    and real labels on a clean-view (PN oracle) run — so an unknown
        #    protocol is NOT guaranteed PU-only labels; only ProtocolPA enforces
        #    that, via its own view check.
        selections = {}
        for proto in self.protocols:
            if isinstance(proto, ProtocolPA):
                art = proto.select(trajectories, pu_val_view, self.threshold_candidates)
            elif isinstance(proto, ProtocolOA):
                art = proto.select(trajectories, clean_val, self.threshold_candidates)
            else:
                art = proto.select(trajectories, pu_val_view, self.threshold_candidates)
            selections[art.protocol] = art

        # 5. independent test evaluation (test never entered selection/training)
        test_metrics = {}
        for name, art in selections.items():
            est = trajectories[art.run_index].model
            s_min = art.metrics.get("val_score_min")
            s_scale = art.metrics.get("val_score_scale")
            if art.threshold is not None and s_min is not None and s_scale:
                # OA picked its threshold in the VAL-side min-max space; reuse
                # the recorded val affine transform here, so the threshold keeps
                # its val semantics. Re-normalising with the test's own min/max
                # applies different affine constants and silently shifts the
                # threshold (F1 fix: fixed val transform, not "same convention").
                scores = (est.decision_function(test.X) - s_min) / s_scale
                pred = (scores >= art.threshold).astype(int)
            else:
                pred = est.predict(test.X)
            auc, auc_unavailable_reason = _auc(est, test)
            test_metrics[name] = {
                "accuracy": float(np.mean(pred == test.labels)),
                "auc": auc,
                "auc_unavailable_reason": auc_unavailable_reason,
            }

        selection_payload = {}
        for name, artifact in selections.items():
            payload = asdict(artifact)
            selected_run = next(
                item for item in candidate_runs if item["trajectory_index"] == artifact.run_index
            )
            payload["candidate_index"] = selected_run["candidate_index"]
            selection_payload[name] = payload

        elapsed = time.perf_counter() - t0
        resources["runner_elapsed_seconds"] = elapsed
        manifest = {
            "seed": self.seed,
            "split_ref": self.config.get("split_ref", {}),
            "generation": {"train": meta_train, "pu_val": meta_val},
            "candidate_runs": candidate_runs,
            "selection": selection_payload,
            "test_results": test_metrics,
            "elapsed": elapsed,
            "failures": failures,
            "resources": resources,
        }
        self._write_manifest(manifest)
        return RunResult(
            selections=selections,
            test_metrics=test_metrics,
            manifest=manifest,
            failures=failures,
            resources=resources,
        )

    def _trainer(self):
        """The configured trainer instance, or the PU-view default."""
        trainer = self.config.get("trainer", DeepFitTrainer())
        if inspect.isclass(trainer):
            # A class satisfies the view guard through its class attributes and
            # then never trains, leaving a failed-run manifest behind.
            raise ValueError(
                f"config['trainer'] must be an instance, got the class "
                f"{trainer.__name__}; pass {trainer.__name__}()."
            )
        return trainer

    def _train(self, est, train_view: DatasetPart, pu_val_view: DatasetPart, trainer):
        """Train one candidate with the trainer the view guard approved."""
        val_pu = (pu_val_view.X, pu_val_view.labels) if pu_val_view is not None else None
        return trainer.fit(
            est,
            train_view.X,
            train_view.labels,
            class_prior=self.class_prior,
            val_pu=val_pu,
        )

    def _write_manifest(self, manifest: dict) -> None:
        if self.manifest_path:
            path = Path(self.manifest_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_manifest(path, manifest)


def _mechanism(generated_meta) -> object:
    """The generator's self-reported mechanism, or None when it reports none."""
    return generated_meta.get("mechanism") if isinstance(generated_meta, dict) else None


def _auc(est, test: DatasetPart) -> tuple[float, str | None]:
    """AUROC on real test labels, with an explicit degenerate-label reason."""
    from pu_toolbox.metrics import pu_auc_roc

    if np.unique(test.labels).size < 2:
        return (
            float("nan"),
            "ROC AUC is unavailable because the test labels contain only one class.",
        )
    # Score-generation and metric bugs are not an expected availability case:
    # let them surface instead of silently converting them to NaN.
    return float(pu_auc_roc(test.labels, est.decision_function(test.X))), None


def _validate_model_capability(
    model,
    bundle: DatasetBundle,
    architecture: str | None,
) -> None:
    """Reject unsupported input/architecture combinations before training."""
    cls = type(model)
    allowed_ndims = frozenset(getattr(cls, "input_ndims", frozenset({2})))
    for role in ("train", "pu_val", "clean_val", "test"):
        ndim = getattr(getattr(bundle, role).X, "ndim", None)
        if ndim not in allowed_ndims:
            raise ValueError(
                f"{cls.__name__} does not support {role} input ndim={ndim}; "
                f"declared input_ndims={sorted(allowed_ndims)!r}."
            )

    if architecture not in (None, "mlp", "cnn"):
        raise ValueError("experiment architecture must be 'mlp', 'cnn', or None.")

    native = frozenset(getattr(cls, "native_architectures", frozenset()))
    if architecture is None:
        encoder_parameter = getattr(cls, "encoder_parameter", None)
        if encoder_parameter and getattr(model, encoder_parameter, None) is not None:
            architecture = "cnn"
        elif "mlp" in native:
            architecture = "mlp"
        else:
            # An empty declaration is the project's explicit tabular-only
            # capability.  The input_ndims check above is its complete gate.
            return

    # Legacy tabular estimators declare no native architecture.  They remain
    # compatible with an explicit ``architecture='mlp'`` when their declared
    # input contract is two-dimensional.
    if architecture == "mlp" and not native and allowed_ndims == frozenset({2}):
        return
    if architecture not in native:
        raise ValueError(
            f"{cls.__name__} does not support architecture={architecture!r}; "
            f"declared native_architectures={sorted(native)!r}."
        )

    if architecture == "cnn":
        encoder_parameter = getattr(cls, "encoder_parameter", None)
        if 4 not in allowed_ndims or not encoder_parameter or not hasattr(model, encoder_parameter):
            raise ValueError(
                f"{cls.__name__} has an inconsistent CNN capability declaration: "
                f"input_ndims={sorted(allowed_ndims)!r}, "
                f"encoder_parameter={encoder_parameter!r}."
            )


def _clone_candidate(model, params: dict, seed: int):
    """Create a clean retry candidate while preserving one run seed."""
    est = clone(model)
    if params:
        est.set_params(**params)
    est_params = est.get_params(deep=False)
    if "random_state" in est_params and est_params["random_state"] is None:
        est.set_params(random_state=seed)
    return est


def _validate_trajectory(trajectory, pu_val: DatasetPart) -> None:
    """Turn non-finite output and trainer interface drift into retryable failures."""
    if not isinstance(trajectory, RunTrajectory):
        raise TypeError("trainer.fit must return a RunTrajectory instance.")
    for record in trajectory.epochs:
        for metric_name, value in record.metrics.items():
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"training history metric {metric_name!r} is not finite at "
                    f"epoch {record.epoch}."
                )
    scores = np.asarray(trajectory.model.decision_function(pu_val.X), dtype=float)
    if scores.shape != pu_val.labels.shape:
        raise ValueError(
            "candidate decision_function must return one score per sample; "
            f"got shape {scores.shape!r} for labels {pu_val.labels.shape!r}."
        )
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("candidate validation scores contain NaN or infinity.")


def _exception_record(exc: Exception, attempt: int) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _artifact_value(value):
    """Convert candidate parameters to stable JSON-compatible artifact values."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _artifact_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_artifact_value(item) for item in value]
    return repr(value)


def _attempt_resources(
    attempt: int,
    status: str,
    started_at: float,
    gpu_device: str | None,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "status": status,
        "elapsed_seconds": time.perf_counter() - started_at,
        "peak_gpu_memory_bytes": resource_tools.peak_gpu_memory_bytes(gpu_device),
    }


def _candidate_resources(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    peaks = [item["peak_gpu_memory_bytes"] for item in attempts]
    successful = next((item for item in attempts if item["status"] == "succeeded"), None)
    return {
        "elapsed_seconds": sum(item["elapsed_seconds"] for item in attempts),
        "successful_attempt_elapsed_seconds": (
            successful["elapsed_seconds"] if successful is not None else None
        ),
        "peak_gpu_memory_bytes": max((item for item in peaks if item is not None), default=None),
        "attempts": attempts,
    }


def _resource_summary(
    candidate_runs: list[dict[str, Any]],
    *,
    generation_elapsed: float,
    tuning_elapsed: float,
) -> dict[str, Any]:
    candidate_costs = [
        {
            "candidate_index": item["candidate_index"],
            "status": item["status"],
            **item["resources"],
        }
        for item in candidate_runs
    ]
    peaks = [item["peak_gpu_memory_bytes"] for item in candidate_costs]
    return {
        "schema_version": "1.0",
        "single_configuration_costs": candidate_costs,
        "tuning": {
            "elapsed_seconds": tuning_elapsed,
            "candidate_count": len(candidate_runs),
            "seed_count": 1,
            "scope": "this ExperimentRunner seed",
        },
        "peak_gpu_memory_bytes": max(
            (item for item in peaks if item is not None),
            default=None,
        ),
        "data_generation_elapsed_seconds": generation_elapsed,
        "shared_preprocessing": {
            "elapsed_seconds": None,
            "status": "outside ExperimentRunner scope",
        },
        "environment": resource_tools.runtime_environment(),
    }
