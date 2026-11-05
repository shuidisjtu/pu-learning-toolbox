"""Opt-in KLDCE execution evidence, without modifying or profiling its solver."""

from __future__ import annotations

import copy
import math
import time
from numbers import Real

import numpy as np

from ..core.validation import validate_pu_X_y
from .classical_candidate import ClassicalCandidateTrainer
from .tracking import RunTrajectory


def validate_kldce_candidate_parameters(model):
    """Reject ambiguous scalar encodings before expensive dense allocations."""
    from ..estimators.risk.kldce import KLDCEClassifier

    if type(model) is not KLDCEClassifier:
        raise TypeError("kernel candidate requires the exact KLDCEClassifier class")
    for key in ("mom_groups", "max_acs_iter", "max_inner_iter"):
        value = getattr(model, key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "flip_probability",
        "reg_strength",
        "centroid_radius",
        "inner_tol",
        "covariance_ridge",
        "tol",
    ):
        value = getattr(model, key)
        if (
            isinstance(value, bool | np.bool_)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be finite real")
        if value < 0 or (key not in {"covariance_ridge", "tol"} and value == 0):
            raise ValueError(f"{key} is outside its native range")
    if not model.flip_probability < 1:
        raise ValueError("flip_probability must be in (0,1)")
    if model.sigma != "scale" and (
        isinstance(model.sigma, bool | np.bool_)
        or not isinstance(model.sigma, Real)
        or not math.isfinite(model.sigma)
        or model.sigma <= 0
    ):
        raise ValueError("sigma must be scale or a finite positive real")


class KLDCECandidateTrainer(ClassicalCandidateTrainer):
    """Record real completed ACS diagnostics, never invent epochs or rollback costs.

    ``prior_policy`` is explicitly ``derived`` (the source k/[n*(1-h)]) or
    ``population_override`` (forward the runner's population prior). The known
    float64 QP-block payload cap is only an exclusion screen, NOT peak RSS.
    Complete-run history stores the final QP per ACS round; earlier rollback
    solves and failed rounds are unobserved and are explicitly not counted.
    """

    def __init__(self, *, prior_policy, kernel_payload_cap_bytes, receipt_dir=None):
        super().__init__(receipt_dir=receipt_dir)
        if prior_policy not in ("derived", "population_override"):
            raise ValueError("explicit prior_policy must be derived or population_override")
        if type(kernel_payload_cap_bytes) is not int or kernel_payload_cap_bytes <= 0:
            raise ValueError("kernel_payload_cap_bytes must be a positive integer")
        self.prior_policy = prior_policy
        self.kernel_payload_cap_bytes = kernel_payload_cap_bytes

    def validate_experiment(self, model, bundle, config, protocols):
        from .strategies import ProtocolOA, ProtocolPA

        if config.get("survey_protocol") is not None or config.get("recipe_registry") is not None:
            raise ValueError("KLDCE candidate is exploratory, not a frozen Survey unit")
        if config.get("capture_epoch_checkpoints", False):
            raise ValueError("KLDCE has no neural epoch checkpoints")
        if config.get("os_or_ts") != "os":
            raise ValueError("KLDCE candidate requires explicit native OS")
        if [type(protocol) for protocol in protocols] != [ProtocolPA, ProtocolOA]:
            raise ValueError("KLDCE candidate requires independent PA then OA selection")
        validate_kldce_candidate_parameters(model)
        self.prediction_screen(bundle)

    def prediction_screen(self, bundle):
        """Screen full-matrix scoring from shapes/dtypes, without reading role labels."""
        rows = {}
        for role in ("pu_val", "clean_val", "test"):
            features = getattr(bundle, role).X
            minimum_itemsize = min(bundle.train.X.dtype.itemsize, features.dtype.itemsize)
            payload = len(bundle.train.X) * len(features) * minimum_itemsize
            if payload > self.kernel_payload_cap_bytes:
                raise ValueError("KLDCE prediction Gram payload exceeds kernel_payload_cap_bytes")
            rows[role] = payload
        return {
            "prediction_gram_payload_lower_bounds_bytes": rows,
            "below_cap_is_not_peak_memory_approval": True,
        }

    def validate_generation(self, train_meta, val_meta):
        self._generation_binding = None
        if any(meta.get("mechanism") != "scar" for meta in (train_meta, val_meta)):
            raise ValueError("KLDCE candidate supports SCAR only")
        rate = train_meta.get("c_realized")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, Real)
            or not math.isfinite(rate)
            or not 0 < rate < 1
        ):
            raise ValueError("KLDCE requires SCAR training c_realized in (0,1)")
        self._generation_binding = {
            "c_requested": train_meta.get("c_requested"),
            "c_realized": float(rate),
            "flip_probability": 1 - float(rate),
            "n_labeled": train_meta["n_labeled"],
            "source": "train_SCAR_generator_metadata",
        }

    def preflight(self, estimator, X, y, *, class_prior=None, os_or_ts=None):  # noqa: N803
        """Use observed PU counts only; no fit, kernel allocation or label inference."""
        from ..estimators.risk._class_prior import (
            solve_prior_from_positive_fraction,
            stable_centroid_denominator,
        )

        validate_kldce_candidate_parameters(estimator)
        if os_or_ts != "os":
            raise ValueError("KLDCE candidate requires explicit native OS; TS is not hooked")
        features, labels = validate_pu_X_y(
            X, y, accept_sparse=False, estimator_name="KLDCE candidate"
        )
        if features.dtype.kind not in "iuf" or not np.isfinite(features).all():
            raise ValueError("KLDCE candidate requires finite real dense features")
        n, d = features.shape
        p = int(np.count_nonzero(labels == 1))
        u = n - p
        # Match the core fit's explicit float(self.flip_probability), including
        # weak scalar-promotion environments; do not let np.float32 round pi.
        flip = float(estimator.flip_probability)
        if not p or not u or estimator.mom_groups > u:
            raise ValueError("positive P/U and mom_groups <= observed U are required")
        binding = getattr(self, "_generation_binding", None)
        if binding is not None and (
            flip != binding["flip_probability"] or p != binding["n_labeled"]
        ):
            raise ValueError("KLDCE flip_probability or P count differs from bound SCAR c_realized")
        if self.prior_policy == "derived":
            prior = solve_prior_from_positive_fraction(p, n, flip)
        else:
            if (
                isinstance(class_prior, bool | np.bool_)
                or not isinstance(class_prior, Real)
                or not math.isfinite(class_prior)
                or not 0 < class_prior < 1
            ):
                raise ValueError("population_override requires finite population prior in (0,1)")
            prior = float(class_prior)
        denominator = stable_centroid_denominator(prior, flip)
        # Same Aeq=[+1_P,-1_U,+1_gamma], alpha<=1/n, gamma<=1/(2n)
        # as the core's Phase-I LP, but this range check allocates no QP.
        rhs = -u / (2 * n * denominator)
        lower, upper = -u / n, p / n + u / (2 * n)
        if rhs < lower - 1e-12 or rhs > upper + 1e-12:
            raise ValueError(
                f"KLDCE QP equality infeasible for its box constraints: rhs={rhs:.12g}, "
                f"reachable=[{lower:.12g}, {upper:.12g}]; no fit started"
            )
        payload = 8 * (n + u) ** 2
        covariance_minimum = d * d * features.dtype.itemsize
        if payload + covariance_minimum > self.kernel_payload_cap_bytes:
            raise ValueError(
                "KLDCE mandatory named QP payload exceeds kernel_payload_cap_bytes; no fit started"
            )
        return {
            "observed_positive_rows": p,
            "observed_unlabeled_rows": u,
            "feature_count": d,
            "training_prior_policy": self.prior_policy,
            "effective_training_prior": prior,
            "external_class_prior_forwarded": self.prior_policy == "population_override",
            "qp_equality_box_feasibility": {
                "rhs": rhs,
                "reachable_min": lower,
                "reachable_max": upper,
                "absolute_tolerance": 1e-12,
            },
            "qp_named_array_payload_bytes": payload,
            "covariance_payload_lower_bound_bytes": covariance_minimum,
            "flip_probability": flip,
            "generation_binding": copy.deepcopy(binding),
            "kernel_payload_cap_bytes": self.kernel_payload_cap_bytes,
            "memory_scope": "named_QP_and_covariance_lower_bound; excludes_Gram_and_temporaries",
            "below_cap_is_not_peak_memory_approval": True,
            "configured_upper_bounds": {
                "acs_rounds": estimator.max_acs_iter,
                "qp_solves": 2 * estimator.max_acs_iter - 1,
                "smo_pair_updates": (2 * estimator.max_acs_iter - 1) * estimator.max_inner_iter,
            },
        }

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):  # noqa: N803
        self.last_receipt = None
        estimator._is_fitted = False
        if hasattr(estimator, "training_provenance_"):
            del estimator.training_provenance_
        receipt = {
            "schema_version": "kldce-candidate-cost-1",
            "method": "kldce",
            "status": "started",
            "budget_unit": "complete_kernel_fit",
            "neural_epoch_count": 0,
            "formal_admission": False,
            "source_reproduction": False,
            "execution_mode": "technical_preintegration",
            "run_view": "os",
            "calibration_applied": False,
            "validation_used_for_training": False,
            "work_started": 0,
            "work_completed": 0,
            "acs_rounds_completed": None,
            "qp_solves_observed": None,
            "total_smo_pair_updates_observed": None,
            "failed_round_progress_observed": False,
        }
        self.last_receipt = receipt
        started = time.perf_counter()
        try:
            bound = self.preflight(estimator, X, y, class_prior=class_prior, os_or_ts=os_or_ts)
            receipt["preflight"] = bound
            kwargs = (
                {"class_prior": class_prior} if self.prior_policy == "population_override" else {}
            )
            receipt["work_started"] = 1
            estimator.fit(X, y, **kwargs)
            if estimator.class_prior_ != bound["effective_training_prior"]:
                raise ValueError("actual KLDCE training prior differs from preflight")
            history = [
                {
                    key: value.item() if isinstance(value, np.generic) else value
                    for key, value in row.items()
                }
                for row in estimator.acs_history_
            ]
            if (
                not 1 <= estimator.n_acs_iter_ <= estimator.max_acs_iter
                or len(history) != estimator.n_acs_iter_
            ):
                raise ValueError("KLDCE completed ACS history/budget mismatch")
            receipt.update(
                status="completed",
                work_completed=1,
                acs_rounds_completed=estimator.n_acs_iter_,
                converged=bool(estimator.converged_),
                effective_training_prior=float(estimator.class_prior_),
                acs_history=history,
                inner_diagnostics_scope="final_QP_per_ACS_round_only; rollback_work_unobserved",
                final_qp_inner_iterations_sum=sum(int(row["inner_n_iter"]) for row in history),
            )
        except BaseException as exc:
            estimator._is_fitted = False
            receipt.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            receipt["elapsed_seconds"] = time.perf_counter() - started
            if self.receipt_dir is not None:
                try:
                    self._persist_receipt(receipt)
                except BaseException:
                    estimator._is_fitted = False
                    raise
        estimator.training_provenance_ = copy.deepcopy(receipt)
        return RunTrajectory(epochs=[], model=estimator)
