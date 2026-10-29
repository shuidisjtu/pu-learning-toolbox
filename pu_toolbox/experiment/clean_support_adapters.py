# ruff: noqa: N803
"""Opt-in full-meta Self-PU and LaGAM, with independent extra-label audit.

These are terminal-model exploratory OA paths, not frozen Survey recipes or
PU-only methods. They never infer support from clean_val or test.
"""

from .pulns_adapter import _IndependentSupportTrainer
from .tracking import EpochRecord, RunTrajectory


class SelfPUCleanSupportTrainer(_IndependentSupportTrainer):
    """Enable Self-PU's clean-meta branch using a separate audited support set.

    Training support chooses its internal teacher; external OA uses clean_val
    afterwards. This wires the existing full-meta implementation, not all
    paper-specific architecture/loss differences, and is not PA eligible.
    """

    method_name = "self_pu"
    display_name = "Self-PU full-meta"
    support_uses = ("meta_reweighting", "internal_teacher_selection")

    @staticmethod
    def estimator_type():
        from ..estimators.deep.self_pu import SelfPUClassifier

        return SelfPUClassifier

    def validate_experiment(self, model, bundle, config, protocols):
        super().validate_experiment(model, bundle, config, protocols)
        if model.max_epochs <= model.self_paced_start:
            self._prepared = False
            raise ValueError("Self-PU full-meta budget must reach a meta-active epoch")
        # The explicitly extra-supervision, full-meta path follows the paper's
        # population-wide consistency domains.  The estimator default remains
        # the historical U-only ablation path used by the frozen pilot.
        model.set_params(distillation_scope="paper_population")

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        self._require_prepared(estimator, X, os_or_ts)
        self._fit_support_estimator(estimator, X, y, class_prior=class_prior)
        self._finish_support_fit(estimator, y)
        return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=estimator)

    def _fit_support_estimator(self, estimator, X, y, *, class_prior, epoch_callback=None):
        callbacks = {} if epoch_callback is None else {"epoch_callback": epoch_callback}
        estimator.fit(
            X,
            y,
            class_prior=class_prior,
            validation_data=(self.support.X, self.support.labels),
            os_or_ts="os",
            **callbacks,
        )

    def _finish_support_fit(self, estimator, y):
        active_records = sum(row["calibration_active"] for row in estimator.reweight_history_)
        if estimator.calibration_mode_ != "clean_validation_meta" or active_records == 0:
            raise ValueError("Self-PU support path unexpectedly ran an ablation")
        self._finish_provenance(estimator, y)
        estimator.training_provenance_.update(
            {
                "method_variant": "clean_meta_extra_support",
                "calibration_mode": estimator.calibration_mode_,
                "teacher_selection_basis": estimator.teacher_selection_basis_,
                "teacher_selection_metrics": list(estimator.teacher_selection_metrics_),
                "meta_reweight_records": len(estimator.reweight_history_),
                "active_meta_reweight_records": active_records,
                "training_population_prior": estimator.class_prior_,
                "executed_epochs": len(estimator.history_["epoch"]),
                "selection_scope": "terminal_after_support_teacher_selection",
                "paper_full_reproduction": False,
                "distillation_scope": estimator.distillation_scope_,
            }
        )


class _SelfPUTSCleanSupportTrainer(SelfPUCleanSupportTrainer):
    """Internal explicit TS fit binding; exposed via the checkpoint variant only."""

    def _validate_training_view(self, view):
        if view != "ts":
            raise ValueError("Self-PU TS clean-support requires explicit os_or_ts=ts")

    def validate_experiment(self, model, bundle, config, protocols):
        self._prepared = False
        if config.get("recipe_registry") is not None or config.get("survey_protocol"):
            raise ValueError("Self-PU TS clean-support is not a versioned Survey recipe")
        super().validate_experiment(model, bundle, config, protocols)
        self._audit.update(
            {
                "training_view": "ts",
                "calibration_applied": True,
                "calibration_scope": "negative_marginal_row_mass_blend_only",
                "native_U_identity_pools_unchanged": True,
                "population_mixture_restoration_verified": False,
                "formal_admission": False,
            }
        )

    def _fit_support_estimator(self, estimator, X, y, *, class_prior, epoch_callback=None):
        callbacks = {} if epoch_callback is None else {"epoch_callback": epoch_callback}
        estimator.fit(
            X,
            y,
            class_prior=class_prior,
            validation_data=(self.support.X, self.support.labels),
            os_or_ts="ts",
            **callbacks,
        )

    def _finish_support_fit(self, estimator, y):
        super()._finish_support_fit(estimator, y)
        # A record is appended only after that student's optimizer.step().
        estimator.training_provenance_.update(
            {
                "student_optimizer_steps": {
                    str(student): sum(
                        row["student"] == student for row in estimator.training_history_
                    )
                    for student in (1, 2)
                },
                "optimizer_steps": len(estimator.training_history_),
                "optimizer_step_scope": "student_updates_only_not_virtual_meta_probes_or_EMA",
            }
        )


class LaGAMCleanSupportTrainer(_IndependentSupportTrainer):
    """LaGAM's clean-support meta labels, separate from external OA selection."""

    method_name = "lagam"
    display_name = "LaGAM"
    support_uses = ("meta_label_disambiguation",)

    @staticmethod
    def estimator_type():
        from ..estimators.deep.lagam import LaGAMClassifier

        return LaGAMClassifier

    def validate_experiment(self, model, bundle, config, protocols):
        super().validate_experiment(model, bundle, config, protocols)
        if model.max_epochs <= model.warmup_epochs:
            self._prepared = False
            raise ValueError("LaGAM meta budget must extend beyond warmup")

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        self._require_prepared(estimator, X, os_or_ts)
        self._fit_support_estimator(estimator, X, y, class_prior=class_prior)
        self._finish_support_fit(estimator, y)
        return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=estimator)

    def _fit_support_estimator(self, estimator, X, y, *, class_prior, epoch_callback=None):
        callbacks = {} if epoch_callback is None else {"epoch_callback": epoch_callback}
        estimator.fit(X, y, support_data=(self.support.X, self.support.labels), **callbacks)

    def _finish_support_fit(self, estimator, y):
        if estimator.virtual_steps_ == 0:
            raise ValueError("LaGAM support path executed no virtual meta update")
        self._finish_provenance(estimator, y)
        estimator.training_provenance_.update(
            {
                "optimizer_steps": estimator.optimizer_steps_,
                "virtual_steps": estimator.virtual_steps_,
                "selection_scope": "terminal",
                "paper_full_reproduction": False,
            }
        )
