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

    def fit(self, estimator, X, y, *, class_prior=None, val_pu=None, os_or_ts=None):
        self._require_prepared(estimator, X, os_or_ts)
        estimator.fit(
            X,
            y,
            class_prior=class_prior,
            validation_data=(self.support.X, self.support.labels),
            os_or_ts="os",
        )
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
            }
        )
        return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=estimator)


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
        estimator.fit(X, y, support_data=(self.support.X, self.support.labels))
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
        return RunTrajectory(epochs=[EpochRecord(epoch=1, metrics={})], model=estimator)
