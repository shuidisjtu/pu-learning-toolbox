"""Public experiment API, loaded on demand without Survey process dependencies."""

from importlib import import_module

_EXPORTS = {
    "BATCH_IDENTITY_FIELDS": ("batch_evidence", "BATCH_IDENTITY_FIELDS"),
    "CleanLabelGenerator": ("strategies", "CleanLabelGenerator"),
    "DatasetBundle": ("bundle", "DatasetBundle"),
    "DatasetPart": ("bundle", "DatasetPart"),
    "DeepFitTrainer": ("strategies", "DeepFitTrainer"),
    "ExperimentRunner": ("runner", "ExperimentRunner"),
    "FitTrainer": ("strategies", "FitTrainer"),
    "GATE_MESSAGE_REASONS": ("survey_summary", "GATE_MESSAGE_REASONS"),
    "GATE_REFUSAL_REASONS": ("survey_summary", "GATE_REFUSAL_REASONS"),
    "LeaderboardRunSpec": ("feature_adapter", "LeaderboardRunSpec"),
    "ProtocolOA": ("strategies", "ProtocolOA"),
    "ProtocolPA": ("strategies", "ProtocolPA"),
    "REASONS": ("survey_summary", "REASONS"),
    "SARLBEAGenerator": ("strategies", "SARLBEAGenerator"),
    "SARLBEBGenerator": ("strategies", "SARLBEBGenerator"),
    "SBERT_EMBEDDING_DIMENSION": ("text", "SBERT_EMBEDDING_DIMENSION"),
    "SBERT_MODEL_NAME": ("text", "SBERT_MODEL_NAME"),
    "SCARGenerator": ("strategies", "SCARGenerator"),
    "STATUSES": ("survey_summary", "STATUSES"),
    "STATUS_PRIORITY": ("survey_summary", "STATUS_PRIORITY"),
    "SummaryError": ("survey_summary", "SummaryError"),
    "SupervisedTrainer": ("strategies", "SupervisedTrainer"),
    "SurveyDatasetSpec": ("datasets", "SurveyDatasetSpec"),
    "SurveyImagePreprocessing": ("image", "SurveyImagePreprocessing"),
    "TSOSBatchView": ("training_views", "TSOSBatchView"),
    "adapt_image_bundle_to_features": ("feature_adapter", "adapt_image_bundle_to_features"),
    "aggregate_metric": ("survey_summary", "aggregate_metric"),
    "aggregate_resource_usage": ("resources", "aggregate_resource_usage"),
    "binaryize_survey_labels": ("datasets", "binaryize_survey_labels"),
    "build_survey_image_augmentation": ("image", "build_survey_image_augmentation"),
    "build_survey_image_encoder": ("image", "build_survey_image_encoder"),
    "calibrate_ts_os_batch": ("training_views", "calibrate_ts_os_batch"),
    "choose_status": ("survey_summary", "choose_status"),
    "encode_survey_texts": ("text", "encode_survey_texts"),
    "fit_survey_image_preprocessing": ("image", "fit_survey_image_preprocessing"),
    "mechanism_of": ("survey_summary", "mechanism_of"),
    "normalize_c_token": ("survey_summary", "normalize_c_token"),
    "partition_fair_leaderboard_runs": ("feature_adapter", "partition_fair_leaderboard_runs"),
    "prepare_survey_dataset": ("datasets", "prepare_survey_dataset"),
    "reasons_for": ("survey_summary", "reasons_for"),
    "resolve_status": ("survey_summary", "resolve_status"),
    "row_key": ("survey_summary", "row_key"),
    "sample_std": ("survey_summary", "sample_std"),
    "select_threshold": ("strategies", "select_threshold"),
    "status_for_completeness": ("survey_summary", "status_for_completeness"),
    "status_for_gate_block": ("survey_summary", "status_for_gate_block"),
    "status_for_gate_error": ("survey_summary", "status_for_gate_error"),
    "status_for_refusal": ("survey_summary", "status_for_refusal"),
    "summarize_costs": ("survey_summary", "summarize_costs"),
    "survey_dataset_catalog": ("datasets", "survey_dataset_catalog"),
    "to_result_summary": ("survey_summary", "to_result_summary"),
    "transform_survey_images": ("image", "transform_survey_images"),
    "validate_bundle": ("bundle", "validate_bundle"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
