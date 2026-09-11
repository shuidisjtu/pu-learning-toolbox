"""PU survey experiment layer (P0 pilot). See implementation_plan.md §1.4."""

from .bundle import DatasetBundle, DatasetPart, validate_bundle
from .datasets import (
    SurveyDatasetSpec,
    binaryize_survey_labels,
    prepare_survey_dataset,
    survey_dataset_catalog,
)
from .feature_adapter import (
    LeaderboardRunSpec,
    adapt_image_bundle_to_features,
    partition_fair_leaderboard_runs,
)
from .image import (
    SurveyImagePreprocessing,
    build_survey_image_augmentation,
    build_survey_image_encoder,
    fit_survey_image_preprocessing,
    transform_survey_images,
)
from .resources import aggregate_resource_usage
from .runner import ExperimentRunner
from .strategies import (
    CleanLabelGenerator,
    DeepFitTrainer,
    FitTrainer,
    ProtocolOA,
    ProtocolPA,
    SARLBEAGenerator,
    SARLBEBGenerator,
    SCARGenerator,
    SupervisedTrainer,
    select_threshold,
)
from .text import (
    SBERT_EMBEDDING_DIMENSION,
    SBERT_MODEL_NAME,
    encode_survey_texts,
)
from .training_views import TSOSBatchView, calibrate_ts_os_batch

__all__ = [
    "CleanLabelGenerator",
    "DatasetBundle",
    "DatasetPart",
    "DeepFitTrainer",
    "ExperimentRunner",
    "FitTrainer",
    "LeaderboardRunSpec",
    "ProtocolOA",
    "ProtocolPA",
    "SARLBEAGenerator",
    "SARLBEBGenerator",
    "SBERT_EMBEDDING_DIMENSION",
    "SBERT_MODEL_NAME",
    "SCARGenerator",
    "SupervisedTrainer",
    "SurveyDatasetSpec",
    "SurveyImagePreprocessing",
    "TSOSBatchView",
    "aggregate_resource_usage",
    "adapt_image_bundle_to_features",
    "binaryize_survey_labels",
    "build_survey_image_augmentation",
    "build_survey_image_encoder",
    "calibrate_ts_os_batch",
    "encode_survey_texts",
    "fit_survey_image_preprocessing",
    "partition_fair_leaderboard_runs",
    "prepare_survey_dataset",
    "select_threshold",
    "survey_dataset_catalog",
    "transform_survey_images",
    "validate_bundle",
]
