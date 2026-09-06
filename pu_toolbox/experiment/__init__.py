"""PU survey experiment layer (P0 pilot). See implementation_plan.md §1.4."""

from .bundle import DatasetBundle, DatasetPart, validate_bundle
from .datasets import (
    SurveyDatasetSpec,
    binaryize_survey_labels,
    prepare_survey_dataset,
    survey_dataset_catalog,
)
from .resources import aggregate_resource_usage
from .runner import ExperimentRunner
from .strategies import (
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

__all__ = [
    "DatasetBundle",
    "DatasetPart",
    "DeepFitTrainer",
    "ExperimentRunner",
    "FitTrainer",
    "ProtocolOA",
    "ProtocolPA",
    "SARLBEAGenerator",
    "SARLBEBGenerator",
    "SBERT_EMBEDDING_DIMENSION",
    "SBERT_MODEL_NAME",
    "SCARGenerator",
    "SupervisedTrainer",
    "SurveyDatasetSpec",
    "aggregate_resource_usage",
    "binaryize_survey_labels",
    "encode_survey_texts",
    "prepare_survey_dataset",
    "select_threshold",
    "survey_dataset_catalog",
    "validate_bundle",
]
